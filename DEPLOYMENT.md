# Deployment Guide — Ubuntu + Docker

This guide deploys **Guru-EWM** on a fresh Ubuntu server using Docker Engine
and the Docker Compose plugin.

> All sources required to run the app live in this repository. The optional
> `hllset-next` / `hllset-cortex` services are **not** required — see
> [Optional services](#8-optional-hllset-services).

---

## 1. Requirements

- Ubuntu 22.04 LTS or 24.04 LTS (x86_64)
- At least **4 GB RAM** (the stack idles at ~2.4 GB; 8 GB recommended)
- ~10 GB free disk (images + models + data volumes)
- Outbound internet access on first run (Docker pulls + model download)

## 2. Install Docker Engine + Compose plugin

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

sudo systemctl enable --now docker
docker --version && docker compose version
```

Add your deploy user to the `docker` group (optional, avoids `sudo`):

```bash
sudo usermod -aG docker "$USER"
# log out and back in for it to take effect
```

## 3. Copy the code to the server

On the **server**, prepare the directory:

```bash
sudo mkdir -p /srv && sudo chown "$USER" /srv
```

Then copy the code from your **local machine** (pick one):

### Option A — rsync (recommended, also used for updates)

```bash
rsync -avz --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '*.pyc' --exclude '.pytest_cache' --exclude '.env' \
  /path/to/guru-ewm/ user@<server-ip>:/srv/guru-ewm/
```

### Option B — scp tarball

```bash
tar -czf guru-ewm.tar.gz \
  --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude='.env' guru-ewm
scp guru-ewm.tar.gz user@<server-ip>:/srv/
ssh user@<server-ip> 'cd /srv && tar -xzf guru-ewm.tar.gz && rm guru-ewm.tar.gz'
```

> If you also want the optional HLLSet services, copy `hllset-next` and
> `hllset-cortex` into `/srv/` and point `.env` at them
> (see [Optional services](#8-optional-hllset-services)).

> **Offline deploy?** Also copy the backup folder (the saved image `.tar` and
> volume `.tar.gz` files, e.g. from `D:\innovation\nanolm images`) to
> `/srv/guru-ewm/backup` — then use Option B in
> [Build & start](#5-build--start).

## 4. Configure

```bash
cp .env.example .env
nano .env
```

The defaults work out of the box. Commonly adjusted values:

| Variable | Default | Purpose |
|---|---|---|
| `UI_PORT` | `8080` | Web UI |
| `GATEWAY_PORT` | `8001` | API gateway |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `DEEPSEEK_OCR_GPU_ENABLED` | `false` | Enable GPU OCR passthrough (needs nvidia-container-toolkit) |

Do **not** commit `.env` — it's gitignored and may contain local settings.

## 5. Build & start

### Option A — online (build images from source)

```bash
docker compose build
docker compose up -d
```

### Option B — offline (use the saved images + data volumes)

Copy the backup folder (produced by `scripts/backup.ps1`) to the server, e.g.
`/srv/guru-ewm/backup`, then run:

```bash
BACKUP_DIR=/srv/guru-ewm/backup bash scripts/deploy-offline.sh
```

This loads the saved Docker images, restores the data volumes (IPFS content
store, NLP/medical snapshots, and the BiomedCLIP model cache — **no
re-ingesting**), builds only `ewm-ui` (its image was not saved), and starts
the stack.

### After either option

```bash
docker compose ps        # wait for all services to be "healthy"
docker compose logs -f   # watch startup (Ctrl+C to stop watching)
```

## 6. Verify

```bash
# UI
curl -fsS http://localhost:8080/health

# API gateway (aggregated status)
curl -fsS http://localhost:8001/health

# Individual services
curl -fsS http://localhost:9095/health    # nlp-model
curl -fsS http://localhost:9094/health    # medical-diagnostic
curl -fsS http://localhost:9093/health    # deepseek-ocr
```

Open `http://<server-ip>:8080` in a browser to use the chat UI.

### First-run note

`medical-diagnostic` downloads the BiomedCLIP model (~750 MB) on its first
image-classification request. It is cached in the `ewm-hf-model-cache` volume,
so this happens only once.

## 7. Resource expectations

Measured at idle:

| Service | Memory |
|---|---|
| `medical-diagnostic` | ~1.2 GiB (torch + BiomedCLIP) |
| `ipfs` | ~450 MiB |
| `nlp-model` | ~330 MiB |
| others | < 200 MiB each |
| **Total** | **~2.4 GiB** |

CPU is negligible at idle (~3%); it spikes during OCR and image classification.

## 8. Optional: HLLSet services

`hllset-next` and `hllset-cortex` are **optional**. Retrieval and diagnosis run
locally in Python; the lattice only adds content-addressed ingestion, and IPFS
snapshots already provide durability.

To enable them (requires the two projects, which are **not** in this repo):

```bash
# 1. Put the sources next to the repo and point .env at them:
#    HLLSET_NEXT_CONTEXT=/srv/hllset-next
#    HLLSET_CORTEX_CONTEXT=/srv/hllset-cortex
# 2. Start with the override file:
docker compose -f docker-compose.yml -f docker-compose.optional.yml up -d --build
```

## 9. Optional: UI resource badge

By default the UI does **not** mount the Docker socket (safe). To show live
CPU/RAM usage in the header:

```bash
# in .env
DOCKER_SOCK_MOUNT=/var/run/docker.sock
```

then:

```bash
docker compose up -d ewm-ui
```

## 10. Production hardening

1. **Reverse proxy + TLS** in front of the UI and gateway (nginx / Caddy):

   ```nginx
   # /etc/nginx/sites-available/guru-ewm  (example)
   server {
       listen 443 ssl;
       server_name your.domain.com;
       # ...ssl_certificate / ssl_certificate_key...

       location / {
           proxy_pass http://127.0.0.1:8080;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```

2. **Firewall** — expose only 80/443 publicly:

   ```bash
   sudo ufw allow OpenSSH
   sudo ufw allow 'Nginx Full'   # or 80,443
   sudo ufw enable
   ```

   Keep `8001`, `8080`, `9093-9095`, `5001`, `8081` bound to localhost or the
   Docker network only.

3. **Run as a service** (auto-start on boot). Create
   `/etc/systemd/system/guru-ewm.service`:

   ```ini
   [Unit]
   Description=Guru-EWM
   Requires=docker.service
   After=docker.service

   [Service]
   Type=oneshot
   RemainAfterExit=yes
   WorkingDirectory=/srv/guru-ewm
   ExecStart=/usr/bin/docker compose up -d
   ExecStop=/usr/bin/docker compose down

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   sudo systemctl enable --now guru-ewm
   ```

## 11. Updating

Copy the new code from your local machine using the same rsync/scp command as
[section 3](#3-copy-the-code-to-the-server), then rebuild:

```bash
cd /srv/guru-ewm
docker compose build
docker compose up -d
```

Named volumes (`ewm-*-data`, `ewm-hf-model-cache`) persist across updates, so
the knowledge snapshot and downloaded models are retained. To roll back,
re-copy the previous code and rebuild.

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| Container unhealthy / crash loop | `docker compose logs <service>` |
| "hllset-next unreachable" in logs | Normal — the lattice is optional; retrieval still works locally |
| First `/classify` times out | BiomedCLIP is downloading (~750 MB) — retry in a minute |
| Port already in use | Change the port in `.env` |
| Build fails resolving `python:3.11-slim-bookworm` | Docker Hub unreachable — check DNS/proxy |
| UI badge empty | Set `DOCKER_SOCK_MOUNT=/var/run/docker.sock` (opt-in) |

For the full stack teardown:

```bash
docker compose down            # stop containers, keep data
docker compose down -v         # stop AND delete data volumes (destructive)
```
