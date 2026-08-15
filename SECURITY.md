# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Report them privately to the maintainers. Include:

- A description of the issue and its potential impact.
- Steps to reproduce (or a proof of concept).
- Affected versions/components.

We will acknowledge receipt, investigate, and release a fix as soon as practical.

## Known design considerations

- The optional UI resource badge mounts the host Docker socket into `ewm-ui`. It is **disabled by default** (`/dev/null`). Only enable it (`DOCKER_SOCK_MOUNT`) on a trusted host.
- `ewm-ui` and `ewm-gateway` are intended to sit behind a reverse proxy with TLS when exposed publicly.
- The diagnostic features are research software and must not be used for real clinical decisions.
