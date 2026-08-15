"""
ewm-ui — NiceGUI frontend for Guru-EWM (NanoLM).

End-user interface: a single page where the public can select a model
(NLP, OCR, or Diagnose) and ask questions in a chat.
"""

import os
import logging
import re
import uuid
from datetime import datetime

import httpx
from nicegui import ui, app

# ── Config ──────────────────────────────────────────────────────────
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://ewm-gateway:8000")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("ewm-ui")

# Serve local assets (the brand logo) to the browser.
_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.jpeg")
LOGO_URL = app.add_static_file(
    local_file=_LOGO_PATH,
    url_path="/static/logo.jpeg",
)

# ── Resource usage (Docker Engine API over the host socket) ─────────
DOCKER_SOCK = os.environ.get("DOCKER_SOCK", "/var/run/docker.sock")
_last_cpu: dict = {}


def _fmt_gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f}"


async def get_resource_usage():
    """Aggregate CPU % and memory usage across the ewm-* containers."""
    info = {"cpu": 0.0, "mem_bytes": 0, "mem_total": 0}
    try:
        transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCK)
        async with httpx.AsyncClient(transport=transport, base_url="http://docker",
                                     timeout=5.0) as client:
            info["mem_total"] = int((await client.get("/info")).json().get("MemTotal", 0))
            containers = (await client.get("/containers/json")).json()
            for ct in containers:
                name = (ct.get("Names") or [""])[0].lstrip("/")
                if not name.startswith("ewm-"):
                    continue
                try:
                    stats = (await client.get(
                        f"/containers/{ct['Id']}/stats", params={"stream": "false"},
                    )).json()
                except Exception:
                    continue

                cpu_stats = stats.get("cpu_stats") or {}
                precpu = stats.get("precpu_stats") or {}
                total_usage = (cpu_stats.get("cpu_usage") or {}).get("total_usage", 0)
                system_usage = cpu_stats.get("system_cpu_usage", 0)
                online = (
                    cpu_stats.get("online_cpus")
                    or len((cpu_stats.get("cpu_usage") or {}).get("percpu_usage") or [0])
                    or 1
                )
                prev = _last_cpu.get(ct["Id"])
                if prev and system_usage > prev[1] and total_usage > prev[0]:
                    cpu_delta = total_usage - prev[0]
                    sys_delta = system_usage - prev[1]
                    if sys_delta > 0:
                        info["cpu"] += (cpu_delta / sys_delta) * online * 100.0
                _last_cpu[ct["Id"]] = (total_usage, system_usage)

                mem = stats.get("memory_stats") or {}
                usage = mem.get("usage", 0)
                cache = (mem.get("stats") or {}).get("cache") or 0
                info["mem_bytes"] += max(0, usage - cache)
        info["cpu"] = round(info["cpu"], 1)
        return info
    except Exception:
        return None

# ── Global styling ───────────────────────────────────────────────────
ui.add_head_html("""
<style>
  :root {
    --bg: #f6f7f9;
    --surface: #ffffff;
    --surface-2: #f1f3f6;
    --border: #e6e8ee;
    --text: #1f2937;
    --muted: #6b7280;
    --accent: #2563eb;
    --user-bubble: #2563eb;
  }
  html, body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  }

  /* ── Header ─────────────────────────────────────────────────────── */
  .app-header {
    position: sticky;
    top: 0;
    z-index: 50;
    width: 100%;
    background: #ffffff;
    border-bottom: 1px solid var(--border);
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
  }
  .app-header-inner {
    max-width: 860px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 11px 20px;
  }
  .brand-logo {
    width: 149px;
    height: 36px;
    border-radius: 8px;
    object-fit: contain;
  }
  .brand-title {
    font-weight: 700;
    font-size: 1rem;
    line-height: 1.15;
    color: var(--text);
    letter-spacing: -0.01em;
  }
  .brand-subtitle {
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.03em;
  }
  .brand-resources {
    font-size: 0.68rem;
    font-weight: 500;
    color: var(--muted);
    white-space: nowrap;
  }

  /* ── Chat ───────────────────────────────────────────────────────── */
  .chat-scroll {
    width: 100%;
    max-width: 860px;
    margin: 0 auto;
    padding: 28px 20px 44px 20px;
    min-height: 62vh;
  }
  .q-message { max-width: 100%; }
  .q-message-name {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 3px;
  }
  .q-message-sent .q-message-name { display: none; }
  .q-message-text-content {
    padding: 10px 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    font-size: 0.95rem;
    word-break: break-word;
  }
  /* Assistant: clean white card */
  .q-message-received .q-message-text-content {
    background: #ffffff;
    color: var(--text);
    border: 1px solid var(--border);
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05);
    border-radius: 14px 14px 14px 4px;
  }
  /* User: right-aligned blue bubble */
  .q-message-sent .q-message-text-content {
    background: var(--user-bubble);
    color: #ffffff;
    border-radius: 14px 14px 4px 14px;
    box-shadow: 0 1px 4px rgba(37, 99, 235, 0.25);
  }
  .q-message-stamp { font-size: 0.66rem; color: var(--muted); opacity: 0.8; margin-top: 4px; }

  /* ── Welcome chips ──────────────────────────────────────────────── */
  .welcome-actions { padding-left: 2px; }
  .suggestion-heading {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 10px;
  }
  .suggestion-chip {
    background: #ffffff !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    font-size: 0.82rem !important;
  }
  .suggestion-chip:hover { background: var(--surface-2) !important; }
  .sample-chip {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
    border: 1px solid #dbeafe !important;
    border-radius: 999px !important;
    font-size: 0.82rem !important;
  }

  /* ── Input bar ──────────────────────────────────────────────────── */
  .input-bar {
    position: sticky;
    bottom: 0;
    z-index: 50;
    width: 100%;
    background: linear-gradient(to top, var(--bg) 82%, rgba(246, 247, 249, 0));
    padding: 10px 20px 18px 20px;
    box-sizing: border-box;
  }
  .input-shell {
    max-width: 820px;
    margin: 0 auto;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 6px 22px rgba(16, 24, 40, 0.07);
    padding: 8px 10px;
  }
  .input-shell textarea { min-height: 44px; max-height: 180px; font-size: 0.95rem; color: var(--text); }
  .input-shell .q-field__native, .input-shell .q-field__label { color: var(--text); }
  .input-hint {
    max-width: 820px;
    margin: 8px auto 0 auto;
    text-align: center;
    font-size: 0.7rem;
    color: var(--muted);
  }
  .copyright {
    max-width: 820px;
    margin: 2px auto 0 auto;
    text-align: center;
    font-size: 0.66rem;
    color: var(--muted);
    opacity: 0.85;
  }

  /* ── Scrollbar ──────────────────────────────────────────────────── */
  ::-webkit-scrollbar { width: 10px; }
  ::-webkit-scrollbar-thumb { background: #cbd2dc; border-radius: 8px; }
  ::-webkit-scrollbar-thumb:hover { background: #b3bcc9; }
</style>
""", shared=True)


# ── Health endpoint for Docker healthcheck ──────────────────────────
@ui.page("/health")
def health_page():
    return {"status": "ok", "service": "ewm-ui"}


# ── Report formatter (OCR / diagnostic model output → chat text) ─────
def _report_to_text(report: dict) -> str:
    if not isinstance(report, dict):
        return str(report)
    if report.get("error"):
        return f"Error: {report['error']}"
    lines = [f"{report.get('modality', '')} report · engine {report.get('engine', '')}"]
    measurements = report.get("measurements") or {}
    if measurements:
        lines.append("")
        lines.append("Measurements:")
        for k, v in measurements.items():
            lines.append(f"  • {k}: {v}")
    findings = report.get("findings") or []
    if findings:
        lines.append("")
        lines.append("Findings:")
        for f in findings:
            lines.append(f"  • {f.get('signal')} [{f.get('severity')}] — {f.get('note')}")
    if report.get("assessment"):
        lines.append("")
        lines.append(f"Assessment: {report['assessment']}")
    if report.get("recommendation"):
        lines.append(f"Recommendation: {report['recommendation']}")
    return "\n".join(lines)


def _safe_json(r) -> dict:
    """Parse a JSON response; fall back to a friendly error dict on failure."""
    try:
        return r.json()
    except Exception:
        return {"error": f"Service returned an invalid response (HTTP {r.status_code})."}


# ── Chat interface ───────────────────────────────────────────────────


def _is_diagnostic_request(text: str) -> bool:
    """True when the message reads as a medical-analysis instruction.

    A diagnostic keyword that is explicitly negated ("without diagnosis",
    "no diagnosis", "don't diagnose") does NOT count — the user is asking
    for OCR-only extraction, not a diagnosis."""
    s = (text or "").lower()
    negation = re.compile(r"\b(without|no|not|dont|don|skip)\b")
    for kw in ("diagnos", "analy"):
        for m in re.finditer(kw, s):
            before = s[max(0, m.start() - 30):m.start()]
            if not negation.search(before):
                return True
    return False


def _is_negated_diagnostic_request(text: str) -> bool:
    """True when the only diagnostic keywords are negated ("without diagnosis")
    — the user wants OCR-only extraction, not a diagnosis."""
    s = (text or "").lower()
    negation = re.compile(r"\b(without|no|not|dont|don|skip)\b")
    found = False
    for kw in ("diagnos", "analy"):
        for m in re.finditer(kw, s):
            found = True
            before = s[max(0, m.start() - 30):m.start()]
            if not negation.search(before):
                return False
    return found


def scroll_to_bottom() -> None:
    """Scroll the chat view to the bottom so the latest message is visible."""
    ui.run_javascript("""
      setTimeout(() => {
        window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' });
      }, 50);
    """)


@ui.page("/")
async def index():
    state = {"session_id": uuid.uuid4().hex, "drafts": [], "draft_idx": 0}

    async def send():
        q = (query.value or "").strip()
        if not q and not attached["content"]:
            return
        with chat:
            ui.chat_message(
                q or f"📎 {attached['name']}",
                sent=True,
                stamp=datetime.now().strftime("%H:%M"),
            )
        scroll_to_bottom()
        query.value = ""
        if q:
            state["drafts"].append(q)
        state["draft_idx"] = len(state["drafts"])
        with chat:
            spinner = ui.spinner(size="sm")
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                mode = model.value  # "NLP" | "OCR" | "Diagnose"
                # Diagnose mode always diagnoses; NLP mode diagnoses only when
                # the query asks to; OCR mode (or a negated request) is OCR-only.
                diagnostic = (mode == "Diagnose") or (mode == "NLP" and _is_diagnostic_request(q))
                ocr_only = (mode == "OCR") or (mode == "NLP" and _is_negated_diagnostic_request(q))

                if attached["content"]:
                    files = {
                        "file": (attached["name"] or "file", attached["content"], "application/octet-stream")
                    }
                    if diagnostic:
                        is_image = (attached["name"] or "").lower().endswith(
                            (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
                             ".webp", ".jfif", ".gif", ".avif", ".heic", ".heif")
                        )
                        is_knee = any(k in q.lower() for k in ("knee", "mri", "meniscus", "acl", "ligament"))
                        if is_image and is_knee:
                            r = await client.post(
                                f"{GATEWAY_URL}/analyze/knee",
                                files=files,
                                data={"instruction": q},
                            )
                        elif is_image:
                            r = await client.post(
                                f"{GATEWAY_URL}/analyze/image",
                                files=files,
                                data={"instruction": q},
                            )
                        else:
                            r = await client.post(
                                f"{GATEWAY_URL}/analyze/document",
                                files=files,
                                data={"instruction": q, "modality": "ECG"},
                            )
                        data = _safe_json(r)
                        if data.get("error") and "report" not in data:
                            reply = f"Error: {data['error']}"
                        else:
                            report = data.get("report") if isinstance(data, dict) else data
                            reply = _report_to_text(report)
                            # Document analysis: also surface the extracted contents.
                            ocr_text = ((data.get("ocr") or {}).get("extracted_text") or "").strip()
                            if ocr_text:
                                reply = f"Document contents:\n\n{ocr_text}\n\n{reply}"
                    elif ocr_only:
                        # deepseek OCR only: extract text and return it verbatim.
                        r = await client.post(f"{GATEWAY_URL}/ocr/extract", files=files)
                        data = _safe_json(r)
                        if data.get("error"):
                            reply = f"Error: {data['error']}"
                        else:
                            text = data.get("page_text") or data.get("text") or ""
                            reply = f"Extracted text:\n\n{text}" if text else "Sorry, I couldn't extract any text from that file."
                    else:
                        # NLP mode + file: OCR the document, then chat about it.
                        r = await client.post(f"{GATEWAY_URL}/ocr/extract", files=files)
                        data = _safe_json(r)
                        if data.get("error"):
                            reply = f"Error: {data['error']}"
                        else:
                            text = data.get("page_text") or data.get("text") or ""
                            if not text:
                                reply = "Sorry, I couldn't extract any text from that file."
                            else:
                                r2 = await client.post(
                                    f"{GATEWAY_URL}/chat",
                                    json={"message": f"{q}\n\n{text}" if q else text,
                                          "session_id": state["session_id"]},
                                )
                                data2 = r2.json()
                                reply = f"Error: {data2.get('error')}" if data2.get("error") else data2.get("reply", "")
                    attached["name"] = None
                    attached["content"] = None
                elif diagnostic:
                    if re.search(r"\b(document|file|upload|attachment|attach)\b", q.lower()):
                        reply = ("Please attach the document or image — I'll extract its contents "
                                 "and diagnose any findings.")
                    else:
                        r = await client.post(
                            f"{GATEWAY_URL}/analyze/text",
                            json={"text": q, "modality": "ECG", "instruction": q},
                        )
                        reply = _report_to_text(_safe_json(r))
                elif ocr_only:
                    reply = "OCR mode is ready — attach a document or image and I'll extract its text."
                else:
                    r = await client.post(
                        f"{GATEWAY_URL}/chat",
                        json={"message": q, "session_id": state["session_id"]},
                    )
                    data = r.json()
                    reply = f"Error: {data.get('error')}" if data.get("error") else data.get("reply", "")
        except Exception as e:
            reply = f"Error: {e}"
        try:
            spinner.delete()
        except Exception:
            pass  # spinner may already be gone (e.g. New chat cleared it mid-flight)
        with chat:
            ui.chat_message(reply, name="NanoLM", stamp=datetime.now().strftime("%H:%M"))
        scroll_to_bottom()

    def recall_previous() -> None:
        drafts = state["drafts"]
        idx = state["draft_idx"]
        if idx > 0:
            idx -= 1
            state["draft_idx"] = idx
            query.value = drafts[idx]

    def recall_next() -> None:
        drafts = state["drafts"]
        idx = state["draft_idx"]
        if idx < len(drafts):
            idx += 1
            state["draft_idx"] = idx
            query.value = drafts[idx] if idx < len(drafts) else ""

    def render_welcome() -> None:
        with chat:
            ui.chat_message(
                "Welcome to Guru Emerging World Model!\n"
                "I'm NanoLM, your AI assistant for NLP, OCR, and clinical analysis. "
                "Ask a question or attach a file to get started.",
                name="NanoLM",
                stamp=datetime.now().strftime("%H:%M"),
            )
        scroll_to_bottom()

    def new_chat() -> None:
        state["session_id"] = uuid.uuid4().hex
        state["drafts"] = []
        state["draft_idx"] = 0
        chat.clear()
        render_welcome()

    # Header
    with ui.element("div").classes("app-header"):
        with ui.element("div").classes("app-header-inner"):
            with ui.row().classes("items-center gap-3"):
                ui.image(LOGO_URL).classes("brand-logo")
                with ui.column().classes("gap-0"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label("Guru Emerging World Model").classes("brand-title")
                        resources = ui.label("").classes("brand-resources")
                    ui.label("NanoLM AI Assistant").classes("brand-subtitle")
            with ui.row().classes("items-center gap-2"):
                ui.button(icon="add_comment", on_click=new_chat).props("round flat color=white")

    async def refresh_resources():
        info = await get_resource_usage()
        if info:
            resources.set_text(
                f"CPU {info['cpu']:.1f}% · RAM "
                f"{_fmt_gb(info['mem_bytes'])} / {_fmt_gb(info['mem_total'])} GB"
            )
        else:
            resources.set_text("")

    ui.timer(5.0, refresh_resources)
    ui.timer(0.3, refresh_resources, once=True)

    chat = ui.column().classes("w-full chat-scroll gap-2")

    with ui.element("div").classes("input-bar"):
        with ui.element("div").classes("input-shell"):
            with ui.row().classes("w-full items-center gap-2"):
                model = ui.select(["NLP", "OCR", "Diagnose"], value="NLP").props("dense borderless").classes("w-28")

                attached = {"name": None, "content": None}

                async def on_upload(e):
                    attached["name"] = e.file.name
                    attached["content"] = await e.file.read()
                    ui.notify(f"Attached: {e.file.name}", type="positive")

                upload = ui.upload(
                    on_upload=on_upload, auto_upload=True, multiple=False,
                ).props("hidden")
                ui.button(icon="attach_file", on_click=lambda: upload.run_method("pickFiles")).props("round flat color=primary")

                query = ui.textarea(placeholder="Ask a question…").props("autogrow borderless").classes("flex-grow")

                ui.button(icon="send", on_click=send).props("round unelevated color=primary")
                query.on("keydown.enter.prevent", send)
                query.on("keydown.up.prevent", recall_previous)
                query.on("keydown.down.prevent", recall_next)
        ui.label("NanoLM is a platform where NLP, OCR, and any other model can be deployed.").classes("input-hint")
        ui.label("© 2026 Guru-EWM contributors · MIT License").classes("copyright")

    render_welcome()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ in {"__main__", "__mp_main__"}:
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"ewm-ui starting on port {port}")
    ui.run(
        host="0.0.0.0",
        port=port,
        title="Guru-EWM",
        favicon=_LOGO_PATH,
    )
