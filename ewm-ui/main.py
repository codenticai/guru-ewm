"""
ewm-ui — NiceGUI Frontend for Emerging World Models.

Generic UI shell that dynamically discovers EWM services from the gateway.
Provides:
  - Dashboard with service health status
  - OCR interface (image upload → process → display)
  - HLLSet pipeline tester
  - IPFS content browser
"""

import os
import logging

import httpx
from nicegui import ui, app

# ── Config ──────────────────────────────────────────────────────────
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://ewm-gateway:8000")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("ewm-ui")


# ── Health endpoint for Docker healthcheck ──────────────────────────
@ui.page("/health")
def health_page():
    return {"status": "ok", "service": "ewm-ui"}


# ═══════════════════════════════════════════════════════════════════════
# Dashboard — Home Page
# ═══════════════════════════════════════════════════════════════════════

@ui.page("/")
async def dashboard():
    ui.markdown("# Guru-EWM — Emerging World Models Platform")
    ui.markdown("### Service Status")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(f"{GATEWAY_URL}/health")
            health_data = r.json()
        except Exception as e:
            health_data = {"status": "gateway unreachable", "error": str(e)}

    # Overall status
    overall = health_data.get("status", "unknown")
    color = "green" if overall == "ok" else "orange" if overall == "degraded" else "red"
    ui.badge(f"Gateway: {overall}", color=color).props("size=lg")

    ui.separator()

    # Service cards
    services = health_data.get("services", {})
    for svc_name, svc_info in services.items():
        with ui.card():
            svc_status = svc_info.get("status", "unknown")
            svc_color = "green" if svc_status in ("ok", "healthy") else "red"
            ui.markdown(f"### {svc_name}")
            ui.badge(svc_status, color=svc_color)
            if svc_info.get("hllset_available") is not None:
                ui.label(f"HLLSet: {'available' if svc_info['hllset_available'] else 'unavailable'}")
            if svc_info.get("active_sessions") is not None:
                ui.label(f"Active sessions: {svc_info['active_sessions']}")

    ui.separator()
    ui.markdown("### Quick Actions")
    with ui.row():
        ui.button("OCR Pipeline", on_click=lambda: ui.navigate.to("/ocr"))
        ui.button("HLLSet Tester", on_click=lambda: ui.navigate.to("/hllset"))
        ui.button("Diagnostic Analysis", on_click=lambda: ui.navigate.to("/diagnose"))
        ui.button("IPFS Browser", on_click=lambda: ui.navigate.to("/ipfs"))


# ═══════════════════════════════════════════════════════════════════════
# OCR Interface
# ═══════════════════════════════════════════════════════════════════════

@ui.page("/ocr")
async def ocr_page():
    ui.markdown("## OCR Pipeline")
    ui.markdown("Upload text for HLLSet semantic compression or process via the gateway.")

    text_input = ui.textarea(
        "Enter encoding IDs or text to process...",
        value="enc10253 enc18278 enc50690 enc10325 enc1805 enc6579 enc18308 enc11347",
    ).props("rows=4")

    format_toggle = ui.toggle(["basic", "debruijn"], value="basic")

    result_display = ui.code("").props("readonly")

    async def process_text():
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                payload = {
                    "text": text_input.value,
                    "format": format_toggle.value,
                }
                r = await client.post(f"{GATEWAY_URL}/ocr/process", json=payload)
                data = r.json()
                result_display.set_content(str(data))
            except Exception as e:
                result_display.set_content(f"Error: {e}")

    ui.button("Process", on_click=process_text, icon="send")

    async def check_health():
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.get(f"{GATEWAY_URL}/health")
                result_display.set_content(str(r.json()))
            except Exception as e:
                result_display.set_content(f"Error: {e}")

    ui.button("Check Health", on_click=check_health, icon="healing")


# ═══════════════════════════════════════════════════════════════════════
# HLLSet Pipeline Tester
# ═══════════════════════════════════════════════════════════════════════

@ui.page("/hllset")
async def hllset_page():
    ui.markdown("## HLLSet Pipeline Tester")
    ui.markdown("Test the HLLSet semantic compression pipeline directly via the gateway.")

    text_input = ui.textarea(
        "Encoding IDs (space-separated)...",
        value="enc10253 enc18278 enc50690 enc10325 enc1805 enc6579",
    ).props("rows=3")

    gate_vocab = ui.textarea(
        "Gate vocabulary (optional, comma or space separated)...",
        value="enc10253 enc18278 enc50690 enc10325 enc1805 enc6579 enc18308 enc11347 enc9042 enc5061",
    ).props("rows=2")

    result_display = ui.code("").props("readonly")

    async def process():
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Set gate first
                vocab = gate_vocab.value.replace(",", " ").split()
                if vocab:
                    await client.post(
                        f"{GATEWAY_URL}/ocr/process",
                        json={"vocab": vocab},
                        params={"url_override": f"{GATEWAY_URL}/ocr/gate"},
                    )

                r = await client.post(f"{GATEWAY_URL}/ocr/process", json={
                    "text": text_input.value,
                    "format": "basic",
                })
                result_display.set_content(str(r.json()))
            except Exception as e:
                result_display.set_content(f"Error: {e}")

    ui.button("Run Pipeline", on_click=process, icon="play_arrow")


# ═══════════════════════════════════════════════════════════════════════
# IPFS Browser
# ═══════════════════════════════════════════════════════════════════════

@ui.page("/ipfs")
async def ipfs_page():
    ui.markdown("## IPFS Content Browser")
    ui.markdown("Browse content-addressed storage.")

    cid_input = ui.input("CID (Content Identifier)", placeholder="h:...")

    result_display = ui.code("").props("readonly")

    async def fetch():
        cid = cid_input.value.strip()
        if not cid:
            result_display.set_content("Enter a CID")
            return
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                r = await client.get(f"{GATEWAY_URL}/ipfs/{cid}")
                result_display.set_content(str(r.json()))
            except Exception as e:
                result_display.set_content(f"Error: {e}")

    ui.button("Fetch", on_click=fetch, icon="download")


# ═══════════════════════════════════════════════════════════════════════
# Diagnostic Analysis — file + instruction → diagnostic report
# ═══════════════════════════════════════════════════════════════════════

def _render_report(report: dict):
    if not isinstance(report, dict) or report.get("error"):
        ui.label(f"Error: {report}").classes("text-red-500")
        return
    with ui.card().classes("w-full"):
        ui.markdown(f"### {report.get('modality', '')} Report · engine `{report.get('engine', '')}`")
        patient = report.get("patient") or {}
        bits = []
        if patient.get("name"):
            bits.append(f"**{patient['name']}**")
        if patient.get("id"):
            bits.append(f"ID {patient['id']}")
        if patient.get("age") is not None:
            bits.append(f"{patient['age']} y")
        if patient.get("sex"):
            bits.append(patient["sex"])
        if bits:
            ui.markdown(" &nbsp;·&nbsp; ".join(bits))
        ui.badge(report.get("assessment", ""), color="teal").props("size=md")

        measurements = report.get("measurements") or {}
        if measurements:
            ui.separator()
            ui.markdown("**Measurements**")
            for k, v in measurements.items():
                ui.label(f"{k}: {v}")

        mf = report.get("measurement_findings") or []
        if mf:
            ui.markdown("**Measurement findings**")
            for f in mf:
                ui.label(f"[{f.get('severity')}] {f.get('note')}")

        findings = report.get("findings") or []
        if findings:
            ui.separator()
            ui.markdown("**Findings**")
            for f in findings:
                bss = f.get("bss")
                bss_s = f"{bss:.2f}" if isinstance(bss, (int, float)) else "-"
                ui.label(f"{bss_s} · {f.get('signal')} · {f.get('note')}")

        ui.separator()
        ui.markdown(f"**Recommendation:** {report.get('recommendation', '')}")
        if report.get("stored_cid"):
            ui.markdown(f"**CID:** `{report.get('stored_cid')}`")


@ui.page("/diagnose")
async def diagnose_page():
    ui.markdown("## Diagnostic Analysis")
    ui.markdown(
        "Attach a medical file (ECG image, X-ray, PDF) and an instruction. "
        "The diagnostic report is returned below."
    )

    instruction = ui.textarea(
        "Instruction (e.g. 'Analyze this ECG and produce a diagnostic report')...",
    ).props("rows=2 outline").classes("w-full")

    uploaded = {"name": None, "content": None}

    async def on_upload(e):
        uploaded["name"] = e.file.name
        uploaded["content"] = await e.file.read()
        ui.notify(f"Attached: {e.file.name}")

    ui.upload(
        on_upload=on_upload,
        auto_upload=True,
        multiple=False,
        label="Attach file",
    ).props("accept='.pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.dcm'").classes("w-full")

    result = ui.column().classes("w-full mt-4")

    async def analyze():
        if not uploaded["content"]:
            ui.notify("Please attach a file first", type="warning")
            return
        result.clear()
        with result:
            ui.spinner(size="lg")
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                files = {
                    "file": (uploaded["name"] or "file", uploaded["content"], "application/octet-stream")
                }
                data = {"instruction": instruction.value or "", "modality": "DOC"}
                r = await client.post(f"{GATEWAY_URL}/analyze/document", files=files, data=data)
            result.clear()
            with result:
                data = r.json()
                report = data.get("report") if isinstance(data, dict) else data
                _render_report(report)
        except Exception as e:
            result.clear()
            with result:
                ui.label(f"Error: {e}").classes("text-red-500")

    ui.button("Analyze", on_click=analyze, icon="medical_services")


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
        favicon="🧠",
    )
