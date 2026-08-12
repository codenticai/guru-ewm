"""Full-pipeline test: ECG image -> gateway /analyze/ecg -> OCR -> hllset lattice."""
import json
import sys
import urllib.request
import uuid

IMAGE_PATH = r"d:\innovation\guru-ewm\docs\ECG.jpeg"
GATEWAY = "http://localhost:8000"


def multipart_post(url, fields, file_field, filename, content_type):
    boundary = "----ewm" + uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    with open(IMAGE_PATH, "rb") as f:
        data = f.read()
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode()
    body += f"Content-Type: {content_type}\r\n\r\n".encode()
    body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read().decode()


def main():
    status, text = multipart_post(
        f"{GATEWAY}/analyze/ecg",
        fields={"modality": "ECG"},
        file_field="file",
        filename="ECG.jpeg",
        content_type="image/jpeg",
    )
    print("HTTP", status)
    if status != 200:
        print(text[:2000])
        sys.exit(1)

    out = json.loads(text)
    print("=== OCR step ===")
    ocr = out.get("ocr", {})
    print("source:", ocr.get("source"))
    print("mode:", ocr.get("mode"))
    print("notice:", ocr.get("notice"))
    print("extracted_text:")
    print(ocr.get("extracted_text", ""))
    print()

    report = out.get("report", {})
    print("=== Report ===")
    print("engine:", report.get("engine"))
    print("modality:", report.get("modality"))
    print("assessment:", report.get("assessment"))
    print("measurements:", json.dumps(report.get("measurements")))
    print("measurement_findings:", json.dumps(report.get("measurement_findings")))
    print()
    print("lattice FINDINGS (ranked by BSS):")
    for f in report.get("findings", []):
        print(f"  {f.get('bss'):.3f}  [{f.get('severity'):8s}] {f.get('signal')}")
    print()
    print("stored_cid:", report.get("stored_cid"))


if __name__ == "__main__":
    main()
