"""Test the new /analyze/document path with file + instruction."""
import json
import urllib.request
import uuid

IMG = r"d:\innovation\guru-ewm\docs\ECG.jpeg"


def multipart_post(url, fields, file_path, filename, content_type):
    boundary = "----ewm" + uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    data = open(file_path, "rb").read()
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += f"Content-Type: {content_type}\r\n\r\n".encode()
    body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.status, json.loads(resp.read().decode())


if __name__ == "__main__":
    status, out = multipart_post(
        "http://localhost:8000/analyze/document",
        fields={"instruction": "Analyze this ECG and produce a diagnostic report", "modality": "DOC"},
        file_path=IMG,
        filename="ECG.jpeg",
        content_type="image/jpeg",
    )
    print("HTTP", status)
    rep = out.get("report", {})
    print("modality (should be ECG via instruction):", rep.get("modality"))
    print("instruction:", rep.get("instruction"))
    print("engine:", rep.get("engine"))
    print("assessment:", rep.get("assessment"))
    print("measurements:", json.dumps(rep.get("measurements")))
    print("findings:", [f["signal"] for f in rep.get("findings", [])])
