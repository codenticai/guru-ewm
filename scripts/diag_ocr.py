"""Diagnose raw Tesseract output for the ECG image."""
import json
import urllib.request
import uuid

p = r"d:\innovation\guru-ewm\docs\ECG.jpeg"
data = open(p, "rb").read()
b = "----x" + uuid.uuid4().hex
body = (
    '--{b}\r\nContent-Disposition: form-data; name="file"; filename="e.jpg"\r\n'
    'Content-Type: image/jpeg\r\n\r\n'.format(b=b).encode()
    + data
    + ("\r\n--{b}--\r\n".format(b=b)).encode()
)
req = urllib.request.Request(
    "http://localhost:9093/ocr/upload",
    data=body,
    headers={"Content-Type": "multipart/form-data; boundary={}".format(b)},
)
r = json.loads(urllib.request.urlopen(req).read().decode())
t = (r.get("page_text") or r.get("text") or "").lower()
for word in ["bradycardia", "tachycardia", "wave", "borderline", "sinus", "normal", "rhythm", "unconfirmed"]:
    print(word, "->", word in t)
print()
print("---- full raw text ----")
print(r.get("page_text") or r.get("text"))
