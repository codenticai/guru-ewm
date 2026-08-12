"""Produce the formal test report from the ECG image's ground-truth text."""
import json
import urllib.request

# Ground-truth text extracted from docs/ECG.jpeg (the 12-lead report)
ECG_TEXT = """ID: 10010001196275
Name: Baibhav
Age: 13 Years
Gender: Male
22-07-2026 01:51:33 PM
Vent. Rate 67 bpm
PR Interval 138 ms
QRS Duration 86 ms
QT/QTc Interval 366/378 ms
P/QRS/T Axes 57/72/60 deg
QTc/Hodges
Interpretation based on pediatric criteria:
Sinus rhythm
Normal ECG
Unconfirmed Diagnosis
I II III aVR aVL aVF V1 V2 V3 V4 V5 V6
25 mm/s 10 mm/mV 50 Hz BDR 20 Hz"""


def post(url, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read().decode())


if __name__ == "__main__":
    r = post(
        "http://localhost:9094/analyze/hllset",
        {
            "text": ECG_TEXT,
            "modality": "ECG",
            "patient": {"name": "Baibhav", "age": 13, "sex": "Male", "id": "10010001196275"},
        },
    )
    print(json.dumps(r, indent=2, ensure_ascii=False))
