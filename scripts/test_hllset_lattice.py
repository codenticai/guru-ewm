"""End-to-end test: ECG text -> hllset-next lattice -> structured report."""
import json
import urllib.request

ECG_TEXT = """ID: 10010001196275  22-07-2026 01:51:33 PM
Name: Baibhav
Age: 13 Years
Gender: Male
Vent. Rate      67 bpm
PR Interval     138 ms
QRS Duration    86 ms
QT/QTc Interval 366/378 ms
P/QRS/T Axes    57/72/60 deg
Interpretation based on pediatric criteria:
Sinus rhythm
Normal ECG
Unconfirmed Diagnosis"""


def post(url, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read().decode())


def get(url):
    return json.loads(urllib.request.urlopen(url).read().decode())


if __name__ == "__main__":
    print("=== hllset-next health ===")
    print(get("http://localhost:9090/api/v1/health"))
    print()

    print("=== ingest corpus ===")
    ing = post("http://localhost:9094/hllset/ingest", {})
    print({"cards": ing["cards"], "ingested": ing["ingested"]})
    print()

    print("=== /analyze/hllset (ECG report) ===")
    r = post(
        "http://localhost:9094/analyze/hllset",
        {
            "text": ECG_TEXT,
            "modality": "ECG",
            "patient": {"name": "Baibhav", "age": 13, "sex": "Male", "id": "10010001196275"},
        },
    )
    for k in ["engine", "modality", "assessment", "query_stats"]:
        print(f"{k}: {json.dumps(r.get(k))}")
    print("measurements:", json.dumps(r.get("measurements")))
    print()
    print("FINDINGS (ranked by BSS):")
    for f in r["findings"]:
        print(f"  {f['bss']:.3f}  [{f['severity']:8s}] {f['signal']:40s} {f['note'][:64]}")
    print()
    print("stored_cid:", r.get("stored_cid"))

    print()
    print("=== negative control: ST-elevation MI text ===")
    stemi = (
        "Vent rate 112 bpm PR 140 QRS 92 QT 400. "
        "ST elevation in leads II III aVF. "
        "Sinus tachycardia with inferior ST elevation, possible acute inferior "
        "myocardial infarction."
    )
    r2 = post("http://localhost:9094/analyze/hllset", {"text": stemi, "modality": "ECG"})
    print("assessment:", r2["assessment"])
    for f in r2["findings"][:5]:
        print(f"  {f['bss']:.3f}  [{f['severity']:8s}] {f['signal']}")
