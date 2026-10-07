"""openrouter_decisions_audit58.py - label the 58 audit reports with OpenRouter DECISION models (7 Oct).

Decision models (inception/mercury-decide:free, respan/span-01-lite:free) do not chat: POST /api/alpha/decisions
takes a `state` and typed questions and returns calibrated probabilities. Each finding is one yes/no ("noul")
question whose instructions and true/false criteria are the panel definitions from data_code/03_label_reports.py's
SYSTEM_SOFT (the prompt behind Qwen's 0.9114). --batch reports share one request (state = {"R1": text, ...},
questions "R1__ACL", ...) because free keys allow 50 requests a day.

Writes data/03_report_labels_dec-<slug>_validate58.csv and appends to data/stats/openrouter_audit58_results.csv.
Keys ONLY from the environment (OPENROUTER_API_KEYS=k1,k2).

    python data_code/label_v2/openrouter_decisions_audit58.py --models inception/mercury-decide:free --smoke
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
D = ROOT / "data"
URL = "https://openrouter.ai/api/alpha/decisions"
RES = D / "stats" / "openrouter_audit58_results.csv"
RAW = D / "stats" / "openrouter_raw"
CTX = ("The reference labels come from an expert panel that re-read the IMAGES, so judge what the knee most likely "
       "shows given the report. A report silent about a finding does not mean absent: fall back on base rates and "
       "indirect evidence.")
Q = {
    "ACL": ("anterior cruciate ligament tear/rupture, any grade including partial and interstitial",
            "mucoid degeneration only, intact or normal ACL"),
    "MCL": ("medial collateral ligament injury, including grade 1-2 sprain (not the LCL)", "intact MCL"),
    "Medial Meniscus": ("a TEAR of the medial meniscus reaching the articular surface",
                        "intact, or degenerative signal not reaching the surface, or post-resection without new tear"),
    "Lateral Meniscus": ("a TEAR of the lateral meniscus reaching the articular surface",
                         "intact, or degenerative signal not reaching the surface, or post-resection without new tear"),
    "Medial OA": ("osteoarthritis of the medial tibiofemoral compartment: substantial cartilage loss (grade 3-4, "
                  "full-thickness defect, bone-on-bone); tricompartmental OA counts",
                  "osteophytes alone, mild fissuring, or normal cartilage"),
    "Lateral OA": ("osteoarthritis of the lateral tibiofemoral compartment: substantial cartilage loss (grade 3-4, "
                   "full-thickness defect, bone-on-bone); tricompartmental OA counts",
                   "osteophytes alone, mild fissuring, or normal cartilage"),
    "PF OA": ("patellofemoral osteoarthritis (patella, facets, trochlea): substantial cartilage loss; "
              "tricompartmental OA counts", "osteophytes alone, mild fissuring, or normal cartilage"),
    "Effusion": ("a MODERATE or LARGE joint effusion, or haemarthrosis (unqualified 'joint effusion' counts)",
                 "no, trace, minimal, physiologic or small effusion"),
    "Synovitis": ("synovitis: synovial thickening/enhancement, reactive synovitis, hoffitis (common in advanced OA "
                  "with effusion even if unmentioned)", "no synovial abnormality"),
    "Baker's": ("a popliteal (Baker's) cyst", "no popliteal cyst"),
    "Contusion": ("bone bruise / bone marrow oedema of TRAUMATIC origin, or muscle contusion",
                  "no oedema, or subchondral oedema beside a degenerative cartilage defect, or cystic change"),
    "Fracture": ("a true fracture of bone, including insufficiency and avulsion fractures",
                 "no fracture; an osteochondral defect or OCD lesion alone"),
}
T = list(Q)


def questions(n):
    out = {}
    for i in range(n):
        for t, (yes, no) in Q.items():
            out[f"R{i + 1}__{re.sub(r'[^A-Za-z]', '', t)}"] = {
                "type": "noul", "instructions": f"Report R{i + 1}: does this knee have {t}? {CTX}",
                "criteria": {"true": yes, "false": no}}
    return out


def call(model, key, texts):
    body = {"model": model, "state": {f"R{i + 1}": t for i, t in enumerate(texts)}, "questions": questions(len(texts))}
    for attempt in range(3):
        r = requests.post(URL, json=body, timeout=600, headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 200:
            return r.json(), 200
        if r.status_code in (429, 502, 503) and attempt < 2:
            time.sleep(30 * (attempt + 1))
            continue
        return {"error": r.text[:400]}, r.status_code
    return {}, -1


def run(model, key, ids, texts, batch, smoke):
    slug = re.sub(r"[/:]", "_", model)
    RAW.mkdir(parents=True, exist_ok=True)
    chunks = [list(range(i, min(i + batch, len(ids)))) for i in range(0, len(ids), batch)]
    chunks = chunks[:1] if smoke else chunks
    P = np.full((len(ids), len(T)), np.nan)
    t0, calls, errs, tin, tout = time.time(), 0, [], 0, 0
    with open(RAW / f"dec-{slug}.jsonl", "a", encoding="utf-8") as fr:
        for ch in chunks:
            j, code = call(model, key, [texts[i] for i in ch])
            calls += 1
            fr.write(json.dumps({"ids": [ids[i] for i in ch], "status": code, "reply": j}) + "\n")
            if code != 200:
                errs.append(f"{code}: {str(j.get('error'))[:150]}")
                continue
            tin += int(j.get("usage", {}).get("input_tokens") or 0)
            tout += int(j.get("usage", {}).get("output_tokens") or 0)
            a = j.get("answers", {})
            for k, i in enumerate(ch):
                for c, t in enumerate(T):
                    v = a.get(f"R{k + 1}__{re.sub(r'[^A-Za-z]', '', t)}", {})
                    if "noul" in v:
                        P[i, c] = float(v["noul"])
    return slug, P, {"calls": calls, "tok_in": tin, "tok_out": tout, "secs": round(time.time() - t0), "errors": errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    keys = [k.strip() for k in os.environ.get("OPENROUTER_API_KEYS", "").split(",") if k.strip()]
    aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
    ids = list(aud.index)
    en = pd.read_csv(D / "02_reports_en.csv").drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    texts = [str(en.loc[s, "Report_en"]) for s in ids]
    Y = (aud.loc[ids, ["official_" + t for t in T]].values >= 0.5).astype(int)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    with ThreadPoolExecutor(len(keys)) as ex:
        futs = [ex.submit(run, m, keys[i % len(keys)], ids, texts, a.batch, a.smoke) for i, m in enumerate(models)]
        for m, f in zip(models, futs):
            slug, P, info = f.result()
            ok = ~np.isnan(P).any(1)
            if a.smoke:
                print(f"SMOKE {m}: {ok.sum()} reports complete, {info}")
                if ok.any():
                    print("  first:", dict(zip(T, np.round(P[np.argmax(ok)], 2))))
                continue
            per = {t: (roc_auc_score(Y[ok, c], P[ok, c]) if 0 < Y[ok, c].sum() < ok.sum() else np.nan)
                   for c, t in enumerate(T)}
            mac = float(np.nanmean(list(per.values()))) if ok.sum() else np.nan
            print(f"{m}: macro AUROC {mac:.4f} on {ok.sum()}/58 | {info}")
            if ok.all():
                pd.DataFrame(P, index=pd.Index(ids, name="StudyInstanceUID"), columns=T).reset_index().to_csv(
                    D / f"03_report_labels_dec-{slug}_validate58.csv", index=False)
            row = {"model": m, "macro_auroc": round(mac, 4), "parsed": int(ok.sum()), "batch": a.batch, **info,
                   **{t: round(v, 4) for t, v in per.items()}, "when": time.strftime("%Y-%m-%d %H:%M")}
            row["errors"] = " | ".join(info["errors"])[:300]
            pd.DataFrame([row]).to_csv(RES, mode="a", header=not RES.exists(), index=False)


if __name__ == "__main__":
    main()
