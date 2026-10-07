"""openrouter_label_audit58.py - label the 58 audit reports with OpenRouter models, BATCHED (7 Oct).

Free-tier OpenRouter keys allow 50 requests a day, so one request carries --batch reports (default 10, 6 calls per
model). Same task as every other labeller here: data_code/03_label_reports.py's SYSTEM_SOFT prompt (Qwen 0.9114),
its tolerant JSON parser (_extract_json + _coerce), the translated report text (data/02_reports_en.csv), one soft
probability per finding. The batch wrapper asks for one JSON object keyed R1..Rn; evidence is optional (short).
Models are spread over the keys round-robin.

Outputs per model (slug = id with / and : replaced):
    data/03_report_labels_or-<slug>_validate58.csv   the 12 probabilities (same columns as every labeller file)
    data/stats/openrouter_raw/<slug>.jsonl           raw replies
    data/stats/openrouter_audit58_results.csv        one row per model: macro AUROC vs the panel, per finding,
                                                     studies parsed, calls, tokens, seconds (appended)
Keys ONLY from the environment (OPENROUTER_API_KEYS=k1,k2).

    python data_code/label_v2/openrouter_label_audit58.py --models google/gemma-4-26b-a4b-it:free --smoke
    python data_code/label_v2/openrouter_label_audit58.py --models m1,m2,m3
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "data_code"))
os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
spec = importlib.util.spec_from_file_location("lab03", ROOT / "data_code" / "03_label_reports.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
from llm_client import _extract_json  # noqa: E402

T = lab.LABELS
URL = "https://openrouter.ai/api/v1/chat/completions"
D = ROOT / "data"
RAW = D / "stats" / "openrouter_raw"
RES = D / "stats" / "openrouter_audit58_results.csv"
BATCH_RULES = """

BATCH FORMAT: you are given several reports at once, each headed "=== R<n> ===". Label EACH report independently,
exactly as you would label it alone. Output ONLY one JSON object (no prose, no code fence) whose keys are the report
ids and whose values are {"unusable": false, "findings": {"<finding>": {"p": <0.0-1.0>, "evidence": "<max 12 words>"}}}
with ALL 12 findings for every report, spelled exactly: """ + ", ".join(T) + "."


def call(model, key, reports, max_tokens, timeout=600):
    user = "\n\n".join(f"=== R{i + 1} ===\n{t}" for i, t in enumerate(reports))
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": lab.SYSTEM_SOFT + BATCH_RULES},
                         {"role": "user", "content": f"{len(reports)} KNEE MRI REPORTS:\n\n{user}"}]}
    for attempt in range(3):
        r = requests.post(URL, json=body, timeout=timeout,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        if r.status_code == 200 and r.json().get("choices"):
            j = r.json()
            return j["choices"][0]["message"].get("content") or "", j.get("usage", {}), r.status_code
        if r.status_code in (429, 502, 503) and attempt < 2:
            time.sleep(30 * (attempt + 1))
            continue
        return "", {"error": r.text[:300]}, r.status_code
    return "", {}, -1


def parse(content, n):
    obj = _extract_json(content)
    if isinstance(obj, list):
        obj = {f"R{i + 1}": v for i, v in enumerate(obj)}
    out = {}
    for i in range(n):
        v = obj.get(f"R{i + 1}") or obj.get(str(i + 1)) or obj.get(f"r{i + 1}")
        try:
            out[i] = {k: f["p"] for k, f in lab._coerce(v, hard=False)["findings"].items()}
        except Exception:  # noqa: BLE001  a missing report stays missing (NaN), counted below
            pass
    return out


def run_model(model, key, ids, texts, batch, max_tokens, smoke):
    slug = re.sub(r"[/:]", "_", model)
    RAW.mkdir(parents=True, exist_ok=True)
    P = np.full((len(ids), len(T)), np.nan)
    pos = {s: i for i, s in enumerate(ids)}
    if (RAW / f"{slug}.jsonl").exists() and not smoke:   # resume: keep every study an earlier reply already labelled
        for line in (RAW / f"{slug}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("content"):
                try:
                    for j, row in parse(r["content"], len(r["ids"])).items():
                        P[pos[r["ids"][j]]] = [row[t] for t in T]
                except Exception:  # noqa: BLE001
                    pass
    todo = [i for i in range(len(ids)) if np.isnan(P[i]).any()]
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    if smoke:
        chunks = [list(range(2))]
    t0, calls, tok_in, tok_out, errs = time.time(), 0, 0, 0, []
    with open(RAW / f"{slug}.jsonl", "a", encoding="utf-8") as fr:
        for ch in chunks:
            content, usage, code = call(model, key, [texts[i] for i in ch], max_tokens)
            calls += 1
            tok_in += int(usage.get("prompt_tokens") or 0)
            tok_out += int(usage.get("completion_tokens") or 0)
            fr.write(json.dumps({"ids": [ids[i] for i in ch], "status": code, "usage": usage, "content": content}) + "\n")
            if not content:
                errs.append(f"{code}: {str(usage.get('error'))[:120]}")
                continue
            try:
                got = parse(content, len(ch))
            except Exception as e:  # noqa: BLE001
                errs.append(f"unparseable: {e}")
                continue
            for j, row in got.items():
                P[ch[j]] = [row[t] for t in T]
    return slug, P, {"calls": calls, "tok_in": tok_in, "tok_out": tok_out, "secs": round(time.time() - t0), "errors": errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--smoke", action="store_true", help="one call, 2 reports, nothing scored or saved")
    a = ap.parse_args()
    keys = [k.strip() for k in os.environ.get("OPENROUTER_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        sys.exit("set OPENROUTER_API_KEYS=k1,k2 in this command's environment")
    aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
    ids = list(aud.index)
    en = pd.read_csv(D / "02_reports_en.csv").drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    texts = [str(en.loc[s, "Report_en"]) for s in ids]
    Y = (aud.loc[ids, ["official_" + t for t in T]].values >= 0.5).astype(int)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    with ThreadPoolExecutor(len(keys)) as ex:   # one model per key at a time
        futs = [ex.submit(run_model, m, keys[i % len(keys)], ids, texts, a.batch, a.max_tokens, a.smoke)
                for i, m in enumerate(models)]
        for m, f in zip(models, futs):
            slug, P, info = f.result()
            ok = ~np.isnan(P).any(1)
            if a.smoke:
                print(f"SMOKE {m}: parsed {ok[:2].sum()}/2, {info}")
                print("  R1:", dict(zip(T, np.round(P[0], 2))) if ok[0] else "-")
                continue
            per = {t: (roc_auc_score(Y[ok, j], P[ok, j]) if ok.sum() and 0 < Y[ok, j].sum() < ok.sum() else np.nan)
                   for j, t in enumerate(T)}
            mac = float(np.nanmean(list(per.values()))) if ok.sum() else np.nan
            print(f"{m}: macro AUROC {mac:.4f} on {ok.sum()}/58 parsed | calls {info['calls']} | "
                  f"tokens {info['tok_in']}+{info['tok_out']} | {info['secs']}s | errors {info['errors'][:2]}")
            if ok.all():
                pd.DataFrame(P, index=pd.Index(ids, name="StudyInstanceUID"), columns=T).reset_index().to_csv(
                    D / f"03_report_labels_or-{slug}_validate58.csv", index=False)
            row = {"model": m, "macro_auroc": round(mac, 4), "parsed": int(ok.sum()), "batch": a.batch, **info,
                   **{t: round(v, 4) for t, v in per.items()}, "when": time.strftime("%Y-%m-%d %H:%M")}
            row["errors"] = " | ".join(info["errors"])[:300]
            pd.DataFrame([row]).to_csv(RES, mode="a", header=not RES.exists(), index=False)


if __name__ == "__main__":
    main()
