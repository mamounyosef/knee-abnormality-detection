"""gemma4_incontext58.py - relabel reports with gemma4:31b, giving it the 58 panel-labelled studies as worked examples.

WHY (user, 13 Oct, final experiment). The labeller has never seen how the panel reads reports. Here the prompt carries
the 58 audit studies' English reports with the panel's 0/1 labels, then asks for the 12 probabilities of a new report.
Prompt rules = data_code/03_label_reports.py SYSTEM_SOFT (the v2 prompt); output parsed by its _coerce.

Modes
  --loo     honest score on the 58: each audit study is labelled with the OTHER 57 as examples (its own answer is
            never in the prompt), then macro AUROC against the panel. Baselines printed from the cached files:
            Qwen3.8-Max v2 labels, gemma4:31b zero-shot, gemma4:31b with the old 45-correction few-shot.
  --corpus  all 58 as examples, label every non-audit study (resumable). Writes data/29_report_labels_gemma4-ic58.csv.

Ollama Cloud, native /api/chat, JSON format, temperature 0, thinking ON (--think true: /api/show lists only false/true for gemma4:31b, default false). API keys ONLY from this command's environment
(OLLAMA_API_KEYS=k1,k2 spreads the requests over several keys); nothing is read from or written to .env.

    $env:OLLAMA_API_KEYS="k1,k2"; python data_code/label_v2/gemma4_incontext58.py --loo --limit 2 --show   # smoke
    $env:OLLAMA_API_KEYS="k1,k2"; python data_code/label_v2/gemma4_incontext58.py --loo
    $env:OLLAMA_API_KEYS="k1,k2"; python data_code/label_v2/gemma4_incontext58.py --corpus
"""
import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"
URL = "https://ollama.com/api/chat"
sys.path.insert(0, str(ROOT / "data_code"))     # 03_label_reports imports llm_client from there
_spec = importlib.util.spec_from_file_location("lab03", ROOT / "data_code" / "03_label_reports.py")
lab03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab03)
LABELS, SYSTEM_SOFT, coerce = lab03.LABELS, lab03.SYSTEM_SOFT, lab03._coerce
RAW_LOO = D / "29_ic58_loo_raw.jsonl"
RAW_CORPUS = D / "29_ic58_corpus_raw.jsonl"
OUT_CORPUS = D / "29_report_labels_gemma4-ic58.csv"
_lock = threading.Lock()
THINK = {"true": True, "false": False}      # gpt-oss style levels ("high") pass through as strings

aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
A58 = list(aud.index)
PANEL = (aud[["official_" + t for t in LABELS]].values >= .5).astype(int)
EN = pd.read_csv(D / "02_reports_en.csv").set_index("StudyInstanceUID")


def example_block(ids):
    parts = []
    for n, s in enumerate(ids, 1):
        lab = ", ".join(f"{t} {int(aud.loc[s, 'official_' + t] >= .5)}" for t in LABELS)
        parts.append(f"EXAMPLE {n}\nREPORT:\n{str(EN.loc[s, 'Report_en']).strip()}\nPANEL LABELS: {lab}")
    return "\n\n".join(parts)


def system(ids):
    return (SYSTEM_SOFT + "\n\nWORKED EXAMPLES. Below are real reports from this dataset with the expert panel's "
            "image-based labels (1 = present, 0 = absent). They are all abnormal knees, chosen by the panel, so they "
            "show how the panel's labels relate to report wording; they do NOT show how common each finding is. Use "
            "them to calibrate, then label the new report.\n\n" + example_block(ids)
            + "\n\nReply with ONE raw JSON object and nothing else: {\"unusable\": false, \"findings\": {<finding>: "
            "{\"p\": <0..1>, \"evidence\": \"<sentence>\"}, ...}} with all 12 findings: " + ", ".join(LABELS))


def keys():
    ks = [k.strip() for k in (os.environ.get("OLLAMA_API_KEYS") or os.environ.get("OLLAMA_API_KEY") or "").split(",")
          if k.strip()]
    if not ks:
        sys.exit("set OLLAMA_API_KEY (or OLLAMA_API_KEYS=k1,k2) in this command's environment")
    return ks


def ask(sys_prompt, text, model, key, think="true", timeout=900, retries=6):
    body = {"model": model, "stream": False, "think": THINK.get(think, think), "format": "json", "options": {"temperature": 0, "num_ctx": 65536},
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": "NEW REPORT:\n" + text}]}
    for a in range(retries):
        try:
            r = requests.post(URL, json=body, timeout=timeout, headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"status {r.status_code}: {r.text[:120]}")
            r.raise_for_status()
            txt = r.json()["message"]["content"]
            i, j = txt.find("{"), txt.rfind("}")
            return coerce(json.loads(txt[i:j + 1]), hard=False)
        except Exception:                                       # noqa: BLE001 - retry anything
            if a == retries - 1:
                raise
            time.sleep(min(5 * 2 ** a, 90))


def done_ids(raw):
    out = set()
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            try:
                out.add(json.loads(line)["StudyInstanceUID"])
            except Exception:                                   # noqa: BLE001 - a torn last line
                pass
    return out


def run(todo, prompt_for, raw, args):
    ks = keys()
    print(f"{len(todo)} to do, model {args.model}, {len(ks)} key(s) x {args.workers} workers", flush=True)
    t0, n, fails = time.time(), 0, 0

    def one(s, k):
        res = ask(prompt_for(s), str(EN.loc[s, "Report_en"]), args.model, k, args.think)
        return {"StudyInstanceUID": s, "model": args.model, "unusable": res["unusable"],
                **{t: res["findings"][t]["p"] for t in LABELS},
                **{"ev_" + t: res["findings"][t]["evidence"] for t in LABELS}}

    with ThreadPoolExecutor(max_workers=args.workers * len(ks)) as pool:
        futs = {pool.submit(one, s, ks[i % len(ks)]): s for i, s in enumerate(todo)}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except Exception as e:                              # noqa: BLE001
                fails += 1
                print(f"  FAIL {futs[f][-12:]}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                continue
            with _lock:
                with raw.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            n += 1
            if args.show:
                print({t: round(rec[t], 2) for t in LABELS}, flush=True)
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} done, {fails} failed, {el / n:.1f} s/report, elapsed {el / 60:.0f} min, "
                      f"ETA {el / n * (len(todo) - n) / 60:.0f} min", flush=True)
    print(f"finished: {n} written, {fails} failed (rerun to retry them)", flush=True)


def mac(P):
    return float(np.mean([roc_auc_score(PANEL[:, j], P[:, j]) for j in range(12)]))


def score_loo():
    recs = [json.loads(x) for x in RAW_LOO.read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(recs).drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")
    miss = [s for s in A58 if s not in df.index]
    if miss:
        print(f"{len(miss)} of 58 missing; rerun --loo to fill them")
        return
    P = df.loc[A58, LABELS].values.astype(float)
    print(f"\naudit-58 macro AUROC, gemma4:31b with the other 57 as examples (leave-one-out): {mac(P):.4f}")
    base = {"Qwen3.8-Max v2 labels": "03_report_labels_qwen24t-med.csv",
            "gemma4:31b zero-shot": "03_report_labels_gemma4-31b_validate58.csv",
            "gemma4:31b + 45 corrections (old few-shot)": "03_report_labels_gemma4-31b-fewshot_validate58.csv"}
    B = {}
    for name, f in base.items():
        p = D / f
        if p.exists():
            b = pd.read_csv(p).set_index("StudyInstanceUID")
            if set(A58) <= set(b.index):
                B[name] = b.loc[A58, LABELS].values.astype(float)
                print(f"   baseline {name:44s} {mac(B[name]):.4f}")
    print("per finding: in-context | " + " | ".join(B))
    for j, t in enumerate(LABELS):
        print(f"  {t:17s} {roc_auc_score(PANEL[:, j], P[:, j]):.4f} | "
              + " | ".join(f"{roc_auc_score(PANEL[:, j], b[:, j]):.4f}" for b in B.values()))
    if "Qwen3.8-Max v2 labels" in B:
        q = B["Qwen3.8-Max v2 labels"]
        rng = np.random.default_rng(0)
        d = []
        for _ in range(2000):
            i = rng.integers(0, 58, 58)
            try:
                d.append(np.mean([roc_auc_score(PANEL[i, j], P[i, j]) - roc_auc_score(PANEL[i, j], q[i, j])
                                  for j in range(12)]))
            except ValueError:
                continue
        d = np.array(d)
        print(f"bootstrap in-context - Qwen v2: mean {d.mean():+.4f}, 90% [{np.percentile(d, 5):+.4f}, "
              f"{np.percentile(d, 95):+.4f}], better in {np.mean(d > 0):.0%}")
        for w in (0.3, 0.5):
            L = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))  # noqa: E731
            print(f"   blend logit {1 - w:.1f} Qwen v2 + {w:.1f} in-context: {mac((1 - w) * L(q) + w * L(P)):.4f}")


def build_corpus():
    recs = [json.loads(x) for x in RAW_CORPUS.read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(recs).drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")
    df[["model", "unusable"] + LABELS].to_csv(OUT_CORPUS)
    print(f"wrote {OUT_CORPUS}: {len(df)} studies")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo", action="store_true")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--score", action="store_true", help="score the cached --loo output only")
    ap.add_argument("--build", action="store_true", help="corpus jsonl -> csv")
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--think", default="true", help="Ollama think value; gemma4:31b supports only true/false (/api/show), so true is its highest")
    ap.add_argument("--workers", type=int, default=4)       # per key
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.loo:
        todo = [s for s in A58 if s not in done_ids(RAW_LOO)][: a.limit or None]
        run(todo, lambda s: system([x for x in A58 if x != s]), RAW_LOO, a)
        score_loo()
    elif a.score:
        score_loo()
    elif a.corpus:
        full = system(A58)
        ids = [s for s in EN.index if s not in set(A58)]
        todo = [s for s in ids if s not in done_ids(RAW_CORPUS)][: a.limit or None]
        run(todo, lambda s: full, RAW_CORPUS, a)
    elif a.build:
        build_corpus()
