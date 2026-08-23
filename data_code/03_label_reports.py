"""
03_label_reports.py — turn the translated reports into 12 soft labels each.
===========================================================================

Stage 2 of label generation. Reads the English text produced by
data_code/02_translate_reports.py and asks the LLM for, per finding, a
probability plus the sentence it relied on.

Two facts shape the prompt, both established in data_code/01_data_analyze.ipynb
and data/stats/report_vs_label_audit.txt:

  * The 58 official labels were assigned by an expert panel re-reading the
    IMAGES (Society of Skeletal Radiology, via MD.ai), not the reports. Reports
    and labels therefore disagree in a substantial minority of cases, and a
    report that is silent about a finding does not mean the finding is absent.
  * The panel applied recoverable criteria: meniscus means a TEAR, osteoarthritis
    means substantial CARTILAGE LOSS rather than osteophytes, "tricompartmental"
    implies all three compartments, and subchondral oedema beside a degenerative
    defect is not a contusion.

Usage
-----
    python data_code/03_label_reports.py --validate      # the 58 only, scored
    python data_code/03_label_reports.py --limit 200     # a sample
    python data_code/03_label_reports.py                 # everything
    python data_code/03_label_reports.py --source original   # ablation: skip translation

Outputs
-------
    data/03_report_labels.csv        one row per study, 12 probabilities
    data/03_report_labels_raw.jsonl  full model output per unique report
    data/stats/03_labeling_log.txt   run summary, plus scores if --validate
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from llm_client import _extract_json, chat, load_env, stats, warmup

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EN_CSV = ROOT / "data" / "02_reports_en.csv"
OUT_CSV = ROOT / "data" / "03_report_labels.csv"
OUT_JSONL = ROOT / "data" / "03_report_labels_raw.jsonl"
LOG_TXT = ROOT / "data" / "stats" / "03_labeling_log.txt"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

SYSTEM = """You are a musculoskeletal radiologist labeling knee MRI reports for a research dataset.

You are given the full text of a knee MRI report. Output a probability (0.0-1.0) that each of
12 findings is present in that knee, plus the exact sentence you relied on.

CRITICAL CONTEXT: the reference labels were assigned by an expert panel who re-read the IMAGES,
not the report. So judge what the knee most likely shows, given what the report describes.
A report that is silent about a finding does NOT mean it is absent - it means unknown; fall back
on the base rate and any indirect evidence. Reserve probabilities below 0.15 for findings the
report explicitly denies.

DEFINITIONS AND SCORING RULES (recovered from the reference panel's own labeling):

ACL  - anterior cruciate ligament tear/rupture, any grade including partial and interstitial.
       Mucoid degeneration alone is NOT a tear (~0.25). "Intact"/"normal" -> 0.05.
MCL  - medial collateral ligament injury, including grade 1-2 sprain. Not the LCL.
Medial Meniscus / Lateral Meniscus - a TEAR specifically. Degenerative signal that does NOT
       reach the articular surface is NOT a tear (~0.2). Post-surgical resection or "marginal
       amputation" without a new tear is NOT a tear. Extrusion alone is not a tear.
Medial OA / Lateral OA / PF OA - osteoarthritis of that compartment, meaning substantial
       CARTILAGE LOSS (grade 3-4 chondropathy, full-thickness defect, chondral ulcer,
       bone-on-bone). Marginal osteophytes alone, or mild fissuring/heterogeneity -> ~0.3.
       IMPORTANT: if the report says "tricompartmental osteoarthritis/chondrosis" without
       naming compartments, score ALL THREE (medial, lateral, patellofemoral) high.
       "Bicompartmental" -> infer which two from context. PF OA covers the patella, patellar
       facets, trochlea and retropatellar cartilage.
Effusion  - a joint effusion the panel would call present, meaning a MODERATE or LARGE amount
       of fluid, or haemarthrosis. Be strict here: nearly every knee report mentions some
       fluid, and the panel does NOT count small amounts. Calibrate by wording:
         "trace" / "minimal" / "physiologic" / "small amount"        -> 0.25
         "mild" / "small effusion"                                    -> 0.40
         "moderate" / "distended suprapatellar bursa" / unqualified
             "joint effusion"                                         -> 0.75
         "large" / "massive" / "tense" / haemarthrosis                -> 0.92
         explicitly "no effusion"                                     -> 0.05
Synovitis - synovial thickening or enhancement, "reactive synovitis", hoffitis, bursitis with
       synovial involvement. Chronic degenerative joints often have it even when unmentioned,
       so a knee with advanced OA and an effusion should not score very low.
Baker's   - popliteal / Baker's cyst. "No popliteal cyst" -> 0.05.
Contusion - bone bruise / bone marrow oedema of TRAUMATIC origin, or muscle contusion.
       Subchondral marrow oedema adjacent to a degenerative cartilage defect is NOT a contusion
       (~0.2). Subchondral cystic change is not a contusion.
Fracture  - a true fracture of bone, including insufficiency and avulsion fractures.
       An osteochondral defect or OCD lesion alone is NOT a fracture (~0.25).

Set "unusable": true only if the report contains NO findings at all (an empty template).
A report stating that everything is normal IS usable - it is informative and negative.

Probabilities must be genuine confidences, not only 0.0 or 1.0."""

SCHEMA = {
    "type": "object",
    "properties": {
        "unusable": {"type": "boolean"},
        "findings": {
            "type": "object",
            "properties": {
                k: {"type": "object",
                    "properties": {"p": {"type": "number"},
                                   "evidence": {"type": "string"}},
                    "required": ["p", "evidence"], "additionalProperties": False}
                for k in LABELS},
            "required": LABELS, "additionalProperties": False,
        },
    },
    "required": ["unusable", "findings"], "additionalProperties": False,
}


P_KEYS = ("p", "probability", "prob", "score", "value", "confidence")
E_KEYS = ("evidence", "sentence", "quote", "text", "snippet")


def _coerce(obj: dict) -> dict:
    """Force any model's answer into {"unusable": bool, "findings": {label: {p, evidence}}}.

    Only Modal enforces the schema. Ollama treats it as a hint, so models invent their
    own shape, and NOT consistently between calls: gemma4:31b puts the 12 findings at
    the TOP level, sometimes as {"probability", "sentence"} and sometimes as the bare
    pair [0.05, "the sentence"]. Rejecting those would throw away usable answers, so
    normalise instead - and fail loudly only when a finding is genuinely missing.
    """
    src = obj.get("findings") if isinstance(obj.get("findings"), dict) else obj
    out = {}
    for k in LABELS:
        v = src.get(k)
        if v is None:
            raise ValueError(f"no entry for {k!r}; got keys {sorted(src)[:14]}")
        if isinstance(v, (list, tuple)):             # [0.05, "the sentence"]
            v = {"p": v[0] if v else None,
                 "evidence": v[1] if len(v) > 1 else ""}
        elif not isinstance(v, dict):                # a bare number
            v = {"p": v}
        p = next((v[a] for a in P_KEYS if a in v and v[a] is not None), None)
        if p is None:
            raise ValueError(f"{k!r} has no probability; got {sorted(v)}")
        ev = next((v[a] for a in E_KEYS if a in v and v[a]), "")
        out[k] = {"p": min(1.0, max(0.0, float(p))), "evidence": str(ev)}
    return {"unusable": bool(obj.get("unusable", False)), "findings": out}


def label(text: str, effort: str = "low", timeout: int = 180) -> dict:
    """One report -> 12 probabilities.

    effort='low' by default: a successful call returns a complete 1,244-token
    labeled output in ~6s and spends ZERO reasoning tokens, so 'medium' bought
    latency and cost without changing the answer. Worth re-checking on the 58.

    The timeout is deliberately short. A request that hangs is occupying a
    container slot; failing fast and retrying is cheaper than waiting 10 minutes,
    and the checkpoint means a retry costs nothing already paid for.
    """
    return _coerce(_extract_json(chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"KNEE MRI REPORT:\n\n{text}"}],
        schema=SCHEMA, schema_name="knee_labels", effort=effort,
        max_tokens=4000, timeout=timeout)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="only the 58 officially labeled studies, and score them")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--source", choices=["english", "original"], default="english",
                    help="'original' skips the translation (ablation)")
    ap.add_argument("--effort", choices=["low", "medium", "xhigh"], default="low")
    ap.add_argument("--timeout", type=int, default=180,
                    help="per-request seconds; short so a hung slot fails fast")
    ap.add_argument("--fewshot", action="store_true",
                    help="prepend label_v2/fewshot_block.txt - 45 real corrections drawn "
                         "from 29 of the 58 panel studies")
    ap.add_argument("--holdout", action="store_true",
                    help="with --validate, score ONLY the panel studies that are NOT "
                         "few-shot examples. Without it, --fewshot --validate scores "
                         "the model on its own prompt and means nothing.")
    ap.add_argument("--tag", default="",
                    help="isolate this run's outputs, e.g. --tag gemma4-31b writes "
                         "03_report_labels_gemma4-31b.csv / _raw_gemma4-31b.jsonl. "
                         "Without it a second labeler would resume from, and append "
                         "to, the first one's checkpoint.")
    args = ap.parse_args()

    if args.tag:
        global OUT_CSV, OUT_JSONL, LOG_TXT
        OUT_CSV = OUT_CSV.with_name(f"03_report_labels_{args.tag}.csv")
        OUT_JSONL = OUT_JSONL.with_name(f"03_report_labels_raw_{args.tag}.jsonl")
        LOG_TXT = LOG_TXT.with_name(f"03_labeling_log_{args.tag}.txt")
        print(f"[tag] outputs isolated as *_{args.tag}")

    global SYSTEM
    if args.fewshot:
        blk = ROOT / "data_code" / "label_v2" / "fewshot_block.txt"
        if not blk.exists():
            raise SystemExit(f"{blk} not found - run label_v2/build_fewshot.py first")
        SYSTEM = SYSTEM + "\n\n" + blk.read_text(encoding="utf-8")
        print(f"[fewshot] +{len(blk.read_text(encoding='utf-8')) // 4:,} tokens of "
              f"calibration examples")

    load_env()
    train = pd.read_csv(RAW / "train.csv")
    train["Report"] = train["Report"].fillna("")

    if args.source == "english":
        if not EN_CSV.exists():
            raise SystemExit(f"{EN_CSV} not found — run 02_translate_reports.py first")
        en = pd.read_csv(EN_CSV)[["StudyInstanceUID", "Report_en"]]
        train = train.merge(en, on="StudyInstanceUID", how="left")
        train["text"] = train["Report_en"].fillna(train["Report"])
        n_fallback = int(train["Report_en"].isna().sum())
        if n_fallback:
            print(f"[warn] {n_fallback} studies have no translation; using the original")
    else:
        train["text"] = train["Report"]

    has_labels = train[LABELS].notna().all(axis=1)
    df = (train[has_labels] if args.validate else train).reset_index(drop=True)
    if args.holdout:
        fs = pd.read_csv(ROOT / "data_code" / "label_v2" / "fewshot.csv")
        before = len(df)
        df = df[~df.StudyInstanceUID.isin(set(fs.StudyInstanceUID))].reset_index(drop=True)
        print(f"[holdout] {before} -> {len(df)} studies (dropped the few-shot examples)")

    uniq = df["text"].drop_duplicates().tolist()
    if args.limit:
        uniq = uniq[:args.limit]
    print(f"[setup] {len(df)} studies -> {len(uniq)} unique texts | source={args.source}")

    # resume from the append-only checkpoint so paid-for work is never lost
    results: dict[str, dict] = {}
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                results[rec["text"]] = {k: v for k, v in rec.items() if k != "text"}
            except json.JSONDecodeError:
                continue          # torn last line from a killed process
        print(f"[resume] {len(results)} labels recovered from {OUT_JSONL.name}")
    uniq = [t for t in uniq if t not in results]
    print(f"[setup] {len(uniq)} texts still to label")

    # A cold 2.4T endpoint needs minutes to load weights. Wait for it once here,
    # patiently, rather than letting every worker's short timeout cancel the boot.
    if uniq and not warmup():
        raise SystemExit("endpoint never became ready — check the Modal dashboard")

    t0, lock = time.time(), threading.Lock()
    fh = OUT_JSONL.open("a", encoding="utf-8")
    consecutive_fails = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(label, t, args.effort, args.timeout): t for t in uniq}
        for i, fut in enumerate(as_completed(futs), 1):
            text = futs[fut]
            try:
                res = fut.result()
                with lock:
                    results[text] = res
                    fh.write(json.dumps({"text": text, **res}, ensure_ascii=False) + "\n")
                    fh.flush()
                    consecutive_fails = 0
            except Exception as exc:                        # noqa: BLE001
                consecutive_fails += 1
                print(f"[fail] {type(exc).__name__}: {str(exc)[:120]}")
                if consecutive_fails >= 40:
                    print(f"\n[abort] {consecutive_fails} consecutive failures — endpoint "
                          f"down or out of credit. {len(results)} labels are safe in "
                          f"{OUT_JSONL.name}; rerun to continue.")
                    for f in futs:
                        f.cancel()
                    break
            if i % 25 == 0 or i == len(uniq):
                el = time.time() - t0
                print(f"[{i:5d}/{len(uniq)}] {i/len(uniq):5.1%}  "
                      f"elapsed {el/60:5.1f}m  ETA {(len(uniq)-i)*el/i/60:5.1f}m  "
                      f"retries {stats['retries']} fails {stats['failures']}", flush=True)

    fh.close()

    rows = []
    for _, r in df.iterrows():
        res = results.get(r["text"])
        if res is None:
            continue
        row = {"StudyInstanceUID": r["StudyInstanceUID"], "unusable": res["unusable"]}
        row.update({k: float(res["findings"][k]["p"]) for k in LABELS})
        rows.append(row)
    out = pd.DataFrame(rows)
    # --validate covers only the 58 studies, so it must NEVER overwrite the
    # full-corpus labels; it writes its own file instead.
    csv_path = (OUT_CSV.with_name(OUT_CSV.stem + "_validate58.csv")
                if args.validate else OUT_CSV)
    out.to_csv(csv_path, index=False)

    L = []
    def w(s=""):
        L.append(str(s)); print(s)

    w("=" * 78)
    w("LLM REPORT LABELING (stage 2)")
    w("=" * 78)
    w(f"source text      : {args.source}")
    w(f"studies labeled  : {len(out)} of {len(df)}")
    w(f"unique texts     : {len(uniq)}   API calls: {stats['calls']}")
    w(f"retries/failures : {stats['retries']} / {stats['failures']}")
    w(f"tokens in/out    : {stats['in_tok']:,} / {stats['out_tok']:,}")
    w(f"wall clock       : {(time.time()-t0)/60:.1f} min")
    w(f"unusable reports : {int(out['unusable'].sum())}")
    w("\nmean predicted probability per finding (%):")
    w((out[LABELS].mean() * 100).round(1).sort_values(ascending=False).to_string())
    w("\nfraction above 0.5 (%):")
    w(((out[LABELS] > 0.5).mean() * 100).round(1).sort_values(ascending=False).to_string())

    if args.validate:
        from sklearn.metrics import roc_auc_score
        truth = df.set_index("StudyInstanceUID")[LABELS].astype(int)
        pred = out.set_index("StudyInstanceUID")[LABELS]
        common = truth.index.intersection(pred.index)
        truth, pred = truth.loc[common], pred.loc[common]
        aur = {c: (roc_auc_score(truth[c].values, pred[c].values)
                   if truth[c].nunique() > 1 else float("nan")) for c in LABELS}
        w(f"\n=== scored against the {len(common)} official studies ===")
        w(pd.DataFrame({"positives": truth.sum(),
                        "mean_p_pos": [pred.loc[truth[c] == 1, c].mean() for c in LABELS],
                        "mean_p_neg": [pred.loc[truth[c] == 0, c].mean() for c in LABELS],
                        "auroc": pd.Series(aur)}).round(3).to_string())
        w(f"\nMACRO AUROC = {pd.Series(aur).mean():.4f}")
        w("keyword-matcher floor = 0.7003")
        w("caveats: 58 studies carries roughly +/-0.05 of noise, and the labels are")
        w("         image-derived, so perfect agreement is not attainable.")

    LOG_TXT.parent.mkdir(parents=True, exist_ok=True)
    LOG_TXT.write_text("\n".join(L) + "\n", encoding="utf-8")
    w(f"\nwritten: {csv_path}\n         {OUT_JSONL}\n         {LOG_TXT}")


if __name__ == "__main__":
    main()
