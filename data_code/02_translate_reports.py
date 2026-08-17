"""
02_translate_reports.py — translate all 4,407 radiology reports into English.
=============================================================================

Stage 1 of label generation. The reports come in 9 languages and English is only
39% of them; open-weights models are documented to label non-English reports
less accurately than English ones, so we translate first, save the result, and
label the English text in stage 2 (data_code/03_label_reports.py).

Keeping translation as its own saved artifact means:
  * we can read the English text ourselves and audit it,
  * stage 2 can be re-run with different prompts/models without paying to
    translate again,
  * the direct-labeling ablation (label the original language) is a fair
    comparison against a fixed translation set.

Reports are deduplicated first: 177 of the 4,407 rows share only 46 distinct
texts, so we translate each unique text once.

Model: Qwen3.8-Max on a Modal Auto Endpoint (OpenAI Chat Completions API under
/v1). Credentials come from the repo .env, which is gitignored.

Usage
-----
    python data_code/02_translate_reports.py --limit 20     # smoke test
    python data_code/02_translate_reports.py                # all of them
    python data_code/02_translate_reports.py --resume       # skip ones already done

Outputs
-------
    data/02_reports_en.csv          StudyInstanceUID, Report, Report_en, lang
    data/stats/02_translation_log.txt
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from llm_client import chat, load_env, stats, warmup   # shared helper, same folder

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_CSV = ROOT / "data" / "02_reports_en.csv"
CKPT = ROOT / "data" / "02_reports_en.jsonl"     # append-only, written as we go
LOG_TXT = ROOT / "data" / "stats" / "02_translation_log.txt"

SYSTEM = """You translate radiology reports into English.

Translate the knee MRI report below into clear clinical English.

Rules:
- Preserve EVERY finding, including all normal/negative statements.
- Preserve every negation exactly ("no tear", "intact", "normal", "unremarkable").
  Negations carry the most important information in this dataset; never drop one.
- Preserve every grade, measurement, and anatomical location, including which
  compartment (medial / lateral / patellofemoral) each finding belongs to.
- Do not summarise, do not omit, do not add interpretation of your own.
- Keep the original section structure and line breaks.
- If the report is already in English, return it unchanged.
- Report the source language as an ISO 639-1 code from the allowed list.

Return strictly the JSON object required by the schema."""

SCHEMA = {
    "type": "object",
    "properties": {
        # an enum, not a description: a free-text field came back as a mix of
        # "English"/"en"/"Spanish"/"es" across the first 1,661 reports
        "language": {"type": "string",
                     "enum": ["en", "es", "tr", "hr", "el", "de", "bg", "nl", "fr", "other"]},
        "already_english": {"type": "boolean"},
        "translation": {"type": "string"},
    },
    "required": ["language", "already_english", "translation"],
    "additionalProperties": False,
}


def translate(text: str) -> dict:
    out = chat([{"role": "system", "content": SYSTEM},
                {"role": "user", "content": text}],
               schema=SCHEMA, schema_name="translation",
               effort="low", max_tokens=6000)
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap unique reports (testing)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--resume", action="store_true",
                    help="reuse translations already present in the output CSV")
    args = ap.parse_args()

    load_env()
    train = pd.read_csv(RAW / "train.csv")
    train["Report"] = train["Report"].fillna("")

    # --- resume from the append-only checkpoint (and the CSV, for older runs) ---
    done: dict[str, dict] = {}
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done[rec["report"]] = {"translation": rec["translation"],
                                       "language": rec.get("language")}
            except json.JSONDecodeError:
                continue          # a torn last line from a killed process
        print(f"[resume] {len(done)} translations recovered from {CKPT.name}")
    if args.resume and OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        for _, r in prev.dropna(subset=["Report_en"]).iterrows():
            done.setdefault(r["Report"], {"translation": r["Report_en"],
                                          "language": r["lang"]})
        print(f"[resume] {len(done)} total after merging {OUT_CSV.name}")

    uniq = [t for t in train["Report"].drop_duplicates().tolist() if t not in done]
    if args.limit:
        uniq = uniq[:args.limit]
    print(f"[setup] {len(train)} studies | {train['Report'].nunique()} unique reports "
          f"| {len(uniq)} to translate")

    # wait out any cold start once, before the workers start (see llm_client.warmup)
    if uniq and not warmup():
        raise SystemExit("endpoint never became ready — check the Modal dashboard")

    t0 = time.time()
    lock = threading.Lock()
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    ckpt_fh = CKPT.open("a", encoding="utf-8")   # append: never lose completed work
    consecutive_fails = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(translate, t): t for t in uniq}
        for i, fut in enumerate(as_completed(futs), 1):
            text = futs[fut]
            try:
                res = fut.result()
                with lock:
                    done[text] = res
                    # flush every record immediately - a killed process, an
                    # exhausted balance or a dropped connection then costs us
                    # nothing already paid for
                    ckpt_fh.write(json.dumps(
                        {"report": text, "translation": res["translation"],
                         "language": res.get("language")}, ensure_ascii=False) + "\n")
                    ckpt_fh.flush()
                    consecutive_fails = 0
            except Exception as exc:                        # noqa: BLE001
                consecutive_fails += 1
                print(f"[fail] {type(exc).__name__}: {str(exc)[:120]}")
                if consecutive_fails >= 40:
                    print(f"\n[abort] {consecutive_fails} consecutive failures — the "
                          f"endpoint is down or out of credit. Stopping so we do not "
                          f"burn an hour on retries.\n[abort] {len(done)} translations "
                          f"are safe in {CKPT.name}; rerun with --resume to continue.")
                    for f in futs:
                        f.cancel()
                    break
            if i % 25 == 0 or i == len(uniq):
                el = time.time() - t0
                print(f"[{i:5d}/{len(uniq)}] {i/len(uniq):5.1%}  "
                      f"elapsed {el/60:5.1f}m  ETA {(len(uniq)-i)*el/i/60:5.1f}m  "
                      f"retries {stats['retries']} fails {stats['failures']}", flush=True)

    ckpt_fh.close()
    train["Report_en"] = train["Report"].map(lambda t: done.get(t, {}).get("translation"))
    train["lang"] = train["Report"].map(lambda t: done.get(t, {}).get("language"))
    train[["StudyInstanceUID", "Report", "Report_en", "lang"]].to_csv(OUT_CSV, index=False)

    L = []
    def w(s=""):
        L.append(str(s)); print(s)

    w("=" * 78)
    w("REPORT TRANSLATION (stage 1 of labeling)")
    w("=" * 78)
    w(f"studies              : {len(train)}")
    w(f"unique reports       : {train['Report'].nunique()}")
    w(f"translated this run  : {len(uniq)}")
    w(f"missing translations : {int(train['Report_en'].isna().sum())}")
    w(f"API calls            : {stats['calls']}   retries {stats['retries']}   "
      f"failures {stats['failures']}")
    w(f"tokens in/out        : {stats['in_tok']:,} / {stats['out_tok']:,}")
    w(f"wall clock           : {(time.time()-t0)/60:.1f} min")
    w("\ndetected source language (per study):")
    w(train["lang"].value_counts(dropna=False).to_string())
    w("\nlength change, original -> English (characters):")
    ok = train.dropna(subset=["Report_en"])
    w(f"  original median {ok['Report'].str.len().median():.0f}, "
      f"English median {ok['Report_en'].str.len().median():.0f}")
    short = ok[ok["Report_en"].str.len() < ok["Report"].str.len() * 0.5]
    w(f"  suspiciously short translations (<50% of original): {len(short)}")
    if len(short):
        w("  -> inspect these; a dropped negation would corrupt the label")
        for _, r in short.head(3).iterrows():
            w(f"     {r['StudyInstanceUID'][-12:]}  {len(r['Report'])} -> {len(r['Report_en'])} chars")

    LOG_TXT.parent.mkdir(parents=True, exist_ok=True)
    LOG_TXT.write_text("\n".join(L) + "\n", encoding="utf-8")
    w(f"\nwritten: {OUT_CSV}\n         {LOG_TXT}")


if __name__ == "__main__":
    main()
