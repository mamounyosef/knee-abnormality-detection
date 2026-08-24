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

SYSTEM_SOFT = """You are a musculoskeletal radiologist labeling knee MRI reports for a research dataset.

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

# ---------------------------------------------------------------------------
# v2 of the soft prompt: Synovitis is the worst-scoring finding (0.832 at medium)
# AND the one the labeler disagrees with itself on most (spread 0.059). It is also
# the only finding whose rule carries no numbers - every other one has anchors.
# This gives it the same wording ladder Effusion has.
SYSTEM_SOFT_V2 = SYSTEM_SOFT.replace('''Synovitis - synovial thickening or enhancement, "reactive synovitis", hoffitis, bursitis with
       synovial involvement. Chronic degenerative joints often have it even when unmentioned,
       so a knee with advanced OA and an effusion should not score very low.''',
'''Synovitis - synovial thickening or enhancement, "reactive synovitis", hoffitis, bursitis with
       synovial involvement. This is the finding reports mention LEAST relative to how often
       the panel sees it, so silence here is weak evidence of absence. Calibrate:
         explicit "synovitis" / "synovial thickening" / "synovial
             enhancement" / "synovial proliferation"                 -> 0.90
         "hoffitis" / Hoffa's fat pad oedema / infrapatellar fat
             pad inflammation                                        -> 0.80
         bursitis described with a thickened or enhancing wall       -> 0.70
         NOT mentioned, but advanced OA (grade 3-4 cartilage loss)
             AND a moderate or large effusion                        -> 0.65
         NOT mentioned, but a moderate or large effusion alone       -> 0.55
         NOT mentioned, but advanced OA alone                        -> 0.50
         NOT mentioned, and the knee is otherwise unremarkable
             (no effusion, no substantial cartilage loss)            -> 0.30
         explicitly "no synovitis" / "no synovial thickening"        -> 0.10''')

# ---------------------------------------------------------------------------
# HARD mode: the same task, but the model must COMMIT to 0 or 1 per finding.
#
# The panel's own 58 rows are hard 0/1, so this matches their format exactly. It is
# NOT a strictly single-variable change against SYSTEM_SOFT, and the difference is
# forced rather than chosen: a soft labeler can answer "0.4" on a report that never
# mentions the finding, and a hard one cannot. It needs a tie-break rule, so the
# prompt has to say which findings are common and which are rare. Read any hard-vs-soft
# comparison with that in mind - two things move at once.
#
# The frequency buckets below are coarse on purpose. Exact per-finding rates would be
# read off the 58 panel studies, which are also the evaluation set, and quoting them
# would leak the answer sheet into the prompt.
SYSTEM_HARD = """You are a musculoskeletal radiologist labeling knee MRI reports for a research dataset.

You are given the full text of a knee MRI report. For each of 12 findings, output 1 if the
finding is PRESENT in that knee and 0 if it is ABSENT, plus the exact sentence you relied on.

CRITICAL CONTEXT: the reference labels were assigned by an expert panel who re-read the IMAGES,
not the report. So judge what the knee most likely shows, given what the report describes.
A report that is silent about a finding does NOT mean it is absent - it means unknown. You must
still commit to 0 or 1. When the report gives you nothing either way, answer with whichever is
more likely in a knee like this one, using the frequency guide at the bottom and any indirect
evidence (a severely degenerative knee, a trauma history, the patient's age).

DEFINITIONS AND DECISION RULES (recovered from the reference panel's own labeling):

ACL  - anterior cruciate ligament tear/rupture, any grade including partial and interstitial.
       Mucoid degeneration alone is NOT a tear -> 0. "Intact"/"normal" -> 0.
MCL  - medial collateral ligament injury, including grade 1-2 sprain -> 1. Not the LCL.
Medial Meniscus / Lateral Meniscus - a TEAR specifically -> 1. Degenerative signal that does
       NOT reach the articular surface -> 0. Post-surgical resection or "marginal amputation"
       without a new tear -> 0. Extrusion alone -> 0.
Medial OA / Lateral OA / PF OA - osteoarthritis of that compartment, meaning substantial
       CARTILAGE LOSS (grade 3-4 chondropathy, full-thickness defect, chondral ulcer,
       bone-on-bone) -> 1. Marginal osteophytes alone, or mild fissuring/heterogeneity -> 0.
       IMPORTANT: "tricompartmental osteoarthritis/chondrosis" without named compartments
       -> 1 for ALL THREE. "Bicompartmental" -> infer which two from context. PF OA covers
       the patella, patellar facets, trochlea and retropatellar cartilage.
Effusion  - the panel counts an effusion only when there is a MODERATE or LARGE amount of
       fluid, or haemarthrosis. Be strict: nearly every knee report mentions some fluid.
         "trace" / "minimal" / "physiologic" / "small amount"        -> 0
         "mild" / "small effusion"                                    -> 0
         "moderate" / "distended suprapatellar bursa" / unqualified
             "joint effusion"                                         -> 1
         "large" / "massive" / "tense" / haemarthrosis                -> 1
         explicitly "no effusion"                                     -> 0
Synovitis - synovial thickening or enhancement, "reactive synovitis", hoffitis, bursitis with
       synovial involvement -> 1. Chronic degenerative joints often have it even when it is
       never mentioned, so a knee with advanced OA and a real effusion -> 1.
Baker's   - popliteal / Baker's cyst -> 1. "No popliteal cyst" -> 0.
Contusion - bone bruise / bone marrow oedema of TRAUMATIC origin, or muscle contusion -> 1.
       Subchondral marrow oedema next to a degenerative cartilage defect -> 0. Subchondral
       cystic change -> 0.
Fracture  - a true fracture of bone, including insufficiency and avulsion fractures -> 1.
       An osteochondral defect or OCD lesion alone -> 0.

Set "unusable": true only if the report contains NO findings at all (an empty template).
A report stating that everything is normal IS usable - it is informative and negative.

WHEN THE REPORT IS SILENT, fall back on how common each finding is in this population:
  common, present in roughly half of these knees   - Effusion, Synovitis
  moderately common, roughly a third               - ACL, Medial Meniscus, Lateral Meniscus,
                                                     PF OA, Contusion, Fracture
  uncommon, well under a third                     - MCL, Medial OA, Lateral OA, Baker's
This is a tie-break for genuine silence only. Never let it override what the report states.

Every value must be exactly 0 or 1. Do not output any other number."""

# the active prompt; main() swaps it for SYSTEM_HARD under --mode hard, and appends the
# output contract / few-shot block on top. Kept under the old name so smoke.py and
# anything else importing `SYSTEM` still resolves to the soft default.
SYSTEM = SYSTEM_SOFT


def _schema(hard: bool = False) -> dict:
    """The JSON schema for one report's answer.

    Modal constrains decoding to this, so in hard mode the enum is what actually stops
    the model emitting 0.35 - the prompt alone would only be a suggestion.
    """
    p = ({"type": "integer", "enum": [0, 1]} if hard else {"type": "number"})
    return {
        "type": "object",
        "properties": {
            "unusable": {"type": "boolean"},
            "findings": {
                "type": "object",
                "properties": {
                    k: {"type": "object",
                        "properties": {"p": p, "evidence": {"type": "string"}},
                        "required": ["p", "evidence"], "additionalProperties": False}
                    for k in LABELS},
                "required": LABELS, "additionalProperties": False,
            },
        },
        "required": ["unusable", "findings"], "additionalProperties": False,
    }


SCHEMA = _schema(False)
SCHEMA_HARD = _schema(True)
MODE = "soft"                      # set by main(); label() reads it


_SNAPPED = [0]                     # hard-mode values the model returned off 0/1

P_KEYS = ("p", "probability", "prob", "score", "value", "confidence")
E_KEYS = ("evidence", "sentence", "quote", "text", "snippet")


def _coerce(obj: dict, hard: bool | None = None) -> dict:
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
    # hard mode: only Modal's constrained decoding actually enforces the 0/1 enum. An
    # unconstrained endpoint will still hand back 0.35 sometimes, so snap - but count
    # it, because a run full of snapped values is a soft run wearing a hard label.
    if (MODE == "hard") if hard is None else hard:
        for v in out.values():
            if v["p"] not in (0.0, 1.0):
                _SNAPPED[0] += 1
                v["p"] = float(v["p"] >= 0.5)
    return {"unusable": bool(obj.get("unusable", False)), "findings": out}


def _one(text: str, effort: str, timeout: int, temperature: float,
         max_tokens: int = 4000, thinking: bool = False) -> dict:
    """A single pass over one report -> 12 probabilities."""
    return _coerce(_extract_json(chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"KNEE MRI REPORT:\n\n{text}"}],
        schema=(SCHEMA_HARD if MODE == "hard" else SCHEMA),
        schema_name="knee_labels", effort=effort,
        max_tokens=max_tokens, timeout=timeout, temperature=temperature,
        chat_template_kwargs={"enable_thinking": True} if thinking else None)))


def label(text: str, effort: str = "low", timeout: int = 180,
          samples: int = 1, temperature: float = 0.0,
          max_tokens: int = 4000, thinking: bool = False) -> dict:
    """One report -> 12 probabilities, optionally averaged over `samples` passes.

    effort='low' by default: a successful call returns a complete 1,244-token
    labeled output in ~6s and spends ZERO reasoning tokens, so 'medium' bought
    latency and cost without changing the answer. Worth re-checking on the 58.

    The timeout is deliberately short. A request that hangs is occupying a
    container slot; failing fast and retrying is cheaper than waiting 10 minutes,
    and the checkpoint means a retry costs nothing already paid for.

    SELF-CONSISTENCY (samples > 1): the same model on the same text disagrees with
    itself on 3.3% of decisions - measured, not assumed, by relabeling 28 studies
    whose text had not changed. That is pure generation variance, and it is larger
    than every bias we went looking for and failed to find. Averaging N passes
    shrinks it by about sqrt(N).

    The SPREAD is kept too, and is arguably the more useful half: passes that
    agree mark a finding the report genuinely determines, passes that scatter mark
    one it does not. 51% of the residual error sits on findings the report never
    mentions, and a single pass states those with false conviction.
    """
    if samples <= 1:
        return _one(text, effort, timeout, temperature, max_tokens, thinking)

    runs, errs = [], []
    for _ in range(samples):
        try:
            runs.append(_one(text, effort, timeout, temperature, max_tokens, thinking))
        except Exception as exc:                             # noqa: BLE001
            errs.append(exc)
    if not runs:                        # every pass failed - let the caller record it
        raise errs[0]

    out = {}
    for k in LABELS:
        ps = [r["findings"][k]["p"] for r in runs]
        out[k] = {"p": float(sum(ps) / len(ps)),
                  # widest disagreement between any two passes, per finding
                  "spread": float(max(ps) - min(ps)),
                  "samples": ps,
                  # the evidence sentence from the first pass that actually cited one
                  "evidence": next((r["findings"][k]["evidence"] for r in runs
                                    if r["findings"][k]["evidence"]), "")}
    return {"unusable": sum(r["unusable"] for r in runs) > len(runs) / 2,
            "findings": out, "n_samples": len(runs)}


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
    ap.add_argument("--thinking", action="store_true",
                    help="send chat_template_kwargs={enable_thinking: true}. Gemma "
                         "IGNORES the OpenAI-style `reasoning` field - its thinking "
                         "lives in the chat template, so --effort alone does nothing "
                         "for it (measured: identical output at medium and xhigh). "
                         "Qwen is the opposite and uses --effort.")
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v1",
                    help="v1 = the prompt that measured 0.9114 at effort=medium. "
                         "v2 = v1 with numeric anchors added to the Synovitis rule.")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=4000,
                    help="completion budget per call. 4000 is ample at effort=low "
                         "(measured 1,435 tokens), but reasoning tokens come out of "
                         "the SAME budget: at effort=medium the model spent all 4000 "
                         "reasoning and returned an empty reply, failing 48 of 58. "
                         "Raise it to ~16000 for medium/xhigh. Truncated generations "
                         "are still billed.")
    ap.add_argument("--samples", type=int, default=1,
                    help="self-consistency: label each report N times and average. "
                         "The labeler disagrees with itself on 3.3%% of decisions, so "
                         "N passes cut that by about sqrt(N). Costs N x the calls.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature. Only meaningful with --samples > 1, "
                         "where it is what makes the passes differ; 0.0 still varies a "
                         "little but explores far less.")
    ap.add_argument("--translations", default="",
                    help="an alternative 02_reports_en_*.csv, layered OVER the default "
                         "one. A study it does not cover keeps its original English "
                         "text, so re-translating a subset changes only that subset.")
    ap.add_argument("--mode", choices=["soft", "hard"], default="soft",
                    help="soft = a probability per finding (the default, and what the "
                         "corpus was labeled with). hard = commit to 0 or 1, matching the "
                         "panel's own format. NOTE: hard labels cannot win an AUROC "
                         "comparison - with two values almost every pair is a tie, worth "
                         "0.5 each - so score them on balanced accuracy and F1 instead.")
    ap.add_argument("--tag", default="",
                    help="isolate this run's outputs, e.g. --tag gemma4-31b writes "
                         "03_report_labels_gemma4-31b.csv / _raw_gemma4-31b.jsonl. "
                         "Without it a second labeler would resume from, and append "
                         "to, the first one's checkpoint.")
    args = ap.parse_args()
    global SYSTEM, MODE

    MODE = args.mode
    SYSTEM = (SYSTEM_HARD if MODE == "hard" else
              (SYSTEM_SOFT_V2 if args.prompt == "v2" else SYSTEM_SOFT))
    print(f"[mode] {MODE} labels | prompt {args.prompt}")

    if args.tag:
        global OUT_CSV, OUT_JSONL, LOG_TXT
        OUT_CSV = OUT_CSV.with_name(f"03_report_labels_{args.tag}.csv")
        OUT_JSONL = OUT_JSONL.with_name(f"03_report_labels_raw_{args.tag}.jsonl")
        LOG_TXT = LOG_TXT.with_name(f"03_labeling_log_{args.tag}.txt")
        print(f"[tag] outputs isolated as *_{args.tag}")

    # Only Modal constrains decoding to SCHEMA. Ollama treats it as a hint, so models
    # answer in whatever shape they like - gpt-oss:120b returns a markdown TABLE, and
    # gemma4:31b changes between {probability, sentence} and [0.05, "sentence"] call to
    # call. Spelling the contract out in the prompt is the only lever left there.
    from llm_client import load_env as _le
    if not _le().get("modal"):
        ex = ('{"unusable": false, "findings": {"ACL": {"p": 0, "evidence": "..."}, ...}}'
              if MODE == "hard" else
              '{"unusable": false, "findings": {"ACL": {"p": 0.05, "evidence": "..."}, ...}}')
        val = ("the integer 0 or the integer 1, nothing else" if MODE == "hard"
               else "a number 0.0-1.0")
        SYSTEM = SYSTEM + (
            "\n\nOUTPUT FORMAT - follow exactly, no exceptions:\n"
            "Reply with ONE raw JSON object and nothing else. No prose, no explanation,\n"
            "no markdown, no ``` fences, no table. The object must be:\n\n"
            + ex + "\n\nOne entry per finding, using EXACTLY these 12 keys and no "
            "others:\n" + ", ".join(LABELS) +
            "\nEach entry has exactly two fields: 'p' (" + val + ") and 'evidence' "
            "(the exact sentence you relied on, or an empty string)."
        )

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
        if args.translations:
            # layered, not swapped: the override file covers a subset, and every study
            # outside it must keep the exact text the baseline run saw, or the
            # comparison would move two things at once
            alt = (pd.read_csv(args.translations)[["StudyInstanceUID", "Report_en"]]
                   .dropna(subset=["Report_en"]).set_index("StudyInstanceUID")["Report_en"])
            hit = train.StudyInstanceUID.map(alt)
            print(f"[translations] {hit.notna().sum()} of {len(train)} studies overridden "
                  f"from {Path(args.translations).name}")
            train["Report_en"] = hit.fillna(train["Report_en"])
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
        futs = {pool.submit(label, t, args.effort, args.timeout,
                            args.samples, args.temperature, args.max_tokens,
                            args.thinking): t for t in uniq}
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
    if args.samples > 1:
        # the spread is as much the point of self-consistency as the mean: findings the
        # passes agree on are ones the report genuinely determines, findings they
        # scatter on are the silent cells a single pass states with false conviction
        sp = pd.DataFrame([{k: v.get("spread", 0.0) for k, v in res["findings"].items()}
                           for res in results.values()
                           if "spread" in next(iter(res["findings"].values()), {})])
        if len(sp):
            w(f"\ndisagreement across the {args.samples} passes (mean max-min):")
            w(sp.mean().round(3).sort_values(ascending=False).to_string())
            w(f"  overall mean spread {sp.values.mean():.3f} | "
              f"cells that moved at all {(sp.values > 0).mean():.1%} | "
              f"cells that moved more than 0.2 {(sp.values > 0.2).mean():.1%}")

    w("\nfraction above 0.5 (%):")
    w(((out[LABELS] > 0.5).mean() * 100).round(1).sort_values(ascending=False).to_string())

    if args.validate:
        from sklearn.metrics import f1_score, roc_auc_score
        truth = df.set_index("StudyInstanceUID")[LABELS].astype(int)
        pred = out.set_index("StudyInstanceUID")[LABELS]
        common = truth.index.intersection(pred.index)
        truth, pred = truth.loc[common], pred.loc[common]
        aur = {c: (roc_auc_score(truth[c].values, pred[c].values)
                   if truth[c].nunique() > 1 else float("nan")) for c in LABELS}
        # AUROC only ranks, so it is not comparable across modes: two distinct values
        # makes almost every pair a tie, and a tie scores 0.5. Binarising the existing
        # soft qwen24t labels at 0.5 costs 0.058 AUROC while leaving every DECISION
        # identical. Balanced accuracy and F1 read the decisions instead, so they are
        # what a hard run should be judged on - and they apply to a soft run too, at
        # the same 0.5 cut.
        hard = (pred.values >= 0.5).astype(float)
        Y = truth.values.astype(float)
        sens = (hard * Y).sum() / max(Y.sum(), 1)
        spec = ((1 - hard) * (1 - Y)).sum() / max((1 - Y).sum(), 1)
        bacc = {c: ((pred[c].values >= 0.5).astype(float) * truth[c].values).sum()
                   / max(truth[c].sum(), 1) for c in LABELS}
        f1 = {c: f1_score(truth[c].values, (pred[c].values >= 0.5).astype(int),
                          zero_division=0) for c in LABELS}
        brier = {c: float(((pred[c].values - truth[c].values) ** 2).mean()) for c in LABELS}
        w(f"\n=== scored against the {len(common)} official studies ===")
        w(pd.DataFrame({"positives": truth.sum(),
                        "mean_p_pos": [pred.loc[truth[c] == 1, c].mean() for c in LABELS],
                        "mean_p_neg": [pred.loc[truth[c] == 0, c].mean() for c in LABELS],
                        "auroc": pd.Series(aur),
                        "recall": pd.Series(bacc),
                        "f1": pd.Series(f1),
                        "brier": pd.Series(brier)}).round(3).to_string())
        w(f"\nMODE            = {MODE}")
        w(f"MACRO AUROC     = {pd.Series(aur).mean():.4f}"
          + ("   <- NOT comparable to a soft run; see below" if MODE == "hard" else ""))
        w(f"MACRO F1        = {pd.Series(f1).mean():.4f}")
        w(f"BALANCED ACC    = {(sens + spec) / 2:.4f}   (sens {sens:.3f} / spec {spec:.3f})")
        w(f"BRIER           = {pd.Series(brier).mean():.4f}   (lower is better)")
        if MODE == "hard":
            w(f"snapped off 0/1 = {_SNAPPED[0]}   "
              + ("(clean - the endpoint enforced the enum)" if not _SNAPPED[0]
                 else "(endpoint did NOT enforce the enum; these were rounded, so this "
                      "is partly a soft run in disguise)"))
            w("compare a hard run on BALANCED ACC / F1, never on AUROC.")
        w("keyword-matcher floor = 0.7003 AUROC")
        w("caveats: 58 studies carries roughly +/-0.05 of noise, and the labels are")
        w("         image-derived, so perfect agreement is not attainable.")

    LOG_TXT.parent.mkdir(parents=True, exist_ok=True)
    LOG_TXT.write_text("\n".join(L) + "\n", encoding="utf-8")
    w(f"\nwritten: {csv_path}\n         {OUT_JSONL}\n         {LOG_TXT}")


if __name__ == "__main__":
    main()
