"""rank_single_models.py - every single run ranked by audit-58, with its validation score and ranks.

For each run with saved predictions, the checkpoint taken is its best-validation step (what
submit.py ships). Scores are recomputed from the saved logits against FIXED labels, so runs
trained on different label sets compare fairly:
  audit-58  vs the expert panel
  val-704   vs the v2 labels (05_val_qwen24t-med), the set most runs trained on
Runs whose validation or audit studies were in training (full-data runs, cross-validation
folds) are excluded, as are runs with fewer than 704 val / 58 audit predictions.

    python runs/others/rank_single_models.py [--top 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

RUNS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUNS / "others/ensembling"))
import sweep_new_runs as sw  # noqa: E402

D = RUNS.parent / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def macro(lg, y):
    r = rankdata(lg, axis=0)
    n1 = y.sum(0).astype(float)
    n0 = len(y) - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        a = ((r * y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(np.nanmean(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
    ya = (aud[["official_" + t for t in T]].values >= 0.5).astype(int)
    val = pd.read_csv(D / "05_val_qwen24t-med.csv").set_index("StudyInstanceUID")
    yv = (val[T].values >= 0.5).astype(int)
    rows = []
    for d in sorted(p for p in RUNS.iterdir() if (p / "config.yaml").exists()):
        n = d.name
        if sw._trains_on_val(n) or sw.LEAKY.search(n) or "_v3_f" in n or n.endswith("_smoke"):
            continue
        fv, fa = sw.best_npz(n, "val"), sw.best_npz(n, "audit")
        if fv is None or fa is None:
            continue
        try:
            zv, za = np.load(fv, allow_pickle=True), np.load(fa, allow_pickle=True)
            if zv["logits"].shape[1] != 12:
                continue
            lv = pd.DataFrame(zv["logits"], index=[str(s) for s in zv["study_ids"]])
            la = pd.DataFrame(za["logits"], index=[str(s) for s in za["study_ids"]])
            if not set(val.index) <= set(lv.index) or not set(aud.index) <= set(la.index):
                continue
            rows.append((n, macro(la.loc[aud.index].values, ya), macro(lv.loc[val.index].values, yv),
                         int(str(fv.stem).split("_s")[-1])))
        except Exception as e:                                         # noqa: BLE001
            print(f"  skipped {n}: {type(e).__name__}: {e}")
    df = pd.DataFrame(rows, columns=["run", "audit58", "val704_v2", "step"])
    df["audit_rank"] = df.audit58.rank(ascending=False, method="min").astype(int)
    df["val_rank"] = df.val704_v2.rank(ascending=False, method="min").astype(int)
    lb = solo_public_scores()
    df["public"] = df.run.map(lb)
    df["public_rank"] = df.public.rank(ascending=False, method="min")
    df = df.sort_values("audit58", ascending=False).reset_index(drop=True)
    print(f"{len(df)} single runs scored; {df.public.notna().sum()} of them have a solo public score\n")
    print("| # | run | audit-58 | audit rank | val-704 (v2 labels) | val rank | public LB | LB rank |")
    print("|---|---|---|---|---|---|---|---|")
    for i, r in df.head(a.top).iterrows():
        pub = f"{r.public:.3f}" if pd.notna(r.public) else "-"
        pr = f"{int(r.public_rank)}" if pd.notna(r.public_rank) else "-"
        print(f"| {i + 1} | `{r.run}` | {r.audit58:.4f} | {r.audit_rank} | {r.val704_v2:.4f} | {r.val_rank} | {pub} | {pr} |")


def solo_public_scores():
    """run -> its solo public score: submissions.json single-member entries (latest wins), else
    the run's own training_summary.json submission block."""
    import json
    import re
    out = {}
    # runs_summary.md single-run table: | `run` | params | val | audit | LB | ...
    for line in (RUNS.parent / "runs_summary.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| `([a-z0-9_]+)` \|[^|]*\|[^|]*\|[^|]*\| \**([0-9]\.[0-9]+)\** \|", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    for d in RUNS.iterdir():
        f = d / "results/training_summary.json"
        if f.exists():
            try:
                sub = json.loads(f.read_text(encoding="utf-8")).get("submission") or {}
                s = sub.get("public_score")
                # a run's block sometimes records the ENSEMBLE it joined; those are not solo
                if s is not None and "not solo" not in str(sub.get("notes") or "").lower():
                    out[d.name] = float(s)
            except Exception:                                           # noqa: BLE001
                pass
    subs = json.loads((RUNS / "others/ensembling/submissions.json").read_text(encoding="utf-8"))
    subs = subs.get("submissions", subs) if isinstance(subs, dict) else subs
    for e in subs:
        if isinstance(e, dict) and len(e.get("members") or []) == 1 and e.get("public_score") is not None:
            out[e["members"][0]] = float(e["public_score"])
    return out


if __name__ == "__main__":
    main()
