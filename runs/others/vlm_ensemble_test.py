"""vlm_ensemble_test.py - does the Qwen3.5-2B VLM help the ensemble, and how different is it? (7 Oct)

Member: qwen35_2b_vlm_sys at STEP (argument, default its best val step so far; it may still be training).
Base: fit-10's core = S6 (zoom_ensemble_test.py) + z030 on the 5 joint findings; also the batch-3 fourteen.
Flat logit mean per finding, qwen added at weight 1, 2, 3 on all 12, and swapped in for each S6 member.
Surfaces: audit-58 (panel labels) and clean-121; paired bootstrap share of resamples where it helps.
Diversity: per-finding Spearman correlation of a member's logits with the mean of the OTHER base members (clean-121),
averaged over findings; qwen vs every S6 member (lower = more different).

    python runs/others/vlm_ensemble_test.py [step]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
src = (HERE / "zoom_ensemble_test.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "zoom_ensemble_test.py"), "__name__": "zoom_scaffold"}
exec(compile(src[:src.index("for bname, base in")], "zoom_ensemble_test.py", "exec"), g)
T, BASE, S6, JM, data, blend, full_auc, auc_cols, R = (g[k] for k in
                                                        ("T", "BASE", "S6", "JM", "data", "blend", "full_auc", "auc_cols", "R"))
Q = "qwen35_2b_vlm_sys"
v = pd.read_csv(R / Q / "results" / "val_log.csv").drop_duplicates(["run_id", "step"], keep="last")
step = int(sys.argv[1]) if len(sys.argv) > 1 else int(v.loc[v.val_macro_auroc.idxmax(), "step"])
g["g"]["STEP"][Q] = step
row = v[v.step == step].iloc[-1]
print(f"qwen @ step {step}: val {row.val_macro_auroc:.4f} | audit {row.audit_macro_auroc:.4f}")
for s, (P, y, I) in data.items():
    ids = g["SURF"][s][0]
    P["qwen"] = g["g"]["get"](Q, ids).loc[ids, T].astype(float).to_numpy()
    print(f"  qwen solo on {s}: {full_auc(y, P['qwen']):.4f}")

core = [m for m in S6 if m in BASE]
Z = [("z030", 1, JM.astype(float))]


def show(bname, mem, extra):
    ref = {s: (full_auc(y, blend(P, mem, extra)), auc_cols(y, blend(P, mem, extra), I)) for s, (P, y, I) in data.items()}
    print(f"\n=== base: {bname}: audit {ref['audit58'][0]:.4f} | clean-121 {ref['clean121'][0]:.4f}")
    vs = {f"qwen add12 w{w}": (mem, extra + [("qwen", w, np.ones(12))]) for w in (1, 2, 3)}
    vs["qwen addJ w1"] = (mem, extra + [("qwen", 1, JM.astype(float))])
    vs["qwen addO w1 (other 7)"] = (mem, extra + [("qwen", 1, (~JM).astype(float))])
    for m in mem:
        vs[f"qwen swap {m[:40]}"] = ([x for x in mem if x != m], extra + [("qwen", 1, np.ones(12))])
    for name, (mm, ex) in vs.items():
        r = []
        for s, (P, y, I) in data.items():
            p = blend(P, mm, ex)
            a = full_auc(y, p)
            r.append((a, a - ref[s][0], ((auc_cols(y, p, I) - ref[s][1]) > 0).mean()))
        (a, da, pa), (c, dc, pc) = r
        flag = "  <- both up" if da > 0 and dc > 0 else ""
        print(f"  {name:48s} audit {a:.4f} ({da:+.4f}, {pa:4.0%}) | clean {c:.4f} ({dc:+.4f}, {pc:4.0%}){flag}")


show("fit-10 core (S6 + z030 joint)", core, Z)
show("all 14", BASE, [])

P, y, _ = data["clean121"]


def corr(name, others):
    o = np.mean([P[m] for m in others], 0)
    return np.nanmean([spearmanr(P[name][:, j], o[:, j])[0] for j in range(len(T))])


print("\n=== diversity (clean-121): Spearman with the mean of the other fit-10 core members, lower = more different")
for m in core + ["z030", "qwen"]:
    print(f"  {m[:60]:60s} {corr(m, [x for x in core if x != m]):.3f}")
