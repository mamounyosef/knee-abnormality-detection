"""manual_build.py - score a HAND-PICKED build order on audit-58 and clean-121, one step at a time (12 Oct, user).

Same members, surfaces and eligibility rules as greedy_from_96slices_per_finding_attn.py (imported from it, so the two cannot drift).
Each step is (member, share): P' = (1 - share) P + share c in logit space. A step whose member is "-name" removes that
member and renormalises. Every step prints both surfaces with the paired-bootstrap P(better) vs the previous step and
vs the start. Edit STEPS and run.

    python runs/others/ensembling/manual_build.py
"""
import sys
from pathlib import Path

import numpy as np

sys.argv = [sys.argv[0]]                         # greedy_from_96slices_per_finding_attn parses argv; no cross-fit on import
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
src = (HERE / "greedy_from_96slices_per_finding_attn.py").read_text(encoding="utf-8")
g = {"__name__": "gfr", "__file__": str(HERE / "greedy_from_96slices_per_finding_attn.py")}
exec(compile(src[:src.index("\n# ---------------------------------------------------------------- greedy")],
             "greedy_from_96slices_per_finding_attn.py", "exec"), g)
POOL, mac, IA, IC, Y, A58, START, rt, solo, aucs = (g[k] for k in
                                                    ("POOL", "mac", "IA", "IC", "Y", "A58", "START", "rt", "solo", "aucs"))
rng = np.random.default_rng(0)
BI = rng.integers(0, len(Y), (2000, len(Y)))
BI = BI[[all(0 < Y[i, j].sum() < len(i) for j in range(12)) for i in BI]]
BA = rng.integers(0, len(A58), (2000, len(A58)))
BA = BA[[all(0 < Y[i, j].sum() < len(i) for j in range(12)) for i in BA]]


def pbetter(p_new, p_old, B, Ys):
    a = np.array([aucs(Ys[i], np.stack([p_new[i], p_old[i]])) for i in B])
    return (a[:, 0] > a[:, 1]).mean()


STEPS = [l.split(":") for l in sys.stdin.read().split()] if not sys.stdin.isatty() else []


def run(steps):
    W = dict(START)
    P = sum(w * POOL[k] for k, w in W.items())
    P0 = P.copy()
    print(f"START {' + '.join(f'{k} x{w}' for k, w in W.items())}: audit {mac(P, IA):.4f} | clean-121 {mac(P, IC):.4f}")
    for name, share in steps:
        share = float(share)
        prev = P
        if name.startswith("-"):
            k = name[1:]
            W.pop(k)
            tot = sum(W.values())
            W = {m: x / tot for m, x in W.items()}
            P = sum(w * POOL[m] for m, w in W.items())
            label = f"REMOVE {k}"
        else:
            P = (1 - share) * P + share * POOL[name]
            W = {m: x * (1 - share) for m, x in W.items()}
            W[name] = W.get(name, 0) + share
            label = f"+ {name} @{share:.2f}"
        pa = pbetter(P[:len(A58)], prev[:len(A58)], BA, Y[:len(A58)])
        pc = pbetter(P, prev, BI, Y)
        print(f"{label:62s} audit {mac(P, IA):.4f} ({mac(P, IA) - mac(prev, IA):+.4f}, {pa:4.0%}) | "
              f"clean {mac(P, IC):.4f} ({mac(P, IC) - mac(prev, IC):+.4f}, {pc:4.0%}) | vs start "
              f"{mac(P, IA) - mac(P0, IA):+.4f} / {mac(P, IC) - mac(P0, IC):+.4f}")
    tot = sum(rt(k) or 0 for k in W)
    print("weights:", ", ".join(f"{k} {w:.3f}" for k, w in sorted(W.items(), key=lambda x: -x[1])))
    print(f"gpu s/study {tot:.2f} -> ~{tot * 1300 / 3600:.2f} h on ~1300 hidden studies (+ decode + 15 min)")


if __name__ == "__main__":
    print("candidates (solo audit / clean):")
    for k in sorted(POOL, key=lambda k: -sum(solo[k])):
        print(f"   {k:60s} {solo[k][0]:.4f} / {solo[k][1]:.4f}  gpu {rt(k) or float('nan'):.2f}")
    if STEPS:
        run(STEPS)
