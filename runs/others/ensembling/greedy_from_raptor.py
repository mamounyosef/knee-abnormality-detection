"""greedy_from_raptor.py - build a NEW ensemble one member at a time, starting from the raptor pair (12 Oct, user).

Start: convnext_base_v5_raptor96_local x0.7 + medimageinsight_davit_huge_v5_raptor96_sam x0.3 (logit mean; LB 0.940).
Step:  try every candidate at every share w (P' = (1 - w) P + w c, logit space), keep the one that raises the
       objective most; a member already in may be picked again (its weight grows, Caruana-style). Stop when the best
       step gains < MIN_GAIN.
Surfaces (unseen by every member used):
  audit-58   panel labels; every member's audit npz at its best-val step
  clean-121  the 58 + the 63 v5-val studies outside the old training split (v3 labels). A member is eligible only if
             none of the 121 is in its training csv (fulldata runs drop out). 5-fold FAMILIES (<name>_f0..f4) are one
             candidate: the fold mean on the 58, each fold's out-of-fold prediction on the 63.
Objective: mean of audit-58 and clean-121 macro AUROC. Each accepted step reports both, with paired-bootstrap P(better).
Honesty check (--crossfit N): the 121 split in random halves N times; the greedy search runs on one half, its picks
are scored on the other half vs the start. Reports how much of the in-sample gain survives.
Runtime: Kaggle T4 gpu s/study per member from runtime_ledger.json / the raptor stresses (n/a when never measured).

    python runs/others/ensembling/greedy_from_raptor.py [--crossfit 40]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent
D = R.parent / "data"
sys.path.insert(0, str(HERE))
import sweep_new_runs as sw  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
START = [("convnext_base_v5_raptor96_local", 0.7), ("medimageinsight_davit_huge_v5_raptor96_sam", 0.3)]
STEP = {"convnext_base_25d_poolattn_v5": 3700, "swinv2_base_in22k_v5_warm": 2000,
        "convnext_base_v5_raptor96_local": 9000, "medimageinsight_davit_huge_v5_raptor96_sam": 5300}
SHARES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
MIN_GAIN = 0.0003
MIN_SOLO = 0.86                     # skip broken / partial members
# gpu s/study on Kaggle T4 measured in the 11 Oct raptor stresses (not in the ledger yet)
RAPTOR_S = {"convnext_base_v5_raptor96_local": 1.03, "medimageinsight_davit_huge_v5_raptor96_sam": 2.91}

ap = argparse.ArgumentParser()
ap.add_argument("--crossfit", type=int, default=0)
ap.add_argument("--exclude", nargs="*", default=[], help="members to leave out (e.g. no checkpoint left anywhere)")
args = ap.parse_args()

# ---------------------------------------------------------------- surfaces
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
v5 = pd.read_csv(D / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
old_train = set(pd.read_csv(D / "05_train_qwen24t-med.csv").StudyInstanceUID)
A58 = list(aud.index)
C63 = [s for s in v5.index if v5.loc[s, "source"] != "audit" and s not in old_train]
IDS = A58 + C63
Y = np.vstack([(aud.loc[A58, ["official_" + t for t in T]].values >= .5),
               (v5.loc[C63, T].values > .5)]).astype(int)
IA = np.arange(len(A58))                     # audit rows inside the 121
IC = np.arange(len(IDS))


SHARED_PATHS = (yaml.safe_load((R / "shared_config.yaml").read_text(encoding="utf-8")) or {}).get("paths") or {}


def cfg(run):
    """the run's own config.yaml, with paths falling back to shared_config's (05_train / 05_val)."""
    p = R / run / "config.yaml"
    c = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}
    if not p.exists():
        return c
    c["paths"] = {**SHARED_PATHS, **(c.get("paths") or {})}
    return c


def csv_set(name):
    p = Path(name) if Path(name).is_absolute() else D / name
    return set(pd.read_csv(p).StudyInstanceUID) if name and p.exists() else None


def load(run, split):
    d = R / run / "results" / "predictions"
    if run in STEP:
        fs = sorted(d.glob(f"*_{split}_s{STEP[run]}.npz"))
        f = fs[-1] if fs else None
    else:
        f = sw.best_npz(run, split)
    if f is None:
        return None
    z = np.load(f, allow_pickle=True)
    if "tasks" in z.files and [str(t) for t in z["tasks"]] != T:
        return None
    if z["logits"].shape[1] != len(T):
        return None
    return pd.DataFrame(z["logits"], index=[str(x) for x in z["study_ids"]], columns=T)


def single(run):
    """(121, 12) logits or None, with the reason."""
    c = cfg(run)
    tr = csv_set(str((c.get("paths") or {}).get("train_csv", "")))
    if tr is None:
        return None, "no train csv"
    if tr & set(IDS):
        return None, "trained on the 121"
    a, v = load(run, "audit"), load(run, "val")
    if a is None or not set(A58) <= set(a.index):
        return None, "no audit npz"
    if v is None or not set(C63) <= set(v.index):
        return None, "val npz misses the 63"
    return np.vstack([a.loc[A58, T].values, v.loc[C63, T].values]), ""


def family(base):
    """5-fold family: fold mean on the 58, out-of-fold on the 63."""
    folds = [f"{base}_f{k}" for k in range(5)]
    aud_p, oof = [], {}
    for f in folds:
        c = cfg(f)
        tr = csv_set(str((c.get("paths") or {}).get("train_csv", "")))
        va = csv_set(str((c.get("paths") or {}).get("val_csv", "")))
        if tr is None or va is None:
            return None, "fold csv missing"
        if tr & set(A58):
            return None, "a fold trained on the audit"
        a, v = load(f, "audit"), load(f, "val")
        if a is None or v is None:
            return None, f"{f} npz missing"
        aud_p.append(a.loc[A58, T].values)
        for s in C63:
            if s in va and s in v.index:
                oof[s] = v.loc[s, T].values
            elif s in tr:
                pass
    if not set(C63) <= set(oof):
        return None, "OOF misses some of the 63"
    return np.vstack([np.mean(aud_p, 0), np.array([oof[s] for s in C63])]), ""


# ---------------------------------------------------------------- pool
runs = sorted(p.parent.parent.name for p in R.glob("*/results/predictions"))
fam = sorted({m.group(1) for r in runs for m in [re.match(r"(.+)_f([0-4])$", r)] if m
              and all(f"{m.group(1)}_f{k}" in runs for k in range(5))})
POOL, why = {}, {}
for r in runs:
    if re.match(r".+_f[0-4]$", r) and r.rsplit("_f", 1)[0] in fam:
        continue
    p, w = single(r)
    (POOL.__setitem__(r, p) if p is not None else why.__setitem__(r, w))
for b in fam:
    p, w = family(b)
    (POOL.__setitem__(b + "_f*", p) if p is not None else why.__setitem__(b + "_f*", w))
for s, _ in START:
    assert s in POOL, f"start member {s} not eligible: {why.get(s)}"


def aucs(y, p):
    """macro AUROC of each row-set in p (k, n, 12) against y (n, 12)."""
    out = np.zeros(p.shape[0])
    for j in range(12):
        yy = y[:, j]; n1 = yy.sum(); n0 = len(yy) - n1
        r = rankdata(p[:, :, j], axis=1)
        out += (r[:, yy == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out / 12


def mac(p, rows):
    return aucs(Y[rows], p[None, rows])[0]


solo = {k: (mac(v, IA), mac(v, IC)) for k, v in POOL.items()}
POOL = {k: v for k, v in POOL.items() if min(solo[k]) >= MIN_SOLO and k not in args.exclude
        and not k.startswith("radimagenet")}         # RadImageNet weights are non-commercial: never shipped
print(f"pool: {len(POOL)} eligible candidates ({len(fam)} fold families); excluded {len(why)}:")
for k in sorted(why):
    print(f"   - {k}: {why[k]}")

# runtime per member (latest ledger figure), s/study gpu
led = json.loads((HERE / "runtime_ledger.json").read_text(encoding="utf-8"))["runs"]
RT = {}
for e in led.values():                       # stress runs only (>= 60 studies): 3-study commit runs carry model load
    if (e.get("n_studies_measured") or 0) < 60:
        continue
    for m, s in (e.get("members") or {}).items():
        RT[m] = min(s, RT.get(m, s))
RT.update(RAPTOR_S)


def rt(k):
    """measured s/study, else the same architecture's measured figure (prefixed ~ when estimated)."""
    if k.endswith("_f*"):
        fs = [RT.get(k[:-1] + str(i)) for i in range(5)]
        return sum(fs) if all(x is not None for x in fs) else None
    if k in RT:
        return RT[k]
    base = re.sub(r"(_v5)?(_warm)?$", "", k)
    if base in RT:
        return RT[base]
    for pre, ref in (("dinov3_convnext_base", "dinov3_convnext_base"), ("mricore_vitb_mae", "mricore_vitb_mae_cross_attention"),
                     ("mricore_vitb", "mricore_vitb_cross_attention"), ("swinv2", "swinv2_base_in22k"),
                     ("convnext_large", "convnext_large"), ("convnext_base", "convnext_base_relabel")):
        if k.startswith(pre):
            return RT[ref]
    return None


# ---------------------------------------------------------------- greedy
def greedy(rows_obj, halves=False):
    """halves: a cross-fit half scores on its own rows only (its audit part can lack a finding's positives)."""
    W = {k: w for k, w in START}
    P = sum(w * POOL[k] for k, w in START)
    rc = rows_obj
    ra = rc if halves else [r for r in rows_obj if r < len(A58)]

    def obj(p):
        return (mac(p, ra) + mac(p, rc)) / 2

    cur = obj(P)
    hist = [("start", None, P.copy(), cur)]
    keys = sorted(POOL)
    while True:
        best = None
        for k in keys:
            c = POOL[k]
            cands = np.stack([(1 - w) * P + w * c for w in SHARES])
            ya, yc = Y[ra], Y[rc]
            sc = (aucs(ya, cands[:, ra]) + aucs(yc, cands[:, rc])) / 2
            i = int(np.argmax(sc))
            if best is None or sc[i] > best[0]:
                best = (sc[i], k, SHARES[i], cands[i])
        if best[0] - cur < MIN_GAIN:
            break
        s, k, w, P = best
        W = {m: x * (1 - w) for m, x in W.items()}
        W[k] = W.get(k, 0) + w
        cur = s
        hist.append((k, w, P.copy(), s))
    return W, hist


rng = np.random.default_rng(0)
BI = rng.integers(0, len(IDS), (2000, len(IDS)))
BI = BI[[all(0 < Y[i, j].sum() < len(i) for j in range(12)) for i in BI]]
BA = rng.integers(0, len(A58), (2000, len(A58)))
BA = BA[[all(0 < Y[i, j].sum() < len(i) for j in range(12)) for i in BA]]


def pbetter(p_new, p_old, B, Ys):
    a = np.array([aucs(Ys[i], np.stack([p_new[i], p_old[i]]))for i in B])
    return (a[:, 0] > a[:, 1]).mean()


W, hist = greedy(IC)
P0 = hist[0][2]
print(f"\nSTART raptor pair 0.7 / 0.3: audit {mac(P0, IA):.4f} | clean-121 {mac(P0, IC):.4f}")
print(f"{'step':>4} {'added':58s} {'share':>5} {'audit-58':>18} {'clean-121':>18} {'P(a)':>5} {'P(c)':>5}  gpu s/study")
prev = P0
Ya, Yc = Y[:len(A58)], Y
for n, (k, w, P, s) in enumerate(hist[1:], 1):
    pa = pbetter(P[:len(A58)], prev[:len(A58)], BA, Ya)
    pc = pbetter(P, prev, BI, Yc)
    print(f"{n:4d} {k:58s} {w:5.2f} {mac(P, IA):.4f} ({mac(P, IA) - mac(prev, IA):+.4f}) "
          f"{mac(P, IC):.4f} ({mac(P, IC) - mac(prev, IC):+.4f}) {pa:5.0%} {pc:5.0%}  {rt(k) or float('nan'):.2f}")
    prev = P
Pf = hist[-1][2]
print(f"\nFINAL vs start: audit {mac(Pf, IA):.4f} ({mac(Pf, IA) - mac(P0, IA):+.4f}, "
      f"P {pbetter(Pf[:len(A58)], P0[:len(A58)], BA, Ya):.0%}) | clean-121 {mac(Pf, IC):.4f} "
      f"({mac(Pf, IC) - mac(P0, IC):+.4f}, P {pbetter(Pf, P0, BI, Yc):.0%})")
tot = 0.0
print("final weights (sum 1):")
for k, w in sorted(W.items(), key=lambda x: -x[1]):
    r_ = rt(k)
    tot += r_ or 0
    print(f"   {w:6.3f}  {k:58s} solo audit {solo[k][0]:.4f} clean {solo[k][1]:.4f}  gpu {r_ or float('nan'):.2f}")
print(f"gpu s/study summed over measured members: {tot:.2f} (x ~1300 hidden studies = {tot * 1300 / 3600:.2f} h, "
      f"+ decode + 15 min fixed); unmeasured: {[k for k in W if rt(k) is None]}")

if args.crossfit:
    print(f"\nCROSS-FIT x{args.crossfit}: greedy on a random half of the 121, scored on the other half vs the start")
    gains, picks = [], {}
    for it in range(args.crossfit):
        perm = rng.permutation(len(IDS))
        A, B = np.sort(perm[: len(IDS) // 2]), np.sort(perm[len(IDS) // 2:])
        if not all(0 < Y[B, j].sum() < len(B) for j in range(12)) or not all(0 < Y[A, j].sum() < len(A) for j in range(12)):
            continue
        Wi, hi = greedy(A, halves=True)
        g_in = hi[-1][3] - hi[0][3]
        g_out = mac(hi[-1][2], B) - mac(hi[0][2], B)
        gains.append((g_in, g_out))
        for k in Wi:
            if k not in dict(START):
                picks[k] = picks.get(k, 0) + 1
    g = np.array(gains)
    print(f"   {len(g)} splits: in-sample gain {g[:, 0].mean():+.4f} | HELD-OUT gain {g[:, 1].mean():+.4f} "
          f"(median {np.median(g[:, 1]):+.4f}, > 0 in {(g[:, 1] > 0).mean():.0%})")
    print("   most-picked members:", ", ".join(f"{k} {v}" for k, v in sorted(picks.items(), key=lambda x: -x[1])[:12]))
