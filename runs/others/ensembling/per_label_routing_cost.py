"""route_cheap.py - the SHIPPABLE version of per-finding routing.

The unrestricted top-3 router needs 27 checkpoints from 13 runs and prices at ~4x the 9 h
Kaggle cap. But most of that cost is new ARCHITECTURES. Restrict the candidate pool to
checkpoints of the seven models the submission already loads, and the extra cost is only
additional forward passes on an already-decoded batch - no new decode group, no new weights
to download, no new geometry.

Prints, for k = 1, 3, 5:
  * the per-finding gain using only in-ensemble checkpoints
  * how many EXTRA forward passes that costs and the runtime against the 9 h cap
  * the cross-fitted (honest) macro delta
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling")
sys.path.insert(0, str(HERE))
import per_label_scan as P  # noqa: E402

SEC_PER_MEMBER = 15.6 / 8          # measured: 8 members at 15.6 s/study
STUDIES = 1300
CAP_H = 9.0

y, cands, order = P.load_all()
n = len(y)
names = [c[0] for c in cands]
runs = [c[1] for c in cands]
mats = [c[2] for c in cands]

pool = [i for i, r in enumerate(runs) if r in P.SEVEN]
print("\ncandidate pool restricted to the 7 shipped models: %d checkpoints "
      "(from %d across the repo)\n" % (len(pool), len(cands)))

base = np.zeros((n, 12))
base_ck = []
for mem in P.SEVEN:
    idx = [i for i, r in enumerate(runs) if r == mem]
    bi, bs = None, -1
    for i in idx:
        s = np.nanmean([P.auc(y[:, j], mats[i][:, j]) for j in range(12)])
        if s == s and s > bs:
            bi, bs = i, s
    base_ck.append(bi)
    for j in range(12):
        base[:, j] += P.rank_norm(mats[bi][:, j])
base /= len(P.SEVEN)
bcols = np.array([P.auc(y[:, j], base[:, j]) for j in range(12)])
print("baseline macro %.4f\n" % np.nanmean(bcols))

sc = np.full((len(cands), 12), np.nan)
for j in range(12):
    for i in pool:
        sc[i, j] = P.auc(y[:, j], mats[i][:, j])

for k in (1, 3, 5):
    print("=" * 88)
    print("TOP-%d from IN-ENSEMBLE checkpoints only" % k)
    print("=" * 88)
    keep, routed = {}, bcols.copy()
    print("  %-18s %8s %8s %9s %8s" % ("finding", "ensemble", "routed", "gain", "SE"))
    for j in range(12):
        good = [i for i in pool if not np.isnan(sc[i, j])]
        top = sorted(good, key=lambda i: -sc[i, j])[:k]
        s = P.auc(y[:, j], P.combine([mats[i] for i in top], j))
        gain = s - bcols[j]
        rng = np.random.default_rng(j)
        d = []
        for _ in range(200):
            b = rng.integers(0, n, n)
            if not (0 < (y[b, j] > 0.5).sum() < n):
                continue
            d.append(P.auc(y[b, j], P.combine([mats[i] for i in top], j, rows=b))
                     - P.auc(y[b, j], base[b, j]))
        se = np.nanstd(d)
        ok = gain > se
        if ok:
            keep[j] = top
            routed[j] = s
        print("  %-18s %8.4f %8.4f %+9.4f %7.2f %s"
              % (P.TASKS[j], bcols[j], s, gain, gain / se if se else 0, "KEEP" if ok else ""))
    need = sorted({i for v in keep.values() for i in v})
    extra = [i for i in need if i not in base_ck]
    hours = (8 + len(extra)) * SEC_PER_MEMBER * STUDIES / 3600
    print("  %-18s %8.4f %8.4f %+9.4f   (%d findings routed)"
          % ("MACRO", np.nanmean(bcols), np.nanmean(routed),
             np.nanmean(routed) - np.nanmean(bcols), len(keep)))
    print("  cost: %d extra forward passes -> %.1f h of the %.0f h cap  %s"
          % (len(extra), hours, CAP_H, "OK" if hours < CAP_H else "OVER CAP"))
    if extra:
        print("  extra checkpoints: %s" % ", ".join(names[i] for i in extra))

    # honest number
    deltas = []
    for seed in range(5):
        r = np.random.default_rng(700 + seed)
        fold = r.integers(0, 5, n)
        oof = base.copy()
        for f in range(5):
            tr = np.where(fold != f)[0]
            te = np.where(fold == f)[0]
            for j in range(12):
                if not (0 < (y[tr, j] > 0.5).sum() < len(tr)):
                    continue
                s_tr = {i: P.auc(y[tr, j], mats[i][tr, j]) for i in pool}
                bl = P.auc(y[tr, j], base[tr, j])
                top = sorted([i for i in pool if s_tr[i] == s_tr[i]],
                             key=lambda i: -s_tr[i])[:k]
                if np.mean([s_tr[i] for i in top]) - bl <= 0:
                    continue
                oof[te, j] = P.combine([mats[i] for i in top], j, rows=te)
        deltas.append(np.nanmean([P.auc(y[:, j], oof[:, j]) for j in range(12)])
                      - np.nanmean(bcols))
    d = np.array(deltas)
    print("  CROSS-FITTED macro delta: %+.5f +/- %.5f  (%d/5 seeds positive)\n"
          % (d.mean(), d.std(), (d > 0).sum()))
