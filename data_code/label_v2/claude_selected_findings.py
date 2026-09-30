"""claude_selected_findings.py - V3 averaged 50/50 (logit) with Opus+Sonnet ONLY on chosen findings.

1. in-sample: the four findings picked by looking at all 58 studies (ACL, Medial Meniscus, PF OA,
   Contusion) - optimistic, the choice saw the answers
2. cross-fitted: on 29 studies pick every finding where the blend beats V3, score those picks on
   the other 29 (300 splits) - the honest estimate of "blend where Claude helps"

    python data_code/label_v2/claude_selected_findings.py
"""
import numpy as np

import claude_label_ensembles as ce   # sources, auc and helpers (prints its own report on import)

V3, C, Y, N, T = ce.V3, ce.SRC["C"], ce.Y, ce.N, ce.T
BL = ce.avg([V3, C], "logit")
PICK = ["ACL", "Medial Meniscus", "PF OA", "Contusion"]
P = V3.copy()
P[:, ce.cols_idx(PICK)] = BL[:, ce.cols_idx(PICK)]
print("\n" + "=" * 70)
print(f"IN-SAMPLE, blend on {PICK}: {ce.macro(P):.4f} vs V3 {ce.macro(V3):.4f} "
      f"({ce.macro(P) - ce.macro(V3):+.4f})  <- optimistic: chosen on these same studies")
rng = np.random.default_rng(1)
g, sizes, done = [], [], 0
while done < 300:
    perm = rng.permutation(N)
    A, B = perm[:29], perm[29:]
    if any(Y[A, j].sum() in (0, 29) or Y[B, j].sum() in (0, 29) for j in range(12)):
        continue
    done += 1
    better = ce.auc_cols(BL, A) > ce.auc_cols(V3, A)
    Q = V3.copy()
    Q[:, better] = BL[:, better]
    sizes.append(int(better.sum()))
    g.append(ce.macro(Q, B) - ce.macro(V3, B))
g = np.array(g)
print(f"CROSS-FITTED (pick findings on 29, score on 29, 300 splits): mean {g.mean():+.4f}, "
      f"median {np.median(g):+.4f}, positive in {100 * (g > 0).mean():.0f}%, "
      f"findings picked per split: median {int(np.median(sizes))}")
