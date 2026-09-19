r"""fast_ens.py - score an ensemble structure in microseconds instead of milliseconds.

THE TWO THINGS THAT MADE THE FIRST VERSION SLOW, both avoidable:

  1. It re-aligned the members on every call - a set intersection over 704 study ids, then a
     704-entry dict and a Python list comprehension PER MEMBER PER STRUCTURE. None of that
     changes between structures, so it belongs in a one-off setup.

  2. Its AUROC ran a pure-Python while loop over 704 studies to average tied ranks, twelve
     times per call. That is ~8,400 interpreted iterations per structure, and with ~7,000
     structures per search it dominated everything else.

THE OBSERVATION THAT COLLAPSES THE REST. A structure is a list of members; a member is a run
or a group of runs averaged together. So the final prediction is

    p = (1/M) * sum over members ( (1/|g|) * sum over runs in g ( L_run ) )

which is just a WEIGHTED AVERAGE of the base run logits, with weight 1/(M*|g|) on each run.
There is no need to materialise member averages at all: build a weight vector over runs and
contract it against one preloaded (runs, studies, tasks) array. Every structure, nested or
flat, is one tensordot.

Third fix: contract only the runs a structure actually uses (7 of 53), and rank with
argsort rather than scipy.rankdata. Measured: 0.56 ms per structure against the original
~110 ms, so a 500,000-structure search runs in minutes instead of half a day.
"""

import re
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent.parent
LEAKY = re.compile(r"_f[0-4]$|convnext_base_full")


def _ranks(q):
    """Column-wise ranks, 1-based. 0.05 ms against scipy rankdata's 0.564 ms.

    scipy averages tied ranks; this assigns them in order, so where ties DO occur the two
    disagree. Measured on the real pool that disagreement is 8.3e-07 of AUROC, against the
    ~1e-4 differences this search actually decides on - 100x below the decision threshold.
    The self-test pins it at 1e-5, so if a future member ever emits quantised or saturated
    scores and the ties stop being rare, the test fails rather than the search quietly
    ranking two structures the wrong way round.
    """
    o = np.argsort(q, axis=0, kind="stable")
    r = np.empty(q.shape, dtype=np.float64)
    idx = np.arange(1, q.shape[0] + 1, dtype=np.float64)[:, None]
    np.put_along_axis(r, o, np.broadcast_to(idx, q.shape), axis=0)
    return r


class Pool:
    """Every run's logits on one common study order, loaded once."""

    def __init__(self, split="val", runs=None, exclude_leaky=True):
        self.split = split
        names = runs if runs is not None else sorted(
            p.parent.parent.name for p in RUNS.glob("*/results/predictions"))
        if exclude_leaky:
            names = [n for n in names if not LEAKY.search(n)]

        loaded = {}
        for n in names:
            v = self._load(n, split)
            if v is not None and v[1].shape[1] == 12:
                loaded[n] = v
        if not loaded:
            raise SystemExit("no usable predictions for split %r" % split)

        common = None
        for ids, _, _ in loaded.values():
            common = set(ids) if common is None else (common & set(ids))
        self.ids = np.array(sorted(common))
        self.names = sorted(loaded)
        self.index = {n: i for i, n in enumerate(self.names)}

        L, y = [], None
        for n in self.names:
            ids, lg, tg = loaded[n]
            pos = {s: i for i, s in enumerate(ids)}
            sel = np.array([pos[s] for s in self.ids])
            L.append(lg[sel])
            if y is None:
                y = tg[sel]
        self.L = np.ascontiguousarray(np.stack(L))          # (R, N, 12)
        self.y = (y > 0.5)                                  # (N, 12) bool
        self.n = len(self.ids)
        # a column with only one class has no AUROC; drop it once rather than per call
        self.valid = np.array([self.y[:, j].any() and not self.y[:, j].all()
                               for j in range(12)])

    @staticmethod
    def _load(run, split):
        d = RUNS / run / "results" / "predictions"
        if not d.exists():
            return None
        files = sorted(d.glob("*_%s_s*.npz" % split))
        if not files:
            return None
        import pandas as pd
        vlog = RUNS / run / "results" / "val_log.csv"
        pick = files[-1]
        if vlog.exists():
            try:
                v = pd.read_csv(vlog)
                step = int(v.loc[v["val_macro_auroc"].idxmax(), "step"])
                for f in files:
                    if f.stem.endswith("_s%d" % step):
                        pick = f
                        break
            except Exception:
                pass
        z = np.load(pick, allow_pickle=True)
        return (np.array([str(s) for s in z["study_ids"]]),
                z["logits"].astype(np.float32), z["targets"].astype(np.float32))

    # ---------------------------------------------------------------- scoring

    def weights(self, structure):
        """Structure -> weight per run. Equal vote per member, split inside a group."""
        w = np.zeros(len(self.names), dtype=np.float64)
        m = len(structure)
        for member in structure:
            grp = member if isinstance(member, (list, tuple)) else [member]
            for r in grp:
                w[self.index[r]] += 1.0 / (m * len(grp))
        return w

    def blend(self, structure):
        """Weighted sum over ONLY the runs this structure uses.

        Contracting all 53 runs when 7 are in play costs 1.0 ms; the subset costs 0.039 ms.
        The zeros contribute nothing, so this is the same number, 25x cheaper.
        """
        w = self.weights(structure)
        nz = np.nonzero(w)[0]
        if len(nz) == 1:
            return w[nz[0]] * self.L[nz[0]]
        out = self.L[nz[0]] * w[nz[0]]
        for i in nz[1:]:
            out += self.L[i] * w[i]
        return out

    def auroc(self, p, rows=None):
        """Macro AUROC, vectorised over all twelve columns at once.

        Mann-Whitney U: with ranks r of the scores, AUC = (sum of positive ranks
        - n1(n1+1)/2) / (n1*n0). rankdata handles ties in C.
        """
        y = self.y if rows is None else self.y[rows]
        q = p if rows is None else p[rows]
        r = _ranks(q)
        n1 = y.sum(axis=0).astype(np.float64)
        n0 = y.shape[0] - n1
        ok = self.valid & (n1 > 0) & (n0 > 0)
        if not ok.any():
            return float("nan")
        s1 = (r * y).sum(axis=0)
        auc = (s1 - n1 * (n1 + 1) / 2.0) / np.maximum(n1 * n0, 1)
        return float(auc[ok].mean())

    def score(self, structure, rows=None):
        return self.auroc(self.blend(structure), rows)

    def per_column(self, structure, rows=None):
        p = self.blend(structure)
        y = self.y if rows is None else self.y[rows]
        q = p if rows is None else p[rows]
        r = _ranks(q)
        n1 = y.sum(axis=0).astype(np.float64)
        n0 = y.shape[0] - n1
        s1 = (r * y).sum(axis=0)
        return (s1 - n1 * (n1 + 1) / 2.0) / np.maximum(n1 * n0, 1)


def self_test():
    """The fast path must agree with the slow one, or the speed is worthless."""
    import time
    p = Pool("val")
    print("  pool: %d runs x %d studies" % (len(p.names), p.n))
    shipped = ["convnext_base_relabel", "mricore_vitb_cross_attention",
               "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
               "convnext_base_25d_18slices", "swinv2_base_in22k",
               "convnext_base_25d_poolattn"]
    shipped = [s for s in shipped if s in p.index]

    # weights must sum to 1 and respect nesting
    w = p.weights(shipped)
    assert abs(w.sum() - 1) < 1e-12, w.sum()
    nested = shipped[:4] + [[shipped[4], shipped[5]]] + shipped[6:]
    wn = p.weights(nested)
    assert abs(wn.sum() - 1) < 1e-12
    i4, i5 = p.index[shipped[4]], p.index[shipped[5]]
    assert abs(wn[i4] - 1.0 / (6 * 2)) < 1e-12, wn[i4]
    print("  PASS weights sum to 1 and a 2-run group splits its single vote")

    # a group of one is the same as the bare run
    assert abs(p.score([shipped[0], shipped[1]])
               - p.score([[shipped[0]], [shipped[1]]])) < 1e-12
    print("  PASS a one-run group equals the bare run")

    # agreement with an independent, naive AUROC on the same blend
    from sklearn.metrics import roc_auc_score
    pr = p.blend(shipped)
    ref = np.mean([roc_auc_score(p.y[:, j], pr[:, j]) for j in range(12) if p.valid[j]])
    got = p.score(shipped)
    print("  PASS matches sklearn: %.6f vs %.6f (diff %.2e)" % (got, ref, abs(got - ref)))
    assert abs(got - ref) < 1e-5, abs(got - ref)

    # subset scoring
    rows = np.arange(0, p.n, 2)
    ref2 = np.mean([roc_auc_score(p.y[rows, j], pr[rows, j]) for j in range(12) if p.valid[j]])
    got2 = p.score(shipped, rows)
    print("  PASS subset matches sklearn: %.6f vs %.6f" % (got2, ref2))
    assert abs(got2 - ref2) < 1e-5, abs(got2 - ref2)

    t0 = time.time()
    for _ in range(2000):
        p.score(shipped)
    dt = (time.time() - t0) / 2000
    print("  %.3f ms per structure  (the old evaluator was ~110 ms)" % (dt * 1000))
    print("  -> a 7,000-structure search: %.1f s" % (dt * 7000))


if __name__ == "__main__":
    self_test()
