r"""seq_head_sweep.py - does reading the slices IN ORDER beat averaging them?

Runs entirely on cached features (extract_features.py), so the 88M backbone is out of the
loop and every head trains on a 4060 in minutes. That is the whole point: the question has
been open since convnext_small_transformer_pool lost (0.8792 against its baseline's 0.8940),
and it lost with lp_ft_epochs 0, so we could never tell whether sequence modelling is useless
here or whether a randomly initialised head simply wrecked the trunk. On frozen features that
confound cannot happen - the trunk is fixed by construction.

THE BASELINE IS THE POINT. `mean` on these same features is exactly what the shipped model
does after its backbone. If a sequence head cannot beat that, on identical inputs, with the
trunk held still, the idea is dead and no GPU run is warranted.

Heads, each reading 4 scans x 24 slices of 1024 features:
    mean        average everything                       (the baseline; order-blind)
    attn        learned weight per image, then sum       (order-blind, but not uniform)
    gru         BiGRU along the 24 slices of each scan, attention-pooled, then over slots
    tfm         small transformer over the 24, with learned slice positions
    gru_joint   one BiGRU over all 96 in slot-major order (the naive flattening)

Every head is trained with the same optimiser, schedule, seed protocol and early stopping,
and scored by macro AUROC on the held-out val studies.

    python runs/others/seq_head_sweep.py
    python runs/others/seq_head_sweep.py --heads mean gru --seeds 5 --epochs 40
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
FEAT = ROOT / "runs" / "others" / "features" / "convnext_base_relabel"


# --------------------------------------------------------------------------- heads
class Mean(nn.Module):
    """What the shipped model does. Order-blind by construction."""

    def __init__(self, d, n_slots, n_slices, n_out):
        super().__init__()
        self.head = nn.Linear(d, n_out)

    def forward(self, x):                      # x: (B, slots, slices, d)
        return self.head(x.flatten(1, 2).mean(1))


class Attn(nn.Module):
    """Learned weight per image. Still order-blind - it cannot see WHICH slice is which."""

    def __init__(self, d, n_slots, n_slices, n_out, h=128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))
        self.head = nn.Linear(d, n_out)

    def forward(self, x):
        f = x.flatten(1, 2)
        w = torch.softmax(self.score(f), dim=1)
        return self.head((f * w).sum(1))


class GRUHead(nn.Module):
    """The real proposal: read each scan's 24 slices IN ORDER, both directions.

    Per scan, so slice 5 of the coronal scan is never mixed with slice 5 of the axial - they
    are different sequences. Weights are shared across scans; the slots are combined after.
    """

    def __init__(self, d, n_slots, n_slices, n_out, dim=256, hid=128):
        super().__init__()
        self.proj = nn.Linear(d, dim)
        self.rnn = nn.GRU(dim, hid, batch_first=True, bidirectional=True)
        self.slice_score = nn.Linear(2 * hid, 1)      # which slices matter, within a scan
        self.slot_score = nn.Linear(2 * hid, 1)       # which scans matter, within a study
        self.norm = nn.LayerNorm(2 * hid)
        self.head = nn.Linear(2 * hid, n_out)

    def forward(self, x):
        b, s, n, d = x.shape
        h, _ = self.rnn(self.proj(x).reshape(b * s, n, -1))        # (b*s, n, 2*hid)
        w = torch.softmax(self.slice_score(h), dim=1)
        per_slot = (h * w).sum(1).view(b, s, -1)                   # (b, slots, 2*hid)
        ws = torch.softmax(self.slot_score(per_slot), dim=1)
        return self.head(self.norm((per_slot * ws).sum(1)))


class TfmHead(nn.Module):
    """Transformer over the 24 slices of a scan, with LEARNED slice positions."""

    def __init__(self, d, n_slots, n_slices, n_out, dim=256, heads=4, layers=1):
        super().__init__()
        self.proj = nn.Linear(d, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_slices, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.1,
                                         batch_first=True, norm_first=True,
                                         activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.slice_score = nn.Linear(dim, 1)
        self.slot_score = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_out)

    def forward(self, x):
        b, s, n, d = x.shape
        h = self.enc(self.proj(x).reshape(b * s, n, -1) + self.pos)
        w = torch.softmax(self.slice_score(h), dim=1)
        per_slot = (h * w).sum(1).view(b, s, -1)
        ws = torch.softmax(self.slot_score(per_slot), dim=1)
        return self.head(self.norm((per_slot * ws).sum(1)))


class GRUJoint(nn.Module):
    """One sequence of all 96, slot-major. The naive flattening, kept as a control."""

    def __init__(self, d, n_slots, n_slices, n_out, dim=256, hid=128):
        super().__init__()
        self.proj = nn.Linear(d, dim)
        self.rnn = nn.GRU(dim, hid, batch_first=True, bidirectional=True)
        self.score = nn.Linear(2 * hid, 1)
        self.norm = nn.LayerNorm(2 * hid)
        self.head = nn.Linear(2 * hid, n_out)

    def forward(self, x):
        b = x.shape[0]
        h, _ = self.rnn(self.proj(x.flatten(1, 2)))
        w = torch.softmax(self.score(h), dim=1)
        return self.head(self.norm((h * w).sum(1)))



class TopK(nn.Module):
    """Mean of the k strongest slices per feature. A focal finding lives on a few slices, so
    a mean over 96 dilutes it; the probe showed Fracture at 0.9244 with k=4 against 0.8981
    with a mean."""

    def __init__(self, d, n_slots, n_slices, n_out, k=4):
        super().__init__()
        self.k = k
        self.head = nn.Linear(d, n_out)

    def forward(self, x):
        f = x.flatten(1, 2)
        return self.head(f.topk(self.k, dim=1).values.mean(1))


class Dual(nn.Module):
    """Give the head BOTH poolings and let it choose per finding through its own weights.

    This is the notebook's per-finding pooling idea without its two problems: no hand-picked
    assignment per class (which cost -0.0090 when copied outright) and no routing (which lost
    on the leaderboard three times). One model, one head, 2048 -> 12 instead of 1024 -> 12.
    """

    def __init__(self, d, n_slots, n_slices, n_out):
        super().__init__()
        self.norm = nn.LayerNorm(2 * d)
        self.head = nn.Linear(2 * d, n_out)

    def forward(self, x):
        f = x.flatten(1, 2)
        return self.head(self.norm(torch.cat([f.mean(1), f.max(1).values], dim=-1)))


class Trio(nn.Module):
    """mean + max + top4, in case top-k carries something max alone does not."""

    def __init__(self, d, n_slots, n_slices, n_out):
        super().__init__()
        self.norm = nn.LayerNorm(3 * d)
        self.head = nn.Linear(3 * d, n_out)

    def forward(self, x):
        f = x.flatten(1, 2)
        return self.head(self.norm(torch.cat(
            [f.mean(1), f.max(1).values, f.topk(4, dim=1).values.mean(1)], dim=-1)))


class MaxPool(nn.Module):
    def __init__(self, d, n_slots, n_slices, n_out):
        super().__init__()
        self.head = nn.Linear(d, n_out)

    def forward(self, x):
        return self.head(x.flatten(1, 2).max(1).values)


class LSE(nn.Module):
    """Per-FINDING log-sum-exp over the per-image logits (slice_aggregation_ideas_research.md idea 1). The shared linear
    head scores every image, then each finding pools its 96 image logits with its own learnable sharpness r_j:
    (1/r) log mean exp(r z). r -> 0 is EXACTLY the mean model (linear head of the mean = mean of the linear head);
    r large -> max. Initialised at r = 0.1 (= the mean), so the head can only move toward focal pooling where it helps."""

    def __init__(self, d, n_slots, n_slices, n_out):
        super().__init__()
        self.head = nn.Linear(d, n_out)
        self.log_r = nn.Parameter(torch.full((n_out,), float(np.log(0.1))))

    def forward(self, x):
        z = self.head(x.flatten(1, 2))                       # (B, 96, 12) per-image logits
        r = self.log_r.exp()
        n = z.shape[1]
        return (torch.logsumexp(r * z, dim=1) - np.log(n)) / r


class ViewW(nn.Module):
    """Per-view mean, one shared head, then a learnable weight per (view, finding), initialised equal (idea 2): exactly
    the mean model at init (views have equal slice counts), able to learn that a finding is read on some views."""

    def __init__(self, d, n_slots, n_slices, n_out):
        super().__init__()
        self.head = nn.Linear(d, n_out)
        self.w = nn.Parameter(torch.zeros(n_slots, n_out))  # softmax over views -> equal at init

    def forward(self, x):
        z = self.head(x.mean(2))                              # (B, slots, 12)
        a = torch.softmax(self.w, dim=0)[None]                # (1, slots, 12)
        return (z * a).sum(1)


class SmoothTop2(nn.Module):
    """Parameter-free (idea 4): smooth each scan's features along the slice axis (window 3), then per feature the mean
    of the 2 strongest smoothed slices of the whole study, concatenated with the mean -> head."""

    def __init__(self, d, n_slots, n_slices, n_out):
        super().__init__()
        self.norm = nn.LayerNorm(2 * d)
        self.head = nn.Linear(2 * d, n_out)

    def forward(self, x):
        b, s, n, d = x.shape
        xs = torch.nn.functional.avg_pool1d(x.reshape(b * s, n, d).transpose(1, 2), 3, 1, 1, count_include_pad=False)
        xs = xs.transpose(1, 2).reshape(b, s * n, d)
        top2 = xs.topk(2, dim=1).values.mean(1)
        return self.head(self.norm(torch.cat([x.flatten(1, 2).mean(1), top2], -1)))


HEADS = {"lse": LSE, "vieww": ViewW, "smooth_top2": SmoothTop2, "mean": Mean, "attn": Attn, "gru": GRUHead, "tfm": TfmHead,
         "gru_joint": GRUJoint, "max": MaxPool, "top4": TopK,
         "dual": Dual, "trio": Trio}


# --------------------------------------------------------------------------- data
def load(split):
    f = np.load(FEAT / ("feats_%s.npy" % split))
    m = np.load(FEAT / ("meta_%s.npz" % split), allow_pickle=True)
    slots, ns = len(m["slots"]), int(m["n_slices"])
    x = torch.from_numpy(f).view(len(f), slots, ns, f.shape[-1])
    return x, torch.from_numpy(m["targets"]), [str(t) for t in m["tasks"]], slots, ns


def macro_auc(y, p):
    out = []
    for j in range(y.shape[1]):
        t = (y[:, j] > 0.5).astype(int)
        if 0 < t.sum() < len(t):
            out.append(roc_auc_score(t, p[:, j]))
    return float(np.mean(out)), np.array(out)


def train_one(kind, tr, va, tasks, slots, ns, seed, epochs, lr, bs, device, patience=6):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xt, yt = tr
    xv, yv = va
    model = HEADS[kind](xt.shape[-1], slots, ns, len(tasks)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.BCEWithLogitsLoss()
    best, best_cols, bad = -1.0, None, 0
    n = len(xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = xt[idx].to(device, non_blocking=True).float()
            yb = yt[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p = torch.cat([model(xv[i:i + 256].to(device).float()).cpu()
                           for i in range(0, len(xv), 256)]).numpy()
        s, cols = macro_auc(yv.numpy(), p)
        if s > best:
            best, best_cols, bad = s, cols, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best, best_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="*", default=list(HEADS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()

    if not (FEAT / "feats_val.npy").exists():
        raise SystemExit("no cached features at %s - run extract_features.py first" % FEAT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt, yt, tasks, slots, ns = load("train")
    xv, yv, _, _, _ = load("val")
    print("train %s   val %s   %d slots x %d slices x %d features   device %s"
          % (tuple(xt.shape), tuple(xv.shape), slots, ns, xt.shape[-1], device))

    res = {}
    for kind in a.heads:
        scores, cols, t0 = [], [], time.time()
        for s in range(a.seeds):
            sc_, c = train_one(kind, (xt, yt), (xv, yv), tasks, slots, ns,
                               s, a.epochs, a.lr, a.bs, device)
            scores.append(sc_)
            cols.append(c)
        res[kind] = (np.mean(scores), np.std(scores), np.mean(cols, axis=0))
        n_par = sum(p.numel() for p in
                    HEADS[kind](xt.shape[-1], slots, ns, len(tasks)).parameters())
        print("  %-10s %.4f +/- %.4f   %6.1fs   %6.2fM params   seeds %s"
              % (kind, res[kind][0], res[kind][1], time.time() - t0, n_par / 1e6,
                 " ".join("%.4f" % v for v in scores)))

    if "mean" in res:
        print("\n  vs the mean-pool baseline (%.4f):" % res["mean"][0])
        for k, (m, s, _) in sorted(res.items(), key=lambda kv: -kv[1][0]):
            if k != "mean":
                print("    %-10s %+.4f" % (k, m - res["mean"][0]))
        print("\n  per finding, best head vs mean:")
        bestk = max((k for k in res if k != "mean"), key=lambda k: res[k][0])
        for j, t in enumerate(tasks):
            print("    %-18s mean %.4f   %s %.4f   %+.4f"
                  % (t, res["mean"][2][j], bestk, res[bestk][2][j],
                     res[bestk][2][j] - res["mean"][2][j]))


if __name__ == "__main__":
    main()
