"""window_variants_test.py - intensity normalisation variants on an EXISTING checkpoint, no training (8 Oct, local GPU).

Model: convnext_base_25d_18slices_v5_aug_base top_step5900.pt (its best; trained on window "volume" [1, 99]).
Scored on the v5 val file (23_val_v3_audit.csv): the 346 v3-labelled studies and the 58 audit studies (panel labels),
each variant alone and as test-time augmentation (TTA: mean logit of the baseline and the variant).
A model never trained on a variant usually LOSES on it alone; the question here is whether any variant is close
enough to be a cheap TTA or a training input worth a run. Paired bootstrap vs baseline (2,000 resamples).
Logits cached in runs/others/window_variants/<variant>.npz, so a rerun only scores what is missing.

    python runs/others/window_variants_test.py
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
RUN = "convnext_base_25d_18slices_v5_aug_base"
sys.path.insert(0, str(ROOT / "runs"))
sys.path.insert(0, str(ROOT / "runs" / RUN))
import shared_code as sc  # noqa: E402
from train import build_model  # noqa: E402

OUT = ROOT / "runs/others/window_variants"
OUT.mkdir(exist_ok=True)
VARIANTS = {
    "base": {},
    "slice": {"window": "slice"},
    "pct_0.5_99.5": {"window_pct": [0.5, 99.5]},
    "pct_1_99.7": {"window_pct": [1.0, 99.7]},
    "pct_2_98": {"window_pct": [2.0, 98.0]},
    "clahe": {"window": "volume_clahe"},
    "zscore": {"window": "zscore"},
    "fgref": {"window": "fgref"},
}
cfg0 = sc.load_config(ROOT / "runs" / RUN, verbose=False)
cfg0["dataloader"].update(num_workers=0, val_num_workers=0, persistent_workers=False, pin_memory=False)
T = cfg0["tasks"]
df = pd.read_csv(ROOT / "data" / cfg0["paths"]["val_csv"])
aud_ids = set(pd.read_csv(ROOT / "data/05_audit58.csv").StudyInstanceUID)
dev = torch.device("cuda")

model = None
for name, ov in VARIANTS.items():
    f = OUT / f"{name}.npz"
    if f.exists():
        continue
    if model is None:
        model = build_model(cfg0)
        ck = torch.load(ROOT / "runs" / RUN / "results/checkpoints/top_step5900.pt", map_location="cpu",
                        weights_only=False)
        sd = ck.get("model") if isinstance(ck, dict) and "model" in ck else ck
        print("load:", model.load_state_dict(sd, strict=True))
    cfg = copy.deepcopy(cfg0)
    cfg["input"].update(ov)
    col = {}
    sc._predict_dataframe(cfg, model, df, dev, batch_size=4, num_workers=0, collect=col, desc=name)
    lg = np.concatenate(col["_lg"]).astype(np.float32)
    np.savez(f, logits=lg, ids=df.StudyInstanceUID.values)
    print(f"  saved {name} {lg.shape}", flush=True)


def auc_cols(y, p, I):
    out = np.zeros(len(I)); cnt = np.zeros(len(I))
    for j in range(y.shape[1]):
        yy, pp = y[I, j], p[I, j]
        n1 = yy.sum(1); n0 = yy.shape[1] - n1
        ok = (n1 > 0) & (n0 > 0)
        r = rankdata(pp, axis=1)
        out += np.where(ok, ((r * yy).sum(1) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1), 0); cnt += ok
    return out / cnt


L = {n: np.load(OUT / f"{n}.npz", allow_pickle=True)["logits"] for n in VARIANTS}
Y = (df[T].values >= 0.5).astype(int)
isa = df.StudyInstanceUID.isin(aud_ids).values
rng = np.random.default_rng(0)
for sname, m in (("val-346 (v3)", ~isa), ("audit-58 (panel)", isa)):
    y = Y[m]
    I = rng.integers(0, m.sum(), (2000, m.sum()))
    full = np.arange(m.sum())[None]
    b = L["base"][m]
    rb, rbI = auc_cols(y, b, full)[0], auc_cols(y, b, I)
    print(f"\n{sname}: baseline {rb:.4f}")
    print(f"  {'variant':14s} {'alone':>16s}   {'TTA with base':>22s}")
    for n in VARIANTS:
        if n == "base":
            continue
        a, t = L[n][m], (L[n][m] + b) / 2
        ra, rt = auc_cols(y, a, full)[0], auc_cols(y, t, full)[0]
        pt = (auc_cols(y, t, I) - rbI > 0).mean()
        print(f"  {n:14s} {ra:.4f} ({ra - rb:+.4f})   {rt:.4f} ({rt - rb:+.4f}, {pt:4.0%})")
