"""ten_mii_folds_audit.py - how should the 5 MII folds join the 0.938 ten: flat (14 equal votes) or nested
(the 5 folds averaged into ONE vote)?

The ten's full-data members trained on the audit studies, so each is replaced by its SPLIT-data
version (same architecture and recipe, never saw val or audit). The MII folds never saw audit either.
Scored on audit-58 (panel labels), flat logit means, each run at its best-val checkpoint.
Paired bootstrap between the variants.

    python runs/others/ten_mii_folds_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mii_vs_convnext_oof as m  # noqa: E402

# the ten, each mapped to its split-data analogue
TEN_SPLIT = {
    "convnext_base_full_all_knees_same_orientation": "convnext_base_relabel",   # its split-data ancestor
    "mricore_vitb_cross_attention": "mricore_vitb_cross_attention",
    "mricore_vitb_mae_cross_attention_fulldata_flip": "mricore_vitb_mae_cross_attention",
    "dinov3_convnext_base": "dinov3_convnext_base",
    "convnext_base_25d_all_knees_same_orientation_fulldata": "convnext_base_25d_all_knees_same_orientation",
    "swinv2_base_in22k": "swinv2_base_in22k",
    "convnext_base_25d_poolattn_fulldata_flip": "convnext_base_25d_poolattn",
    "convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip": "convnext_base_25d_perfinding_slots_and_pooling",
    "medimageinsight_davit_huge_336_fulldata": "medimageinsight_davit_huge_336",
    "medimageinsight_davit_huge_perslice_480_warm": "medimageinsight_davit_huge_perslice_480_warm",
}
MII = "medimageinsight_davit_huge_336_fulldata"
FOLDS = [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)]


def main():
    aud = pd.read_csv(m.D / "05_audit58.csv").set_index("StudyInstanceUID")
    u = list(aud.index)
    Y = (aud.loc[u, ["official_" + t for t in m.T]].values >= 0.5).astype(int)
    L = {}
    for full, split in TEN_SPLIT.items():
        df, f = m.load(split, "audit")
        L[full] = df.loc[u].values
        print(f"  {full[:52]:52s} <- {split} ({f})")
    for r in FOLDS:
        L[r] = m.load(r, "audit")[0].loc[u].values
    others = [k for k in TEN_SPLIT if k != MII]
    fold_mean = np.mean([L[r] for r in FOLDS], 0)
    V = {"the ten (single split MII)": np.mean([L[k] for k in TEN_SPLIT], 0),
         "nested: 9 + fold mean as ONE vote (1/10)": np.mean([L[k] for k in others] + [fold_mean], 0),
         "flat: 9 + 5 folds (5/14 MII)": np.mean([L[k] for k in others] + [L[r] for r in FOLDS], 0),
         "9 + fold mean at 2/10": (np.sum([L[k] for k in others], 0) + 2 * fold_mean) / 11,
         "MII folds alone": fold_mean,
         "the 9 without any MII 336": np.mean([L[k] for k in others], 0)}
    print("\naudit-58 (panel labels):")
    sc = {k: m.mac(Y, v) for k, v in V.items()}
    for k, v in sc.items():
        print(f"  {k:44s} {v:.4f}")
    rng = np.random.default_rng(0)
    pairs = [("flat: 9 + 5 folds (5/14 MII)", "nested: 9 + fold mean as ONE vote (1/10)"),
             ("nested: 9 + fold mean as ONE vote (1/10)", "the ten (single split MII)"),
             ("flat: 9 + 5 folds (5/14 MII)", "the ten (single split MII)")]
    for a, b in pairs:
        d = []
        for _ in range(2000):
            i = rng.integers(0, len(u), len(u))
            if (Y[i].sum(0) == 0).any() or (Y[i].sum(0) == len(i)).any():
                continue
            d.append(m.mac(Y[i], V[a][i]) - m.mac(Y[i], V[b][i]))
        d = np.array(d)
        print(f"  bootstrap {a[:20]} minus {b[:20]}: {d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, "
              f"{np.percentile(d, 97.5):+.4f}], P(>0) {100 * (d > 0).mean():.0f}%")


if __name__ == "__main__":
    main()
