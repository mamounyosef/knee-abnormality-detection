"""
11_report_embeddings.py - turn each radiology report into one fixed vector.

WHY. Every model in this project is trained on 12 binary labels, so a study delivers at
most 12 bits of supervision to an 86M-parameter network across 3,638 training studies.
The reports those labels were DERIVED from are a median 1,013 characters each and contain
far more: laterality, location, severity, and findings that have no label column at all
(the first report in the file mentions subchondral avascular necrosis of the medial femoral
condyle, which is nowhere in our 12).

This script encodes each report ONCE with a frozen text encoder. The resulting vectors are
then targets, exactly like the label columns: during training an auxiliary head predicts
the vector for a study and pays a cosine penalty for missing it, so the image trunk is
pushed to represent everything the report says, not just the 12 bits. The head is dropped
at inference, so this costs nothing at submission time and no text is needed there.

LICENCE. The encoder never ships. It runs here, offline, and only its OUTPUT vectors reach
training. Even so the default is Apache-2.0, so nothing about the trained weights is
encumbered.

    python data_code/11_report_embeddings.py
    python data_code/11_report_embeddings.py --model sentence-transformers/all-mpnet-base-v2
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
# all-mpnet-base-v2: Apache-2.0, 768-dim. Chosen by measurement, not by domain intuition:
# 12_report_emb_probe.py recovers the 12 findings from these vectors at 0.9072 macro AUROC
# against PubMedBERT's 0.8986 and BGE's 0.8914. The biomedical encoder was the obvious pick
# and it lost.
ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
ap.add_argument("--reports", default="data/02_reports_en.csv")
ap.add_argument("--text-col", default="Report_en")
ap.add_argument("--out", default="data/11_report_emb.npz")
ap.add_argument("--batch-size", type=int, default=32)
a = ap.parse_args()

# Avast intercepts TLS on this machine; without this the model download fails
cert = Path.home() / ".certs" / "ca-bundle.pem"
if cert.exists():
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(cert))
    os.environ.setdefault("SSL_CERT_FILE", str(cert))

R = pd.read_csv(ROOT / a.reports)
R = R[R[a.text_col].notna()].copy()
R[a.text_col] = R[a.text_col].astype(str)
print(f"{len(R)} reports, median {int(R[a.text_col].str.len().median())} characters")

from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer(a.model)
print(f"encoder: {a.model}  dim={model.get_sentence_embedding_dimension()}")

E = model.encode(R[a.text_col].tolist(), batch_size=a.batch_size,
                 convert_to_numpy=True, normalize_embeddings=True,
                 show_progress_bar=True)
ids = R["StudyInstanceUID"].astype(str).to_numpy()
print(f"encoded {E.shape[0]} x {E.shape[1]}")

# ------------------------------------------------------------------ sanity check
# NOT raw cosine similarity. That was tried and it is the wrong instrument: it is dominated
# by the highest-variance directions, which encode report STYLE and length, and it reported
# a NEGATIVE separation (-0.020) on vectors that in fact carry the findings perfectly well.
# The same mistake as reading a contrastive model's cone as feature collapse.
#
# The trustworthy test is whether the 12 labels can be recovered from the vector. They were
# written from these reports, so the ceiling is ~1.0 and anything near chance would mean the
# encoder discarded the content. See 12_report_emb_probe.py for the full comparison.
rng = np.random.default_rng(0)
idx = rng.choice(len(E), size=min(2000, len(E)), replace=False)
sims = (E[idx] @ E[idx].T)[np.triu_indices(len(idx), k=1)]
print(f"\npairwise cosine: mean {sims.mean():.3f}, sd {sims.std():.3f} "
      f"(reference only - see the note above, it does NOT validate the target)")

# Per-dimension standardisation stats, computed on the TRAIN split only so validation
# reports never influence the target's scaling. The engine must apply these before the
# auxiliary loss: on the raw vector the loss would chase the style directions instead of
# the clinical ones, which is exactly what the cosine test was picking up.
tr = pd.read_csv(ROOT / "data" / "05_train_qwen24t-med.csv")
train_ids = set(tr.StudyInstanceUID.astype(str))
mask = np.array([i in train_ids for i in ids])
mu = E[mask].mean(0)
sd_raw = E[mask].std(0)
# Drop DEAD dimensions before standardising. A handful of this encoder's outputs are
# constant, and dividing by their ~0 standard deviation would turn rounding noise into a
# unit-variance target the image model would then chase. Everything else sits in a narrow
# band, so the surviving dimensions are all genuinely informative.
keep = np.where(sd_raw > 0.1 * np.median(sd_raw))[0]
E, mu, sd = E[:, keep], mu[keep], sd_raw[keep]
print(f"standardisation stats from {int(mask.sum())} TRAIN reports "
      f"(held-out reports never influence the scaling)")
print(f"kept {len(keep)} of {len(sd_raw)} dimensions; dropped {len(sd_raw) - len(keep)} dead "
      f"ones whose sd was ~0")
print(f"surviving per-dimension sd: min {sd.min():.4f}, max {sd.max():.4f}, "
      f"ratio {sd.max() / sd.min():.1f}x")
print("  Standardising matters because the findings live in LOWER-variance directions than")
print("  report style; on the raw vector the loss would chase style. See the note above.")

out = ROOT / a.out
np.savez_compressed(out, study_ids=ids, emb=E.astype(np.float16),
                    mean=mu.astype(np.float32), std=sd.astype(np.float32),
                    keep=keep.astype(np.int32), model=a.model)
print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB, float16)")
