"""
12_report_emb_probe.py - can the 12 labels be recovered FROM a report embedding?

This is the decisive test of whether a report embedding is a usable auxiliary target. The
labels were produced by an LLM reading these exact reports, so all twelve are, by
construction, present in the text. A text encoder that preserves the clinical content
should let a linear model recover them almost perfectly. One that scores near chance has
thrown the content away, and training an image model to predict such a vector would add
noise, not supervision.

Compares several encoders on the same reports and the same folds.

    python data_code/12_report_emb_probe.py
    python data_code/12_report_emb_probe.py --models "a,b" --n 2000
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
cert = Path.home() / ".certs" / "ca-bundle.pem"
if cert.exists():
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(cert))
    os.environ.setdefault("SSL_CERT_FILE", str(cert))

DEFAULT = [
    "NeuML/pubmedbert-base-embeddings",          # biomedical, Apache-2.0, the first try
    "sentence-transformers/all-mpnet-base-v2",   # general sentence encoder, Apache-2.0
    "BAAI/bge-base-en-v1.5",                     # strong general retrieval encoder, MIT
]

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=",".join(DEFAULT))
ap.add_argument("--n", type=int, default=2500, help="studies to probe")
ap.add_argument("--strip-boilerplate", action="store_true",
                help="drop the Technique/Results/Impression scaffolding before encoding")
a = ap.parse_args()

R = pd.read_csv(ROOT / "data" / "02_reports_en.csv")
R = R[R.Report_en.notna()].copy()
R["StudyInstanceUID"] = R.StudyInstanceUID.astype(str)
lab = pd.read_csv(ROOT / "data" / "03_report_labels_qwen24t-med.csv")
lab["StudyInstanceUID"] = lab.StudyInstanceUID.astype(str)
cols = [c for c in lab.columns if c not in ("StudyInstanceUID", "lang", "group")]

M = R.merge(lab, on="StudyInstanceUID", how="inner")
P = M[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
keep = ~np.isnan(P).any(1)
M, P = M[keep].iloc[:a.n], P[keep][:a.n]
Y = (P > 0.5).astype(int)
text = M.Report_en.astype(str).tolist()
if a.strip_boilerplate:
    import re
    text = [re.sub(r"(?i)\b(technique|results|impression)\s*:", " ", t) for t in text]
    text = [re.sub(r"(?i)^.*?knee mri\.?", "", t).strip() or t for t in text]
print(f"{len(text)} reports x {len(cols)} findings, positives per finding:")
print("  " + "  ".join("%s %d" % (c[:9], Y[:, j].sum()) for j, c in enumerate(cols)))

from sentence_transformers import SentenceTransformer  # noqa: E402


def probe(E):
    """Cross-validated macro AUROC of a linear model predicting the 12 labels from E."""
    E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    aucs = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if y.sum() < 20 or (1 - y).sum() < 20:
            continue
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=0).split(E, y):
            clf = LogisticRegression(max_iter=3000, C=1.0)
            clf.fit(E[tr], y[tr])
            oof[te] = clf.predict_proba(E[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), aucs


print()
results = {}
for name in [s.strip() for s in a.models.split(",") if s.strip()]:
    try:
        m = SentenceTransformer(name)
        E = m.encode(text, batch_size=32, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
        del m
    except Exception as e:
        print(f"SKIPPED {name}: {type(e).__name__}: {str(e)[:80]}")
        continue
    macro, aucs = probe(E)
    results[name] = macro
    print(f"{macro:.4f}  {name}  dim={E.shape[1]}")

print("\n" + "=" * 74)
print("RECOVERING THE 12 LABELS FROM THE REPORT EMBEDDING (cross-validated)")
for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
    print("  %.4f  %s" % (v, k))
print("=" * 74)
print("The labels were WRITTEN from these reports, so the ceiling is ~1.0. Anything near")
print("0.5-0.7 means the encoder discarded the clinical content and is not a usable")
print("auxiliary target. Above ~0.90 means the vector really does carry the findings.")
