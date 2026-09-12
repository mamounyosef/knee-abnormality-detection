"""
report_encoder_bench.py - pick the report encoder by measurement, on Modal.

The encoder NEVER SHIPS. It runs once, offline, and only its output vectors reach training;
the auxiliary head is dropped at inference. So there is no runtime budget, no submission
size limit, and no reason to prefer a small encoder. The only real constraint is licence:
the trained weights are derived from these targets, so non-commercial encoders (NV-Embed-v2
and friends) are excluded exactly as non-commercial backbone weights are.

SELECTION METRIC. The 12 labels were written by an LLM reading these exact reports, so all
twelve are present in the text by construction and a perfect encoder would let a linear
model recover them at ~1.0. Cross-validated macro AUROC of that recovery is therefore a
direct measure of how much clinical content survives encoding. It is also the metric that
matters: an auxiliary target the image model cannot learn anything real from is noise.

Do NOT use raw cosine similarity to judge these. It is dominated by report style and
length, and it scored a NEGATIVE separation on vectors that recover the findings at 0.907.

    modal run runs/others/report_encoder_bench.py
    modal run runs/others/report_encoder_bench.py --models "a,b" --n-studies 3000
"""

import sys
from pathlib import Path

# (name, licence, note). Every one permissive; anything non-commercial is excluded above.
CANDIDATES = [
    ("sentence-transformers/all-mpnet-base-v2", "apache-2.0", "110M, the local baseline"),
    ("BAAI/bge-large-en-v1.5", "mit", "335M"),
    ("intfloat/e5-large-v2", "mit", "335M"),
    ("Alibaba-NLP/gte-large-en-v1.5", "apache-2.0", "434M, 8k context"),
    ("Qwen/Qwen3-Embedding-0.6B", "apache-2.0", "0.6B"),
    ("Qwen/Qwen3-Embedding-4B", "apache-2.0", "4B"),
    ("Qwen/Qwen3-Embedding-8B", "apache-2.0", "8B, top of MTEB at release"),
    ("Alibaba-NLP/gte-Qwen2-7B-instruct", "apache-2.0", "7B"),
]

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    ROOT = Path(__file__).resolve().parent.parent.parent
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("sentence-transformers>=3.0", "transformers>=4.44", "torch",
                          "scikit-learn", "pandas", "numpy", "einops", "accelerate")
             .add_local_dir(str(ROOT / "data"), "/data_local",
                            ignore=lambda p: p.suffix not in (".csv",)))
    app = modal.App("report-encoder-bench")

    @app.function(image=image, gpu="H100", timeout=60 * 45,
                  secrets=[modal.Secret.from_dict({"HF_HUB_DISABLE_TELEMETRY": "1"})])
    def bench_one(spec, n_studies: int):
        """ONE encoder, in its own container.

        Isolation is not tidiness: a CUDA device-side assert poisons the whole context, so
        when Qwen3-Embedding-0.6B tripped one, every model queued behind it in the same
        container failed identically and looked broken. Separate containers give each
        candidate an honest attempt.
        """
        name, lic, note = spec
        try:
            return (name, lic, note) + _run_one(name, n_studies)
        except Exception as e:
            print(f"[bench] FAILED {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            return (name, lic, note, None, 0)

    def _run_one(name, n_studies):
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sentence_transformers import SentenceTransformer

        R = pd.read_csv("/data_local/02_reports_en.csv")
        R = R[R.Report_en.notna()].copy()
        R["StudyInstanceUID"] = R.StudyInstanceUID.astype(str)
        lab = pd.read_csv("/data_local/03_report_labels_qwen24t-med.csv")
        lab["StudyInstanceUID"] = lab.StudyInstanceUID.astype(str)
        cols = [c for c in lab.columns
                if c not in ("StudyInstanceUID", "lang", "group", "unusable")]

        M = R.merge(lab, on="StudyInstanceUID", how="inner")
        P = M[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        keep = ~np.isnan(P).any(1)
        M, P = M[keep].iloc[:n_studies], P[keep][:n_studies]
        Y = (P > 0.5).astype(int)
        text = M.Report_en.astype(str).tolist()
        print(f"[bench] {len(text)} reports x {len(cols)} findings", flush=True)

        def probe(E):
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
            return float(np.mean(aucs))

        m = SentenceTransformer(name, trust_remote_code=True,
                                model_kwargs={"torch_dtype": torch.float16})
        m.max_seq_length = min(getattr(m, "max_seq_length", 512) or 512, 1024)
        E = m.encode(text, batch_size=4, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
        del m
        torch.cuda.empty_cache()
        macro = probe(E.astype(np.float32))
        print(f"[bench] {macro:.4f}  {name}  dim={E.shape[1]}", flush=True)
        return macro, int(E.shape[1])

    @app.local_entrypoint()
    def main(models: str = "", n_studies: int = 2500):
        chosen = CANDIDATES
        if models:
            want = [s.strip() for s in models.split(",") if s.strip()]
            chosen = [c for c in CANDIDATES if c[0] in want]
            chosen += [(w, "?", "requested") for w in want
                       if not any(c[0] == w for c in chosen)]
        rows = list(bench_one.starmap([(c, n_studies) for c in chosen]))
        ok = [r for r in rows if r[3] is not None]
        print("\n" + "=" * 88)
        print("RECOVERING THE 12 FINDINGS FROM THE REPORT EMBEDDING (4-fold)")
        for name, lic, note, macro, dim in sorted(ok, key=lambda r: -r[3]):
            print("  %.4f  %-44s dim=%-5d %-11s %s" % (macro, name, dim, lic, note))
        for name, lic, note, macro, dim in rows:
            if macro is None:
                print("  FAILED    %-44s %s" % (name, note))
        print("=" * 88)
        print("Ceiling is ~1.0: the labels were written FROM these reports. The encoder never")
        print("ships, so size and speed are irrelevant - only content retention matters.")
