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
    # from the MTEB leaderboard, not from memory. Qwen3-8B is #2 overall (70.58); these
    # are the models above or beside it that are not API-only.
    ("tencent/KaLM-Embedding-Gemma3-12B-2511", "tencent-comm",
     "11.8B, MTEB #1 at 72.32 - LICENCE NEEDS CHECKING before use"),
    ("nvidia/llama-3.2-nemoretriever-300m-embed-v1", "nvidia-open", "300M"),
    ("intfloat/multilingual-e5-large-instruct", "mit", "560M, MTEB #9, the best MIT option"),
]

def _hf_secret():
    """The local Hugging Face token, as a Modal secret. Needed for gated repos.

    Read from the file the huggingface CLI already wrote; never printed or committed.
    Returns an empty secret when there is no token, so ungated models still run.
    """
    import modal as _m
    t = Path.home() / ".cache" / "huggingface" / "token"
    tok = t.read_text().strip() if t.exists() else ""
    return _m.Secret.from_dict({"HF_TOKEN": tok, "HUGGING_FACE_HUB_TOKEN": tok,
                                "HF_HUB_DISABLE_TELEMETRY": "1"})


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
                  secrets=[_hf_secret()])
    def bench_one(spec, n_studies: int, common_dim: int):
        """ONE encoder, in its own container.

        Isolation is not tidiness: a CUDA device-side assert poisons the whole context, so
        when Qwen3-Embedding-0.6B tripped one, every model queued behind it in the same
        container failed identically and looked broken. Separate containers give each
        candidate an honest attempt.
        """
        name, lic, note = spec
        try:
            return (name, lic, note) + _run_one(name, n_studies, common_dim)
        except Exception as e:
            print(f"[bench] FAILED {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            return (name, lic, note, None, 0, None)

    def _run_one(name, n_studies, common_dim):
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
        E = E.astype(np.float32)
        macro = probe(E)

        # DIMENSION CONTROL. A linear probe with 4,096 features on 2,500 samples has more
        # room to fit than one with 768, so a wider encoder can win on width rather than on
        # content. Reduce every candidate to the SAME width and probe again: if the ranking
        # survives, the margin is real.
        red = macro
        if common_dim and E.shape[1] > common_dim:
            X = E - E.mean(0)
            # randomised SVD would do; full is fine at this size and is deterministic
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            red = probe(U[:, :common_dim] * S[:common_dim])
        print(f"[bench] {macro:.4f} (full dim={E.shape[1]})  "
              f"{red:.4f} (at {min(common_dim, E.shape[1])}d)  {name}", flush=True)
        return macro, int(E.shape[1]), red

    @app.local_entrypoint()
    def main(models: str = "", n_studies: int = 2500, common_dim: int = 768):
        chosen = CANDIDATES
        if models:
            want = [s.strip() for s in models.split(",") if s.strip()]
            chosen = [c for c in CANDIDATES if c[0] in want]
            chosen += [(w, "?", "requested") for w in want
                       if not any(c[0] == w for c in chosen)]
        rows = list(bench_one.starmap([(c, n_studies, common_dim) for c in chosen]))
        ok = [r for r in rows if r[3] is not None]
        print("\n" + "=" * 88)
        print("RECOVERING THE 12 FINDINGS FROM THE REPORT EMBEDDING (4-fold)")
        print("  %-9s %-9s %-44s %-6s %-11s" % ("full", "@%dd" % common_dim, "model", "dim", "licence"))
        for name, lic, note, macro, dim, red in sorted(ok, key=lambda r: -r[3]):
            print("  %-9.4f %-9.4f %-44s %-6d %-11s %s"
                  % (macro, red if red is not None else macro, name, dim, lic, note))
        for name, lic, note, macro, dim, red in rows:
            if macro is None:
                print("  FAILED    %-44s %s" % (name, note))
        print("=" * 88)
        print("Ceiling is ~1.0: the labels were written FROM these reports. The encoder never")
        print("ships, so size and speed are irrelevant - only content retention matters.")
