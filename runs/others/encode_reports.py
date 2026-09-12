"""
encode_reports.py - build the auxiliary report-embedding target, on Modal.

Runs the chosen encoder over every report and writes ONE npz that training consumes. The
encoder never ships: only this file reaches the image model, and the auxiliary head that
consumes it is dropped at inference.

Qwen3-Embedding-8B is the default because it was measured to be the best, not assumed:
report_encoder_bench.py recovers the 12 findings from its vectors at 0.9634 against
0.9548 (4B), 0.9312 (0.6B), 0.9112 (bge-large) and 0.9072 (all-mpnet-base-v2). It is
Apache-2.0, and since it never ships its 8B size costs the submission nothing.

    modal run runs/others/encode_reports.py
    modal volume get knee-runs /_reports/11_report_emb.npz data/11_report_emb.npz
"""

from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-Embedding-8B"
OUT_REMOTE = "/runs_vol/_reports/11_report_emb.npz"

ROOT = Path(__file__).resolve().parent.parent.parent
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("sentence-transformers>=3.0", "transformers>=4.44", "torch",
                      "pandas", "numpy", "accelerate")
         .add_local_dir(str(ROOT / "data"), "/data_local",
                        ignore=lambda p: p.suffix not in (".csv",)))
app = modal.App("encode-reports")
runs_vol = modal.Volume.from_name("knee-runs", create_if_missing=True)


@app.function(image=image, gpu="H100", timeout=60 * 60,
              volumes={"/runs_vol": runs_vol})
def encode(model_name: str, batch_size: int):
    import os

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    R = pd.read_csv("/data_local/02_reports_en.csv")
    R["StudyInstanceUID"] = R.StudyInstanceUID.astype(str)
    # Two studies have an EMPTY Report_en because the translation step skipped reports that
    # were already in English, leaving the text only in Report. Dropping them would silently
    # cost two studies their target, so fall back to the original.
    n_fallback = int(R.Report_en.isna().sum())
    R["_text"] = R.Report_en.fillna(R.Report)
    R = R[R._text.notna()].copy()
    text = R._text.astype(str).tolist()
    print(f"[enc] {n_fallback} report(s) fell back to the untranslated original", flush=True)
    ids = R.StudyInstanceUID.to_numpy()
    print(f"[enc] {len(text)} reports, median {int(R.Report_en.str.len().median())} chars",
          flush=True)

    m = SentenceTransformer(model_name, trust_remote_code=True,
                            model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = min(getattr(m, "max_seq_length", 512) or 512, 1024)
    E = m.encode(text, batch_size=batch_size, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    print(f"[enc] encoded {E.shape[0]} x {E.shape[1]} with {model_name}", flush=True)

    # Standardisation stats from the TRAIN split only, so held-out reports never influence
    # the target's scaling. The engine must apply these: the findings live in LOWER-variance
    # directions than report style, so on the raw vector the auxiliary loss would spend its
    # gradient on writing style. A raw-cosine check misses this entirely and reported a
    # NEGATIVE separation on vectors that in fact recover the findings at 0.96.
    tr = pd.read_csv("/data_local/05_train_qwen24t-med.csv")
    train_ids = set(tr.StudyInstanceUID.astype(str))
    mask = np.array([i in train_ids for i in ids])
    mu, sd_raw = E[mask].mean(0), E[mask].std(0)

    # drop dead dimensions: dividing a constant output by its ~0 sd turns rounding noise
    # into a unit-variance target the image model would then chase
    keep = np.where(sd_raw > 0.1 * np.median(sd_raw))[0]
    E, mu, sd = E[:, keep], mu[keep], sd_raw[keep]
    print(f"[enc] train reports {int(mask.sum())}; kept {len(keep)} of {len(sd_raw)} dims; "
          f"sd ratio {sd.max() / sd.min():.1f}x", flush=True)

    os.makedirs(os.path.dirname(OUT_REMOTE), exist_ok=True)
    np.savez_compressed(OUT_REMOTE, study_ids=ids, emb=E.astype(np.float16),
                        mean=mu.astype(np.float32), std=sd.astype(np.float32),
                        keep=keep.astype(np.int32), model=model_name)
    runs_vol.commit()
    print(f"[enc] wrote {OUT_REMOTE}  ({os.path.getsize(OUT_REMOTE) / 1e6:.1f} MB)", flush=True)
    return {"n": int(E.shape[0]), "dim": int(E.shape[1]), "model": model_name}


@app.local_entrypoint()
def main(model: str = MODEL, batch_size: int = 4):
    print(encode.remote(model, batch_size))
    print("\nnow pull it down:")
    print("  modal volume get knee-runs /_reports/11_report_emb.npz "
          "data/11_report_emb.npz --force")
