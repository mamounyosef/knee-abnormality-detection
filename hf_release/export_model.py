"""export_model.py - build the Hugging Face folder for ONE model. Uploads nothing.

Steps:
    1. download the submitted weights + config from the private Kaggle dataset
    2. save the weights as model.safetensors (the standard Hugging Face format)
    3. write config.json, metrics.json, val_history.csv, modeling.py and README.md next to it
    4. load the folder back to check that it works

    python hf_release/export_model.py
"""
import csv
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file   # safetensors: weights-only format, can't hide code like .pt can

ROOT = Path(__file__).resolve().parent.parent      # the repo
HERE = Path(__file__).resolve().parent             # hf_release/

# ---- the model to export (change these to export a different model) ----
RUN = "convnext_base_v5_96slices_per_finding_attn"   # run folder name under runs/
KAGGLE_DATASET = "mamounhussam/knee-raptor4-folds"   # private dataset holding the weights we submitted
MEMBER = "m0"                                         # this model's file prefix in that dataset (m0_model.pt, ...)
STEP = 9000                                           # training step of those weights
PUBLIC_LB = 0.939                                     # Kaggle scores of this model on its own
PRIVATE_LB = 0.922

OUT = HERE / "out" / RUN                   # the folder that gets uploaded
DOWNLOADS = HERE / "out" / "_download"     # raw Kaggle files, kept so re-runs skip the download


def download():
    """Step 1: get mN_model.pt and mN_config.json from Kaggle (skipped if already downloaded)."""
    # Avast intercepts HTTPS on this PC, so point Python at the matching certificate bundle
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
    os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])
    import kaggle   # imported here: it logs in on import, and only this step needs it

    for name in [f"{MEMBER}_model.pt", f"{MEMBER}_config.json"]:
        if (DOWNLOADS / name).exists():
            continue
        print("downloading", name)
        kaggle.api.dataset_download_file(KAGGLE_DATASET, name, path=str(DOWNLOADS))
        zipped = DOWNLOADS / (name + ".zip")   # Kaggle zips big files
        if zipped.exists():
            shutil.unpack_archive(zipped, DOWNLOADS)
            zipped.unlink()


def save_weights():
    """Step 2: .pt -> model.safetensors. The .pt holds {"model": weights, "cfg", "epoch", "step", "score"}."""
    checkpoint = torch.load(DOWNLOADS / f"{MEMBER}_model.pt", map_location="cpu")   # cpu: no GPU needed
    # make sure this really is the checkpoint we think it is
    assert checkpoint["step"] == STEP, f"expected step {STEP}, the file has {checkpoint['step']}"
    # "model" maps each layer's name to its tensor; safetensors wants each tensor in one contiguous block
    weights = {name: tensor.contiguous() for name, tensor in checkpoint["model"].items()}
    save_file(weights, OUT / "model.safetensors")   # THE conversion: same tensors, written in safetensors format
    n_params = sum(t.numel() for t in weights.values())
    print(f"model.safetensors: {n_params / 1e6:.1f}M parameters")


def save_config():
    """Step 3a: config.json, keeping only what is needed to rebuild and understand the model."""
    full = json.loads((DOWNLOADS / f"{MEMBER}_config.json").read_text(encoding="utf-8"))
    # drop paths, cloud settings and logging options: they only mattered on our machines
    config = {
        "experiment": {"name": RUN, "checkpoint_step": STEP},
        "tasks": full["tasks"],                 # the 12 findings, in output order
        "model": full["model"],                 # backbone name
        "input": full["input"],                 # series, slices, resolution, windows
        "augmentation": full["augmentation"],
        "target": full["target"],
        "training": full["training"],           # optimiser, schedule, loss, epochs
    }
    for unused in ["aux_report", "aux_attributes", "log_ud_ratio", "perf"]:   # switched-off extras
        config["training"].pop(unused, None)
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def save_metrics():
    """Step 3b: metrics.json (scores at STEP) and val_history.csv (the whole validation log)."""
    val_log = ROOT / "runs" / RUN / "results" / "val_log.csv"   # one row per validation during training
    shutil.copy(val_log, OUT / "val_history.csv")

    # the row for our step (the last one, in case that step was validated twice)
    with open(val_log, encoding="utf-8") as f:
        row = [r for r in csv.DictReader(f) if int(r["step"]) == STEP][-1]

    metrics = {
        "step": STEP,
        "epoch": int(row["epoch"]),
        "public_lb": PUBLIC_LB,
        "private_lb": PRIVATE_LB,
        "val_macro_auroc": round(float(row["val_macro_auroc"]), 4),        # 346 studies, report labels
        "audit58_macro_auroc": round(float(row["audit_macro_auroc"]), 4),  # 58 studies, expert labels
        "val_auroc_per_finding": {},
        "audit58_auroc_per_finding": {},
    }
    # per-finding columns look like "val_auroc_ACL" / "audit_auroc_ACL"
    for column, value in row.items():
        if column.startswith("val_auroc_"):
            metrics["val_auroc_per_finding"][column.removeprefix("val_auroc_")] = round(float(value), 4)
        elif column.startswith("audit_auroc_"):
            metrics["audit58_auroc_per_finding"][column.removeprefix("audit_auroc_")] = round(float(value), 4)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def copy_code_and_card():
    """Step 3c: the standalone model code and the model card."""
    shutil.copy(HERE / "modeling_per_finding_attn.py", OUT / "modeling.py")
    shutil.copy(HERE / "cards" / f"{RUN}.md", OUT / "README.md")   # Hugging Face shows README.md as the page


def check():
    """Step 4: rebuild the model from the folder alone and run a dummy input through it."""
    import sys
    sys.dont_write_bytecode = True             # no __pycache__ in the upload folder
    sys.path.insert(0, str(OUT))               # import modeling.py from the exported folder, like a user would
    from modeling import load_model

    model = load_model(OUT)                    # fails if any weight does not match
    with torch.no_grad():                      # inference only, no gradients
        # one fake study: 94 windows of 3 x 64 x 64; small images keep the check fast
        out = model(torch.zeros(model.windows_per_study, 3, 64, 64))
    print("check ok: one study ->", tuple(out.shape), "scores")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    download()
    save_weights()
    save_config()
    save_metrics()
    copy_code_and_card()
    check()
    print("\nready to upload:", OUT)
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:20s} {f.stat().st_size / 2**20:7.1f} MB")


if __name__ == "__main__":
    main()
