"""
evaluate.py — held-out reconstruction loss for a saved MAE checkpoint.

This is a sanity number, NOT a verdict. Reconstruction loss falls almost monotonically and
says nothing about whether the trunk got better for our 12 findings. The real evaluation is
the downstream fine-tune: load the encoder into the cross-attention run and compare against
mricore_vitb_cross_attention's 0.8962.

    python -m modal run runs/mricore_vitb_mae/evaluate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXP_NAME = "mricore_vitb_mae"
PKG_ROOT = next((c for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent)
                 if (c / "shared_code.py").exists()), Path(__file__).resolve().parent.parent)
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(EXP_DIR))
import shared_code as sc  # noqa: E402
from train import MAE, MAESliceDataset  # noqa: E402


def _run(cfg, out_dir: Path, checkpoint: str = "last"):
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = Path(out_dir) / cfg["output"]["run_dir"]
    ck_path = res / cfg["output"]["checkpoints_dir"] / f"{checkpoint}.pt"
    if not ck_path.exists():
        raise SystemExit(f"no checkpoint at {ck_path}")

    model = MAE(cfg).to(device)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    # the val split, exactly what the training loop reports each epoch
    series = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    ho = MAESliceDataset(pd.read_csv(sc.resolve(cfg["paths"]["val_csv"], cfg)),
                         cfg, series, train=False)

    dl = cfg["dataloader"]
    loader = DataLoader(ho, batch_size=dl["batch_size"], shuffle=False,
                        num_workers=dl.get("val_num_workers") or dl["num_workers"])
    tot, n = 0.0, 0
    with torch.no_grad():
        for x in loader:
            with torch.autocast("cuda", enabled=cfg["training"]["amp"]):
                tot += float(model(sc.prepare_batch(x, cfg, device))) * x.shape[0]
            n += x.shape[0]
    print(f"\n  checkpoint {ck_path.name} (epoch {ck.get('epoch')}, step {ck.get('step')})")
    print(f"  held-out masked-patch MSE: {tot / max(n, 1):.4f}  over {n * ho.k:,} slices")


def main():
    _run(sc.load_config(EXP_DIR, verbose=False), EXP_DIR)


try:
    import modal

    cfg = sc.load_config(EXP_DIR, verbose=False)
    app = modal.App(f"{EXP_NAME}-eval")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={cfg["modal"]["shards_mount"]: _shards,
                           cfg["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(cfg))
    def eval_remote(checkpoint: str):
        _run(sc.remote_cfg(cfg), Path(cfg["modal"]["runs_mount"]) / EXP_NAME, checkpoint)

    @app.local_entrypoint()
    def modal_main(checkpoint: str = "last"):
        try:
            eval_remote.remote(checkpoint)
        finally:
            sc.sync_back(cfg, "mae-evaluated")

except ImportError:
    pass


if __name__ == "__main__":
    main()
