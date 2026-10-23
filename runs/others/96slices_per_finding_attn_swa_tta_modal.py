"""96slices_per_finding_attn_swa_tta_modal.py - SWA and anatomical-mirror TTA for convnext_base_v5_96slices_per_finding_attn, scored on Modal (11 Oct).

Variants on the v5 val file (346 v3-labelled + the 58 audit studies), 94 windows per study as at inference:
  s9000        the best-val checkpoint alone (the submitted one)
  swa3         uniform weight average of top_step8750 / 9000 / 9250 (the public 0.949 averaged 3 epochs)
  *_mirror     the same weights on ANATOMICALLY MIRRORED studies: laterality_flip "right" instead of "left", so every
               knee arrives in the opposite orientation (sagittal slice order reversed, coronal/axial width flipped)
  *_tta        mean logit of the variant and its mirror (the public notebook's +0.001)
Logits come back to this PC: runs/others/96slices_per_finding_attn_swa_tta/<variant>.npz, then scored on val-346 / audit-58 / clean-121.

    (select the Modal profile that holds the run first)
    python -m modal run runs/others/96slices_per_finding_attn_swa_tta_modal.py
"""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent   # repo locally; /root in the container
for _p in (str(ROOT / "runs"), "/root/runs"):
    if Path(_p).exists():
        sys.path.insert(0, _p)
import shared_code as sc  # noqa: E402

RUN = "convnext_base_v5_96slices_per_finding_attn"
STEPS = [8750, 9000, 9250]
app = modal.App("attn96-swa-tta")
image = sc.modal_image()
shards = modal.Volume.from_name("knee-shards")
runs_vol = modal.Volume.from_name("knee-runs")


@app.function(image=image, gpu="H100", cpu=16.0, memory=65536, timeout=3 * 3600,
              volumes={"/shards": shards, "/runs": runs_vol}, retries=0)
def score() -> dict:
    import copy
    import importlib.util

    import numpy as np
    import pandas as pd
    import torch

    sys.path.insert(0, "/root/runs")
    import shared_code as rsc
    cfg = rsc.remote_cfg(rsc.load_config(Path("/root/runs") / RUN, verbose=False))
    cfg["dataloader"].update(num_workers=12, val_num_workers=12)
    spec = importlib.util.spec_from_file_location("attn96_train", f"/root/runs/{RUN}/train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ck = Path("/runs") / RUN / "results" / "checkpoints"
    sds = {}
    for s in STEPS:
        c = torch.load(ck / f"top_step{s}.pt", map_location="cpu", weights_only=False)
        sds[s] = c.get("model", c) if isinstance(c, dict) else c
    swa = {k: (sum(sds[s][k].float() for s in STEPS) / len(STEPS)).to(sds[9000][k].dtype)
           if sds[9000][k].is_floating_point() else sds[9000][k] for k in sds[9000]}
    df = pd.read_csv("/root/data/" + cfg["paths"]["val_csv"])
    dev = torch.device("cuda")
    out = {"ids": df.StudyInstanceUID.values}
    for wname, sd in (("s9000", sds[9000]), ("swa3", swa)):
        m = mod.build_model(copy.deepcopy(cfg))
        print(wname, m.load_state_dict(sd, strict=True), flush=True)
        for mir in (False, True):
            c2 = copy.deepcopy(cfg)
            if mir:
                c2["input"]["laterality_flip"] = "right"
            col = {}
            rsc._predict_dataframe(c2, m, df, dev, batch_size=4, collect=col, desc=wname + ("_mirror" if mir else ""))
            out[wname + ("_mirror" if mir else "")] = np.concatenate(col["_lg"]).astype(np.float32)
    return out


@app.local_entrypoint()
def main():
    import numpy as np
    res = score.remote()
    d = ROOT / "runs/others/96slices_per_finding_attn_swa_tta"
    d.mkdir(exist_ok=True)
    for k, v in res.items():
        if k != "ids":
            np.savez(d / f"{k}.npz", logits=v, ids=res["ids"])
    print("saved", sorted(k for k in res if k != "ids"))
