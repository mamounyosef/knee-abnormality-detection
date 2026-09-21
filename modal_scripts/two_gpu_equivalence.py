r"""two_gpu_equivalence.py - score the SAME studies on 1 T4 and on 2 T4s and diff the output.

checks/two_gpu_split_check.py proves the plumbing in isolation. This proves the thing that
actually protects a submission: the real ensemble, the real loader, real DICOMs, scored both
ways, giving the same predictions.

It runs on Modal with gpu="T4:2" so the second card is real, and forces the single-device
path with inference.max_gpus = 1 for the reference pass. Kaggle's "GPU T4 x2" is the same
hardware, so the timing here is a fair preview of the submission.

    python -m modal run modal_scripts/two_gpu_equivalence.py --studies 6
"""

import sys
from pathlib import Path

# The container re-imports this file as /root/smoke.py, where __file__.parent.parent/"runs"
# is /runs and does not exist - the runs tree is mounted at /root/runs. Try that first, the
# way every train.py does, or the module-level import dies before anything runs.
RUNS = next((p for p in (Path("/root/runs"),
                         Path(__file__).resolve().parent.parent / "runs")
             if (p / "shared_code.py").exists()),
            Path(__file__).resolve().parent.parent / "runs")
sys.path.insert(0, str(RUNS))
import shared_code as sc  # noqa: E402

import modal  # noqa: E402

MEMBERS = ["convnext_base_relabel", "mricore_vitb_cross_attention",
           "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
           "convnext_base_25d_all_knees_same_orientation", "swinv2_base_in22k",
           "convnext_base_25d_poolattn",
           "convnext_base_25d_perfinding_slots_and_pooling"]

cfg0 = sc.load_config(RUNS / MEMBERS[0], verbose=False)
app = modal.App("two-gpu-equivalence")
_image = sc.modal_image()
_shards = modal.Volume.from_name(cfg0["modal"]["shards_volume"], create_if_missing=True)
_runs = modal.Volume.from_name(cfg0["modal"]["runs_volume"], create_if_missing=True)
_data = modal.Volume.from_name("knee-data", create_if_missing=True)


@app.function(image=_image, gpu="T4:2", timeout=5400, cpu=8.0, memory=32768,
              volumes={cfg0["modal"]["shards_mount"]: _shards,
                       cfg0["modal"]["runs_mount"]: _runs, "/data": _data})
def compare(members: list, studies: int = 6):
    import time

    import numpy as np
    import pandas as pd
    import torch

    sys.path.insert(0, "/root/runs")
    import shared_code as s  # noqa: F811

    print("  visible GPUs: %d  (%s)" % (torch.cuda.device_count(),
                                        torch.cuda.get_device_name(0)), flush=True)
    assert torch.cuda.device_count() >= 2, "this test needs two cards"

    cfgs, models = [], []
    for name in members:
        c = s.remote_cfg(s.load_config(Path("/root/runs") / name, verbose=False))
        ck = Path(c["modal"]["runs_mount"]) / name / "results" / "checkpoints" / "best.pt"
        m = s.load_model_for_inference(c, ck) if hasattr(s, "load_model_for_inference") else None
        if m is None:
            import importlib
            sys.path.insert(0, "/root/runs/" + name)
            tm = importlib.import_module("train")
            m = tm.build_model(c)
            sd = torch.load(str(ck), map_location="cpu", weights_only=False)
            sd = sd.get("model", sd.get("state_dict", sd))
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            m.load_state_dict(sd, strict=False)
            for mod in list(sys.modules):
                if mod == "train":
                    del sys.modules[mod]
        cfgs.append(c)
        models.append(m)
    print("  %d members loaded" % len(models), flush=True)

    df = pd.read_csv(s.resolve(cfgs[0]["paths"]["val_csv"], cfgs[0])).head(studies)
    idx = pd.read_csv(s.resolve(cfgs[0]["paths"]["series_index"], cfgs[0]))
    plan = s.plan_slots(idx, cfgs[0])
    zipp = "/data/rsna-knee-abnormality-detection.zip"
    index, files = s.build_dicom_index(zipp, df) if Path(zipp).exists() else (None, None)
    if index is None:
        print("  no DICOM archive on this volume - scoring from SHARDS instead")

    def run(max_gpus):
        for c in cfgs:
            c.setdefault("inference", {})["max_gpus"] = max_gpus
        torch.cuda.empty_cache()
        t0 = time.time()
        out = s.predict_dicom_ensemble_grouped(
            cfgs, [m for m in models], df, plan, index, files) if index is not None else None
        return out, time.time() - t0

    one, t1 = run(1)
    two, t2 = run(2)
    cols = [c for c in one.columns if c != "StudyInstanceUID"]
    same_ids = list(one["StudyInstanceUID"]) == list(two["StudyInstanceUID"])
    d = np.abs(one[cols].values - two[cols].values)
    print("\n  study order identical: %s" % same_ids)
    print("  max abs difference over %d x %d predictions: %.3e" % (d.shape[0], d.shape[1], d.max()))
    print("  mean abs difference: %.3e" % d.mean())
    print("  1 GPU %.1f s   2 GPUs %.1f s   speedup %.2fx" % (t1, t2, t1 / max(t2, 1e-9)))
    return {"same_ids": same_ids, "max_diff": float(d.max()), "mean_diff": float(d.mean()),
            "t1": t1, "t2": t2}


@app.local_entrypoint()
def main(studies: int = 6):
    r = compare.remote(MEMBERS, studies)
    ok = r["same_ids"] and r["max_diff"] < 1e-5
    print("\n  %s  max diff %.3e, %.2fx faster"
          % ("EQUIVALENT" if ok else "MISMATCH - DO NOT SUBMIT",
             r["max_diff"], r["t1"] / max(r["t2"], 1e-9)))
