"""member_preds_modal.py - the heavy ensemble members' logits on the 4,400 studies, on Modal.

Companion of data_code/19_ens_member_preds.py for the members that spill out of an 8 GB card.
ONE A100 container runs the members one after another (never parallel containers). Each member
is built by its OWN runs/<member>/train.py inside the container, loaded with the exact local
best.pt weights (uploaded weights-only to knee-runs:/_ens_infer/<member>.pt), and scored with the
engine's validation prediction function. Every finished member is written to the volume at once
(_ens_infer/preds/<member>.npz) so a crash loses nothing, and pulled home to
data/19_member_preds/<member>.npz in the same format as the local script, then checked the same
way (saved validation predictions of the same step, audit-58 AUROC).

    (select the Modal workspace first)
    python -m modal run modal_scripts/member_preds_modal.py
"""
import io
import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
sys.path.insert(0, "/root/runs")
import shared_code as sc  # noqa: E402

HEAVY = ["convnext_base_full_all_knees_same_orientation", "dinov3_convnext_base", "swinv2_base_in22k",
         "mricore_vitb_cross_attention", "mricore_vitb_mae_cross_attention_fulldata_flip"]
PRETRAINED = ["MRI_CORE_vitb.pth", "MRI_CORE_vitb_mae.pth"]
STUDIES_CSV = "17_train_full_qwen-mg-raw.csv"
app = modal.App("ens-member-preds")
runs_vol = modal.Volume.from_name("knee-runs")
shards = modal.Volume.from_name("knee-shards")


@app.function(image=sc.modal_image(), gpu="A100-80GB", cpu=12, memory=65536, timeout=5 * 3600,
              volumes={"/runs": runs_vol, "/shards": shards})
def predict(members: list):
    import importlib.util
    import time

    import numpy as np
    import pandas as pd
    import torch

    dev = torch.device("cuda")
    out = Path("/runs/_ens_infer/preds")
    out.mkdir(parents=True, exist_ok=True)
    for name in members:
        f = out / f"{name}.npz"
        if f.exists():
            print(f"[{name}] already on the volume, skipped", flush=True)
            continue
        sys.path.insert(0, f"/root/runs/{name}")
        spec = importlib.util.spec_from_file_location(f"train_{name}", f"/root/runs/{name}/train.py")
        tm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tm)
        rcfg = sc.remote_cfg(tm.cfg)
        assert rcfg["experiment"]["name"] == name
        ck = torch.load(f"/runs/_ens_infer/{name}.pt", map_location="cpu", weights_only=False)
        model = tm.build_model(rcfg)
        sc._load_model_state(model, ck["model"])
        model = model.to(dev).eval()
        df = sc._eval_df(rcfg, STUDIES_CSV)
        plan = sc.plan_slots(pd.read_csv(sc.resolve(rcfg["paths"]["series_index"], rcfg)), rcfg)
        torch.cuda.reset_peak_memory_stats()
        pk, t0 = {}, time.time()
        sc._predict_dataframe(rcfg, model, df, dev, plan=plan, desc=name[:40], collect=pk)
        secs = time.time() - t0
        logits = np.concatenate(pk["_lg"]).astype(np.float32)
        ids = df.StudyInstanceUID.astype(str).values
        np.savez_compressed(f, logits=logits, study_ids=ids, tasks=np.array(rcfg["tasks"]),
                            step=ck.get("step"), seconds=secs)
        runs_vol.commit()
        print(f"[{name}] step {ck.get('step')}: {len(ids)} studies in {secs:.0f}s, peak VRAM "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, saved to the volume", flush=True)
        del model
        torch.cuda.empty_cache()
        sys.path.remove(f"/root/runs/{name}")
    return {m: (out / f"{m}.npz").read_bytes() for m in members if (out / f"{m}.npz").exists()}


def upload_inputs():
    """Weights-only copies of the local best.pt (exact tensors) + the MRI-CORE trunks."""
    import tempfile

    import torch
    have = {e.path for e in runs_vol.listdir("/_ens_infer")} if _exists("/_ens_infer") else set()
    pre = {e.path for e in runs_vol.listdir("/pretrained")} if _exists("/pretrained") else set()
    tmp = Path(tempfile.mkdtemp())
    for m in HEAVY:
        if f"_ens_infer/{m}.pt" in have:
            print(f"  {m}: weights already on the volume")
            continue
        ck = torch.load(ROOT / f"runs/{m}/results/checkpoints/best.pt", map_location="cpu", weights_only=False)
        p = tmp / f"{m}.pt"
        torch.save({k: ck[k] for k in ("model", "cfg", "epoch", "step", "score") if k in ck}, p)
        with runs_vol.batch_upload(force=True) as b:
            b.put_file(str(p), f"/_ens_infer/{m}.pt")
        print(f"  {m}: uploaded step {ck.get('step')} ({p.stat().st_size / 1e9:.2f} GB)")
        p.unlink()
    for w in PRETRAINED:
        if f"pretrained/{w}" in pre:
            print(f"  {w}: already on the volume")
            continue
        with runs_vol.batch_upload(force=True) as b:
            b.put_file(str(ROOT / f"data/pretrained/{w}"), f"/pretrained/{w}")
        print(f"  {w}: uploaded")


def _exists(path):
    try:
        runs_vol.listdir(path)
        return True
    except Exception:                                                   # noqa: BLE001
        return False


@app.local_entrypoint()
def main():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    print("uploading inputs:")
    upload_inputs()
    print("predicting (one container, members in sequence):", HEAVY)
    res = predict.remote(HEAVY)
    dst = ROOT / "data/19_member_preds"
    dst.mkdir(parents=True, exist_ok=True)
    aud = pd.read_csv(ROOT / "data/05_audit58.csv").set_index("StudyInstanceUID")
    summ_f = dst / "summary.json"
    summ = json.loads(summ_f.read_text()) if summ_f.exists() else {}
    for m, blob in res.items():
        (dst / f"{m}.npz").write_bytes(blob)
        z = np.load(io.BytesIO(blob), allow_pickle=True)
        logits, ids, step = z["logits"], [str(s) for s in z["study_ids"]], int(z["step"])
        rec = {"step": step, "studies": len(ids), "seconds": float(z["seconds"]), "where": "modal A100"}
        pdir = ROOT / f"runs/{m}/results/predictions"
        cands = sorted(pdir.glob(f"*_val_s{step}.npz")) if pdir.exists() else []
        if cands:
            v = np.load(cands[-1], allow_pickle=True)
            old = pd.DataFrame(v["logits"], index=[str(s) for s in v["study_ids"]])
            new = pd.DataFrame(logits, index=ids)
            common = [s for s in old.index if s in new.index]
            d = np.abs(old.loc[common].values - new.loc[common].values)
            rec["val_check"] = {"studies": len(common), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}
        new = pd.DataFrame(logits, index=ids, columns=list(z["tasks"]))
        u = list(aud.index)
        y = aud.loc[u, ["official_" + t for t in z["tasks"]]].values
        rec["audit_auroc"] = float(np.mean([roc_auc_score(y[:, j], new.loc[u].values[:, j]) for j in range(12)]))
        summ[m] = rec
        print(f"[{m}] saved locally; {rec}")
    summ_f.write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(f"DONE: {len(res)}/{len(HEAVY)} heavy members saved in {dst}")
