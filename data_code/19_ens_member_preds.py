"""
19_ens_member_preds.py - every 0.938-ensemble member's logits on all 4,400 labelled studies.

For the "train on the ensemble's labels" test: the ensemble is a flat average of its members'
logits, so saving each member's logits on the train + val + audit studies is enough to rebuild
it exactly. Runs LOCALLY, one member at a time, at below-normal priority. Each member is loaded
through its OWN runs/<member>/train.py (its config and build_model) with its best.pt, and scored
with the engine's validation prediction function (_predict_dataframe, eval transforms, the run's
own amp / slot plan), so these are the same numbers a validation would produce.

Saved per member: data/19_member_preds/<member>.npz (logits, study_ids, tasks, step, seconds).
A member already saved is skipped (resume); --force redoes it.

Checks after each member:
  - members with saved validation predictions: the new logits on those val studies must match
    the saved ones from the same checkpoint step
  - audit-58 AUROC against the panel (full-data members trained on these, so theirs is inflated)

    python data_code/19_ens_member_preds.py --test          # 6 audit studies per member: VRAM + speed
    python data_code/19_ens_member_preds.py                 # the full 4,400 studies
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "data/19_member_preds"
STUDIES_CSV = "17_train_full_qwen-mg-raw.csv"                  # train 3,638 + val 704 + audit 58
MEMBERS = ["convnext_base_full_all_knees_same_orientation", "mricore_vitb_cross_attention",
           "mricore_vitb_mae_cross_attention_fulldata_flip", "dinov3_convnext_base",
           "convnext_base_25d_all_knees_same_orientation_fulldata", "swinv2_base_in22k",
           "convnext_base_25d_poolattn_fulldata_flip",
           "convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip",
           "medimageinsight_davit_huge_336_fulldata", "medimageinsight_davit_huge_perslice_480_warm"]


class ChunkedBackbone(torch.nn.Module):
    """Feed the backbone `n` images at a time instead of the whole (B*windows) stack. Exact for
    backbones that treat every image independently (ConvNeXt / SwinV2 / DaViT: LayerNorm, no
    batch statistics): the per-slice 480 member otherwise needs ~13 GB and spills 5.5 GB into
    shared system memory on an 8 GB card, 20x slower (27 Sep). NOT for the MRI-CORE members,
    whose inserted cross-slice attention mixes images on purpose."""

    def __init__(self, bb, n):
        super().__init__()
        self.bb, self.n = bb, n

    def forward(self, x):
        return torch.cat([self.bb(c) for c in x.split(self.n)], dim=0)


NO_CHUNK = ("mricore",)


def load_member(name):
    sys.path.insert(0, str(RUNS))
    sys.path.insert(0, str(RUNS / name))
    spec = importlib.util.spec_from_file_location(f"train_{name}", RUNS / name / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    sys.path.remove(str(RUNS / name))
    assert tm.cfg["experiment"]["name"] == name, tm.cfg["experiment"]["name"]
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="6 audit studies per member, nothing saved")
    ap.add_argument("--members", default="", help="comma list; default all ten")
    ap.add_argument("--batch", type=int, default=2, help="studies per forward pass")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="", help="comma list for the one-process-per-member loop")
    ap.add_argument("--chunk", type=int, default=0, help="backbone images per forward (0 = all)")
    ap.add_argument("--out", default="", help="write to this folder instead (checks)")
    a = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:                                                # noqa: BLE001
        pass
    from sklearn.metrics import roc_auc_score
    if not a.members and not a.test:
        # one FRESH process per member: eight models loaded one after another in one process
        # ran this PC out of RAM with no error printed (27 Sep); a child exits and frees it all
        import subprocess
        for m in [x for x in a.only.split(",") if x] or MEMBERS:
            if (OUT / f"{m}.npz").exists() and not a.force:
                print(f"[{m}] already saved, skipped", flush=True)
                continue
            cmd = [sys.executable, "-u", __file__, "--members", m, "--batch", str(a.batch),
                   "--workers", str(a.workers), "--chunk", str(a.chunk)] + (["--force"] if a.force else [])
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"[{m}] FAILED (exit {r.returncode}); the others continue", flush=True)
        done = [m for m in MEMBERS if (OUT / f"{m}.npz").exists()]
        print(f"\nALL DONE: {len(done)}/{len(MEMBERS)} members saved in {OUT}", flush=True)
        return
    members = [m for m in a.members.split(",") if m] or MEMBERS
    OUT.mkdir(parents=True, exist_ok=True)
    aud = pd.read_csv(ROOT / "data/05_audit58.csv").set_index("StudyInstanceUID")
    dev = torch.device("cuda")
    summary = {}
    for name in members:
        f = OUT / f"{name}.npz"
        if f.exists() and not a.force and not a.test:
            print(f"[{name}] already saved, skipped", flush=True)
            continue
        tm = load_member(name)
        sc, cfg = tm.sc, tm.cfg
        ck = torch.load(RUNS / name / "results/checkpoints/best.pt", map_location="cpu", weights_only=False)
        model = tm.build_model(cfg)
        sc._load_model_state(model, ck["model"])
        if a.chunk and not name.startswith(NO_CHUNK) and hasattr(model, "backbone"):
            model.backbone = ChunkedBackbone(model.backbone, a.chunk)
            print(f"[{name}] backbone fed {a.chunk} images at a time", flush=True)
        model = model.to(dev).eval()
        del ck["model"]
        df = sc._eval_df(cfg, STUDIES_CSV)
        if a.test:
            df = df[df.StudyInstanceUID.isin(aud.index)].head(6).reset_index(drop=True)
        plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pk = {}
        t0 = time.time()
        sc._predict_dataframe(cfg, model, df, dev, plan=plan, batch_size=a.batch,
                              num_workers=a.workers, desc=f"{name[:40]}", collect=pk)
        secs = time.time() - t0
        logits = np.concatenate(pk["_lg"]).astype(np.float32)
        ids = df.StudyInstanceUID.astype(str).values
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"[{name}] step {ck.get('step')}: {len(ids)} studies in {secs:.0f}s "
              f"({secs / len(ids):.2f} s/study), peak VRAM {peak:.2f} GiB", flush=True)
        rec = {"step": ck.get("step"), "studies": len(ids), "seconds": round(secs, 1),
               "s_per_study": round(secs / len(ids), 3), "peak_vram_gib": round(peak, 2)}
        if not a.test:
            np.savez_compressed(f, logits=logits, study_ids=ids, tasks=np.array(cfg["tasks"]),
                                step=ck.get("step"), seconds=secs)
            # check 1: the saved validation predictions of the same checkpoint step
            pdir = RUNS / name / "results/predictions"
            cands = sorted(pdir.glob(f"*_val_s{ck.get('step')}.npz")) if pdir.exists() else []
            if cands:
                z = np.load(cands[-1], allow_pickle=True)
                old = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]])
                new = pd.DataFrame(logits, index=ids)
                common = [s for s in old.index if s in new.index]
                d = np.abs(old.loc[common].values - new.loc[common].values)
                rec["val_check"] = {"studies": len(common), "max_abs_diff": float(d.max()),
                                    "mean_abs_diff": float(d.mean())}
                print(f"    val check vs {cands[-1].name}: {len(common)} studies, max |logit diff| "
                      f"{d.max():.4f}, mean {d.mean():.5f}", flush=True)
            # check 2: audit-58 AUROC against the panel
            new = pd.DataFrame(logits, index=ids, columns=cfg["tasks"])
            u = list(aud.index)
            y = aud.loc[u, ["official_" + t for t in cfg["tasks"]]].values
            auc = float(np.mean([roc_auc_score(y[:, j], new.loc[u].values[:, j]) for j in range(12)]))
            rec["audit_auroc"] = auc
            print(f"    audit-58 macro AUROC {auc:.4f}" +
                  ("  (trained on these: in-sample)" if "full" in cfg["paths"]["train_csv"] else ""), flush=True)
        summary[name] = rec
        del model
        torch.cuda.empty_cache()
    if not a.test:
        prev = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
        prev.update(summary)
        (OUT / "summary.json").write_text(json.dumps(prev, indent=1), encoding="utf-8")
    else:
        est = sum(r["s_per_study"] for r in summary.values()) * 4400 / 3600
        print(f"\nTEST done. Estimated full run: {est:.1f} h for 4,400 studies x {len(summary)} members "
              f"(largest peak VRAM {max(r['peak_vram_gib'] for r in summary.values()):.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
