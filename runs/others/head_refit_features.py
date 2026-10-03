"""head_refit_features.py - step 1 of the head-refit test: the feature vector each study feeds into the head.

Loads a finished run's checkpoint and, for EVERY labelled study (train + val + audit, 4,400), records the pooled
feature vector right before the final linear head, plus the original head's logits. No augmentation (train=False
loaders), knee flip as the run trained, mixed precision for the backbone, features kept in fp32. Step 2
(head_refit_fit.py) refits only the head on these, in fp32, on the CPU: with the backbone frozen and no augmentation
every epoch would recompute these exact vectors, so computing them once IS normal head-only fine-tuning, done cheaply.

Default: convnext_base_25d_18slices_v5_aug_base at top_step5900 (v4 labels, v5 split, flipped; val 0.9174, LB 0.930).
Local recipe: batch 4 studies, 1 loader worker, no pinned memory (8 GB card, 16 GB RAM).
Everything runs under main(): on Windows each loader worker re-imports this file, and top-level code would rerun.

    python runs/others/head_refit_features.py                     # all 4,400 studies (~20-30 min locally)
    python runs/others/head_refit_features.py --limit 9           # smoke test

Output: runs/<run>/results/features_s<step>.npz  (study_ids, split, feats [N, F] fp32, logits [N, 12], head_w, head_b)
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

RUNS = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="convnext_base_25d_18slices_v5_aug_base")
    ap.add_argument("--ckpt", default="top_step5900.pt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=2)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location(a.run, RUNS / a.run / "train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg, sc = m.cfg, m.sc
    cfg["model"]["pretrained"] = False                    # the checkpoint carries every weight
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(RUNS / a.run / "results" / "checkpoints" / a.ckpt, map_location="cpu", weights_only=False)
    model = m.build_model(cfg)
    res = sc._load_model_state(model, ck.get("model", ck))
    assert not res.missing_keys and not res.unexpected_keys, res
    model = model.to(dev).eval()
    print(f"{a.run} / {a.ckpt}: step {ck.get('step')}, score {ck.get('score')}, "
          f"flip {cfg['input'].get('laterality_flip')}, device {dev}", flush=True)

    # every labelled study once: the v5 train file (v4 labels), the v5 val (346 v3 + 58 audit)
    tr = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg)).assign(split="train")
    va = pd.read_csv(sc.resolve(cfg["paths"]["val_csv"], cfg))
    va["split"] = np.where(va["source"] == "audit", "audit", "val")
    df = pd.concat([tr, va], ignore_index=True)
    assert df.StudyInstanceUID.is_unique and len(df) == 4400, len(df)
    if a.limit:
        df = df.groupby("split", group_keys=False).head(max(1, a.limit // 3)).reset_index(drop=True)
    idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    plan = sc.plan_slots(idx, cfg)

    # RESUMABLE: progress is saved every 200 studies; a restart skips what is already done
    out = RUNS / a.run / "results" / (f"features_s{ck.get('step')}" + ("_smoke" if a.limit else "") + ".npz")
    part = out.with_name(out.stem + "_partial.npz")
    done = {"ids": [], "feats": [], "logits": []}
    if part.exists():
        p = np.load(part, allow_pickle=True)
        done = {"ids": list(p["ids"].astype(str)), "feats": [p["feats"]], "logits": [p["logits"]]}
        print(f"resuming: {len(done['ids'])} studies already done ({part.name})", flush=True)
    todo = df[~df.StudyInstanceUID.isin(set(done["ids"]))].reset_index(drop=True)
    ds = sc.KneeShardDataset(todo, cfg, plan, False)      # train=False: no augmentation, fixed slice picks
    assert ds.aug is None
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=False,
                    persistent_workers=False, prefetch_factor=a.prefetch if a.workers else None)
    print(f"{len(todo)} studies to do, batch {a.batch}, {a.workers} worker(s), prefetch {a.prefetch}", flush=True)

    def save_partial(n_new):
        ids = done["ids"] + list(todo.StudyInstanceUID.iloc[:n_new])
        Fp = np.concatenate(done["feats"] + feats) if (done["feats"] or feats) else np.zeros((0, 1))
        Lp = np.concatenate(done["logits"] + logits) if (done["logits"] or logits) else np.zeros((0, 12))
        assert len(ids) == len(Fp) == len(Lp)
        tmp = part.with_name(part.stem + "_tmp.npz")
        np.savez(tmp, ids=np.array(ids), feats=Fp, logits=Lp)
        tmp.replace(part)                                 # atomic: a kill mid-write cannot corrupt it

    feats, logits, grab = [], [], {}
    h = model.head.register_forward_hook(lambda mod, inp, out: grab.__setitem__("f", inp[0].detach()))
    t0, n = time.time(), 0
    with torch.no_grad():
        for b, batch in enumerate(dl):
            x = batch[0]
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                o = model(sc.prepare_batch(x, cfg, dev))
            feats.append(grab["f"].float().cpu().numpy())
            logits.append(o.float().cpu().numpy())
            n += len(x)
            if b % 50 == 0 or n == len(todo):
                save_partial(n)
                print(f"  {len(done['ids']) + n}/{len(df)} studies, {(time.time() - t0) / n:.2f} s/study (saved)", flush=True)
    h.remove()
    got = dict(zip(done["ids"] + list(todo.StudyInstanceUID),
                   zip(np.concatenate(done["feats"] + feats), np.concatenate(done["logits"] + logits))))
    F = np.stack([got[s][0] for s in df.StudyInstanceUID])  # back in the full table's order
    L = np.stack([got[s][1] for s in df.StudyInstanceUID])
    assert F.shape[0] == len(df) and np.isfinite(F).all() and np.isfinite(L).all()

    # the hook must have caught exactly what the head sees: head(F) == the model's logits
    with torch.no_grad():
        re = model.head(torch.from_numpy(F).to(dev)).float().cpu().numpy()
    print(f"head(features) vs model logits: max |diff| {np.abs(re - L).max():.2e}", flush=True)
    assert np.abs(re - L).max() < 5e-2, "the captured features do not reproduce the logits"

    np.savez_compressed(out, study_ids=df.StudyInstanceUID.values.astype(str), split=df.split.values.astype(str),
                        feats=F.astype(np.float32), logits=L.astype(np.float32),
                        head_w=model.head.weight.detach().float().cpu().numpy(),
                        head_b=model.head.bias.detach().float().cpu().numpy())
    part.unlink(missing_ok=True)
    print(f"saved {out} : {F.shape[0]} studies x {F.shape[1]} features, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
