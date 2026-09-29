"""local_train_bench.py - will this run train on THIS PC, and how long would it take?

Mirrors the engine's training step (run_experiment): the run's OWN train.py (its cfg and
build_model), channels_last, gradient checkpointing when the config asks for it, fp16 autocast
with a GradScaler, AdamW, soft-label BCE, the config's micro-batch and accumulation, real batches
from the local shards. Then a no-grad pass at the config's val batch size. Reports peak VRAM
(must stay below the card, or Windows silently spills into shared system memory and crawls)
and extrapolates a full run from the measured step times and the run's own val_log cadence.

    python checks/local_train_bench.py <run> [--steps 12]
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd
import torch



def main():
    # a function behind a __main__ guard: on Windows every DataLoader worker re-imports
    # this file, and module-level code would start training again in each worker
    ROOT = Path(__file__).resolve().parent.parent
    RUNS = ROOT / "runs"
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=None, help="override training.backbone_chunk")
    ap.add_argument("--eight-bit", type=int, default=None, help="override optimizer.eight_bit (0/1)")
    ap.add_argument("--inner-ckpt", type=int, default=None, help="override grad_checkpointing (0/1)")
    ap.add_argument("--no-val", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(RUNS))
    sys.path.insert(0, str(RUNS / a.run))
    spec = importlib.util.spec_from_file_location("train", RUNS / a.run / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    sc, cfg = tm.sc, tm.cfg
    assert cfg["experiment"]["name"] == a.run, cfg["experiment"]["name"]
    print(f"run {a.run}: train {cfg['paths']['train_csv']}, val {cfg['paths']['val_csv']}")
    if a.chunk is not None:
        cfg["training"]["backbone_chunk"] = a.chunk
    if a.eight_bit is not None:
        cfg["training"]["optimizer"]["eight_bit"] = bool(a.eight_bit)
    if a.inner_ckpt is not None:
        cfg["training"]["grad_checkpointing"] = bool(a.inner_ckpt)
        cfg["model"]["grad_checkpointing"] = bool(a.inner_ckpt)
    ckpt = bool(cfg["training"].get("grad_checkpointing") or cfg["model"].get("grad_checkpointing"))
    print(f"variant: backbone_chunk {cfg['training'].get('backbone_chunk')}, eight_bit "
          f"{cfg['training']['optimizer'].get('eight_bit')}, inner grad checkpointing {ckpt}")
    dl = cfg["dataloader"]
    B, ACC, VB = dl["batch_size"], dl.get("accum_steps", 1), dl.get("val_batch_size") or dl["batch_size"]
    dev = torch.device("cuda")
    print(f"GPU {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB; "
          f"micro-batch {B} x accum {ACC} = {B * ACC} per update; val batch {VB}; "
          f"grad_checkpointing {cfg['training'].get('grad_checkpointing')}; amp {cfg['training']['amp']}")

    df = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg))
    shards = Path(cfg["paths"]["shards_dir"])
    have = [s for s in df.StudyInstanceUID if (shards / f"{s}.npz").exists()]
    n_need = B * (a.steps + 2)
    assert len(have) >= n_need, f"only {len(have)} local shards, need {n_need}"
    sub = df[df.StudyInstanceUID.isin(have[:n_need])].reset_index(drop=True)
    plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
    ds = sc.KneeShardDataset(sub, cfg, plan, train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=False, num_workers=dl["num_workers"], persistent_workers=True, prefetch_factor=dl.get("prefetch_factor", 2))

    model = tm.build_model(cfg).to(dev).to(memory_format=torch.channels_last)
    getattr(model, "backbone", model).set_grad_checkpointing(ckpt)   # build_model may read either key
    if cfg["training"].get("backbone_chunk"):
        sc.chunk_backbone(model, int(cfg["training"]["backbone_chunk"]))
    opt = sc.build_optimizer(model, cfg)                              # the engine's own groups
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"])
    lossf = torch.nn.BCEWithLogitsLoss()
    torch.cuda.reset_peak_memory_stats()
    times = []
    model.train()
    for i, bt in enumerate(loader):
        xb = sc.prepare_batch(bt[0], cfg, dev)
        yb = bt[1].to(dev).float()
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", enabled=scaler.is_enabled()):
            out = model(xb)
            loss = lossf(out.float(), yb) / ACC
        scaler.scale(loss).backward()
        if (i + 1) % ACC == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.time() - t)
    peak = torch.cuda.max_memory_allocated() / 2**30
    import psutil
    _me = psutil.Process()
    ram = (_me.memory_info().rss + sum(c.memory_info().rss for c in _me.children(recursive=True))) / 2**30
    print(f"RAM: this process + {len(_me.children(recursive=True))} loader workers = {ram:.2f} GiB; "
          f"system available now {psutil.virtual_memory().available / 2**30:.1f} GiB")
    card = torch.cuda.get_device_properties(0).total_memory / 2**30
    step = sum(times) / len(times)
    print(f"TRAIN: {len(times)} timed micro-steps, {step:.2f} s each (GPU only), peak VRAM {peak:.2f} GiB "
          f"of {card:.1f} ({'fits' if peak < card - 0.4 else 'TOO CLOSE / spills to system memory'})")

    print(f"TRAIN seconds per study: {step / B:.2f}; {cfg['training']['epochs']} epochs x {len(df)} "
          f"studies = {cfg['training']['epochs'] * len(df) * step / B / 3600:.1f} h GPU time")
    if a.no_val:
        return
    model.eval()
    vds = sc.KneeShardDataset(sub, cfg, plan, train=False)
    vl = torch.utils.data.DataLoader(vds, batch_size=VB, shuffle=False, num_workers=dl["num_workers"], persistent_workers=True, prefetch_factor=dl.get("prefetch_factor", 2))
    vt, nv = [], 0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i, bt in enumerate(vl):
            xb = sc.prepare_batch(bt[0], cfg, dev)
            torch.cuda.synchronize()
            t = time.time()
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                model(xb)
            torch.cuda.synchronize()
            if i >= 1:
                vt.append(time.time() - t)
                nv += len(bt[0])
    vs = sum(vt) / max(nv, 1)
    print(f"VAL:   {vs:.3f} s per study (GPU only), peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    n_train = len(df)
    n_val = len(pd.read_csv(sc.resolve(cfg["paths"]["val_csv"], cfg))) + 58
    epochs = cfg["training"]["epochs"]
    per_epoch = n_train / B * step
    ref = RUNS / "convnext_base_25d_18slices_mglabels_raw/results/val_log.csv"
    n_vals = len(pd.read_csv(ref).drop_duplicates("step")) if ref.exists() else 50
    val_total = n_vals * n_val * vs
    print(f"\nESTIMATE (GPU time only; data loading can add more): {epochs} epochs x {per_epoch / 3600:.2f} h "
          f"= {epochs * per_epoch / 3600:.1f} h training + {n_vals} validations x {n_val * vs / 60:.1f} min "
          f"= {val_total / 3600:.1f} h -> about {(epochs * per_epoch + val_total) / 3600:.1f} h "
          f"(the A100 run took 1.5 h and early-stopped after {n_vals} validations)")


if __name__ == "__main__":
    main()
