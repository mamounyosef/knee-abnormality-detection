"""
shared_code.py — the knee-MRI training engine.

Everything shared across runs lives here: config loading, the shard dataset,
metrics, the train loop and the Modal glue. Run folders only supply a config
delta and a build_model().

Shards are one .npz per study, one uint8 array per series at native resolution
(modal_scripts/modal_preprocess.py). All the modelling decisions preprocessing
left open — slot choice, duplicates, slice count, mm/px, size, windowing — are
applied here from config.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    import cv2
    cv2.setNumThreads(0)          # workers do their own parallelism
except ImportError:
    cv2 = None


# =============================================================================
# config
# =============================================================================
def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(experiment_dir, config_name: str = "config.yaml", verbose: bool = True) -> dict:
    exp_dir = Path(experiment_dir)
    shared = yaml.safe_load((exp_dir.parent / "shared_config.yaml").read_text(encoding="utf-8"))
    own = yaml.safe_load((exp_dir / config_name).read_text(encoding="utf-8")) or {}
    cfg = _deep_merge(shared, own)
    cfg.setdefault("experiment", {}).setdefault("name", exp_dir.name)
    if verbose:
        print_config(cfg)
    return cfg


def print_config(cfg: dict):
    i = cfg["input"]
    t = cfg["training"]
    print("=" * 70)
    print(f"experiment : {cfg['experiment']['name']}")
    print(f"model      : {cfg.get('model', {}).get('name', '?')}")
    print(f"input      : {len(i['slots'])} slots {i['slots']}")
    print(f"             {i['n_slices']} slices @ {i['size']}px, {i['mm_per_px']} mm/px")
    print(f"target     : {cfg['target']['mode']}   loss: {t['loss']['name']}")
    print(f"training   : {t.get('epochs')} epochs, batch {t.get('batch_size')}, "
          f"lr {t['optimizer'].get('lr')}")
    print("=" * 70)


def num_output_logits(cfg: dict) -> int:
    return len(cfg["tasks"])


def resolve(name: str, cfg: dict) -> Path:
    p = Path(name)
    return p if p.is_absolute() else Path(cfg["paths"]["data_dir"]) / name


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# =============================================================================
# slot planning
# =============================================================================
PLANE_OF = {"sag_plain": "sagittal", "sag_fluid": "sagittal",
            "cor_plain": "coronal", "cor_fluid": "coronal",
            "ax_plain": "axial", "ax_fluid": "axial"}


def plan_slots(index: pd.DataFrame, cfg: dict) -> dict[str, dict[str, str]]:
    """study -> {slot: series}. Applies fallback and the duplicate rule."""
    slots, rule = cfg["input"]["slots"], cfg["input"]["duplicate_rule"]
    fallback = cfg["input"]["fallback_same_plane"]

    # 44% of studies have >1 series competing for a slot; first row after the sort wins
    sort_by, asc = {
        "coverage":      (["depth_mm", "pixel_spacing_c"], [False, True]),
        "resolution":    (["pixel_spacing_c", "depth_mm"], [True, False]),
        "slices":        (["n_slices", "depth_mm"], [False, False]),
        "series_number": (["series", "depth_mm"], [True, False]),
        "random":        (["series"], [True]),
    }[rule]

    plan: dict[str, dict[str, str]] = {}
    for study, g in index.groupby("study", sort=False):
        chosen = {}
        for slot in slots:
            cand = g[g.slot == slot]
            if cand.empty and fallback:
                cand = g[g.plane == PLANE_OF[slot]]      # same plane, either setting
            if cand.empty:
                continue
            chosen[slot] = cand.sort_values(sort_by, ascending=asc).iloc[0]["series"]
        plan[study] = chosen
    return plan


def pick_slices(n_available: int, n_want: int, mode: str, rng,
                margin: float = 0.0) -> np.ndarray:
    """Indices of real slices. Never interpolates; repeats when too few exist."""
    # the first/last slices of a stack are usually past the edge of the knee
    lo = int(n_available * margin)
    hi = n_available - lo
    if hi - lo >= n_want:
        edges = np.linspace(lo, hi, n_want + 1)
        if mode == "random":
            lo, hi = np.floor(edges[:-1]).astype(int), np.ceil(edges[1:]).astype(int)
            return np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(lo, hi)])
        return np.clip(((edges[:-1] + edges[1:]) / 2).astype(int), 0, n_available - 1)
    # too few slices: repeat rather than interpolate (17.8% of series at n=24)
    idx = np.linspace(0, n_available - 1, n_want)
    return np.round(idx).astype(int)


# =============================================================================
# dataset
# =============================================================================
class KneeShardDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict, plan: dict, train: bool):
        self.df = df.reset_index(drop=True)
        self.cfg, self.plan, self.train = cfg, plan, train
        self.tasks = cfg["tasks"]
        self.shards = Path(cfg["paths"]["shards_dir"])
        i = cfg["input"]
        self.slots, self.n_slices, self.size = i["slots"], i["n_slices"], i["size"]
        self.mm_per_px, self.window = i["mm_per_px"], i["window"]
        self.mode = i["train_slice_sampling"] if train else i["slice_sampling"]
        self.margin = i.get("slice_margin", 0.0)
        self.aug = cfg["augmentation"] if train and cfg["augmentation"]["enable"] else None
        self.geo = (pd.read_csv(resolve(cfg["paths"]["series_index"], cfg))
                    .set_index("series")[["pixel_spacing_c", "pixel_spacing_r"]])

    def __len__(self):
        return len(self.df)

    def _load(self, study: str, series: str, rng) -> np.ndarray:
        with np.load(self.shards / f"{study}.npz") as z:
            if series not in z.files:
                return np.zeros((self.n_slices, self.size, self.size), np.uint8)
            arr = z[series]
            idx = pick_slices(arr.shape[0], self.n_slices, self.mode, rng, self.margin)
            vol = arr[idx].astype(np.float32)

        # native spacing varies 5x across scanners; rescale so a knee is the same size
        if self.mm_per_px:
            sp = float(self.geo["pixel_spacing_c"].get(series, self.mm_per_px) or self.mm_per_px)
            f = sp / self.mm_per_px
            if abs(f - 1) > 0.02:
                h, w = int(round(vol.shape[1] * f)), int(round(vol.shape[2] * f))
                vol = np.stack([cv2.resize(s, (w, h), interpolation=cv2.INTER_AREA)
                                for s in vol])

        vol = self._fit(vol)
        # shards are already windowed 0.1-99.9; re-window only if asked
        if self.window != "shard":
            lo_p, hi_p = self.cfg["input"]["window_pct"]
            axis = (1, 2) if self.window == "slice" else None
            lo = np.percentile(vol, lo_p, axis=axis, keepdims=axis is not None)
            hi = np.percentile(vol, hi_p, axis=axis, keepdims=axis is not None)
            vol = np.clip((vol - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255.0
        return vol

    def _fit(self, vol: np.ndarray) -> np.ndarray:
        """Centre crop or pad each slice to size x size."""
        s, (h, w) = self.size, vol.shape[1:]
        if h > s:
            t = (h - s) // 2; vol = vol[:, t:t + s]
        if w > s:
            l = (w - s) // 2; vol = vol[:, :, l:l + s]
        h, w = vol.shape[1:]
        if h < s or w < s:
            ph, pw = max(0, s - h), max(0, s - w)
            vol = np.pad(vol, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
        return vol

    def _augment(self, vol: np.ndarray, rng) -> np.ndarray:
        a, s = self.aug, self.size
        M = np.eye(2, 3, dtype=np.float32)
        touched = False
        if rng.random() < a["rotation"]["p"]:
            deg = rng.uniform(-a["rotation"]["deg"], a["rotation"]["deg"])
            M = cv2.getRotationMatrix2D((s / 2, s / 2), deg, 1.0).astype(np.float32); touched = True
        if rng.random() < a["scale"]["p"]:
            lo, hi = a["scale"]["range"]
            f = rng.uniform(lo, hi)
            M[:, :2] *= f; M[:, 2] += (1 - f) * s / 2; touched = True
        if rng.random() < a["translate"]["p"]:
            f = a["translate"]["frac"]
            M[0, 2] += rng.uniform(-f, f) * s; M[1, 2] += rng.uniform(-f, f) * s; touched = True
        if touched:
            vol = np.stack([cv2.warpAffine(sl, M, (s, s), flags=cv2.INTER_LINEAR) for sl in vol])
        if rng.random() < a["brightness"]["p"]:
            vol = vol + rng.uniform(-1, 1) * a["brightness"]["factor"] * 255
        if rng.random() < a["contrast"]["p"]:
            m = vol.mean()
            vol = (vol - m) * (1 + rng.uniform(-1, 1) * a["contrast"]["factor"]) + m
        if rng.random() < a["hflip"]["p"]:
            vol = vol[:, :, ::-1].copy()
        sd = a["slice_dropout"]
        if rng.random() < sd["p"]:
            k = max(1, int(len(vol) * sd["frac"]))
            vol[rng.choice(len(vol), k, replace=False)] = 0
        return np.clip(vol, 0, 255)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        study = row["StudyInstanceUID"]
        # val/audit get a per-study fixed seed so their sampling is reproducible
        rng = np.random.default_rng(None if self.train else (hash(study) & 0xFFFFFFFF))
        chosen = self.plan.get(study, {})

        vols = []
        for slot in self.slots:
            series = chosen.get(slot)
            v = (self._load(study, series, rng) if series is not None
                 else np.zeros((self.n_slices, self.size, self.size), np.float32))
            if self.aug is not None:
                v = self._augment(v, rng)
            vols.append(v)

        x = torch.from_numpy(np.stack(vols).astype(np.uint8))     # (slots, D, H, W) uint8
        y = torch.tensor([float(row[t]) for t in self.tasks], dtype=torch.float32)
        if self.cfg["target"]["mode"] == "hard":
            thr, ls = self.cfg["target"]["hard_threshold"], self.cfg["target"]["label_smoothing"]
            y = (y > thr).float() * (1 - ls) + ls / 2
        return x, y


def prepare_batch(x: torch.Tensor, cfg: dict, device) -> torch.Tensor:
    """uint8 (B, slots, D, H, W) -> normalized float (B*slots*D, C, H, W)."""
    # ship uint8 to the GPU (4x smaller transfer), normalize there
    b, s, d, h, w = x.shape
    x = x.to(device, non_blocking=True).float().div_(255.0)
    x = x.view(b * s * d, 1, h, w)
    c = cfg["input"]["channels"]
    if c > 1:
        x = x.expand(-1, c, -1, -1)
    mean = torch.tensor(cfg["input"]["norm_mean"][:c], device=device).view(1, c, 1, 1)
    std = torch.tensor(cfg["input"]["norm_std"][:c], device=device).view(1, c, 1, 1)
    return (x - mean) / std


def make_loaders(cfg: dict):
    idx = pd.read_csv(resolve(cfg["paths"]["series_index"], cfg))
    plan = plan_slots(idx, cfg)
    dl = cfg.get("dataloader", {})
    out = {}
    for split, key, train in [("train", "train_csv", True), ("val", "val_csv", False)]:
        df = pd.read_csv(resolve(cfg["paths"][key], cfg))
        ds = KneeShardDataset(df, cfg, plan, train)
        out[split] = DataLoader(
            ds, batch_size=cfg["training"]["batch_size"], shuffle=train,
            num_workers=dl.get("num_workers", 4),
            pin_memory=dl.get("pin_memory", True),
            persistent_workers=dl.get("persistent_workers", True) and dl.get("num_workers", 4) > 0,
            prefetch_factor=dl.get("prefetch_factor", 2) if dl.get("num_workers", 4) > 0 else None,
            drop_last=dl.get("drop_last", True) and train)
    if cfg["evaluation"]["audit"]["enable"]:
        df = pd.read_csv(resolve(cfg["evaluation"]["audit"]["csv"], cfg))
        df = df.rename(columns={f"official_{t}": t for t in cfg["tasks"]})
        out["audit"] = DataLoader(KneeShardDataset(df, cfg, plan, False),
                                  batch_size=cfg["training"]["batch_size"],
                                  shuffle=False, num_workers=dl.get("num_workers", 4))
    return out


# =============================================================================
# metrics / loss / optim
# =============================================================================
def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray, tasks: list, thr: float = 0.5) -> dict:
    from sklearn.metrics import roc_auc_score
    per = {}
    for i, t in enumerate(tasks):
        yt = (y_true[:, i] > thr).astype(int)   # soft targets need a threshold to score
        per[t] = roc_auc_score(yt, y_prob[:, i]) if yt.min() != yt.max() else float("nan")
    vals = [v for v in per.values() if not math.isnan(v)]
    return {"macro_auroc": float(np.mean(vals)) if vals else float("nan"), "per_task": per}


class FocalBCE(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__(); self.gamma = gamma; self.register_buffer("pw", pos_weight)

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pw, reduction="none")
        p = torch.sigmoid(logits)
        mod = (target * (1 - p) + (1 - target) * p) ** self.gamma
        return (mod * bce).mean()


def build_loss(cfg: dict, train_df: pd.DataFrame | None, device) -> nn.Module:
    lc = cfg["training"]["loss"]
    pw = None
    if lc.get("pos_weight") == "auto" and train_df is not None:
        prev = train_df[cfg["tasks"]].mean().values
        pw = torch.tensor((1 - prev) / np.maximum(prev, 1e-3), dtype=torch.float32, device=device)
    if lc["name"] == "focal":
        return FocalBCE(lc.get("focal_gamma", 2.0), pw).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def build_optimizer(model: nn.Module, cfg: dict):
    o = cfg["training"]["optimizer"]
    lr, mult = o["lr"], o.get("backbone_lr_mult", 1.0)
    head, backbone = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # fresh head learns at full lr, pretrained backbone slower
        (head if any(k in n for k in ("head", "classifier", "fc", "pool_attn")) else backbone).append(p)
    groups = [{"params": backbone, "lr": lr * mult}, {"params": head, "lr": lr}]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=o.get("weight_decay", 0.05))


def build_scheduler(opt, cfg: dict, total_steps: int):
    s = cfg["training"]["scheduler"]
    warm = max(1, int(total_steps * s.get("warmup_frac", 0.05)))
    lo = s.get("min_lr_mult", 0.01)

    def fn(step):
        if step < warm:
            return step / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# =============================================================================
# eval / train
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, loss_fn, cfg, device) -> dict:
    model.eval()
    P, Y, losses = [], [], []
    amp = cfg["training"]["amp"] and device.type == "cuda"
    for x, y in loader:
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            logits = model(prepare_batch(x, cfg, device))
            losses.append(loss_fn(logits, y).item())
        P.append(torch.sigmoid(logits.float()).cpu().numpy()); Y.append(y.cpu().numpy())
    m = macro_auroc(np.concatenate(Y), np.concatenate(P), cfg["tasks"])
    m["loss"] = float(np.mean(losses))
    return m


def run_experiment(cfg: dict, model: nn.Module, out_dir: Path, persist_fn=None, log_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["reproducibility"]["seed"], cfg["reproducibility"]["deterministic"])

    out = Path(out_dir) / cfg["output"]["run_dir"]
    ckpt_dir = out / cfg["output"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loaders = make_loaders(cfg)
    train_df = pd.read_csv(resolve(cfg["paths"]["train_csv"], cfg))
    model = model.to(device)
    if cfg["training"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    loss_fn = build_loss(cfg, train_df, device)
    opt = build_optimizer(model, cfg)
    epochs = cfg["training"]["epochs"]
    accum = cfg["training"].get("accum_steps", 1)
    total = max(1, len(loaders["train"]) // accum) * epochs
    sched = build_scheduler(opt, cfg, total)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda")

    es = cfg["training"]["early_stopping"]
    best, bad, gstep = -math.inf, 0, 0
    train_rows, val_rows = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for it, (x, y) in enumerate(loaders["train"], 1):
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss = loss_fn(model(prepare_batch(x, cfg, device)), y) / accum
            scaler.scale(loss).backward()
            if it % accum == 0:
                if cfg["training"]["grad_clip"]:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                gstep += 1
                if gstep % cfg["output"]["console_log_every"] == 0:
                    lr = sched.get_last_lr()[-1]
                    msg = (f"e{epoch} step {gstep}/{total} loss {loss.item()*accum:.4f} "
                           f"lr {lr:.2e} {fmt(time.time()-t0)}")
                    print(msg, flush=True)
                    train_rows.append({"epoch": epoch, "step": gstep,
                                       "loss": loss.item() * accum, "lr": lr})
                    if log_fn:
                        log_fn({"train/loss": loss.item() * accum, "train/lr": lr}, gstep)

        # ---- validation -------------------------------------------------
        vm = evaluate(model, loaders["val"], loss_fn, cfg, device)
        row = {"epoch": epoch, "step": gstep, "val_loss": vm["loss"],
               "val_macro_auroc": vm["macro_auroc"]}
        row |= {f"val_{k}": v for k, v in vm["per_task"].items()}
        print(f"\n[val] epoch {epoch}  loss {vm['loss']:.4f}  "
              f"macro AUROC {vm['macro_auroc']:.4f}")
        for k, v in vm["per_task"].items():
            print(f"       {k:18s} {v:.3f}")

        # the 58 panel-labeled studies: reported, never used for best.pt
        if "audit" in loaders:
            am = evaluate(model, loaders["audit"], loss_fn, cfg, device)
            row["audit_macro_auroc"] = am["macro_auroc"]
            print(f"[audit 58] macro AUROC {am['macro_auroc']:.4f}  (report only)")

        val_rows.append(row)
        if log_fn:
            log_fn({f"val/{k}": v for k, v in row.items() if isinstance(v, float)}, gstep)

        pd.DataFrame(train_rows).to_csv(out / cfg["output"]["train_log_csv"], index=False)
        pd.DataFrame(val_rows).to_csv(out / cfg["output"]["val_log_csv"], index=False)

        score = row.get(es["monitor"], vm["macro_auroc"])
        improved = score > best if es["mode"] == "max" else score < best
        if improved:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "score": score}, ckpt_dir / cfg["output"]["best_name"])
            print(f"       new best {es['monitor']} = {score:.4f}")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                   ckpt_dir / f"last.pt")

        (out / cfg["output"]["summary_json"]).write_text(json.dumps({
            "experiment": cfg["experiment"]["name"], "epoch": epoch,
            "best": best, "monitor": es["monitor"],
            "last": {k: v for k, v in row.items() if isinstance(v, float)},
            "elapsed": fmt(time.time() - t0)}, indent=2))
        if persist_fn:
            persist_fn()
        if es["enable"] and bad >= es["patience"]:
            print(f"early stop: no improvement for {bad} epochs")
            break

    print(f"\ndone in {fmt(time.time()-t0)}   best {es['monitor']} = {best:.4f}")
    return best


def fmt(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# =============================================================================
# modal glue
# =============================================================================
def modal_image():
    import modal
    return (modal.Image.debian_slim(python_version="3.11")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install("torch", "torchvision", "timm", "numpy", "pandas",
                         "opencv-python-headless", "scikit-learn", "pyyaml", "wandb")
            .add_local_dir(str(Path(__file__).parent), "/root/runs")
            .add_local_dir(str(Path(__file__).parent.parent / "data"), "/root/data",
                           ignore=["shards", "raw", "backups", "*.jsonl"]))


def modal_resources(cfg: dict) -> dict:
    return {"gpu": cfg["modal"]["gpu"],
            "timeout": int(cfg["modal"].get("timeout_hours", 8) * 3600)}


def remote_cfg(cfg: dict) -> dict:
    """Repoint paths at the mounted volumes."""
    r = json.loads(json.dumps(cfg))
    r["paths"]["data_dir"] = "/root/data"
    r["paths"]["shards_dir"] = cfg["modal"]["shards_mount"]
    return r
