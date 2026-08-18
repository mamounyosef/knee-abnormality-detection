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

import csv
import io
import json
import math
import os
import random
import sys
import time
from datetime import datetime
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

# box-drawing glyphs need utf-8; Windows consoles default to cp1252
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# pretty printing
# =============================================================================
W = 78


def _rule(ch: str = "\u2500"):
    print(ch * W)


def _banner(title: str, emoji: str = ""):
    """Big section header."""
    print()
    print("\u2554" + "\u2550" * (W - 2) + "\u2557")
    label = f" {emoji} {title} ".ljust(W - 2)
    print("\u2551" + label + "\u2551")
    print("\u255A" + "\u2550" * (W - 2) + "\u255D")


def _kv(title: str, pairs, emoji: str = ""):
    """Boxed key/value block. `pairs` may contain (k, v) or None for a separator."""
    head = f" {emoji} {title} ".rstrip()
    print("\u250C\u2500" + head + "\u2500" * max(0, W - len(head) - 3) + "\u2510")
    for row in pairs:
        if row is None:
            print("\u251C" + "\u2500" * (W - 2) + "\u2524")
            continue
        k, v = row
        text = f" {k:<22}\u2502 {v}"
        if len(text) > W - 2:
            text = text[:W - 5] + "..."
        print("\u2502" + text.ljust(W - 2) + "\u2502")
    print("\u2514" + "\u2500" * (W - 2) + "\u2518")


def _table(title: str, headers, rows, emoji: str = ""):
    """Boxed table. Title sits above the box so it never overflows the border."""
    cols = [[str(h)] + [str(r[i]) for r in rows] for i, h in enumerate(headers)]
    wid = [max(len(c) for c in col) for col in cols]
    print(f"  {emoji} {title}".rstrip())
    print("┌" + "┬".join("─" * (w + 2) for w in wid) + "┐")
    print("│ " + " │ ".join(str(h).ljust(w) for h, w in zip(headers, wid)) + " │")
    print("├" + "┼".join("─" * (w + 2) for w in wid) + "┤")
    for r in rows:
        print("│ " + " │ ".join(str(c).ljust(w) for c, w in zip(r, wid)) + " │")
    print("└" + "┴".join("─" * (w + 2) for w in wid) + "┘")


# =============================================================================
# config
# =============================================================================
def _deep_merge(base: dict, over: dict) -> dict:
    # recursive dict merge; `over` wins on scalars, dicts get merged key by key
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(experiment_dir, config_name: str = "config.yaml", verbose: bool = True) -> dict:
    """shared_config.yaml + the run's config.yaml, run wins."""
    exp_dir = Path(experiment_dir)
    shared = yaml.safe_load((exp_dir.parent / "shared_config.yaml").read_text(encoding="utf-8"))
    own = yaml.safe_load((exp_dir / config_name).read_text(encoding="utf-8")) or {}
    cfg = _deep_merge(shared, own)
    cfg.setdefault("experiment", {}).setdefault("name", exp_dir.name)   # folder name is the default run name
    if verbose:
        print_config(cfg)
    return cfg


def print_config(cfg: dict):
    """Everything this run is doing, as boxed tables, then the raw merged YAML."""
    i, t, o = cfg["input"], cfg["training"], cfg["output"]
    m, dl, aug = cfg.get("modal", {}), cfg.get("dataloader", {}), cfg["augmentation"]
    mo = cfg.get("model", {})
    n_img = len(i["slots"]) * i["n_slices"]
    field = i["size"] * (i["mm_per_px"] or 0)

    _banner("CONFIGURATION", "\u2699\uFE0F")

    _kv("run", [
        ("experiment", cfg["experiment"]["name"]),
        ("model", f"{mo.get('name', '?')}  (pretrained={mo.get('pretrained')})"),
        ("pooling", mo.get("pool")),
        ("findings", f"{len(cfg['tasks'])}  {', '.join(cfg['tasks'][:4])} ..."),
        ("resume", cfg.get("resume")),
        ("seed", f"{cfg['reproducibility']['seed']}  "
                 f"deterministic={cfg['reproducibility']['deterministic']}"),
    ], "\U0001F3AF")

    _kv("paths", [
        ("data_dir", cfg["paths"]["data_dir"]),
        ("shards_dir", cfg["paths"]["shards_dir"]),
        ("train csv", cfg["paths"]["train_csv"]),
        ("val csv", cfg["paths"]["val_csv"]),
        ("audit csv", cfg["paths"]["audit_csv"]),
    ], "\U0001F4C1")

    _table("input slots", ["#", "slot", "plane", "setting"],
           [[n + 1, sl, PLANE_OF.get(sl, "?"),
             "fluid (bright water)" if sl.endswith("fluid") else "plain (anatomy)"]
            for n, sl in enumerate(i["slots"])], "\U0001F5BC\uFE0F")

    _kv("geometry & intensity", [
        ("images per study", f"{len(i['slots'])} slots x {i['n_slices']} slices = {n_img}"),
        ("resolution", f"{i['size']} x {i['size']} px @ {i['mm_per_px']} mm/px "
                       f"= {field:.0f} mm field"),
        ("slice sampling", f"train={i['train_slice_sampling']}  eval={i['slice_sampling']}  "
                           f"margin={i.get('slice_margin', 0)}"),
        ("missing slots", f"fallback_same_plane={i['fallback_same_plane']}"),
        ("duplicate rule", i["duplicate_rule"]),
        None,
        ("window", f"{i['window']}  pct={i['window_pct']}"),
        ("channels", i["channels"]),
        ("norm mean/std", f"{i['norm_mean']} / {i['norm_std']}"),
    ], "\U0001F4D0")

    if aug["enable"]:
        rows = [[k, v.get("p", "-"),
                 ", ".join(f"{kk}={vv}" for kk, vv in v.items() if kk != "p") or "-"]
                for k, v in aug.items() if isinstance(v, dict)]
        _table("augmentation (train split only)", ["transform", "p", "params"], rows,
               "\U0001F3B2")
    else:
        _kv("augmentation", [("enabled", False)], "\U0001F3B2")

    eff = t.get("batch_size", 0) * t.get("accum_steps", 1)
    _kv("training", [
        ("epochs", t.get("epochs")),
        ("batch", f"{t.get('batch_size')} studies x accum {t.get('accum_steps', 1)} "
                  f"= {eff} studies ({eff * n_img} images) per update"),
        ("optimizer", f"{t['optimizer']['name']}  lr={t['optimizer'].get('lr'):.2e}  "
                      f"wd={t['optimizer'].get('weight_decay')}  "
                      f"backbone x{t['optimizer'].get('backbone_lr_mult')}"),
        ("scheduler", f"{t['scheduler']['name']}  warmup={t['scheduler']['warmup_frac']:.0%}  "
                      f"floor={t['scheduler']['min_lr_mult']:.0%} of base"),
        ("loss", f"{t['loss']['name']}  pos_weight={t['loss']['pos_weight']}"),
        ("target", f"{cfg['target']['mode']}  (LLM probabilities, 0.895 macro AUROC vs the 58)"),
        ("precision", f"amp={t['amp']}  channels_last={t['channels_last']}  "
                      f"grad_clip={t['grad_clip']}"),
        ("early stopping", f"{t['early_stopping']['monitor']} "
                           f"({t['early_stopping']['mode']}), "
                           f"patience={t['early_stopping']['patience']}"),
    ], "\U0001F3CB\uFE0F")

    _kv("dataloader", [
        ("num_workers", dl.get("num_workers")),
        ("prefetch/persist", f"{dl.get('prefetch_factor')} / {dl.get('persistent_workers')}"),
        ("pin_memory", dl.get("pin_memory")),
        ("drop_last", dl.get("drop_last")),
    ], "\U0001F69A")

    _kv("evaluation & output", [
        ("eval cadence", f"every {cfg['evaluation']['eval_every_epochs']} epoch(s), "
                         f"every_steps={cfg['evaluation']['eval_every_steps']}"),
        ("audit set", f"enable={cfg['evaluation']['audit']['enable']}  "
                      f"(58 studies, report only)"),
        ("run_dir", o["run_dir"]),
        ("checkpoints", f"{o['checkpoints_dir']}/  best={o['best_name']}  "
                        f"keep_last_n={o['keep_last_n']}"),
        ("logs", f"{o['train_log_csv']}, {o['val_log_csv']}, {o['summary_json']}"),
        ("console", f"log_every={o['console_log_every']}  "
                    f"capture={o.get('console_capture_dir')}"),
    ], "\U0001F4CB")

    if m:
        _kv("modal (cloud)", [
            ("gpu", m.get("gpu")),
            ("timeout", f"{m.get('timeout_hours')} h"),
            ("shards volume", f"{m.get('shards_volume')} -> {m.get('shards_mount')}"),
            ("runs volume", f"{m.get('runs_volume')} -> {m.get('runs_mount')}"),
        ], "\u2601\uFE0F")

    print()
    print("\u2500\u2500 raw merged config (verbatim) " + "\u2500" * (W - 31))
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True).rstrip())
    _rule()


def num_output_logits(cfg: dict) -> int:
    return len(cfg["tasks"])          # one logit per label, 12


def resolve(name: str, cfg: dict) -> Path:
    """Bare filename -> data_dir/<name>; absolute paths pass through."""
    p = Path(name)
    return p if p.is_absolute() else Path(cfg["paths"]["data_dir"]) / name


def set_seed(seed: int, deterministic: bool = False):
    # deterministic=False keeps cudnn benchmark on, which is much faster
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# =============================================================================
# slot planning
# study = one knee exam, series = one scan in it, slot = the role we want filled,
# plane = slicing direction: sagittal (side), coronal (front), axial (top-down)
# =============================================================================
PLANE_OF = {"sag_plain": "sagittal", "sag_fluid": "sagittal",
            "cor_plain": "coronal", "cor_fluid": "coronal",
            "ax_plain": "axial", "ax_fluid": "axial"}


def plan_slots(index: pd.DataFrame, cfg: dict) -> dict[str, dict[str, str]]:
    """Pick one series per slot for every study -> {study: {slot: series}}.
    Duplicate rule breaks ties; fallback accepts any series of the right plane."""
    slots, rule = cfg["input"]["slots"], cfg["input"]["duplicate_rule"]
    fallback = cfg["input"]["fallback_same_plane"]      # accept any series of the right plane

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
            cand = g[g.slot == slot]                     # exact slot match first
            if cand.empty and fallback:
                cand = g[g.plane == PLANE_OF[slot]]      # same plane, either setting
            if cand.empty:
                continue                                 # slot stays empty -> zeros at load time
            chosen[slot] = cand.sort_values(sort_by, ascending=asc).iloc[0]["series"]
        plan[study] = chosen
    return plan


def pick_slices(n_available: int, n_want: int, mode: str, rng,
                margin: float = 0.0) -> np.ndarray:
    """Pick n_want slice indices from a stack of n_available real slices.
    Never blends slices into fake ones; repeats indices if the stack is short."""
    # the first/last slices of a stack are usually past the edge of the knee
    lo = int(n_available * margin)
    hi = n_available - lo
    if hi - lo >= n_want:
        edges = np.linspace(lo, hi, n_want + 1)          # n_want equal bins across the stack
        if mode == "random":
            # one random slice per bin: same coverage, different pick each epoch
            lo, hi = np.floor(edges[:-1]).astype(int), np.ceil(edges[1:]).astype(int)
            return np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(lo, hi)])
        return np.clip(((edges[:-1] + edges[1:]) / 2).astype(int), 0, n_available - 1)  # bin centres
    # too few slices: repeat rather than interpolate (17.8% of series at n=24)
    idx = np.linspace(0, n_available - 1, n_want)
    return np.round(idx).astype(int)


_LAST_PLAN: dict = {}          # slot plan of the most recent make_loaders call


# =============================================================================
# dataset
# =============================================================================
class KneeShardDataset(Dataset):
    """One item = one study: 4 slot volumes of n_slices each, plus its 12 labels."""

    def __init__(self, df: pd.DataFrame, cfg: dict, plan: dict, train: bool):
        self.df = df.reset_index(drop=True)
        self.cfg, self.plan, self.train = cfg, plan, train
        self.tasks = cfg["tasks"]
        self.shards = Path(cfg["paths"]["shards_dir"])
        i = cfg["input"]
        self.slots, self.n_slices, self.size = i["slots"], i["n_slices"], i["size"]
        self.mm_per_px, self.window = i["mm_per_px"], i["window"]
        # random slice picks while training, fixed even spacing while evaluating
        self.mode = i["train_slice_sampling"] if train else i["slice_sampling"]
        self.margin = i.get("slice_margin", 0.0)
        self.aug = cfg["augmentation"] if train and cfg["augmentation"]["enable"] else None
        # per-series mm-per-pixel, needed to rescale every scan to one physical scale
        self.geo = (pd.read_csv(resolve(cfg["paths"]["series_index"], cfg))
                    .set_index("series")[["pixel_spacing_c", "pixel_spacing_r"]])

    def __len__(self):
        return len(self.df)

    def _load(self, study: str, series: str, rng) -> np.ndarray:
        """One series -> (n_slices, size, size) float array, resampled and windowed."""
        with np.load(self.shards / f"{study}.npz") as z:
            if series not in z.files:
                return np.zeros((self.n_slices, self.size, self.size), np.uint8)   # missing series
            arr = z[series]                                  # (all_slices, H, W) uint8, native res
            idx = pick_slices(arr.shape[0], self.n_slices, self.mode, rng, self.margin)
            vol = arr[idx].astype(np.float32)

        # native spacing varies 5x across scanners; rescale so a knee is the same size
        if self.mm_per_px:
            sp = float(self.geo["pixel_spacing_c"].get(series, self.mm_per_px) or self.mm_per_px)
            f = sp / self.mm_per_px
            if abs(f - 1) > 0.02:                            # skip resize when already within 2%
                h, w = int(round(vol.shape[1] * f)), int(round(vol.shape[2] * f))
                vol = np.stack([cv2.resize(s, (w, h), interpolation=cv2.INTER_AREA)
                                for s in vol])

        vol = self._fit(vol)                                 # force every slice to size x size
        # shards are already windowed 0.1-99.9; re-window only if asked
        if self.window != "shard":
            lo_p, hi_p = self.cfg["input"]["window_pct"]
            # "volume" = one contrast window for the whole stack, "slice" = per slice
            axis = (1, 2) if self.window == "slice" else None
            lo = np.percentile(vol, lo_p, axis=axis, keepdims=axis is not None)
            hi = np.percentile(vol, hi_p, axis=axis, keepdims=axis is not None)
            # stretch the [lo, hi] intensity range back over the full 0-255 range
            vol = np.clip((vol - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255.0
        return vol

    def _fit(self, vol: np.ndarray) -> np.ndarray:
        """Centre crop or pad each slice to size x size."""
        # s = target side in px (cfg input.size); h, w = this slice's current height/width
        s, (h, w) = self.size, vol.shape[1:]      # vol is (n_slices, h, w), so skip axis 0
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
        """Random train-time distortions so the model doesn't memorize exact pixels."""
        a, s = self.aug, self.size
        # one 2x3 affine matrix collects rotation + scale + shift, applied in a single warp
        M = np.eye(2, 3, dtype=np.float32)
        touched = False
        if rng.random() < a["rotation"]["p"]:
            deg = rng.uniform(-a["rotation"]["deg"], a["rotation"]["deg"])
            M = cv2.getRotationMatrix2D((s / 2, s / 2), deg, 1.0).astype(np.float32); touched = True
        if rng.random() < a["scale"]["p"]:
            lo, hi = a["scale"]["range"]
            f = rng.uniform(lo, hi)
            M[:, :2] *= f; M[:, 2] += (1 - f) * s / 2; touched = True   # zoom about the centre
        if rng.random() < a["translate"]["p"]:
            f = a["translate"]["frac"]
            M[0, 2] += rng.uniform(-f, f) * s; M[1, 2] += rng.uniform(-f, f) * s; touched = True
        if touched:
            # same matrix for every slice, so the stack stays geometrically consistent
            vol = np.stack([cv2.warpAffine(sl, M, (s, s), flags=cv2.INTER_LINEAR) for sl in vol])
        if rng.random() < a["brightness"]["p"]:
            vol = vol + rng.uniform(-1, 1) * a["brightness"]["factor"] * 255
        if rng.random() < a["contrast"]["p"]:
            m = vol.mean()
            vol = (vol - m) * (1 + rng.uniform(-1, 1) * a["contrast"]["factor"]) + m
        if rng.random() < a["hflip"]["p"]:
            vol = vol[:, :, ::-1].copy()
        sd = a["slice_dropout"]                       # blank a few slices: mimics gaps/artifacts
        if rng.random() < sd["p"]:
            k = max(1, int(len(vol) * sd["frac"]))
            vol[rng.choice(len(vol), k, replace=False)] = 0
        return np.clip(vol, 0, 255)

    def __getitem__(self, i: int):
        """Build one study's tensor: stack the 4 slot volumes, attach its 12 labels."""
        row = self.df.iloc[i]
        study = row["StudyInstanceUID"]                 # one study = one patient's knee exam
        # val/audit get a per-study fixed seed so their sampling is reproducible
        rng = np.random.default_rng(None if self.train else (hash(study) & 0xFFFFFFFF))
        chosen = self.plan.get(study, {})               # {slot: series} picked by plan_slots

        vols = []
        for slot in self.slots:
            series = chosen.get(slot)                   # None when this study lacks that view
            v = (self._load(study, series, rng) if series is not None
                 else np.zeros((self.n_slices, self.size, self.size), np.float32))
            if self.aug is not None:
                v = self._augment(v, rng)
            vols.append(v)

        x = torch.from_numpy(np.stack(vols).astype(np.uint8))     # (slots, D, H, W) uint8
        y = torch.tensor([float(row[t]) for t in self.tasks], dtype=torch.float32)   # 12 probs in [0,1]
        # "hard" mode turns the soft label probabilities into crisp 0/1 targets
        if self.cfg["target"]["mode"] == "hard":
            thr, ls = self.cfg["target"]["hard_threshold"], self.cfg["target"]["label_smoothing"]
            y = (y > thr).float() * (1 - ls) + ls / 2
        return x, y


def prepare_batch(x: torch.Tensor, cfg: dict, device) -> torch.Tensor:
    """uint8 (B, slots, D, H, W) -> normalized float (B*slots*D, C, H, W)."""
    # ship uint8 to the GPU (4x smaller transfer), normalize there
    b, s, d, h, w = x.shape
    x = x.to(device, non_blocking=True).float().div_(255.0)
    x = x.view(b * s * d, 1, h, w)                 # every slice becomes its own 2D image
    c = cfg["input"]["channels"]
    if c > 1:
        x = x.expand(-1, c, -1, -1)                # grey copied to 3 channels for ImageNet backbones
    mean = torch.tensor(cfg["input"]["norm_mean"][:c], device=device).view(1, c, 1, 1)
    std = torch.tensor(cfg["input"]["norm_std"][:c], device=device).view(1, c, 1, 1)
    return (x - mean) / std


def make_loaders(cfg: dict):
    """Build the train / val (/ audit) DataLoaders, all sharing one slot plan."""
    idx = pd.read_csv(resolve(cfg["paths"]["series_index"], cfg))   # one row per series
    plan = plan_slots(idx, cfg)
    _LAST_PLAN.clear(); _LAST_PLAN.update(plan)   # so the data summary can report coverage
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
    # audit = the 58 studies with real panel labels, scored but never used to pick best.pt
    if cfg["evaluation"]["audit"]["enable"]:
        df = pd.read_csv(resolve(cfg["evaluation"]["audit"]["csv"], cfg))
        df = df.rename(columns={f"official_{t}": t for t in cfg["tasks"]})   # match task column names
        out["audit"] = DataLoader(KneeShardDataset(df, cfg, plan, False),
                                  batch_size=cfg["training"]["batch_size"],
                                  shuffle=False, num_workers=dl.get("num_workers", 4))
    return out


# =============================================================================
# logging
# =============================================================================
class _Tee:
    """Write to the console and the capture file at once, flushing every time."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for st in self._streams:
            try:
                st.write(data); st.flush()
            except Exception:
                pass                      # a broken capture file must never kill a run

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


class CSVLogger:
    """Append-only CSV, header on first row, flushed per row so a crash keeps the log."""

    def __init__(self, path: Path):
        self.path = Path(path); self._f = None; self._w = None

    def log(self, row: dict):
        try:
            if self._f is None:
                new = (not self.path.exists()) or self.path.stat().st_size == 0
                self._f = open(self.path, "a", newline="", encoding="utf-8")
                self._w = csv.DictWriter(self._f, fieldnames=list(row.keys()))
                if new:
                    self._w.writeheader()
            self._w.writerow(row); self._f.flush()
        except Exception as e:                # logging must never crash training
            print(f"  [warn] CSV log failed: {type(e).__name__}: {e}")
            self._f, self._w = None, None

    def close(self):
        try:
            if self._f:
                self._f.close()
        except Exception:
            pass


# =============================================================================
# metrics / loss / optim
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, tasks: list,
                    thr: float = 0.5) -> dict:
    """Per-task AUROC + AUPRC and the operating-point stats at `thr`."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    per = {}
    for i, t in enumerate(tasks):
        yt = (y_true[:, i] > thr).astype(int)          # soft targets need a threshold to score
        yp = y_prob[:, i]
        one_class = yt.min() == yt.max()
        per[t] = {
            "auroc": float("nan") if one_class else roc_auc_score(yt, yp),
            "auprc": float("nan") if one_class else average_precision_score(yt, yp),
            "pos": int(yt.sum()),
        }
    au = [v["auroc"] for v in per.values() if not math.isnan(v["auroc"])]
    ap = [v["auprc"] for v in per.values() if not math.isnan(v["auprc"])]

    yb, pb = (y_true > thr).astype(int), (y_prob > thr).astype(int)
    tp = ((yb == 1) & (pb == 1)).sum(); fp = ((yb == 0) & (pb == 1)).sum()
    fn = ((yb == 1) & (pb == 0)).sum(); tn = ((yb == 0) & (pb == 0)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {
        "macro_auroc": float(np.mean(au)) if au else float("nan"),
        "macro_auprc": float(np.mean(ap)) if ap else float("nan"),
        "precision": float(prec), "recall": float(rec),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
        "specificity": float(tn / max(tn + fp, 1)),
        "per_task": per,
    }


class FocalBCE(nn.Module):
    """Binary cross-entropy that down-weights easy examples (gamma) to focus on hard ones."""

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__(); self.gamma = gamma; self.register_buffer("pw", pos_weight)

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pw, reduction="none")
        p = torch.sigmoid(logits)
        mod = (target * (1 - p) + (1 - target) * p) ** self.gamma   # small when the guess is right
        return (mod * bce).mean()


def build_loss(cfg: dict, train_df: pd.DataFrame | None, device) -> nn.Module:
    """Plain or focal binary cross-entropy, optionally re-weighted for rare labels."""
    lc = cfg["training"]["loss"]
    pw = None
    if lc.get("pos_weight") == "auto" and train_df is not None:
        # rare labels get a bigger weight so they aren't drowned out by the common ones
        prev = train_df[cfg["tasks"]].mean().values
        pw = torch.tensor((1 - prev) / np.maximum(prev, 1e-3), dtype=torch.float32, device=device)
    if lc["name"] == "focal":
        return FocalBCE(lc.get("focal_gamma", 2.0), pw).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def build_optimizer(model: nn.Module, cfg: dict):
    """AdamW with two learning rates: full for the new head, slower for the backbone."""
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
    """Learning-rate schedule: ramp up over the first few %, then cosine-decay to min_lr_mult."""
    s = cfg["training"]["scheduler"]
    warm = max(1, int(total_steps * s.get("warmup_frac", 0.05)))
    lo = s.get("min_lr_mult", 0.01)

    def fn(step):                                    # returns a multiplier on the base lr
        if step < warm:
            return step / warm                       # linear warm-up from 0
        prog = (step - warm) / max(1, total_steps - warm)
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# =============================================================================
# eval / train
# =============================================================================
@torch.no_grad()                                   # no gradients: eval only, saves memory
def evaluate(model, loader, loss_fn, cfg, device) -> dict:
    """Run the whole loader and return macro AUROC, per-label AUROC and mean loss."""
    model.eval()
    P, Y, losses = [], [], []                      # predictions, targets, per-batch losses
    amp = cfg["training"]["amp"] and device.type == "cuda"
    for x, y in loader:
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            logits = model(prepare_batch(x, cfg, device))
            losses.append(loss_fn(logits, y).item())
        # sigmoid turns raw logits into 0-1 probabilities
        P.append(torch.sigmoid(logits.float()).cpu().numpy()); Y.append(y.cpu().numpy())
    m = compute_metrics(np.concatenate(Y), np.concatenate(P), cfg["tasks"])
    m["loss"] = float(np.mean(losses))
    m["n"] = int(np.concatenate(Y).shape[0])
    return m


def print_environment(cfg: dict):
    """Hardware and software this run is actually executing on."""
    import platform

    _banner("ENVIRONMENT", "\U0001F5A5\uFE0F")

    rows = [("host", platform.node()),
            ("platform", f"{platform.system()} {platform.release()}"),
            ("python", platform.python_version()),
            ("torch", torch.__version__),
            ("cuda (torch)", torch.version.cuda or "cpu build")]
    try:
        import timm
        rows.append(("timm", timm.__version__))
    except Exception:
        pass
    rows.append(("numpy / pandas", f"{np.__version__} / {pd.__version__}"))
    if cv2 is not None:
        rows.append(("opencv", cv2.__version__))
    _kv("software", rows, "\U0001F4E6")

    # ---- CPU / RAM ---------------------------------------------------------
    cpu_rows = [("logical cores", os.cpu_count())]
    try:
        import psutil
        vm = psutil.virtual_memory()
        cpu_rows += [("physical cores", psutil.cpu_count(logical=False)),
                     ("RAM total", f"{vm.total / 1024**3:.1f} GiB"),
                     ("RAM available", f"{vm.available / 1024**3:.1f} GiB")]
    except ImportError:
        cpu_rows.append(("RAM", "install psutil for memory details"))
    try:                                   # the cgroup limit is what Modal actually gives us
        with open("/sys/fs/cgroup/memory.max") as fh:
            v = fh.read().strip()
            if v.isdigit():
                cpu_rows.append(("container RAM limit", f"{int(v) / 1024**3:.1f} GiB"))
    except Exception:
        pass
    _kv("cpu / memory", cpu_rows, "\U0001F9EE")

    # ---- GPU ---------------------------------------------------------------
    if torch.cuda.is_available():
        rows = []
        for i in range(torch.cuda.device_count()):
            d = torch.cuda.get_device_properties(i)
            rows.append([i, d.name, f"{d.total_memory / 1024**3:.1f} GiB",
                         f"{d.major}.{d.minor}", d.multi_processor_count])
        _table("gpu", ["#", "device", "memory", "capability", "SMs"], rows,
               "\U0001F3AE")
        print(f"  cudnn benchmark={torch.backends.cudnn.benchmark}  "
              f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32}  "
              f"tf32_cudnn={torch.backends.cudnn.allow_tf32}")
    else:
        _kv("gpu", [("available", "NO - running on CPU")], "\U0001F3AE")


def print_data_summary(loaders: dict, cfg: dict, plan: dict | None = None):
    """What actually got loaded: split sizes, batches, label balance, slot coverage."""
    _banner("DATA", "\U0001F4E6")

    rows = []
    for name in ("train", "val", "audit"):
        if name not in loaders:
            continue
        ld = loaders[name]
        n = len(ld.dataset)
        rows.append([name, f"{n:,}", f"{len(ld):,}",
                     f"{n * len(cfg['input']['slots']) * cfg['input']['n_slices']:,}",
                     "soft LLM" if name != "audit" else "panel 0/1"])
    _table("splits", ["split", "studies", "batches", "images", "labels"], rows,
           "\U0001F4CA")

    # label balance, so a collapsed prediction is obvious later
    rows = []
    for t in cfg["tasks"]:
        cells = [t]
        for name in ("train", "val"):
            if name in loaders:
                col = loaders[name].dataset.df[t].astype(float)
                cells += [f"{col.mean():.3f}", f"{(col > 0.5).mean():.1%}"]
        if "audit" in loaders:
            cells.append(f"{loaders['audit'].dataset.df[t].astype(float).mean():.1%}")
        rows.append(cells)
    heads = ["finding", "train mean p", "train >0.5", "val mean p", "val >0.5"]
    if "audit" in loaders:
        heads.append("audit pos")
    _table("label balance", heads, rows, "\u2696\uFE0F")

    if plan:
        filled = {}
        for slots in plan.values():
            for sl in slots:
                filled[sl] = filled.get(sl, 0) + 1
        n_studies = len(plan)
        _table("slot coverage (after fallback)", ["slot", "studies", "%"],
               [[sl, filled.get(sl, 0), f"{filled.get(sl, 0) / n_studies:.1%}"]
                for sl in cfg["input"]["slots"]], "\U0001F9E9")


def run_experiment(cfg: dict, model: nn.Module, out_dir: Path, persist_fn=None, log_fn=None):
    """Public entry: tee the whole console into results/train_config/<timestamp>.txt,
    then train. Streams are restored even if the run crashes, traceback included."""
    out_dir = Path(out_dir)
    cap_dir = (out_dir / cfg["output"]["run_dir"]
               / cfg["output"].get("console_capture_dir", "train_config"))
    cap_dir.mkdir(parents=True, exist_ok=True)
    cap_path = cap_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    try:
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    orig_out, orig_err = sys.stdout, sys.stderr
    log_f = open(cap_path, "w", encoding="utf-8")
    sys.stdout, sys.stderr = _Tee(orig_out, log_f), _Tee(orig_err, log_f)
    print(f"[capture] full console of this run -> {cap_path}")
    print(f"[capture] command: {' '.join(sys.argv)}")
    print(f"[capture] started {datetime.now():%Y-%m-%d %H:%M:%S}")
    try:
        return _run_experiment(cfg, model, out_dir, persist_fn, log_fn)
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_f.close()
        print(f"[capture] saved -> {cap_path}")


def _print_eval(tag: str, epoch: int, epochs: int, gstep: int, m: dict,
                elapsed: float, eta: float, tasks: list, emoji: str = "\U0001F50E"):
    """One validation block: headline metrics, then every finding."""
    print("-" * 70)
    print(f"  {emoji} [{tag}] epoch={epoch}/{epochs} step={gstep}  n={m['n']}  "
          f"loss={m['loss']:.4f}  elapsed={fmt(elapsed)}  ETA={fmt(eta)}")
    print(f"     \U0001F4CA macro AUROC={m['macro_auroc']:.4f}  AUPRC={m['macro_auprc']:.4f}  "
          f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
          f"Spec={m['specificity']:.3f}")
    for t in tasks:
        v = m["per_task"][t]
        print(f"        {t:18s} AUROC={v['auroc']:.4f}  AUPRC={v['auprc']:.4f}  "
              f"(pos={v['pos']})")


def _run_experiment(cfg: dict, model: nn.Module, out_dir: Path, persist_fn=None, log_fn=None):
    """Full training run: loaders, loss, optimizer, epoch loop, eval, checkpoints, logs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg["training"]["epochs"]
    print("#" * 70)
    print(f"# \U0001F680 run_experiment: {cfg['experiment']['name']}")
    print(f"# \U0001F5A5\uFE0F  device={device}  epochs={epochs}")
    print("#" * 70)
    set_seed(cfg["reproducibility"]["seed"], cfg["reproducibility"]["deterministic"])
    print(f"[set_seed] seed={cfg['reproducibility']['seed']}  "
          f"deterministic={cfg['reproducibility']['deterministic']}")
    print_environment(cfg)
    print_config(cfg)

    out = Path(out_dir) / cfg["output"]["run_dir"]
    ckpt_dir = out / cfg["output"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paths] results -> {out}")

    loaders = make_loaders(cfg)
    print_data_summary(loaders, cfg, _LAST_PLAN)
    train_df = pd.read_csv(resolve(cfg["paths"]["train_csv"], cfg))

    model = model.to(device)
    if cfg["training"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    tot_p = sum(q.numel() for q in model.parameters())
    tr_p = sum(q.numel() for q in model.parameters() if q.requires_grad)
    _banner("MODEL", "\U0001F9E0")
    _kv("architecture", [
        ("backbone", cfg.get("model", {}).get("name", "?")),
        ("pretrained", cfg.get("model", {}).get("pretrained")),
        ("pooling", f"{cfg.get('model', {}).get('pool')} over "
                    f"{len(cfg['input']['slots']) * cfg['input']['n_slices']} images/study"),
        ("output logits", f"{num_output_logits(cfg)}  (one per finding)"),
        None,
        ("total params", f"{tot_p:,}  ({tot_p/1e6:.2f} M)"),
        ("trainable", f"{tr_p:,}  ({tr_p/1e6:.2f} M, {100*tr_p/max(tot_p,1):.1f}%)"),
        ("frozen", f"{tot_p-tr_p:,}  ({(tot_p-tr_p)/1e6:.2f} M)"),
    ], "\U0001F9F1")

    loss_fn = build_loss(cfg, train_df, device)
    opt = build_optimizer(model, cfg)
    accum = cfg["training"].get("accum_steps", 1)
    steps_per_epoch = max(1, len(loaders["train"]) // accum)
    total = steps_per_epoch * epochs
    sched = build_scheduler(opt, cfg, total)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda")
    imgs_per_batch = (cfg["training"]["batch_size"] * len(cfg["input"]["slots"])
                      * cfg["input"]["n_slices"])
    print(f"[setup] optimizer={cfg['training']['optimizer']['name']}  "
          f"scheduler={cfg['training']['scheduler']['name']}  amp={scaler.is_enabled()}")
    print(f"[setup] steps/epoch={steps_per_epoch}  total_steps={total}  accum={accum}  "
          f"images/optimizer-step={imgs_per_batch * accum}")

    es = cfg["training"]["early_stopping"]
    best, best_epoch, bad, gstep = -math.inf, 0, 0, 0
    train_log = CSVLogger(out / cfg["output"]["train_log_csv"])
    val_log = CSVLogger(out / cfg["output"]["val_log_csv"])
    history = []
    t0 = time.time()
    prev_loss = None

    for epoch in range(1, epochs + 1):
        print("\n" + "=" * 70)
        print(f"\U0001F4DA EPOCH {epoch}/{epochs}")
        print("=" * 70)
        model.train()
        ep_t0, seen = time.time(), 0
        for it, (x, y) in enumerate(loaders["train"], 1):
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss = loss_fn(model(prepare_batch(x, cfg, device)), y) / accum
            scaler.scale(loss).backward()
            seen += imgs_per_batch
            if it % accum == 0:
                gnorm = float("nan")
                if cfg["training"]["grad_clip"]:
                    scaler.unscale_(opt)
                    gnorm = float(nn.utils.clip_grad_norm_(model.parameters(),
                                                           cfg["training"]["grad_clip"]))
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                gstep += 1
                lr = sched.get_last_lr()[-1]
                l = loss.item() * accum
                ips = seen / max(time.time() - ep_t0, 1e-9)
                mem = (torch.cuda.max_memory_allocated() / 1024**3
                       if device.type == "cuda" else 0.0)
                el = time.time() - t0
                eta = el * (total / gstep - 1)
                train_log.log({"epoch": epoch, "epoch_step": it // accum,
                               "global_step": gstep, "loss": round(l, 6), "lr": lr,
                               "gnorm": round(gnorm, 4), "img_s": round(ips, 1),
                               "gpu_mem_gb": round(mem, 2), "elapsed_s": round(el, 1)})
                if gstep % cfg["output"]["console_log_every"] == 0 or gstep == 1:
                    d = "" if prev_loss is None else f" ({l - prev_loss:+.4f})"
                    print(f"  Epoch {epoch}/{epochs}  epoch_step: {it // accum}/{steps_per_epoch}"
                          f"  global_step: {gstep}/{total}  loss={l:.4f}{d}  lr={lr:.4e}"
                          f"  gnorm={gnorm:.2f}  {ips:.0f} img/s  elapsed={fmt(el)}"
                          f"  ETA={fmt(eta)}  gpu_mem(peak)={mem:.2f}GB", flush=True)
                    prev_loss = l
                    if log_fn:
                        log_fn({"train/loss": l, "train/lr": lr, "train/gnorm": gnorm}, gstep)

        el = time.time() - t0
        eta = el * (total / max(gstep, 1) - 1)
        print("=" * 70)
        print(f"\U0001F50E VALIDATION - epoch {epoch}/{epochs}  step {gstep}  "
              f"({len(loaders['val'])} batches, {len(loaders['val'].dataset)} studies) ...")
        vm = evaluate(model, loaders["val"], loss_fn, cfg, device)
        _print_eval("VAL", epoch, epochs, gstep, vm, el, eta, cfg["tasks"])

        row = {"epoch": epoch, "step": gstep, "val_loss": round(vm["loss"], 6),
               "val_macro_auroc": round(vm["macro_auroc"], 6),
               "val_macro_auprc": round(vm["macro_auprc"], 6),
               "val_f1": round(vm["f1"], 4), "val_precision": round(vm["precision"], 4),
               "val_recall": round(vm["recall"], 4),
               "val_specificity": round(vm["specificity"], 4)}
        for t in cfg["tasks"]:
            row[f"val_auroc_{t}"] = round(vm["per_task"][t]["auroc"], 6)
            row[f"val_auprc_{t}"] = round(vm["per_task"][t]["auprc"], 6)

        if "audit" in loaders:
            am = evaluate(model, loaders["audit"], loss_fn, cfg, device)
            _print_eval("AUDIT-58", epoch, epochs, gstep, am, el, eta, cfg["tasks"],
                        "\U0001F9EA")
            print("     \u2139\uFE0F  report only - the panel's own labels, never drives best.pt")
            row["audit_macro_auroc"] = round(am["macro_auroc"], 6)
            for t in cfg["tasks"]:
                row[f"audit_auroc_{t}"] = round(am["per_task"][t]["auroc"], 6)

        row["elapsed_s"] = round(el, 1)
        val_log.log(row)
        history.append(row)
        if log_fn:
            log_fn({f"val/{k}": v for k, v in row.items() if isinstance(v, (int, float))}, gstep)

        score = row.get(es["monitor"], vm["macro_auroc"])
        improved = score > best if es["mode"] == "max" else score < best
        if improved:
            prev = best
            best, best_epoch, bad = score, epoch, 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "step": gstep, "score": score}, ckpt_dir / cfg["output"]["best_name"])
            delta = "" if prev == -math.inf else f"  (was {prev:.4f}, +{score - prev:.4f})"
            print(f"  \u2B50 new best {es['monitor']} = {score:.4f}{delta} -> best.pt")
        else:
            bad += 1
            print(f"  \u23F3 no improvement ({bad}/{es['patience']})  "
                  f"best {best:.4f} @ epoch {best_epoch}")
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "step": gstep},
                   ckpt_dir / "last.pt")
        print(f"  \U0001F4BE saved last.pt (epoch {epoch})")

        (out / cfg["output"]["summary_json"]).write_text(json.dumps({
            "experiment": cfg["experiment"]["name"], "epoch": epoch, "epochs": epochs,
            "best_score": best, "best_epoch": best_epoch, "monitor": es["monitor"],
            "last": row, "elapsed": fmt(time.time() - t0)}, indent=2))
        if persist_fn:
            persist_fn()

        if es["enable"] and bad >= es["patience"]:
            print(f"\n\U0001F6D1 early stop: no improvement for {bad} epochs")
            break

    train_log.close(); val_log.close()
    total_time = time.time() - t0
    best_row = next((r for r in history if r["epoch"] == best_epoch),
                    history[-1] if history else {})
    print("\n" + "=" * 70)
    print(f"\U0001F3C1 FINISHED - {cfg['experiment']['name']}")
    print("=" * 70)
    print(f"  \u23F1\uFE0F  total time       : {fmt(total_time)}  "
          f"({len(history)} epochs, {gstep} steps)")
    print(f"  \u2B50 best {es['monitor']:<15s}: {best:.4f}  @ epoch {best_epoch}")
    if best_row:
        print(f"  \U0001F4C9 val loss at best  : {best_row.get('val_loss')}")
        print(f"  \U0001F4CA macro AUPRC       : {best_row.get('val_macro_auprc')}")
        if "audit_macro_auroc" in best_row:
            print(f"  \U0001F9EA audit-58 AUROC    : {best_row['audit_macro_auroc']}  (report only)")
        print("\n  per-finding AUROC at the best epoch (bar = AUROC above chance):")
        ranked = sorted(((t, best_row.get(f"val_auroc_{t}", float("nan")))
                         for t in cfg["tasks"]), key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0))
        for t, v in ranked:
            bar = "\u2588" * int(max(0.0, v - 0.5) * 40) if v == v else ""
            print(f"    {t:18s} {v:.4f}  {bar}")
        print("\n  \U0001F4C8 reference: keyword labeler 0.700, LLM labels 0.895 (vs the 58)")
    print(f"  \U0001F4BE checkpoints       : {ckpt_dir}")
    print(f"  \U0001F4DD logs              : {out / cfg['output']['train_log_csv']}")
    print(f"                        {out / cfg['output']['val_log_csv']}")
    print("=" * 70)
    return best


def fmt(sec: float) -> str:
    """Seconds -> "1h04m09s" / "4m09s"."""
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# =============================================================================
# modal glue
# =============================================================================
def modal_image():
    """Container image for remote runs: deps + this runs/ folder + the small CSVs."""
    import modal
    return (modal.Image.debian_slim(python_version="3.11")
            .apt_install("libgl1", "libglib2.0-0")     # OpenCV needs these system libs
            .pip_install("torch", "torchvision", "timm", "numpy", "pandas",
                         "opencv-python-headless", "scikit-learn", "pyyaml", "wandb")
            .add_local_dir(str(Path(__file__).parent), "/root/runs")
            .add_local_dir(str(Path(__file__).parent.parent / "data"), "/root/data",
                           ignore=["shards", "raw", "backups", "*.jsonl"]))  # big data is a volume


def modal_resources(cfg: dict) -> dict:
    return {"gpu": cfg["modal"]["gpu"],
            "timeout": int(cfg["modal"].get("timeout_hours", 8) * 3600)}


def remote_cfg(cfg: dict) -> dict:
    """Copy of the config with local Windows paths swapped for the container's mounts."""
    r = json.loads(json.dumps(cfg))               # cheap deep copy, leaves the local cfg alone
    r["paths"]["data_dir"] = "/root/data"
    r["paths"]["shards_dir"] = cfg["modal"]["shards_mount"]
    return r
