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

import copy
import csv
import io
import json
import math
import multiprocessing as _mp
import os
import random
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
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
    # recursive dict merge; `over` wins on scalars, dicts get merged key by key.
    # copy.deepcopy so an untouched nested dict isn't shared with `base` — otherwise
    # two configs off one load alias each other and swap run_ids.
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
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


def train_batch_size(cfg: dict) -> int:
    """Studies per training batch. Lives under dataloader; training.batch_size still
    works so older run configs keep loading."""
    dl = cfg.get("dataloader", {})
    bs = dl.get("batch_size")
    if bs is None:
        bs = cfg["training"].get("batch_size")
    if bs is None:
        raise KeyError("batch_size missing: set dataloader.batch_size in the run config")
    return int(bs)


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
        ("run_id", cfg["experiment"].get("run_id", "(allocated at launch)")),
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

    bs = train_batch_size(cfg)
    eff = bs * t.get("accum_steps", 1)
    _kv("training", [
        ("epochs", t.get("epochs")),
        ("batch", f"{bs} studies x accum {t.get('accum_steps', 1)} "
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


# =============================================================================
# run identity
# =============================================================================
# Every run of an experiment gets a sequential id: run001, run002, ... It tags
# every CSV row and prefixes every file the run writes, so artifacts from two
# runs can be merged across machines without ever blending into each other.
RUN_ID_RE = re.compile(r"run\d+")


def run_id_of(name: str) -> str | None:
    """Leading runNNN of a filename, e.g. run007_e1_s800.png -> run007."""
    m = re.match(r"(run\d+)", str(name))
    return m.group(1) if m else None


def scan_run_ids(results_dir) -> set[str]:
    """Every runNNN this results/ folder already shows evidence of."""
    results_dir = Path(results_dir)
    ids = set()
    for sub in (".run_ids", "train_config", "debugged_images"):
        d = results_dir / sub
        if d.is_dir():
            ids |= {r for q in d.glob("run*") if (r := run_id_of(q.name))}
    for name in ("train_log.csv", "val_log.csv"):
        f = results_dir / name
        if f.exists():
            try:                                   # a truncated log must not block a launch
                ids |= set(pd.read_csv(f, usecols=["run_id"], engine="python",
                                       on_bad_lines="skip")["run_id"]
                           .dropna().astype(str))
            except Exception:
                pass
    return {i for i in ids if RUN_ID_RE.fullmatch(str(i))}


def next_run_no(ids) -> int:
    return max((int(str(i)[3:]) for i in ids if RUN_ID_RE.fullmatch(str(i))), default=0) + 1


def allocate_run_id(results_dir, known=()) -> str:
    """Reserve the next free runNNN for this experiment.

    Called on THIS PC at launch, never inside the container: the number is then
    passed into the run. `known` carries ids seen on the Modal volume so two
    workspaces can't land on the same one. The reservation itself is an
    exclusive-create, so two concurrent launches here can't collide either.
    """
    results_dir = Path(results_dir)
    lock_dir = results_dir / ".run_ids"            # markers only, never synced
    lock_dir.mkdir(parents=True, exist_ok=True)
    n = next_run_no(scan_run_ids(results_dir) | {str(k) for k in known})
    import platform
    while True:
        cand = f"run{n:03d}"
        try:
            with open(lock_dir / cand, "x", encoding="utf-8") as fh:
                fh.write(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  "
                         f"{platform.node()}  {workspace_name()}\n")
            return cand
        except FileExistsError:                    # someone else took it a moment ago
            n += 1


def workspace_name() -> str:
    """Which Modal workspace this is, from MODAL_CONFIG_PATH + modal_setup_commands.txt."""
    ws = os.environ.get("MODAL_WORKSPACE") or os.environ.get("MODAL_PROFILE")
    if ws:
        return ws
    m = re.search(r"modal_session(\d+)", os.environ.get("MODAL_CONFIG_PATH", ""))
    if not m:
        return "local"
    try:
        txt = (Path(__file__).resolve().parent.parent
               / "modal_setup_commands.txt").read_text(encoding="utf-8-sig")
        name = None
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("#"):
                name = line.lstrip("# ").rstrip(":").strip()
            elif f".modal_session{m.group(1)}.toml" in line and name:
                return name
    except Exception:
        pass
    return f"session{m.group(1)}"


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
# cross-worker counters isolating the raw shard read (npz open + decompress) from
# the resample/window/augment CPU work. Created at import, before the DataLoader
# forks, so every worker bumps the same three ints. Windows spawns instead of
# forking, so locally they stay 0 and the retrieval line simply does not print.
_RETR_NS = _mp.Value("q", 0)       # summed read time, nanoseconds
_RETR_BYTES = _mp.Value("q", 0)    # summed bytes decompressed
_RETR_IMGS = _mp.Value("q", 0)     # summed slices actually used


def _retr_snapshot():
    """(imgs, bytes, ns) totals so far - cheap lock-free read."""
    return _RETR_IMGS.value, _RETR_BYTES.value, _RETR_NS.value


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
        _t = time.perf_counter()
        with np.load(self.shards / f"{study}.npz") as z:
            if series not in z.files:
                return np.zeros((self.n_slices, self.size, self.size), np.uint8)   # missing series
            arr = z[series]                                  # (all_slices, H, W) uint8, native res
            idx = pick_slices(arr.shape[0], self.n_slices, self.mode, rng, self.margin)
            vol = arr[idx].astype(np.float32)
        # only the volume read is timed; resample/window/augment come after
        with _RETR_NS.get_lock():
            _RETR_NS.value += int((time.perf_counter() - _t) * 1e9)
            _RETR_BYTES.value += int(arr.nbytes)
            _RETR_IMGS.value += len(idx)

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


def val_loader_settings(cfg: dict) -> tuple[int, int]:
    """(batch_size, num_workers) for eval loaders — val_* keys, else the train values."""
    dl = cfg.get("dataloader", {})
    nw = dl.get("val_num_workers")
    return (int(dl.get("val_batch_size") or train_batch_size(cfg)),
            int(dl.get("num_workers", 4) if nw is None else nw))


def make_loaders(cfg: dict):
    """Build the train / val (/ audit) DataLoaders, all sharing one slot plan."""
    idx = pd.read_csv(resolve(cfg["paths"]["series_index"], cfg))   # one row per series
    plan = plan_slots(idx, cfg)
    _LAST_PLAN.clear(); _LAST_PLAN.update(plan)   # so the data summary can report coverage
    dl = cfg.get("dataloader", {})
    vbs, vnw = val_loader_settings(cfg)      # no backward pass -> val can go much wider
    out = {}
    for split, key, train in [("train", "train_csv", True), ("val", "val_csv", False)]:
        df = pd.read_csv(resolve(cfg["paths"][key], cfg))
        ds = KneeShardDataset(df, cfg, plan, train)
        nw = dl.get("num_workers", 4) if train else vnw
        out[split] = DataLoader(
            ds, batch_size=train_batch_size(cfg) if train else vbs, shuffle=train,
            num_workers=nw,
            pin_memory=dl.get("pin_memory", True),
            persistent_workers=dl.get("persistent_workers", True) and nw > 0,
            worker_init_fn=seed_worker,
            prefetch_factor=dl.get("prefetch_factor", 2) if nw > 0 else None,
            drop_last=dl.get("drop_last", True) and train)
    # audit = the 58 studies with real panel labels, scored but never used to pick best.pt
    if cfg["evaluation"]["audit"]["enable"]:
        df = pd.read_csv(resolve(cfg["evaluation"]["audit"]["csv"], cfg))
        # the audit CSV carries BOTH official_<task> (panel 0/1) and <task> (LLM soft).
        # drop the soft ones first, or the rename makes duplicate column names.
        df = df.drop(columns=[t for t in cfg["tasks"] if t in df.columns], errors="ignore")
        df = df.rename(columns={f"official_{t}": t for t in cfg["tasks"]})
        out["audit"] = DataLoader(KneeShardDataset(df, cfg, plan, False),
                                  batch_size=vbs, shuffle=False, num_workers=vnw)
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

    def __init__(self, path: Path, run_id: str = ""):
        self.path = Path(path); self.run_id = run_id; self._f = None; self._w = None

    def log(self, row: dict):
        row = {"run_id": self.run_id, **row}       # first column: which run wrote this line
        try:
            if self._f is None:
                new = (not self.path.exists()) or self.path.stat().st_size == 0
                fields = list(row.keys())
                if not new:                        # reuse the file's own header, never mix two
                    try:
                        with open(self.path, newline="", encoding="utf-8") as fh:
                            old = next(csv.reader(fh), [])
                        if old and old != fields:
                            print(f"  [warn] {self.path.name} has a different header, "
                                  f"appending in its column order")
                            fields = old
                    except Exception:
                        pass
                self._f = open(self.path, "a", newline="", encoding="utf-8")
                self._w = csv.DictWriter(self._f, fieldnames=fields, restval="",
                                         extrasaction="ignore")
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
                    thr: float = 0.5, pred_thr=None) -> dict:
    """Per-task AUROC + AUPRC and the operating-point stats at `thr`.

    pred_thr: scalar or one value per task for cutting the PREDICTIONS; defaults to thr.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    # targets always binarize at thr; predictions may use calibrated per-task cuts
    pt = np.asarray(thr if pred_thr is None else pred_thr, dtype=float)
    pt = np.broadcast_to(pt, (len(tasks),)).astype(float)

    per = {}
    for i, t in enumerate(tasks):
        yt = (y_true[:, i] > thr).astype(int)          # soft targets need a threshold to score
        yp = y_prob[:, i]
        one_class = yt.min() == yt.max()
        pbi = (yp > pt[i]).astype(int)
        tp_ = int(((yt == 1) & (pbi == 1)).sum()); fp_ = int(((yt == 0) & (pbi == 1)).sum())
        fn_ = int(((yt == 1) & (pbi == 0)).sum()); tn_ = int(((yt == 0) & (pbi == 0)).sum())
        p_ = tp_ / max(tp_ + fp_, 1); r_ = tp_ / max(tp_ + fn_, 1)
        per[t] = {
            "auroc": float("nan") if one_class else roc_auc_score(yt, yp),
            "auprc": float("nan") if one_class else average_precision_score(yt, yp),
            "pos": int(yt.sum()),
            "threshold": float(pt[i]),
            "precision": float(p_), "recall": float(r_),
            "f1": float(2 * p_ * r_ / max(p_ + r_, 1e-9)),
            "specificity": float(tn_ / max(tn_ + fp_, 1)),
        }
    au = [v["auroc"] for v in per.values() if not math.isnan(v["auroc"])]
    ap = [v["auprc"] for v in per.values() if not math.isnan(v["auprc"])]

    yb, pb = (y_true > thr).astype(int), (y_prob > pt[None, :]).astype(int)
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


class AUCMarginLoss(nn.Module):
    """Pairwise squared-hinge surrogate for AUROC — the competition metric itself.

    Every positive/negative pair in the batch is pushed `margin` apart; the pair
    weight is y_i*(1-y_j), so soft LLM targets work without rounding. Self-contained
    (no libauc), so it needs no extra dependency and plain AdamW drives it. A small
    BCE mix keeps the probabilities calibrated and covers batches with no pairs.
    """

    def __init__(self, margin: float = 1.0, bce_mix: float = 0.1, pos_weight=None):
        super().__init__()
        self.margin = float(margin); self.bce_mix = float(bce_mix)
        self.register_buffer("pw", pos_weight)

    def forward(self, logits, target):
        p = torch.sigmoid(logits)                       # (B, T) in 0-1, the AUC is over these
        d = p.unsqueeze(1) - p.unsqueeze(0)             # (B, B, T) pairwise score gaps
        w = target.unsqueeze(1) * (1 - target).unsqueeze(0)     # i positive, j negative
        hinge = torch.clamp(self.margin - d, min=0) ** 2
        loss = (w * hinge).sum() / w.sum().clamp(min=1e-6)
        if self.bce_mix > 0 or w.sum() <= 0:
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pw)
            loss = loss + max(self.bce_mix, 1e-3) * bce
        return loss


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
    if lc["name"] == "aucm":
        return AUCMarginLoss(lc.get("aucm_margin", 1.0), lc.get("aucm_bce_mix", 0.1), pw).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def _llrd_depths(backbone: nn.Module) -> tuple[dict, int]:
    """Map backbone param name -> depth (0 = nearest the pixels), plus the head depth.

    Handles the two timm layouts we use: ViT-style `blocks.<i>` and ConvNeXt-style
    `stages.<i>.blocks.<j>`. Anything unrecognized sits at the deepest backbone level.
    """
    names = [n for n, _ in backbone.named_parameters()]
    blocks = getattr(backbone, "blocks", None)
    stages = getattr(backbone, "stages", None)
    depth = {}
    if blocks is not None and hasattr(blocks, "__len__"):          # ViT / Swin-ish
        n_layers = len(blocks) + 1                                 # + the head
        for n in names:
            m = re.match(r"blocks\.(\d+)\.", n)
            if m:
                depth[n] = int(m.group(1)) + 1
            elif any(n.startswith(k) for k in ("patch_embed", "cls_token", "pos_embed", "stem")):
                depth[n] = 0
            else:
                depth[n] = n_layers - 1                            # final norm rides with the last block
        return depth, n_layers
    if stages is not None and hasattr(stages, "__len__"):          # ConvNeXt / ResNet-ish
        counts = [len(getattr(s, "blocks", [])) for s in stages]
        offset = np.cumsum([0] + counts).tolist()                  # blocks before each stage
        n_layers = int(offset[-1]) + 1
        for n in names:
            m = re.match(r"stages\.(\d+)\.blocks\.(\d+)\.", n)
            if m:
                depth[n] = offset[int(m.group(1))] + int(m.group(2)) + 1
            elif (m2 := re.match(r"stages\.(\d+)\.", n)):          # that stage's downsample
                depth[n] = offset[int(m2.group(1))]
            elif n.startswith("stem"):
                depth[n] = 0
            else:
                depth[n] = n_layers - 1
        return depth, n_layers
    return {n: 0 for n in names}, 1                                # unknown layout: one backbone level


def _param_groups_llrd(model: nn.Module, base_lr: float, weight_decay: float,
                       layer_decay: float) -> list:
    """One param group per (depth, decay flag): lr = base_lr * layer_decay**(top - depth).

    Early layers barely move, the head trains at full lr. 1-D params (norms, biases)
    get no weight decay, the usual fine-tuning convention.
    """
    backbone = getattr(model, "backbone", None)
    bdepth, n_layers = _llrd_depths(backbone) if backbone is not None else ({}, 1)
    top = n_layers                                    # head sits one level above the backbone
    scales = [layer_decay ** (top - d) for d in range(top + 1)]
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        d = bdepth.get(n[len("backbone."):], top) if n.startswith("backbone.") else top
        nd = p.ndim == 1
        key = (d, nd)
        g = groups.setdefault(key, {"params": [], "lr": base_lr * scales[d],
                                    "weight_decay": 0.0 if nd else weight_decay,
                                    "lr_scale": scales[d], "layer_id": d})
        g["params"].append(p)
    return [groups[k] for k in sorted(groups)]


def _print_llrd(groups: list, base_lr: float, layer_decay: float, top: int):
    """Boxed per-depth learning rates, so the decay can be eyeballed before it runs."""
    per = {}
    for g in groups:
        per.setdefault(g["layer_id"], g["lr"])
        per[g["layer_id"]] = g["lr"]
    depths = sorted(per)
    show = depths if len(depths) <= 8 else depths[:3] + [None] + depths[-3:]
    rows = []
    for d in show:
        if d is None:
            rows.append(["...", "...", "..."])
            continue
        label = "stem / embed" if d == 0 else ("head + pool" if d == top else f"block {d - 1}")
        rows.append([d, label, f"{per[d]:.3e}"])
    _table(f"LLRD layer_decay={layer_decay}  base_lr={base_lr:.2e}  "
           f"({len(depths)} depths, {len(groups)} groups)",
           ["depth", "what", "lr"], rows, "\U0001FA9C")


def build_optimizer(model: nn.Module, cfg: dict):
    """AdamW with two learning rates: full for the new head, slower for the backbone."""
    o = cfg["training"]["optimizer"]
    lr, mult = o["lr"], o.get("backbone_lr_mult", 1.0)
    ld = o.get("layer_decay")
    if ld:
        # LLRD replaces the flat backbone_lr_mult: every depth gets its own lr
        groups = _param_groups_llrd(model, lr, o.get("weight_decay", 0.05), float(ld))
        _print_llrd(groups, lr, float(ld), max(g["layer_id"] for g in groups))
        return torch.optim.AdamW(groups, lr=lr, weight_decay=o.get("weight_decay", 0.05))
    # split structurally, not by name: everything under .backbone is the pretrained
    # trunk, the rest is freshly initialized. Substring matching got this wrong -
    # "fc" also matches every mlp.fc1/fc2 inside a ConvNeXt block.
    has_bb = hasattr(model, "backbone")
    head, backbone = [], []
    n_head = n_bb = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = (n.startswith("backbone.") if has_bb
                 else not any(k in n for k in ("head", "classifier", "pool_attn")))
        (backbone if is_bb else head).append(p)
        if is_bb:
            n_bb += p.numel()
        else:
            n_head += p.numel()
    groups = [{"params": backbone, "lr": lr * mult}, {"params": head, "lr": lr}]
    print(f"  \U0001F9EE lr groups: head {n_head / 1e6:.2f}M @ {lr:.2e}  |  "
          f"backbone {n_bb / 1e6:.2f}M @ {lr * mult:.2e}  (x{mult})")
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
# checkpoints & resume
# =============================================================================
def _rng_state(device) -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _restore_rng(st: dict, device):
    try:
        random.setstate(st["python"]); np.random.set_state(st["numpy"])
        torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
        if device.type == "cuda" and "cuda" in st:
            torch.cuda.set_rng_state_all([t.cpu() for t in st["cuda"]])
    except Exception as e:                     # a resumed run must not die over RNG
        print(f"  [warn] could not restore RNG: {type(e).__name__}: {e}")


def save_checkpoint(path: Path, cfg, model, opt=None, sched=None, scaler=None,
                    epoch=0, step=0, score=None, track=None, device=None,
                    weights_only=False):
    """Full state by default, so a run can resume exactly where it stopped."""
    payload = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
               "step": step, "score": score, "track": track}
    if not weights_only:
        if opt is not None:
            payload["optimizer"] = opt.state_dict()
        if sched is not None:
            payload["scheduler"] = sched.state_dict()
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        if device is not None:
            payload["rng"] = _rng_state(device)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)                          # atomic: never leave a half-written ckpt
    return path


def _resolve_resume(resume, ckpt_dir: Path) -> Path | None:
    """null | 'best' | 'last' | <step int> | <path> -> a file to load."""
    if resume in (None, False, ""):
        return None
    if isinstance(resume, str) and resume not in ("best", "last"):
        pth = Path(resume)
        if pth.exists():
            return pth
    if resume == "best":
        pth = ckpt_dir / "best.pt"
        return pth if pth.exists() else None
    if resume == "last":
        cands = sorted(ckpt_dir.glob("step*.pt"), key=lambda q: q.stat().st_mtime)
        pth = ckpt_dir / "last.pt"
        if pth.exists():
            return pth
        return cands[-1] if cands else None
    if isinstance(resume, int) or str(resume).isdigit():
        pth = ckpt_dir / f"step{int(resume)}.pt"
        return pth if pth.exists() else None
    return None


def load_checkpoint(resume, model, ckpt_dir: Path, opt=None, sched=None,
                    scaler=None, device=None) -> dict:
    """Restore whatever the checkpoint holds. Returns {} when there is nothing to do."""
    path = _resolve_resume(resume, ckpt_dir)
    if path is None:
        if resume not in (None, False, ""):
            print(f"  \u26A0\uFE0F  resume={resume!r} requested but no checkpoint found "
                  f"in {ckpt_dir}")
        return {}
    print(f"  \u21A9\uFE0F  resuming from {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    for obj, key in ((opt, "optimizer"), (sched, "scheduler"), (scaler, "scaler")):
        if obj is not None and key in ck and ck[key] is not None:
            try:
                obj.load_state_dict(ck[key])
            except Exception as e:
                print(f"  [warn] could not restore {key}: {type(e).__name__}: {e}")
    if device is not None and "rng" in ck:
        _restore_rng(ck["rng"], device)
    print(f"  \u21A9\uFE0F  resumed at epoch {ck.get('epoch')}, step {ck.get('step')}, "
          f"score={ck.get('score')}")
    return ck


def prune_checkpoints(ckpt_dir: Path, keep_last_n: int, keep_names=()):
    """Rolling window over the periodic step checkpoints."""
    if keep_last_n <= 0:
        return
    files = sorted(ckpt_dir.glob("step*.pt"), key=lambda q: q.stat().st_mtime)
    for f in files[:-keep_last_n]:
        if f.name not in keep_names:
            try:
                f.unlink()
            except OSError:
                pass


def update_top_k(ckpt_dir: Path, cfg, model, epoch: int, step: int, score: float,
                 track: list, k: int, weights_only: bool = True) -> list:
    """Keep the K best validations of the whole run, not just the single best.

    best.pt is overwritten every improvement, so every runner-up is otherwise
    lost -- and those are exactly what ensembling wants.
    """
    if k <= 0 or score != score:                # k=0 disables; NaN never ranks
        return track
    track = list(track or [])
    track.append({"step": step, "epoch": epoch, "score": float(score),
                  "file": f"top_step{step}.pt"})
    track.sort(key=lambda r: -r["score"])
    keep, drop = track[:k], track[k:]
    if any(r["step"] == step for r in keep):
        save_checkpoint(ckpt_dir / f"top_step{step}.pt", cfg, model, epoch=epoch,
                        step=step, score=score, weights_only=weights_only)
    for r in drop:
        f = ckpt_dir / r["file"]
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass
    (ckpt_dir / "top_k.json").write_text(json.dumps(keep, indent=2))
    return keep


def seed_worker(worker_id: int):
    """Give every dataloader worker its own reproducible stream."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


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


def save_debug_batch(x_prep: torch.Tensor, cfg: dict, out_dir: Path,
                     epoch: int, step: int, y: torch.Tensor | None = None,
                     prefix: str = ""):
    """Dump one study from the batch EXACTLY as the model sees it.

    x_prep is the normalized tensor fed to the backbone, so we invert the
    ImageNet normalization to make it viewable. Anything wrong with windowing,
    augmentation, the mm/px rescale or the crop shows up here immediately.
    """
    if cv2 is None:
        return
    slots, d = len(cfg["input"]["slots"]), cfg["input"]["n_slices"]
    per = slots * d
    if x_prep.shape[0] < per:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor(cfg["input"]["norm_mean"][:x_prep.shape[1]]).view(1, -1, 1, 1)
    std = torch.tensor(cfg["input"]["norm_std"][:x_prep.shape[1]]).view(1, -1, 1, 1)
    v = x_prep[:per].detach().float().cpu() * std + mean       # undo normalization
    v = (v.clamp(0, 1) * 255).to(torch.uint8)[:, 0].numpy()    # first channel is enough

    cols = min(8, d)
    picks = np.linspace(0, d - 1, cols).astype(int)
    h, w = v.shape[1:]
    sheet = np.zeros((slots * h, cols * w), np.uint8)
    for r in range(slots):
        for c, sl in enumerate(picks):
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = v[r * d + sl]

    sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    for r, name in enumerate(cfg["input"]["slots"]):
        cv2.putText(sheet, name, (6, r * h + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1, cv2.LINE_AA)
    if y is not None:
        pos = [t for t, p_ in zip(cfg["tasks"], y[0].tolist()) if p_ > 0.5]
        cv2.putText(sheet, "labels>0.5: " + (", ".join(pos) or "none"),
                    (6, sheet.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)

    path = out_dir / f"{prefix}e{epoch}_s{step}.png"
    cv2.imwrite(str(path), sheet)
    print(f"  \U0001F5BC\uFE0F  debug image -> {path}")


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
    res_dir = out_dir / cfg["output"]["run_dir"]
    exp = cfg.setdefault("experiment", {})
    # normally allocated on the launching PC and passed in; only a bare local run gets here
    exp["run_id"] = exp.get("run_id") or allocate_run_id(res_dir)
    cap_dir = res_dir / cfg["output"].get("console_capture_dir", "train_config")
    cap_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cap_path = cap_dir / f"{exp['run_id']}_{stamp}.txt"
    try:
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    orig_out, orig_err = sys.stdout, sys.stderr
    log_f = open(cap_path, "a", encoding="utf-8")
    sys.stdout, sys.stderr = _Tee(orig_out, log_f), _Tee(orig_err, log_f)
    print(f"[capture] full console of this run -> {cap_path}")
    print(f"[capture] run_id: {exp['run_id']}  workspace: {workspace_name()}")
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
    run_id = cfg.setdefault("experiment", {}).get("run_id") or ""
    print("#" * 70)
    print(f"# 🚀 run_experiment: {cfg['experiment']['name']}")
    print(f"# 🆔 run_id={run_id}  workspace={workspace_name()}")
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
    # every file this run writes carries its run_id as a prefix, so nothing collides
    dbg_dir = out / cfg["output"].get("debug_dir", "debugged_images")
    cap_dir = out / cfg["output"].get("console_capture_dir", "train_config")
    cap_dir.mkdir(parents=True, exist_ok=True)

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
    perf = cfg["training"].get("perf", {})
    if device.type == "cuda" and perf.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if perf.get("compile", False):
        try:
            model = torch.compile(model, mode=perf.get("compile_mode", "default"))
            print(f"  \u26A1 torch.compile enabled (mode={perf.get('compile_mode', 'default')})")
        except Exception as e:
            print(f"  [warn] torch.compile failed: {type(e).__name__}: {e}")
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda")
    imgs_per_batch = (train_batch_size(cfg) * len(cfg["input"]["slots"])
                      * cfg["input"]["n_slices"])
    print(f"[setup] optimizer={cfg['training']['optimizer']['name']}  "
          f"scheduler={cfg['training']['scheduler']['name']}  amp={scaler.is_enabled()}")
    print(f"[setup] steps/epoch={steps_per_epoch}  total_steps={total}  accum={accum}  "
          f"images/optimizer-step={imgs_per_batch * accum}")
    _vbs, _vnw = val_loader_settings(cfg)
    print(f"[setup] loss={cfg['training']['loss']['name']}  "
          f"train bs={train_batch_size(cfg)}/{cfg.get('dataloader', {}).get('num_workers', 4)}w  "
          f"val bs={_vbs}/{_vnw}w")

    es = cfg["training"]["early_stopping"]
    best, best_epoch, bad, gstep = -math.inf, 0, 0, 0
    start_epoch, top_k_track = 1, []
    ck = load_checkpoint(cfg.get("resume"), model, ckpt_dir, opt, sched, scaler, device)
    if ck:
        gstep = int(ck.get("step") or 0)
        start_epoch = int(ck.get("epoch") or 0) + 1
        tr = ck.get("track") or {}
        best = tr.get("best", -math.inf)
        best_epoch = tr.get("best_epoch", 0)
        top_k_track = tr.get("top_k", [])
    train_log = CSVLogger(out / cfg["output"]["train_log_csv"], run_id)
    val_log = CSVLogger(out / cfg["output"]["val_log_csv"], run_id)
    history = []
    t0 = time.time()
    prev_loss = None

    last_val_step = -1

    def _validate_and_checkpoint(epoch: int, gstep: int, tag: str = "epoch"):
        """Score val (and audit), log, checkpoint, update early stopping."""
        nonlocal best, best_epoch, bad, top_k_track, last_val_step
        if gstep == last_val_step:            # a step-trigger already covered this step
            return None
        last_val_step = gstep
        el = time.time() - t0
        eta = el * (total / max(gstep, 1) - 1)
        print("=" * 70)
        print(f"\U0001F50E VALIDATION ({tag}) - epoch {epoch}/{epochs}  step {gstep}  "
              f"({len(loaders['val'])} batches, {len(loaders['val'].dataset)} studies) ...")
        vm = evaluate(model, loaders["val"], loss_fn, cfg, device)
        _print_eval("VAL", epoch, epochs, gstep, vm, el, eta, cfg["tasks"])

        row = {"epoch": epoch, "step": gstep, "trigger": tag,
               "val_loss": round(vm["loss"], 6),
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
            print("     \u2139\uFE0F  report only - the panel's labels, never drives best.pt")
            row["audit_macro_auroc"] = round(am["macro_auroc"], 6)
            for t in cfg["tasks"]:
                row[f"audit_auroc_{t}"] = round(am["per_task"][t]["auroc"], 6)

        row["elapsed_s"] = round(el, 1)
        val_log.log(row)
        history.append(row)
        if log_fn:
            log_fn({f"val/{k}": v for k, v in row.items() if isinstance(v, (int, float))},
                   gstep)

        score = row.get(es["monitor"], vm["macro_auroc"])
        improved = score > best if es["mode"] == "max" else score < best
        if improved:
            prev = best
            best, best_epoch, bad = score, epoch, 0
            save_checkpoint(ckpt_dir / cfg["output"]["best_name"], cfg, model, opt, sched,
                            scaler, epoch, gstep, score,
                            track={"best": best, "best_epoch": best_epoch,
                                   "top_k": top_k_track}, device=device)
            delta = "" if prev == -math.inf else f"  (was {prev:.4f}, +{score - prev:.4f})"
            print(f"  \u2B50 new best {es['monitor']} = {score:.4f}{delta} -> best.pt")
        else:
            bad += 1
            print(f"  \u23F3 no improvement ({bad}/{es['patience']})  "
                  f"best {best:.4f} @ epoch {best_epoch}")

        save_checkpoint(ckpt_dir / "last.pt", cfg, model, opt, sched, scaler, epoch, gstep,
                        score, track={"best": best, "best_epoch": best_epoch,
                                      "top_k": top_k_track}, device=device)
        top_k_track = update_top_k(ckpt_dir, cfg, model, epoch, gstep, score, top_k_track,
                                   cfg["output"].get("keep_top_k", 0),
                                   cfg["output"].get("top_k_weights_only", True))
        print(f"  \U0001F4BE saved last.pt (epoch {epoch})"
              + (f"  \U0001F3C6 top-{len(top_k_track)} kept" if top_k_track else ""))

        summary = json.dumps({
            "experiment": cfg["experiment"]["name"], "run_id": run_id,
            "epoch": epoch, "epochs": epochs,
            "best_score": best, "best_epoch": best_epoch, "monitor": es["monitor"],
            "last": row, "elapsed": fmt(time.time() - t0)}, indent=2)
        (out / cfg["output"]["summary_json"]).write_text(summary)
        # a per-run copy too, so the sync can carry it without clobbering the other side
        (cap_dir / f"{run_id}_summary.json").write_text(summary)
        if persist_fn:
            persist_fn()
        model.train()
        return score


    retr_prev = (0, 0, 0, 0.0)      # last retrieval snapshot
    for epoch in range(start_epoch, epochs + 1):
        print("\n" + "=" * 70)
        print(f"\U0001F4DA EPOCH {epoch}/{epochs}")
        print("=" * 70)
        model.train()
        ep_t0, seen = time.time(), 0
        # two throughput numbers so a slow volume can't hide behind a fast GPU:
        #   compute = images / GPU time only, real = images / wall incl. the data wait.
        # next() is timed by hand (not via `for`) so eval/checkpoint time between
        # steps never gets miscounted as waiting on data.
        roll = deque(maxlen=20)          # (imgs, wall, data_wait) for the last 20 steps
        win_imgs, win_t0, win_data, win_first = 0, time.time(), 0.0, 0.0
        _train_it, it = iter(loaders["train"]), 0
        while True:
            _tf = time.time()
            try:
                x, y = next(_train_it)
            except StopIteration:
                break
            data_dt = time.time() - _tf              # blocked purely on this batch
            it += 1
            if (it - 1) % accum == 0:                # fresh accumulation window
                win_imgs, win_t0 = 0, time.time()
                win_data, win_first = 0.0, data_dt
            win_data += data_dt
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss = loss_fn(model(prepare_batch(x, cfg, device)), y) / accum
            scaler.scale(loss).backward()
            seen += imgs_per_batch
            win_imgs += imgs_per_batch
            if it % accum == 0:
                gnorm = float("nan")
                if cfg["training"]["grad_clip"]:
                    scaler.unscale_(opt)
                    gnorm = float(nn.utils.clip_grad_norm_(model.parameters(),
                                                           cfg["training"]["grad_clip"]))
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                gstep += 1
                lrs = sched.get_last_lr()
                lr, lr_lo, lr_hi = lrs[-1], min(lrs), max(lrs)
                l = loss.item() * accum
                # dt spans the whole window; the waits inside it aren't compute
                dt = time.time() - win_t0
                wall = win_first + dt
                comp_t = max(dt - (win_data - win_first), 1e-9)
                roll.append((win_imgs, wall, win_data))
                r_i = sum(a for a, _, _ in roll)
                r_w = max(sum(b for _, b, _ in roll), 1e-9)
                r_d = sum(c for _, _, c in roll)
                ips = win_imgs / comp_t                      # compute-only
                real_ips = r_i / r_w                         # incl. data wait, 20-step
                wait_pct = 100.0 * r_d / r_w
                mem = (torch.cuda.max_memory_allocated() / 1024**3
                       if device.type == "cuda" else 0.0)
                el = time.time() - t0
                eta = el * (total / gstep - 1)
                train_log.log({"epoch": epoch, "epoch_step": it // accum,
                               "global_step": gstep, "loss": round(l, 6), "lr": lr,
                               "lr_min": lr_lo, "lr_max": lr_hi,
                               "gnorm": round(gnorm, 4), "img_s": round(ips, 1),
                               "real_img_s": round(real_ips, 1),
                               "data_wait_pct": round(wait_pct, 1),
                               "gpu_mem_gb": round(mem, 2), "elapsed_s": round(el, 1)})
                ev = cfg["evaluation"].get("eval_every_steps", 0)
                if ev and gstep % ev == 0:
                    _validate_and_checkpoint(epoch, gstep, f"every {ev} steps")
                every = cfg["output"].get("checkpoint_every_steps", 0)
                if every and gstep % every == 0:
                    save_checkpoint(ckpt_dir / f"step{gstep}.pt", cfg, model, opt, sched,
                                    scaler, epoch, gstep,
                                    track={"best": best, "best_epoch": best_epoch,
                                           "top_k": top_k_track}, device=device)
                    prune_checkpoints(ckpt_dir, cfg["output"].get("keep_last_n", 3))
                    print(f"  \U0001F4BE periodic checkpoint @ step {gstep} "
                          f"(rolling, keep_last_n={cfg['output'].get('keep_last_n', 3)})")
                dbg = cfg["output"].get("save_debug_images", 0)
                if dbg and (gstep % dbg == 0 or gstep == 1):
                    save_debug_batch(prepare_batch(x, cfg, device), cfg, dbg_dir,
                                     epoch, gstep, y, prefix=f"{run_id}_")
                if gstep % cfg["output"]["console_log_every"] == 0 or gstep == 1:
                    d = "" if prev_loss is None else f" ({l - prev_loss:+.4f})"
                    # one lr when every group shares it, both ends when LLRD or
                    # backbone_lr_mult splits them
                    lr_s = (f"lr={lr_hi:.3e}" if lr_lo == lr_hi
                            else f"lr[head={lr_hi:.3e} backbone={lr_lo:.3e}]")
                    print("─" * 70)
                    print(f"  Epoch {epoch}/{epochs}  epoch_step: {it // accum}/{steps_per_epoch}"
                          f"  global_step: {gstep}/{total}")
                    print(f"     loss={l:.4f}{d}  {lr_s}  gnorm={gnorm:.2f}  "
                          f"elapsed={fmt(el)}  ETA={fmt(eta)}  gpu_mem(peak)={mem:.2f}GB")
                    print(f"     throughput: {ips:.0f} img/s compute | {real_ips:.0f} img/s real "
                          f"({len(roll)}-step) | data-wait {wait_pct:.0f}%")
                    # slices/s and MB/s off the shard volume, summed over every worker,
                    # for the window since the last print. ms/img is per-slice latency.
                    _ri, _rb, _rns = _retr_snapshot()
                    _pi, _pb, _pns, _pel = retr_prev
                    d_i, d_b, d_ns = _ri - _pi, _rb - _pb, _rns - _pns
                    d_el = max(el - _pel, 1e-6)
                    retr_prev = (_ri, _rb, _rns, el)
                    if d_i > 0:
                        print(f"     retrieval:  {d_i / d_el:.0f} img/s | "
                              f"{d_b / 1e6 / d_el:.0f} MB/s | "
                              f"{d_ns / 1e6 / d_i:.1f} ms/img avg read (shards)", flush=True)
                    prev_loss = l
                    if log_fn:
                        log_fn({"train/loss": l, "train/lr": lr, "train/gnorm": gnorm}, gstep)

        _validate_and_checkpoint(epoch, gstep, "epoch")

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

    # curves for every run, and a plotting bug must never sink a finished run
    if cfg["output"].get("make_plots", True):
        _safe("make_plots", make_plots, cfg, out,
              cfg["output"].get("plots_dir", "plots"))
        if persist_fn:
            persist_fn()
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
                         "opencv-python-headless", "scikit-learn", "pyyaml", "wandb",
                         "matplotlib")
            .add_local_dir(str(Path(__file__).parent), "/root/runs")
            .add_local_dir(str(Path(__file__).parent.parent / "data"), "/root/data",
                           ignore=["shards", "raw", "backups", "*.jsonl"]))  # big data is a volume


def modal_resources(cfg: dict) -> dict:
    return {"gpu": cfg["modal"]["gpu"],
            "timeout": int(cfg["modal"].get("timeout_hours", 8) * 3600)}


def _sync_module():
    """modal_scripts/sync_runs.py — lazy, because it only exists on this PC."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modal_scripts"))
    import sync_runs
    return sync_runs


def launch_run_id(cfg: dict, exp_dir) -> str:
    """Allocate this launch's runNNN, here on the launching PC.

    The volume is consulted too (a `modal volume ls`, no container), so two
    workspaces can never hand out the same number.
    """
    known = set()
    try:
        sr = _sync_module()
        s = sr.session_from_env()
        if s:
            known = sr.remote_run_ids(s, cfg["experiment"]["name"])
    except Exception as e:                     # offline / no credit: local state still works
        print(f"  ⚠️  couldn't read run ids off the volume "
              f"({type(e).__name__}: {e}) - numbering from local state only")
    rid = allocate_run_id(Path(exp_dir) / cfg["output"]["run_dir"], known)
    print(f"  🆔 allocated {rid} for {cfg['experiment']['name']}"
          + (f"  (volume already had {len(known)})" if known else ""))
    return rid


def sync_back(cfg: dict, status: str = "") -> None:
    """Pull this run's artifacts home. Fires on success AND on crash; never raises."""
    try:
        sr = _sync_module()
        s = sr.session_from_env()
        if s is None:
            print("  ⚠️  no MODAL_CONFIG_PATH - skipping the artifact sync")
            return
        print(f"\n  \U0001F504 syncing artifacts back ({status}) ...")
        sr.sync_session(s, cfg["experiment"]["name"])
    except Exception as e:                     # a failed sync must never mask a training error
        print(f"  ⚠️  artifact sync failed: {type(e).__name__}: {e}")


def remote_cfg(cfg: dict) -> dict:
    """Copy of the config with local Windows paths swapped for the container's mounts."""
    r = json.loads(json.dumps(cfg))               # cheap deep copy, leaves the local cfg alone
    r["paths"]["data_dir"] = "/root/data"
    r["paths"]["shards_dir"] = cfg["modal"]["shards_mount"]
    return r


# =============================================================================
# plots — post-hoc PNGs from the two CSVs. CPU only, no torch needed.
# =============================================================================
def _safe(label: str, fn, *args, **kwargs):
    """Run fn but never let it kill a run; warn and carry on."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  ⚠️  {label} failed (continuing): {type(e).__name__}: {e}")
        return None


def _smooth(series, window: int):
    """Trailing rolling mean; min_periods=1 so the start of the curve survives."""
    if window and window > 1:
        return pd.Series(series).rolling(window, min_periods=1).mean()
    return pd.Series(series)


def _epoch_boundary_steps(df) -> list:
    """Steps where a new epoch starts — drawn as faint guides."""
    steps, prev = [], None
    if df is None or "epoch" not in df.columns:
        return steps
    for st, ep in zip(df["global_step"], df["epoch"]):
        if prev is not None and ep != prev:
            steps.append(st)
        prev = ep
    return steps


def _vlines(ax, steps):
    for s in steps:
        ax.axvline(s, color="0.85", lw=0.8, zorder=0)


def _mark_extreme(ax, x, y, mode: str = "max", color: str = "crimson"):
    """Star the best point of a curve and label it."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or np.all(np.isnan(y)):
        return
    i = int(np.nanargmax(y)) if mode == "max" else int(np.nanargmin(y))
    xi = float(np.asarray(x)[i])
    ax.scatter([xi], [y[i]], marker="*", s=200, color=color, zorder=6)
    ax.annotate(f"{y[i]:.4f}\n@ step {int(xi)}", (xi, y[i]), textcoords="offset points",
                xytext=(8, -14), fontsize=8, color=color)


def _runs_in(df, col="run_id") -> list:
    """Run ids present, in first-seen order. Never blend two runs into one line."""
    if df is None or col not in df.columns:
        return [None]
    ids = [r for r in dict.fromkeys(df[col].astype(str).tolist()) if r and r != "nan"]
    return ids or [None]


def _run_slice(df, rid):
    if df is None:
        return None
    if rid is None or "run_id" not in df.columns:
        return df
    sub = df[df["run_id"].astype(str) == rid]
    return sub if len(sub) else None


def make_plots(cfg: dict, results_dir, out_subdir: str = "plots",
               smooth_window: int = 25, dpi: int = 130) -> list:
    """Training-curve PNGs from train_log.csv + val_log.csv -> results/plots/.

    Both CSVs can hold several runs (the run_id column), so every run is drawn as
    its own series and never averaged together.
    """
    import matplotlib
    matplotlib.use("Agg")                     # headless: write files, never open a window
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    o, tasks = cfg["output"], cfg["tasks"]
    name = cfg.get("experiment", {}).get("name", "experiment")
    tr_path, va_path = results_dir / o["train_log_csv"], results_dir / o["val_log_csv"]
    plots_dir = results_dir / out_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    _banner("PLOTS", "\U0001F4C8")
    print(f"  train log : {tr_path}")
    print(f"  val   log : {va_path}")
    print(f"  output    : {plots_dir}")
    tr = pd.read_csv(tr_path) if tr_path.exists() else None
    va = pd.read_csv(va_path) if va_path.exists() else None
    if tr is None:
        print("  ⚠️  no train_log.csv - skipping the train-based plots")
    if va is None:
        print("  ⚠️  no val_log.csv - skipping the validation-based plots")
    if tr is None and va is None:
        return []

    runs = _runs_in(tr if tr is not None else va)
    for df in (tr, va):                       # a run may appear in only one of the two
        for r in _runs_in(df):
            if r not in runs:
                runs.append(r)
    colors = plt.get_cmap("tab10").colors
    cmap = {r: colors[i % len(colors)] for i, r in enumerate(runs)}
    single = len(runs) == 1
    bsteps = _epoch_boundary_steps(tr) if (single and tr is not None) else []
    print(f"  runs      : {', '.join(str(r) for r in runs)}")
    saved = []

    def _save(fig, fname):
        fig.tight_layout(); fig.savefig(plots_dir / fname, dpi=dpi); plt.close(fig)
        saved.append(fname); print(f"  ✅ {fname}")

    def _lbl(rid, extra=""):
        base = str(rid) if rid else "run"
        return f"{base}{extra}"

    # 1) loss: raw + smoothed train, val as markers, one colour per run
    def _plot_loss():
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in runs:
            c = cmap[r]
            t = _run_slice(tr, r)
            if t is not None and "loss" in t.columns:
                ax.plot(t["global_step"], t["loss"], color=c, alpha=0.2, lw=0.8)
                ax.plot(t["global_step"], _smooth(t["loss"].values, smooth_window),
                        color=c, lw=1.8, label=_lbl(r, " train"))
            v = _run_slice(va, r)
            if v is not None and "val_loss" in v.columns:
                ax.plot(v["step"], v["val_loss"], color=c, ls="--", marker="o", ms=4,
                        lw=1.4, label=_lbl(r, " val"))
        _vlines(ax, bsteps)
        ax.set_xlabel("global step"); ax.set_ylabel("loss")
        ax.set_title(f"{name} - loss (train smoothed x{smooth_window})")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        _save(fig, "loss_curves.png")

    # 2) one macro metric over steps, best point starred per run
    def _plot_macro(col, title, fname, mode="max", extra_col=None):
        if va is None or col not in va.columns:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in runs:
            v = _run_slice(va, r)
            if v is None:
                continue
            ax.plot(v["step"], v[col], color=cmap[r], marker="o", ms=4, lw=1.8,
                    label=_lbl(r))
            _mark_extreme(ax, v["step"].values, v[col].values, mode, cmap[r])
            if extra_col and extra_col in v.columns and v[extra_col].notna().any():
                ax.plot(v["step"], v[extra_col], color=cmap[r], ls=":", marker="^",
                        ms=4, lw=1.2, alpha=0.7, label=_lbl(r, " audit-58"))
        _vlines(ax, bsteps)
        ax.set_xlabel("global step"); ax.set_ylabel(col)
        ax.set_title(f"{name} - {title}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        _save(fig, fname)

    # 3) per-task small multiples, one panel per finding, one line per run
    def _plot_per_task(prefix, title, fname):
        if va is None:
            return
        cols = [(t, f"val_{prefix}_{t}") for t in tasks if f"val_{prefix}_{t}" in va.columns]
        if not cols:
            return
        ncol = 4
        nrow = int(math.ceil(len(cols) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, (t, c) in zip(axes, cols):
            for r in runs:
                v = _run_slice(va, r)
                if v is not None:
                    ax.plot(v["step"], v[c], color=cmap[r], marker="o", ms=2.5, lw=1.3,
                            label=_lbl(r))
            ax.axhline(0.5, color="0.6", lw=0.7, ls=":")     # chance level
            ax.set_title(t, fontsize=9); ax.grid(True, alpha=0.3)
        for ax in axes[len(cols):]:
            ax.axis("off")
        if len(runs) > 1 or runs[0]:
            axes[0].legend(fontsize=7, loc="lower right", framealpha=0.7)
        fig.suptitle(f"{name} - {title}", fontsize=11)
        _save(fig, fname)

    # 4) macro F1 / precision / recall / specificity at 0.5
    def _plot_op_point():
        if va is None:
            return
        series = [("val_f1", "F1"), ("val_precision", "precision"),
                  ("val_recall", "recall"), ("val_specificity", "specificity")]
        series = [s for s in series if s[0] in va.columns]
        if not series:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        styles = ["-", "--", ":", "-."]
        for r in runs:
            v = _run_slice(va, r)
            if v is None:
                continue
            for (col, lbl), st in zip(series, styles):
                ax.plot(v["step"], v[col], color=cmap[r], ls=st, lw=1.5, marker="o",
                        ms=3, label=_lbl(r, f" {lbl}"))
        _vlines(ax, bsteps)
        ax.set_xlabel("global step"); ax.set_ylabel("score @ threshold 0.5")
        ax.set_title(f"{name} - operating point (threshold 0.5)")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
        _save(fig, "macro_f1_precision_recall.png")

    # 5) learning rate
    def _plot_lr():
        if tr is None or "lr" not in tr.columns:
            return
        fig, ax = plt.subplots(figsize=(9, 4))
        for r in runs:
            t = _run_slice(tr, r)
            if t is not None:
                ax.plot(t["global_step"], t["lr"], color=cmap[r], lw=1.6, label=_lbl(r))
        _vlines(ax, bsteps)
        ax.set_xlabel("global step"); ax.set_ylabel("learning rate")
        ax.set_title(f"{name} - learning rate (warmup + cosine)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        _save(fig, "learning_rate.png")

    # 6) throughput / GPU memory / gradient norm, stacked
    def _plot_diagnostics():
        if tr is None:
            return
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        panels = [("img_s", "images / s"), ("gpu_mem_gb", "GPU mem peak (GB)"),
                  ("gnorm", "grad norm")]
        for ax, (col, ylab) in zip(axes, panels):
            for r in runs:
                t = _run_slice(tr, r)
                if t is None or col not in t.columns:
                    continue
                ax.plot(t["global_step"], t[col], color=cmap[r], alpha=0.25, lw=0.8)
                ax.plot(t["global_step"], _smooth(t[col].values, smooth_window),
                        color=cmap[r], lw=1.6, label=_lbl(r))
            ax.set_ylabel(ylab); ax.grid(True, alpha=0.3); _vlines(ax, bsteps)
        axes[0].set_title(f"{name} - throughput / memory / gradients")
        axes[0].legend(fontsize=8); axes[-1].set_xlabel("global step")
        _save(fig, "training_diagnostics.png")

    _safe("plot:loss", _plot_loss)
    _safe("plot:macro_auroc", _plot_macro, "val_macro_auroc", "validation macro AUROC",
          "macro_auroc.png", "max", "audit_macro_auroc")
    _safe("plot:macro_auprc", _plot_macro, "val_macro_auprc", "validation macro AUPRC",
          "macro_auprc.png", "max")
    _safe("plot:per_task_auroc", _plot_per_task, "auroc", "per-finding AUROC",
          "per_task_auroc.png")
    _safe("plot:per_task_auprc", _plot_per_task, "auprc", "per-finding AUPRC",
          "per_task_auprc.png")
    _safe("plot:op_point", _plot_op_point)
    _safe("plot:lr", _plot_lr)
    _safe("plot:diagnostics", _plot_diagnostics)
    print(f"  \U0001F4C8 wrote {len(saved)} plot(s) -> {plots_dir}")
    return saved


# =============================================================================
# standalone inference, threshold calibration, official scoring
# =============================================================================
def _json_safe(obj):
    """Make a dict JSON-valid: NaN/inf -> null, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def _resolve_eval_ckpt(checkpoint, ckpt_dir: Path) -> Path | None:
    """best | last | top1..topK | <step> | <path> -> a file."""
    if isinstance(checkpoint, str) and re.fullmatch(r"top\d+", checkpoint):
        rank = int(checkpoint[3:])
        try:
            track = json.loads((ckpt_dir / "top_k.json").read_text())
            return ckpt_dir / track[rank - 1]["file"]
        except Exception as e:
            print(f"  ⚠️  no {checkpoint} in top_k.json ({type(e).__name__}: {e})")
            return None
    return _resolve_resume(checkpoint, ckpt_dir)


def _load_ckpt_weights(model, ckpt_dir: Path, checkpoint, device):
    """Load the requested checkpoint's weights. Returns (ckpt_dict, path)."""
    path = _resolve_eval_ckpt(checkpoint, ckpt_dir)
    if path is None:
        raise FileNotFoundError(f"no checkpoint {checkpoint!r} under {ckpt_dir}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return ck, path


def _eval_df(cfg: dict, csv_name: str) -> pd.DataFrame:
    """Read a split CSV; the audit file names its columns official_<task>."""
    df = pd.read_csv(resolve(csv_name, cfg))
    if any(f"official_{t}" in df.columns for t in cfg["tasks"]):
        df = df.drop(columns=[t for t in cfg["tasks"] if t in df.columns], errors="ignore")
    return df.rename(columns={f"official_{t}": t for t in cfg["tasks"]})


@torch.no_grad()
def _predict_dataframe(cfg: dict, model, df: pd.DataFrame, device, loss_fn=None,
                       plan: dict | None = None, batch_size: int | None = None,
                       num_workers: int | None = None, amp: bool | None = None,
                       desc: str = "") -> tuple:
    """Score one dataframe with the eval transforms. -> (y_true, y_prob, mean_loss)."""
    if plan is None:
        plan = plan_slots(pd.read_csv(resolve(cfg["paths"]["series_index"], cfg)), cfg)
    vbs, vnw = val_loader_settings(cfg)
    bs = int(batch_size or vbs)
    nw = vnw if num_workers is None else int(num_workers)
    amp = (cfg["training"]["amp"] and device.type == "cuda") if amp is None else amp
    has_y = all(t in df.columns for t in cfg["tasks"])
    if not has_y:                          # a test split has no labels; feed zeros
        df = df.copy()
        for t in cfg["tasks"]:
            df[t] = 0.0
    loader = DataLoader(KneeShardDataset(df, cfg, plan, False), batch_size=bs,
                        shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")
    model.eval().to(device)
    P, Y, losses, seen = [], [], [], 0
    t0 = last = time.time()
    print(f"     [infer] {desc}: {len(df)} studies | {len(loader)} batches x{bs} | "
          f"{nw} workers | amp={amp}", flush=True)
    for bi, (x, y) in enumerate(loader, 1):
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            logits = model(prepare_batch(x, cfg, device))
            if loss_fn is not None:
                losses.append(float(loss_fn(logits, y)) * y.shape[0])
        P.append(torch.sigmoid(logits.float()).cpu().numpy()); Y.append(y.cpu().numpy())
        seen += y.shape[0]
        now = time.time()
        if now - last >= 5.0 or bi == len(loader):     # throttled progress line
            sps = seen / max(now - t0, 1e-9)
            print(f"     [infer] {desc}: batch {bi}/{len(loader)} | study {seen}/{len(df)} "
                  f"({100 * seen / max(len(df), 1):.0f}%) | {sps:.1f} studies/s | "
                  f"elapsed {fmt(now - t0)} | ETA {fmt((len(df) - seen) / max(sps, 1e-9))}",
                  flush=True)
            last = now
    y_true = np.concatenate(Y) if has_y else None
    return y_true, np.concatenate(P), (float(np.sum(losses) / max(seen, 1)) if losses else float("nan"))


def _best_f1_threshold(y_true_col, y_prob_col) -> tuple:
    """Exact max-F1 cut for one task, read off the precision-recall curve."""
    from sklearn.metrics import precision_recall_curve

    yt, yp = np.asarray(y_true_col), np.asarray(y_prob_col)
    n_pos = int(yt.sum())
    if n_pos == 0 or n_pos == len(yt):            # degenerate task: nothing to optimize
        return 0.5, float("nan")
    prec, rec, thr = precision_recall_curve(yt, yp)
    f1 = (2 * prec * rec / (prec + rec + 1e-12))[:-1]    # last point has no threshold
    if len(f1) == 0 or np.all(np.isnan(f1)):
        return 0.5, float("nan")
    i = int(np.nanargmax(f1))
    # sklearn cuts at p >= thr, compute_metrics at p > thr; nudge down so they agree
    return float(np.nextafter(thr[i], -np.inf)), float(f1[i])


def calibrate_thresholds(cfg: dict, model, df_val: pd.DataFrame, device, **kw) -> tuple:
    """Per-task max-F1 threshold on the validation split. -> (thresholds, y_true, y_prob)."""
    y_true, y_prob, _ = _predict_dataframe(cfg, model, df_val, device,
                                           desc="calibrate/val", **kw)
    cut = float(cfg.get("target", {}).get("hard_threshold", 0.5))
    thr = {}
    for i, t in enumerate(cfg["tasks"]):
        thr[t], _ = _best_f1_threshold((y_true[:, i] > cut).astype(int), y_prob[:, i])
    return thr, y_true, y_prob


def _thresholds_path(results_dir, cfg) -> Path:
    return Path(results_dir) / cfg["output"].get("thresholds_json", "thresholds.json")


def load_thresholds(results_dir, cfg) -> tuple:
    """Frozen per-task thresholds from results/thresholds.json, or (None, path)."""
    path = _thresholds_path(results_dir, cfg)
    if not path.exists():
        return None, path
    try:
        thr = json.loads(path.read_text(encoding="utf-8")).get("thresholds", {})
    except Exception as e:
        print(f"  ⚠️  unreadable {path.name}: {type(e).__name__}: {e}")
        return None, path
    if not all(t in thr for t in cfg["tasks"]):
        return None, path
    return {t: float(thr[t]) for t in cfg["tasks"]}, path


def _threshold_payload(cfg, thresholds, y_true, y_prob, ckpt, ckpt_path, n_val) -> tuple:
    """thresholds.json content + the val metrics those thresholds achieve."""
    tasks = cfg["tasks"]
    cut = float(cfg.get("target", {}).get("hard_threshold", 0.5))
    m = compute_metrics(y_true, y_prob, tasks, cut, [thresholds[t] for t in tasks])
    payload = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "run_id": cfg.get("experiment", {}).get("run_id"),
        "objective": "f1", "scope": "per_task",
        "calibration_set": str(cfg["paths"]["val_csv"]),
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step"), "checkpoint_epoch": ckpt.get("epoch"),
        "n_val_studies": int(n_val),
        "thresholds": {t: float(thresholds[t]) for t in tasks},
        "val_metrics_at_threshold": {k: m[k] for k in
                                     ("macro_auroc", "macro_auprc", "f1", "precision",
                                      "recall", "specificity", "per_task")},
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return payload, m


def run_calibration(cfg: dict, model, experiment_dir, device=None, checkpoint: str = "best",
                    **kw) -> dict:
    """Pick and freeze the per-task max-F1 thresholds -> results/thresholds.json.

    AUROC is threshold-free, so this only affects the reported F1/precision/recall
    and any future hard-label output (submission, error analysis).
    """
    experiment_dir = Path(experiment_dir)
    res = experiment_dir / cfg["output"]["run_dir"]
    ckpt_dir = res / cfg["output"]["checkpoints_dir"]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = cfg["tasks"]

    _banner("THRESHOLD CALIBRATION", "\U0001F39A️")
    ck, ck_path = _load_ckpt_weights(model, ckpt_dir, checkpoint, device)
    _kv("setup", [("experiment", cfg.get("experiment", {}).get("name")),
                  ("checkpoint", ck_path.name),
                  ("from", f"epoch {ck.get('epoch')}, step {ck.get('step')}, "
                           f"score={ck.get('score')}"),
                  ("device", str(device)),
                  ("set", str(cfg["paths"]["val_csv"])),
                  ("objective", "max F1 per finding")], "⚙️")

    df_val = _eval_df(cfg, cfg["paths"]["val_csv"])
    thr, y_true, y_prob = calibrate_thresholds(cfg, model, df_val, device, **kw)
    payload, m = _threshold_payload(cfg, thr, y_true, y_prob, ck, ck_path, len(df_val))

    def f(x):
        return f"{x:.4f}" if isinstance(x, float) and math.isfinite(x) else "nan"

    _table("per-finding thresholds", ["finding", "thr", "F1", "Prec", "Rec", "AUROC"],
           [[t, f"{thr[t]:.3f}", f(m["per_task"][t]["f1"]), f(m["per_task"][t]["precision"]),
             f(m["per_task"][t]["recall"]), f(m["per_task"][t]["auroc"])] for t in tasks],
           "\U0001F39A️")
    print(f"  \U0001F4CA macro @ thresholds: F1={m['f1']:.4f}  P={m['precision']:.4f}  "
          f"R={m['recall']:.4f}   (AUROC={m['macro_auroc']:.4f}, threshold-free)")
    path = _thresholds_path(res, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"  \U0001F4BE saved frozen thresholds -> {path}")
    return thr


def _render_report_txt(rep: dict) -> str:
    """Readable twin of <set>_results.json."""
    L = ["=" * 86,
         f"EVALUATION - {rep['set']}   ({rep['csv']})",
         "=" * 86,
         f"  experiment : {rep['experiment']}   run_id: {rep.get('run_id')}",
         f"  checkpoint : {rep['checkpoint']}  (epoch {rep.get('checkpoint_epoch')}, "
         f"step {rep.get('checkpoint_step')})",
         f"  studies    : {rep['n_studies']}    device: {rep['device']}    amp: {rep['amp']}",
         f"  thresholds : {rep['threshold_source']}"]
    if rep.get("macro") is None:
        L += ["-" * 86, "  no label columns in this CSV - predictions only.", "=" * 86]
        return "\n".join(L)
    mac = rep["macro"]
    L += ["-" * 86,
          "  MACRO (mean over the 12 findings)   [AUROC / AUPRC are threshold-free]",
          f"    macro AUROC : {mac['macro_auroc']:.4f}        macro AUPRC : {mac['macro_auprc']:.4f}",
          f"    F1 : {mac['f1']:.4f}    precision : {mac['precision']:.4f}    "
          f"recall : {mac['recall']:.4f}    specificity : {mac['specificity']:.4f}",
          f"    loss : {rep['loss']:.5f}",
          "-" * 86, "  PER-FINDING",
          f"    {'finding':<18}{'thr':>7}{'AUROC':>8}{'AUPRC':>8}{'F1':>8}{'Prec':>8}"
          f"{'Rec':>8}{'Spec':>8}{'n_pos':>8}"]
    for t, m in rep["per_task"].items():
        def f(x):
            return f"{x:.4f}" if isinstance(x, float) and math.isfinite(x) else "nan"
        L.append(f"    {t:<18}{m['threshold']:>7.3f}{f(m['auroc']):>8}{f(m['auprc']):>8}"
                 f"{f(m['f1']):>8}{f(m['precision']):>8}{f(m['recall']):>8}"
                 f"{f(m['specificity']):>8}{m['pos']:>8}")
    L.append("=" * 86)
    return "\n".join(L)


def _print_eval_comparison(reports: dict, cfg: dict, res: Path):
    """Side-by-side of every scored set, so val and the audit-58 read as one picture."""
    labels = [k for k in reports if reports[k].get("macro")]
    if not labels:
        return
    tasks = cfg["tasks"]

    _banner("SUMMARY", "\U0001F4CB")

    meta = {"val": "LLM labels - larger, so statistically solid",
            "audit": "the expert panel's own 0/1 - the real target, but noisy"}
    _table("what was scored", ["set", "studies", "checkpoint", "what it measures"],
           [[k, reports[k]["n_studies"], Path(reports[k]["checkpoint"]).name,
             meta.get(k, "custom split")] for k in labels], "\U0001F9ED")

    rows = [[m.replace("macro_", "").replace("_", " "),
             *[f"{reports[k]['macro'][m]:.4f}" for k in labels]]
            for m in ("macro_auroc", "macro_auprc", "f1", "precision",
                      "recall", "specificity")]
    _table("headline", ["metric", *labels], rows, "\U0001F3AF")

    rows = []
    for t in tasks:
        cells = [t]
        for k in labels:
            pt = reports[k]["per_task"][t]
            cells.append("  n/a" if pt["auroc"] != pt["auroc"] else f"{pt['auroc']:.4f}")
        if len(labels) == 2:
            a, b = (reports[k]["per_task"][t]["auroc"] for k in labels)
            cells.append("     -" if (a != a or b != b) else f"{b - a:+.4f}")
        rows.append(cells)
    heads = ["finding", *[f"{k} AUROC" for k in labels]]
    if len(labels) == 2:
        heads.append("delta")
    _table("per-finding AUROC", heads, rows, "\U0001F50D")

    best = max(labels, key=lambda k: reports[k]["macro"]["macro_auroc"])
    print(f"\n  \U0001F4C8 reference points: keyword labeler 0.700, LLM labels 0.895 "
          f"(both vs the 58)")
    for k in labels:
        v = reports[k]["macro"]["macro_auroc"]
        bar = "\u2588" * int(max(0.0, v - 0.5) * 60)
        print(f"    {k:<8} {v:.4f}  {bar}")
    if "audit" in labels:
        print("  \u2139\uFE0F  audit-58 is the honest external check: never trained on, "
              "never used to pick a checkpoint.")
    print(f"  \U0001F4C1 reports written to {res}")


def evaluate_official(cfg: dict, model, experiment_dir, device=None,
                      checkpoint: str = "best", eval_sets=("val", "audit"),
                      calibrate_if_missing: bool = True, batch_size: int | None = None,
                      num_workers: int | None = None, amp: bool | None = None) -> dict:
    """Score a saved checkpoint on whole splits, outside any training run.

    eval_sets names a split ("train"/"val"/"audit") or any CSV under data_dir. Each
    writes results/<set>_predictions.csv plus a <set>_results.json/.txt report. A CSV
    with no label columns (the test set) gets predictions only - that is the submission.
    """
    experiment_dir = Path(experiment_dir)
    res = experiment_dir / cfg["output"]["run_dir"]
    ckpt_dir = res / cfg["output"]["checkpoints_dir"]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = cfg["tasks"]

    _banner("EVALUATION", "\U0001F9EA")
    ck, ck_path = _load_ckpt_weights(model, ckpt_dir, checkpoint, device)
    model = model.to(device)
    _kv("setup", [("experiment", cfg.get("experiment", {}).get("name")),
                  ("checkpoint", str(ck_path)),
                  ("from", f"epoch {ck.get('epoch')}, step {ck.get('step')}, "
                           f"score={ck.get('score')}"),
                  ("device", str(device)), ("sets", ", ".join(eval_sets))], "⚙️")

    plan = plan_slots(pd.read_csv(resolve(cfg["paths"]["series_index"], cfg)), cfg)
    loss_fn = build_loss(cfg, None, device)
    kw = dict(batch_size=batch_size, num_workers=num_workers, amp=amp, plan=plan)

    thr_map, thr_path = load_thresholds(res, cfg)
    thr_source = f"per-task max-F1 on {cfg['paths']['val_csv']}"
    if thr_map is None and calibrate_if_missing:
        print(f"  ⚠️  no usable {thr_path.name} - calibrating now on "
              f"{cfg['paths']['val_csv']} ...")
        df_val = _eval_df(cfg, cfg["paths"]["val_csv"])
        thr_map, y_t, y_p = calibrate_thresholds(cfg, model, df_val, device, **kw)
        payload, _ = _threshold_payload(cfg, thr_map, y_t, y_p, ck, ck_path, len(df_val))
        thr_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(f"  \U0001F4BE saved frozen thresholds -> {thr_path}")
    elif thr_map is None:
        thr_map, thr_source = {t: 0.5 for t in tasks}, "uncalibrated (0.5)"
        print("  ⚠️  no thresholds.json - using 0.5 (AUROC/AUPRC unaffected)")
    else:
        print(f"  \U0001F39A️  frozen thresholds from {thr_path}")

    named = {"train": cfg["paths"]["train_csv"], "val": cfg["paths"]["val_csv"],
             "audit": cfg["evaluation"]["audit"]["csv"]}
    cut = float(cfg.get("target", {}).get("hard_threshold", 0.5))
    reports = {}
    for name in eval_sets:
        csv_name = named.get(name, name)
        # a raw csv path is labelled by its stem, so nothing gets written outside results/
        label = name if name in named else Path(name).stem
        path = resolve(csv_name, cfg)
        print("=" * 70)
        print(f"\U0001F9EA evaluating on {label}  ({path})")
        if not path.exists():
            print(f"  ⚠️  CSV not found - skipping {label}")
            continue
        df = _eval_df(cfg, csv_name)
        y_true, y_prob, loss = _predict_dataframe(cfg, model, df, device, loss_fn,
                                                  desc=f"eval/{label}", **kw)
        ids = df.get("StudyInstanceUID", pd.Series(range(len(df)))).astype(str).values
        pred = pd.DataFrame(y_prob, columns=[f"p_{t}" for t in tasks])
        pred.insert(0, "StudyInstanceUID", ids)
        pred_path = res / f"{label}_predictions.csv"
        pred.to_csv(pred_path, index=False)

        m = (compute_metrics(y_true, y_prob, tasks, cut, [thr_map[t] for t in tasks])
             if y_true is not None else None)
        rep = {"experiment": cfg.get("experiment", {}).get("name"),
               # outside a training run the live cfg has no run_id: use the checkpoint's
               "run_id": (cfg.get("experiment", {}).get("run_id")
                          or (ck.get("cfg", {}).get("experiment", {}) or {}).get("run_id")
                          or "?"),
               "set": label, "csv": str(csv_name), "checkpoint": str(ck_path),
               "checkpoint_step": ck.get("step"), "checkpoint_epoch": ck.get("epoch"),
               "n_studies": int(len(df)), "device": str(device),
               "amp": bool(cfg["training"]["amp"] if amp is None else amp),
               "threshold_source": thr_source,
               "thresholds": {t: float(thr_map[t]) for t in tasks},
               "loss": loss,
               "macro": None if m is None else {k: m[k] for k in
                                                ("macro_auroc", "macro_auprc", "f1",
                                                 "precision", "recall", "specificity")},
               "per_task": None if m is None else m["per_task"],
               "predictions_csv": str(pred_path),
               "evaluated_at": datetime.now().isoformat(timespec="seconds")}
        txt = _render_report_txt(rep)
        (res / f"{label}_results.json").write_text(json.dumps(_json_safe(rep), indent=2),
                                                  encoding="utf-8")
        (res / f"{label}_results.txt").write_text(txt + "\n", encoding="utf-8")
        print(txt)
        print(f"  \U0001F4BE {res / f'{label}_results.json'}")
        print(f"  \U0001F4BE {res / f'{label}_results.txt'}")
        print(f"  \U0001F4BE {pred_path}")
        reports[label] = rep

    _safe("eval summary", _print_eval_comparison, reports, cfg, res)
    print(f"\n\U0001F3C1 evaluation done - {len(reports)} set(s) scored.")
    return reports
