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
import shutil
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
from torch.utils.data import DataLoader, Dataset, Sampler

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


def _coerce_sci(node):
    """Turn YAML's accidental strings back into floats.

    YAML only reads scientific notation as a number when the mantissa has a decimal point:
    `8.0e-6` is a float, `8e-6` is the STRING "8e-6". Every learning rate in this repo is
    written in that notation, so one missing dot silently hands a string to the optimizer.
    float() happens to rescue most call sites, which is worse - it means the typo survives
    into a run and only shows up as a config print that looks subtly wrong.

    Only strings carrying an exponent marker are touched, so genuine strings ("last",
    "auto_clamped", "cosine") and plain digits (a resume step) are left alone.
    """
    items = node.items() if isinstance(node, dict) else enumerate(node)
    for k, v in list(items):
        if isinstance(v, (dict, list)):
            _coerce_sci(v)
        elif isinstance(v, str) and ("e" in v or "E" in v):
            try:
                node[k] = float(v)
            except ValueError:
                pass


def load_config(experiment_dir, config_name: str = "config.yaml", verbose: bool = True) -> dict:
    """shared_config.yaml + the run's config.yaml, run wins."""
    exp_dir = Path(experiment_dir)
    shared = yaml.safe_load((exp_dir.parent / "shared_config.yaml").read_text(encoding="utf-8"))
    own = yaml.safe_load((exp_dir / config_name).read_text(encoding="utf-8")) or {}
    cfg = _deep_merge(shared, own)
    _coerce_sci(cfg)
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


def accum_steps(cfg: dict) -> int:
    """Micro-batches per weight update. Lives under dataloader, next to batch_size;
    training.accum_steps still works for older run configs."""
    dl = cfg.get("dataloader", {})
    return int(dl.get("accum_steps", cfg["training"].get("accum_steps", 1)))


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
    acc = accum_steps(cfg)
    eff = bs * acc
    _kv("training", [
        ("epochs", t.get("epochs")),
        ("batch", f"{bs} studies x accum {acc} "
                  f"= {eff} studies ({eff * n_img} images) per update"),
        ("optimizer", f"{t['optimizer']['name']}  "
                      f"backbone lr={t['optimizer'].get('lr'):.2e}  "
                      f"head lr={t['optimizer'].get('lr') / (t['optimizer'].get('backbone_lr_mult') or 1):.2e}  "
                      f"wd={t['optimizer'].get('weight_decay')}"),
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
        ("eval cadence", f"{cfg['evaluation'].get('eval_every_steps_before')} steps "
                         f"until epoch {cfg['evaluation'].get('eval_every_steps_epoch_threshold')}, "
                         f"then every {cfg['evaluation'].get('eval_every_steps_after')} steps"
                         + (f"  + every {cfg['evaluation']['eval_every_epochs']} epoch(s)"
                            if cfg['evaluation'].get('eval_every_epochs') else "")),
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
            ("cpu / memory", f"{m.get('cpu') or 'modal default'} cores / "
                             f"{str(m.get('memory_gb')) + ' GB' if m.get('memory_gb') else 'modal default'}"),
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
    passed into the run. `known` carries ids already seen on the Modal volume, so a
    relaunch can't reuse one. The reservation itself is an exclusive-create, so two
    concurrent launches here can't collide either.
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


def hardware_info(cfg: dict) -> dict:
    """What this run actually trained on. Asked of the driver, not the config.

    The config records what we REQUESTED; a run that silently landed on a different
    card, or fell back to CPU, would otherwise look identical in the summary. That
    matters here because runs get resumed on fresh hardware mid-flight.
    """
    m = cfg.get("modal", {}) or {}
    dl = cfg.get("dataloader", {}) or {}
    # runtime_side(), not workspace_name(): this dict is written into the COMMITTED
    # training_summary.json, and which account a job landed on is nobody's business
    hw = {"gpu_requested": m.get("gpu"), "workspace": runtime_side(),
          "cpu_requested": m.get("cpu"), "memory_gb_requested": m.get("memory_gb"),
          "num_workers": dl.get("num_workers"),
          "batch_size": dl.get("batch_size"), "accum_steps": dl.get("accum_steps")}
    try:
        hw["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["gpu_count"] = torch.cuda.device_count()
            hw["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
            hw["gpu_capability"] = f"sm_{props.major}{props.minor}"
            hw["cuda"] = torch.version.cuda
            hw["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        else:
            hw["gpu"] = "cpu"
    except Exception as e:                        # never let bookkeeping kill a run
        hw["gpu"] = f"unknown ({type(e).__name__})"
    return hw


def workspace_name() -> str:
    """Which Modal workspace this is, from MODAL_CONFIG_PATH + modal_setup_commands.txt.

    For the CONSOLE and for results/.run_ids/*, both of which stay on this PC - the
    launch log lands in results/train_config/, and that plus .run_ids are git-ignored.
    Never put this in a committed artifact: use runtime_side() for those.
    """
    ws = os.environ.get("MODAL_WORKSPACE") or os.environ.get("MODAL_PROFILE")
    if ws:
        return ws
    m = re.search(r"modal_session(\d+)", os.environ.get("MODAL_CONFIG_PATH", ""))
    if not m:
        return runtime_side()
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


def runtime_side() -> str:
    """Which side of the wire this is, with no account name in it.

    training_summary.json is committed, so it gets this rather than workspace_name().
    MODAL_TASK_ID exists only inside a container, so "local" would be exactly backwards
    there - say which side we are on instead.
    """
    return "modal-container" if os.environ.get("MODAL_TASK_ID") else "local"


def num_output_logits(cfg: dict) -> int:
    return len(cfg["tasks"])          # one logit per label, 12


class SlicePool(nn.Module):
    """Collapse a study's per-slice features into ONE vector. `pool` picks how.

    The 12 labels are per STUDY, but the backbone sees 96 images (4 slots x 24 slices),
    so something has to combine them. That step is the whole architecture:

      mean   plain average. A fracture on 2 of 96 slices is diluted 48x, and each slice
             gets 1/96 of the gradient. The weakest findings are the FOCAL ones
             (Fracture AUPRC 0.43, MCL 0.50) while diffuse ones (Effusion 0.93) do
             fine - which is what that dilution looks like.
      max    strongest slice wins. Undoes the dilution, but one slice decides.
      attn   learned weight per slice, softmaxed (attention-MIL, Ilse et al. 2018).
             Permutation-invariant: it can weight slice 40 highly but cannot know it
             sits between 39 and 41.
      transformer  self-attention over all 96, with slice-position AND slot-type
             embeddings. The only option where a slice on one view can look at a slice
             on ANOTHER - a meniscus tear corroborated on sagittal and coronal, say.
             Our 4-slot input creates that opportunity; a single-volume model has no
             such thing, which is why the RSNA competition solutions (CNN + GRU over
             slices) never needed it.

    `out_dim` is what the classifier head should take - the transformer pools in its
    own d_model, everything else in the backbone's width.
    """

    def __init__(self, cfg: dict, feat: int):
        super().__init__()
        m = cfg["model"]
        self.kind = m.get("pool", "mean")
        self.n_slots = len(cfg["input"]["slots"])
        self.n_slices = int(cfg["input"]["n_slices"])
        self.out_dim = feat

        if self.kind == "attn":
            self.pool_attn = nn.Sequential(nn.Linear(feat, 128), nn.Tanh(),
                                           nn.Linear(128, 1))
        elif self.kind == "transformer":
            t = m.get("pool_transformer", {}) or {}
            d = int(t.get("d_model", 448))
            heads = int(t.get("n_heads", 7))
            if d % heads:
                raise ValueError(f"d_model {d} must divide by n_heads {heads}")
            layers = int(t.get("n_layers", 1))
            ratio = float(t.get("ffn_ratio", 2))
            drop = float(t.get("dropout", 0.1))
            # project first: attention costs scale with d^2, and 768 would make this
            # block a large fraction of a model that has already shown it gains
            # nothing from capacity (Base: +0.0008 for 38M params)
            self.proj = nn.Linear(feat, d)
            # position here is a PAIR, (slot, slice), so the embedding is factorised
            # rather than one flat table of 96. Flat would make the model relearn
            # "5th slice through the knee" separately for each of the 4 views; this
            # way the slice meaning is shared across slots and the view identity is
            # shared across slices - 28 vectors instead of 96. Learned, not
            # sinusoidal: there are only 24 and 4 of them.
            self.pos_slice = nn.Parameter(torch.zeros(1, self.n_slices, d))
            self.pos_slot = nn.Parameter(torch.zeros(1, self.n_slots, d))
            nn.init.trunc_normal_(self.pos_slice, std=0.02)
            nn.init.trunc_normal_(self.pos_slot, std=0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=int(d * ratio),
                dropout=drop, batch_first=True, norm_first=True, activation="gelu")
            # enable_nested_tensor is a no-op under norm_first and only warns
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers,
                                                 enable_nested_tensor=False)
            # norm_first=True is PRE-LN: each sublayer normalises its INPUT, so the
            # encoder's output never passes through a final norm. This is that norm.
            # (The residual/skip connections are inside TransformerEncoderLayer -
            # there are none at this level.)
            self.norm = nn.LayerNorm(d)
            self.score = nn.Linear(d, 1)      # attention pool over the 96 outputs
            self.out_dim = d
        elif self.kind not in ("mean", "max"):
            raise ValueError(f"unknown pool {self.kind!r}")

    def forward(self, f):                     # f: (B, slots*slices, F)
        if self.kind == "max":
            return f.max(dim=1).values
        if self.kind == "attn":
            w = torch.softmax(self.pool_attn(f), dim=1)
            return (f * w).sum(dim=1)
        if self.kind == "transformer":
            # shapes for the default 4 slots x 24 slices, d_model 448:
            b, n, _ = f.shape                        # f:  (B, 96, 768)
            x = self.proj(f)                         #     (B, 96, 448)
            x = x.view(b, self.n_slots, self.n_slices, -1)   # (B, 4, 24, 448)
            # (1,1,24,448) + (1,4,1,448) broadcast to all 96 (slot, slice) positions
            x = x + self.pos_slice[:, None, :, :] + self.pos_slot[:, :, None, :]
            x = x.reshape(b, n, -1)                  #     (B, 96, 448)  96 tokens
            x = self.encoder(x)                      #     (B, 96, 448)  each sees all 95
            x = self.norm(x)                         #     (B, 96, 448)  pre-LN final norm
            # weighted average over the 96 slices: score each one, softmax the scores
            # into weights summing to 1, blend. Same as mean pooling except the
            # weights are learned instead of all being 1/96.
            slice_scores = self.score(x)             #     (B, 96, 1)  "how much does
            slice_w = torch.softmax(slice_scores, 1) #     (B, 96, 1)   this slice matter"
            return (x * slice_w).sum(dim=1)          # ->  (B, 448)    one per study
        return f.mean(dim=1)                  # mean


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
            ranked = cand.sort_values(sort_by, ascending=asc)
            # slot_variant > 0 takes the 2nd/3rd/... best candidate instead of the
            # winner. Only used for test-time augmentation: 38% of studies have a
            # slot with several series and we normally discard all but one, so this
            # is a second look at the SAME knee through different images. Clamped, so
            # a slot with one candidate simply keeps it.
            k = min(int(cfg["input"].get("slot_variant", 0)), len(ranked) - 1)
            chosen[slot] = ranked.iloc[k]["series"]
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
_EPOCH = _mp.Value("q", 0)         # current epoch, so train augmentation varies per epoch


def _retr_snapshot():
    """(imgs, bytes, ns) totals so far - cheap lock-free read."""
    return _RETR_IMGS.value, _RETR_BYTES.value, _RETR_NS.value


class KneeShardDataset(Dataset):
    """One item = one study: 4 slot volumes of n_slices each, plus its 12 labels."""

    def __init__(self, df: pd.DataFrame, cfg: dict, plan: dict, train: bool,
                 dicom: tuple | None = None):
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
        # augmentation is a TRAIN thing by default. eval_enable turns it on for the eval
        # split too, which is only useful for test-time augmentation: score the same
        # checkpoint several times under mild transforms and average. The eval rng below
        # keys off self.seed, so a TTA pass just varies reproducibility.seed.
        _aug_on = cfg["augmentation"]["enable"] and (
            train or cfg["augmentation"].get("eval_enable", False))
        self.aug = cfg["augmentation"] if _aug_on else None
        self.seed = int(cfg["reproducibility"]["seed"])
        # 2.5D: instead of copying one slice into 3 identical channels, fill them with
        # neighbouring slices of the SAME series, so the backbone sees through-plane
        # context. Costs nothing extra - those channels already existed, just wasted.
        self.ctx = 3 if i.get("slice_context") == "neighbours" else 1
        self.ctx_stride = int(i.get("slice_context_stride", 1))
        # (index, {series: [files]}) at submission time: the same volumes come from
        # raw DICOM instead of a shard, everything downstream is identical
        self.dicom_files = None
        if dicom is not None:
            dicom_index, self.dicom_files = dicom
            self.geo = dicom_index.set_index("series")[["pixel_spacing_c", "pixel_spacing_r"]]
            return
        # per-series mm-per-pixel, needed to rescale every scan to one physical scale
        self.geo = (pd.read_csv(resolve(cfg["paths"]["series_index"], cfg))
                    .set_index("series")[["pixel_spacing_c", "pixel_spacing_r"]])

    def __len__(self):
        return len(self.df)

    def _read_series(self, study: str, series: str) -> np.ndarray | None:
        """The stored volume for one series, from a shard or straight from DICOM."""
        if self.dicom_files is not None:                  # submission path
            arr = dicom_series_volume(self.dicom_files.get(series, []))
            return arr if arr.size else None
        with np.load(self.shards / f"{study}.npz") as z:
            return z[series] if series in z.files else None

    def _load(self, study: str, series: str, rng) -> np.ndarray:
        """One series -> (n_slices * ctx, size, size) float array, resampled and windowed."""
        _t = time.perf_counter()
        arr = self._read_series(study, series)
        if arr is None:                                   # series missing for this study
            return np.zeros((self.n_slices * self.ctx, self.size, self.size), np.float32)
        idx = pick_slices(arr.shape[0], self.n_slices, self.mode, rng, self.margin)
        if self.ctx > 1:
            # k-stride, k, k+stride, clamped at the ends; flattened slice-major so a
            # later reshape gives (n_slices, ctx, H, W)
            off = np.arange(-(self.ctx // 2), self.ctx // 2 + 1) * self.ctx_stride
            idx = np.clip(np.asarray(idx)[:, None] + off[None, :],
                          0, arr.shape[0] - 1).reshape(-1)
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
            n_sl = len(vol) // self.ctx               # groups, not raw planes
            k = max(1, int(n_sl * sd["frac"]))
            drop = rng.choice(n_sl, k, replace=False)
            for d0 in drop:                           # zero every channel of that slice
                vol[d0 * self.ctx:(d0 + 1) * self.ctx] = 0
        return np.clip(vol, 0, 255)

    def __getitem__(self, i: int):
        """Build one study's tensor: stack the 4 slot volumes, attach its 12 labels."""
        row = self.df.iloc[i]
        study = row["StudyInstanceUID"]                 # one study = one patient's knee exam
        # every item's randomness is a pure function of (seed, epoch, index) for train
        # and (seed, study) for val/audit - so a run is reproducible and a resumed run
        # sees the same stream it would have seen uninterrupted
        rng = np.random.default_rng([self.seed, int(_EPOCH.value), i] if self.train
                                    else [self.seed, hash(study) & 0xFFFFFFFF])
        chosen = self.plan.get(study, {})               # {slot: series} picked by plan_slots

        vols = []
        for slot in self.slots:
            series = chosen.get(slot)                   # None when this study lacks that view
            blank = np.zeros((self.n_slices * self.ctx, self.size, self.size), np.float32)
            if series is None:
                v = blank
            else:
                try:
                    v = self._load(study, series, rng)
                except Exception as e:
                    # a worker that raises kills the whole DataLoader, and at submission
                    # time that costs every study, not just this one. an unreadable
                    # series becomes an empty slot, exactly like a missing one.
                    if not self.train:
                        print(f"  ⚠️  {study[:16]}… {slot}: "
                              f"{type(e).__name__}: {e} -> blank slot", flush=True)
                        v = blank
                    else:
                        raise
            if self.aug is not None:
                v = self._augment(v, rng)             # same warp across a slice's channels
            if self.ctx > 1:
                v = v.reshape(self.n_slices, self.ctx, self.size, self.size)
            vols.append(v)

        # (slots, D, H, W) normally, (slots, D, ctx, H, W) with 2.5D
        x = torch.from_numpy(np.stack(vols).astype(np.uint8))
        y = torch.tensor([float(row[t]) for t in self.tasks], dtype=torch.float32)   # 12 probs in [0,1]
        # "hard" mode turns the soft label probabilities into crisp 0/1 targets
        if self.cfg["target"]["mode"] == "hard":
            thr, ls = self.cfg["target"]["hard_threshold"], self.cfg["target"]["label_smoothing"]
            y = (y > thr).float() * (1 - ls) + ls / 2
        return x, y


def prepare_batch(x: torch.Tensor, cfg: dict, device) -> torch.Tensor:
    """uint8 (B, slots, D, H, W) -> normalized float (B*slots*D, C, H, W)."""
    # ship uint8 to the GPU (4x smaller transfer), normalize there
    c = cfg["input"]["channels"]
    x = x.to(device, non_blocking=True).float().div_(255.0)
    if x.dim() == 6:                               # 2.5D: channels already hold neighbours
        b, s, d, ctx, h, w = x.shape
        x = x.reshape(b * s * d, ctx, h, w)
    else:
        b, s, d, h, w = x.shape
        x = x.view(b * s * d, 1, h, w)             # every slice becomes its own 2D image
        if c > 1:
            x = x.expand(-1, c, -1, -1)            # grey copied to 3 channels for ImageNet backbones
    mean = torch.tensor(cfg["input"]["norm_mean"][:c], device=device).view(1, c, 1, 1)
    std = torch.tensor(cfg["input"]["norm_std"][:c], device=device).view(1, c, 1, 1)
    return (x - mean) / std


class ResumableSampler(Sampler):
    """Shuffling sampler whose permutation is a pure function of (seed, epoch).

    torch's default RandomSampler draws from the global RNG, so the order an epoch
    sees depends on everything consumed before it — a resumed run gets a different
    permutation and there is no way to say "continue where you stopped".

    Here each epoch's order is torch.randperm under a generator seeded seed+epoch,
    identical whether or not the run was interrupted. set_epoch(epoch, skip=k) then
    drops the first k indices, so the workers never even load the studies already
    trained on this epoch, and the epoch finishes on exactly the tail it would have.
    The skip clears after that one epoch.

    __len__ stays the full epoch size so steps_per_epoch and the lr schedule are
    unaffected; only the resumed epoch yields fewer batches.
    """

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.skip = int(n), int(seed), 0, 0

    def set_epoch(self, epoch: int, skip: int = 0):
        self.epoch, self.skip = int(epoch), max(0, int(skip))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        if self.skip:
            perm = perm[self.skip:]
            self.skip = 0
        return iter(perm)

    def __len__(self):
        return self.n


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
        # train shuffles through ResumableSampler so a mid-epoch resume can skip
        # exactly the studies it already saw; val keeps its fixed order
        smp = ResumableSampler(len(ds), cfg["reproducibility"]["seed"]) if train else None
        out[split] = DataLoader(
            ds, batch_size=train_batch_size(cfg) if train else vbs,
            sampler=smp, shuffle=False,
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
        # 58 studies is ~15 batches, so vnw (24 here) would spawn two dozen processes
        # for one batch each, at every validation, against the train loader's workers.
        anw = max(0, min(vnw, 4))
        out["audit"] = DataLoader(KneeShardDataset(df, cfg, plan, False),
                                  batch_size=vbs, shuffle=False, num_workers=anw,
                                  pin_memory=dl.get("pin_memory", True),
                                  persistent_workers=anw > 0,
                                  worker_init_fn=seed_worker,
                                  prefetch_factor=(dl.get("prefetch_factor", 2)
                                                   if anw > 0 else None))
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
                            extra = [c for c in fields if c not in old]
                            if extra:
                                # The run now logs columns this file predates - a new loss
                                # reporting its internals, say. Keeping the old header would
                                # silently DROP them (extrasaction="ignore"), so widen the
                                # file instead: rewrite it once with the union header and
                                # blanks for the rows that never had these values.
                                self._migrate(old, old + extra)
                                fields = old + extra
                                print(f"  📝 {self.path.name}: added {len(extra)} new "
                                      f"column(s), existing rows back-filled blank")
                            else:
                                print(f"  [warn] {self.path.name} has a different header, "
                                      f"appending in its column order")
                                fields = old
                    except Exception as e:
                        print(f"  [warn] header check on {self.path.name} failed: "
                              f"{type(e).__name__}: {e}")
                self._f = open(self.path, "a", newline="", encoding="utf-8")
                self._w = csv.DictWriter(self._f, fieldnames=fields, restval="",
                                         extrasaction="ignore")
                if new:
                    self._w.writeheader()
            else:
                # Columns can appear MID-RUN: gate_* rows are only built on the diagnostic
                # cadence, so row 1 does not have them and the header locked without them.
                # DictWriter(extrasaction="ignore") would drop them silently for the whole
                # run - which is exactly how mricore_vitb_cross_attention_ft lost every
                # gamma value it printed. Widen the file instead, same as on resume.
                late = [k for k in row if k not in self._w.fieldnames]
                if late:
                    fields = list(self._w.fieldnames) + late
                    self._f.close()
                    self._migrate(list(self._w.fieldnames), fields)
                    self._f = open(self.path, "a", newline="", encoding="utf-8")
                    self._w = csv.DictWriter(self._f, fieldnames=fields, restval="",
                                             extrasaction="ignore")
                    print(f"  📝 {self.path.name}: added {len(late)} new "
                          f"column(s) mid-run, earlier rows back-filled blank")
            self._w.writerow(row); self._f.flush()
        except Exception as e:                # logging must never crash training
            print(f"  [warn] CSV log failed: {type(e).__name__}: {e}")
            self._f, self._w = None, None

    def _migrate(self, old: list, new: list):
        """Rewrite the CSV under a wider header, via a temp file so a kill cannot eat it."""
        tmp = self.path.with_suffix(".migrating")
        with open(self.path, newline="", encoding="utf-8") as fh, \
                open(tmp, "w", newline="", encoding="utf-8") as out:
            r = csv.DictReader(fh)
            w = csv.DictWriter(out, fieldnames=new, restval="", extrasaction="ignore")
            w.writeheader()
            for row in r:
                w.writerow(row)
        tmp.replace(self.path)

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


class AUCMOfficial(nn.Module):
    """LibAUC's MultiLabelAUCMLoss, wired to this engine's conventions.

    Faithful to the published setup, which our earlier `aucm` was NOT:
      * the loss is libauc's own, min-max with learnable per-task a, b and dual alpha
      * it is driven by PESG, not AdamW - see build_optimizer
      * PESG finds the duals by attribute, so a/b/alpha/margin are re-exported here
      * libauc takes PROBABILITIES, so sigmoid is applied before the call
      * libauc masks with `1 == y_true`, so targets MUST be binary. Our soft LLM
        targets would make every pos_mask all-zero and the loss silently meaningless,
        so they are thresholded here. That is a real change to the objective and the
        run config says so.
    """

    def __init__(self, n_tasks: int, margin: float = 1.0, thr: float = 0.5,
                 imratio=None, version: str = "v1", device=None):
        super().__init__()
        from libauc.losses import MultiLabelAUCMLoss
        self.inner = MultiLabelAUCMLoss(margin=margin, version=version,
                                        imratio=imratio, num_labels=n_tasks,
                                        device=device)
        self.thr = float(thr)
        self.margin = margin

    # PESG does `self.a = loss_fn.a` etc, so they have to be reachable from the wrapper
    @property
    def a(self): return self.inner.a
    @property
    def b(self): return self.inner.b
    @property
    def alpha(self): return self.inner.alpha

    def forward(self, logits, target):
        return self.inner(torch.sigmoid(logits), (target > self.thr).float())

    # a, b and alpha are LEARNED, and libauc holds them as bare requires_grad tensors
    # rather than nn.Parameters - so model.state_dict() does not contain them and a
    # plain resume would silently restart the loss's own state at zero while the
    # weights carried on. a/b are the per-task positive/negative score means and alpha
    # is the dual variable; losing them mid-run resets the min-max problem underneath a
    # model that has already moved. Fetched by attribute so this works whichever type
    # the installed libauc uses.
    _STATE = ("a", "b", "alpha")

    def state(self):
        out = {}
        for k in self._STATE:
            t = getattr(self.inner, k, None)
            if torch.is_tensor(t):
                out[k] = t.detach().cpu().clone()
        return out

    def stats(self, tasks):
        """Flat per-task scalars for the CSV: the objective's own internal state.

        alpha is the dual. Pinned at 0 means the max player never engages and the loss
        has collapsed to a plain squared error; growing without bound means the min-max
        is diverging. b - a is the margin the model has actually opened between the
        negative and positive score means, which is the quantity AUC-M is maximizing.
        Without these a plateau is indistinguishable from a broken objective.
        """
        st = self.state()
        row = {}
        for k, v in st.items():
            f = v.flatten()
            row[f"aucm_{k}_mean"] = float(f.mean())
            row[f"aucm_{k}_min"] = float(f.min())
            row[f"aucm_{k}_max"] = float(f.max())
            if len(f) == len(tasks):
                for t, x in zip(tasks, f.tolist()):
                    row[f"aucm_{k}__{t}"] = float(x)
        a, b = st.get("a"), st.get("b")
        if a is not None and b is not None and a.numel() == b.numel():
            gap = (b - a).flatten()
            row["aucm_gap_mean"] = float(gap.mean())
            if len(gap) == len(tasks):
                for t, x in zip(tasks, gap.tolist()):
                    row[f"aucm_gap__{t}"] = float(x)
        return row

    def load_state(self, d):
        for k, v in (d or {}).items():
            t = getattr(self.inner, k, None)
            if torch.is_tensor(t) and torch.is_tensor(v) and t.shape == v.shape:
                with torch.no_grad():
                    t.copy_(v.to(t.device))


def build_loss(cfg: dict, train_df: pd.DataFrame | None, device) -> nn.Module:
    """Plain or focal binary cross-entropy, optionally re-weighted for rare labels."""
    lc = cfg["training"]["loss"]
    pw = None
    if lc.get("pos_weight") in ("auto", "auto_clamped") and train_df is not None:
        # rare labels get a bigger weight so they aren't drowned out by the common ones.
        # Targets are soft LLM probabilities, so "prevalence" is a mean probability, not
        # a positive count - which is why Fracture lands at 7.6x rather than ~16x.
        prev = train_df[cfg["tasks"]].mean().values
        w = (1 - prev) / np.maximum(prev, 1e-3)
        if lc["pos_weight"] == "auto_clamped":
            # floor at 1.0: boost the rare findings, but never DOWN-weight a common one.
            # Only Effusion is above 50% prevalence, and it is our strongest finding
            # (AUROC 0.89) - nothing to gain there and something to lose.
            w = np.maximum(w, 1.0)
        pw = torch.tensor(w, dtype=torch.float32, device=device)
    if lc["name"] == "focal":
        return FocalBCE(lc.get("focal_gamma", 2.0), pw).to(device)
    if lc["name"] == "aucm_official":
        return AUCMOfficial(len(cfg["tasks"]), lc.get("aucm_margin", 1.0),
                            float(cfg.get("target", {}).get("hard_threshold", 0.5)),
                            lc.get("aucm_imratio"), lc.get("aucm_version", "v1"),
                            device).to(device)
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


class _NullScheduler:
    """A scheduler that truly does nothing - PESG owns the learning rate.

    LambdaLR(lambda: 1.0) is NOT a no-op: it recomputes lr = base_lr * 1.0 on every
    step, so it silently reverts PESG's stagewise drop the instant it happens. Measured:
    update_regularizer(3) took 0.01 -> 0.00333, and the next sched.step() put it back to
    0.01. Both decays would have been dead and the run would have trained at a constant
    lr while printing the drops.

    Only three methods are ever called on a scheduler here: step, get_last_lr and
    state_dict.
    """

    def __init__(self, opt):
        self.opt = opt

    def step(self, *a, **k):
        return None

    def get_last_lr(self):
        return [g["lr"] for g in self.opt.param_groups]

    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        return None


def build_optimizer(model: nn.Module, cfg: dict, loss_fn: nn.Module | None = None):
    """AdamW with two learning rates.

    `optimizer.lr` is the BACKBONE rate — the pretrained trunk is ~99.9% of the
    parameters, so that is the number worth setting directly. The fresh head is
    derived: head_lr = lr / backbone_lr_mult (mult 0.5 -> head runs 2x faster).
    """
    o = cfg["training"]["optimizer"]
    # AUC-M is a min-max objective and libauc ships PESG for it: a proximal epoch
    # stochastic gradient method that pulls the weights toward a periodically-reset
    # reference point (epoch_decay), ascends the dual alpha with projection to
    # non-negative, and drops the lr stagewise via update_regularizer. AdamW is not
    # built for that saddle point - running the surrogate under AdamW is what the
    # earlier convnext_base_aucm did, and it is NOT the published method.
    # PESG reads a/b/alpha off loss_fn by attribute, so it has to be handed the loss.
    if isinstance(loss_fn, AUCMOfficial):
        from libauc.optimizers import PESG
        pe = o.get("pesg", {}) or {}
        opt = PESG(model.parameters(), loss_fn=loss_fn, lr=o["lr"],
                   mode=pe.get("mode", "sgd"),          # 'sgd' as published, 'adam' also supported
                   momentum=pe.get("momentum", 0.9),
                   clip_value=pe.get("clip_value", 1.0),
                   epoch_decay=pe.get("epoch_decay", 2e-3),   # = 1/gamma
                   weight_decay=o.get("weight_decay", 1e-5),
                   verbose=True, device=None)
        print(f"  🔁 PESG  lr={o['lr']:.2e}  mode={pe.get('mode','sgd')}  "
              f"epoch_decay={pe.get('epoch_decay', 2e-3)}  margin={loss_fn.margin}  "
              f"weight_decay={o.get('weight_decay', 1e-5)}")

        # Per-layer rates, via gradients rather than param groups. PESG does not accept
        # param groups at all - it walks the params expecting tensors and dies on a dict
        # with "'dict' object has no attribute 'shape'" - so the usual two-lr split is
        # unavailable. For SGD-with-momentum the update is linear in the gradient, so
        # scaling a parameter's grad by c is exactly scaling its lr by c.
        #
        # Why it is needed: the AUC stage reinitializes the head, which leaves a RANDOM
        # 1024->12 layer that needs a big lr, bolted onto a CONVERGED trunk that needs a
        # tiny one. One flat rate cannot serve both, and both failures were measured:
        #   lr 1e-2  head learns, trunk drifts   -> plateaus ~0.82
        #   lr 1e-3  neither moves usefully      -> 0.4855 at the first validation
        # So the head keeps optimizer.lr and everything under .backbone is scaled down.
        #
        # Only the gradient term is affected. PESG's proximal pull gamma*(v - v_ref) and
        # its weight decay still act at full lr on the trunk - with gamma 2e-3 and wd 0
        # that is a ~2e-5 nudge toward a reference we refresh anyway, not drift.
        gs = float(pe.get("backbone_grad_scale", 1.0) or 1.0)
        ld = pe.get("backbone_layer_decay")     # None -> flat scale across the whole trunk
        bb = getattr(model, "backbone", None)
        if gs != 1.0 and bb is not None:
            def _scaler(c):
                return lambda g: g * c
            # gs caps the TOP of the trunk, and layer_decay walks it further down from
            # there, so no backbone layer ever exceeds gs and the early ones get far less.
            # Strictly more conservative than a flat gs, which is the point: the top of a
            # converged trunk is the part most able to drift.
            bd, nl = (_llrd_depths(bb) if ld else ({}, 1))
            top = nl - 1
            n, lo, hi = 0, float("inf"), 0.0
            for q_name, q in bb.named_parameters():
                if not q.requires_grad:
                    continue
                c = gs * (float(ld) ** (top - bd.get(q_name, top))) if ld else gs
                q.register_hook(_scaler(c))
                n += 1; lo = min(lo, c); hi = max(hi, c)
            if ld:
                print(f"  🔁 backbone grads x{hi:.3g} (top) down to x{lo:.3g} "
                      f"(first layer), decay {ld} over {nl} depths, {n} tensors")
                print(f"     effective trunk lr {o['lr'] * hi:.2e} .. {o['lr'] * lo:.2e} "
                      f"against the head's {o['lr']:.2e}")
            else:
                print(f"  🔁 backbone grads x{gs:g} on {n} tensors -> effective "
                      f"trunk lr {o['lr'] * gs:.2e} against the head's {o['lr']:.2e}")
        return opt
    lr = o["lr"]                                      # backbone
    mult = o.get("backbone_lr_mult", 1.0) or 1.0      # 0 would divide by zero
    head_lr = lr / mult
    ld = o.get("layer_decay")
    if ld:
        # LLRD replaces the flat mult: depths decay downward from the head, so the
        # head still sits at head_lr and `lr` stays "what the trunk's top layer gets"
        groups = _param_groups_llrd(model, head_lr, o.get("weight_decay", 0.05), float(ld))
        _print_llrd(groups, head_lr, float(ld), max(g["layer_id"] for g in groups))
        return torch.optim.AdamW(groups, lr=head_lr, weight_decay=o.get("weight_decay", 0.05))
    # split structurally, not by name: everything under .backbone is the pretrained
    # trunk, the rest is freshly initialized. Substring matching got this wrong -
    # "fc" also matches every mlp.fc1/fc2 inside a ConvNeXt block.
    has_bb = hasattr(model, "backbone")

    # A module can declare itself freshly initialised by setting `is_fresh = True`. Blocks
    # spliced INTO a pretrained trunk live under `backbone.` and would otherwise inherit the
    # trunk's learning rate - which is chosen for weights that are already converged and is
    # exactly wrong for weights that are still random. mricore_vitb_cross_attention_ft put 21
    # random tensors on a 6e-6 trunk rate that way.
    fresh_ids = set()
    for _, mod_ in model.named_modules():
        if getattr(mod_, "is_fresh", False):
            fresh_ids.update(id(q) for q in mod_.parameters())
    fresh_lr = o.get("fresh_lr")

    head, backbone, fresh = [], [], []
    n_head = n_bb = n_fr = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if fresh_lr and id(p) in fresh_ids:
            fresh.append(p); n_fr += p.numel(); continue
        is_bb = (n.startswith("backbone.") if has_bb
                 else not any(k in n for k in ("head", "classifier", "pool_attn")))
        (backbone if is_bb else head).append(p)
        if is_bb:
            n_bb += p.numel()
        else:
            n_head += p.numel()
    groups = [{"params": backbone, "lr": lr}, {"params": head, "lr": head_lr}]
    msg = (f"  \U0001F9EE lr groups: backbone {n_bb / 1e6:.2f}M @ {lr:.2e} (config lr)  |  "
           f"head {n_head / 1e6:.2f}M @ {head_lr:.2e}  (lr / {mult})")
    if fresh:
        groups.append({"params": fresh, "lr": float(fresh_lr)})
        msg += f"  |  fresh {n_fr / 1e6:.2f}M @ {float(fresh_lr):.2e}  (optimizer.fresh_lr)"
    elif fresh_ids and not fresh_lr:
        msg += (f"  |  \u26a0\ufe0f {len(fresh_ids)} tensors declare is_fresh but "
                f"optimizer.fresh_lr is unset - they are on the trunk rate")
    print(msg)
    return torch.optim.AdamW(groups, lr=head_lr, weight_decay=o.get("weight_decay", 0.05))


def warmup_steps(cfg: dict, total_steps: int) -> int:
    """How many steps the lr ramp lasts — same formula build_scheduler uses."""
    return max(1, int(total_steps * cfg["training"]["scheduler"].get("warmup_frac", 0.05)))


def build_scheduler(opt, cfg: dict, total_steps: int):
    """Learning-rate schedule: ramp up over the first few %, then cosine-decay to min_lr_mult."""
    s = cfg["training"]["scheduler"]
    warm = warmup_steps(cfg, total_steps)
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


class ScaleProbe:
    """Feature and logit scale at the head, every N steps.

    The diagnostic that was missing when dinov3_convnext_base underfitted by 0.11 AUROC.
    Its trunk was fine; its FEATURES were 4.8x larger than the ImageNet trunk's, because
    a backbone's final LayerNorm carries a learned gain that ships inside the downloaded
    weights and nothing in self-supervised pretraining ever pressured that gain toward a
    scale a linear head wants. Bigger features mean bigger logits (a head confident before
    it has learned anything, where the sigmoid slope is shallow) AND proportionally bigger
    head gradients, since dL/dW = delta @ f.T carries f. Neither is visible in the loss.

    Healthy: feature std near 1, logit |max| under ~2 early on. A feature std far from 1
    means the head's learning rate does not mean what the config says it means.
    """

    def __init__(self, model, every: int = 200):
        self.every = int(every)
        self.buf, self.handle, self.head_name = None, None, None
        head = None
        for n, m in model.named_modules():          # the last Linear IS the head
            if isinstance(m, nn.Linear):
                head, self.head_name = m, n
        if head is not None:
            self.handle = head.register_forward_hook(self._hook)

    def _hook(self, _mod, inp, out):
        if self.buf is not False:                   # False = not an active step
            f, o = inp[0].detach().float(), out.detach().float()
            self.buf = (float(f.std()), float(f.norm(dim=-1).mean()),
                        float(o.std()), float(o.abs().max()))

    def active(self, step: int) -> bool:
        return self.handle is not None and self.every > 0 and step % self.every == 0

    def arm(self, on: bool):
        self.buf = None if on else False            # None = armed, waiting for a forward

    def read(self):
        b = self.buf if isinstance(self.buf, tuple) else None
        self.buf = False
        return b

    @staticmethod
    def row(b):
        if not b:
            return {}
        return {"feat_std": round(b[0], 4), "feat_l2": round(b[1], 3),
                "logit_std": round(b[2], 4), "logit_absmax": round(b[3], 3)}

    @staticmethod
    def summarize(b):
        if not b:
            return ""
        fs, fl, ls, lm = b
        flag = ""
        if fs > 2.5 or fs < 0.15:
            flag = ("   <-- feature scale is far from 1: the head's effective lr is not "
                    "what the config says. Consider model.feat_norm")
        return (f"  \U0001F4CF scale  features std {fs:.3f}  L2 {fl:.1f}   |   "
                f"logits std {ls:.3f}  |max| {lm:.2f}{flag}")


class UDRatio:
    """log10 of the update-to-data ratio per weight matrix, the Karpathy diagnostic.

    The usual form is log10(lr * grad.std() / param.std()), which assumes the step IS
    lr*grad. Under PESG that is false - there is momentum, a proximal term and a
    weight-decay term - and under AMP the grads are still loss-scaled at that point.
    So this snapshots the parameter BEFORE the step and diffs it AFTER, which is the
    real update whatever the optimizer did to produce it.

    Roughly -3 is healthy. Much below and the layer is effectively frozen; much above
    and the step is large against the weights it is moving, which is how a trunk gets
    wrecked. Only 2D+ weights are tracked - biases and norms have their own scales and
    read as noise.
    """

    def __init__(self, model, every: int = 200, top_n: int = 8):
        self.every, self.top_n = int(every), int(top_n)
        self.params = [(n, q) for n, q in model.named_parameters()
                       if q.requires_grad and q.ndim >= 2]
        # HEAD and TRUNK are reported apart, because the healthy band differs. A
        # reinitialized head starts random and SHOULD sit at or above -3; a converged
        # pretrained trunk should only drift, so -5 or -6 is right for it, not a
        # problem. Pooling them hides both facts behind one median.
        self.head = {n for n, _ in self.params
                     if any(k in n.lower() for k in ("head", "classifier", ".fc."))
                     and "backbone" not in n.lower()}
        self.snap, self.rows = None, []

    def active(self, gstep: int) -> bool:
        return self.every > 0 and gstep % self.every == 0

    def before(self):
        self.snap = [q.detach().clone() for _, q in self.params]

    def after(self, epoch: int, gstep: int):
        if self.snap is None:
            return None
        out = []
        for (n, q), old in zip(self.params, self.snap):
            d = (q.detach() - old).std().item()
            w = q.detach().std().item()
            if w > 0 and d > 0:
                out.append((n, math.log10(d / w)))
        self.snap = None
        for n, r in out:
            self.rows.append({"epoch": epoch, "step": gstep, "param": n, "log10_ud": r})
        return out

    @staticmethod
    def _fmt(label, vals, want):
        v = sorted(vals)
        return ("     %-7s n=%-3d min %6.2f  mean %6.2f  median %6.2f  max %6.2f   %s"
                % (label, len(v), v[0], sum(v) / len(v), v[len(v) // 2], v[-1], want))

    def summarize(self, out):
        if not out:
            return ""
        hv = [r for n, r in out if n in self.head]
        bv = [r for n, r in out if n not in self.head]
        lines = ["  📏 ud log10 (update / weight)"]
        if hv:
            lines.append(self._fmt("head", hv, "(random head: want ~-3 or above)"))
        if bv:
            lines.append(self._fmt("trunk", bv, "(pretrained: -5..-6 is drift, not frozen)"))
        # Only flag what is genuinely wrong for its own role.
        dead = [n for n, r in out if n not in self.head and r < -7.0]
        hot = [n for n, r in out if n not in self.head and r > -3.0]
        cold_head = [n for n, r in out if n in self.head and r < -4.5]
        if dead:
            lines.append("     ⚠️  trunk frozen (< -7): %d tensors, e.g. %s"
                         % (len(dead), dead[0]))
        if hot:
            lines.append("     ⚠️  trunk hot (> -3, may damage the pretrained "
                         "features): %d tensors, e.g. %s" % (len(hot), hot[0]))
        if cold_head:
            lines.append("     ⚠️  head barely moving (< -4.5): %s" % cold_head[0])
        return chr(10).join(lines)

    def dump(self, path):
        """MERGE with what is already there, never truncate.

        self.rows only holds this process's samples, so a plain to_csv would wipe every
        earlier epoch on the first save after a workspace hand-off. Dedupes on
        (step, param) so a replayed mid-epoch resume cannot double-count.
        """
        if not self.rows:
            return
        try:
            df = pd.DataFrame(self.rows)
            if path.exists():
                try:
                    df = pd.concat([pd.read_csv(path), df], ignore_index=True)
                except Exception as e:
                    print(f"  [warn] {path.name} unreadable, starting fresh: "
                          f"{type(e).__name__}: {e}")
            df = (df.drop_duplicates(subset=["step", "param"], keep="last")
                    .sort_values(["step", "param"]))
            df.to_csv(path, index=False)
        except Exception as e:
            print(f"  [warn] could not write {path.name}: {type(e).__name__}: {e}")


def save_checkpoint(path: Path, cfg, model, opt=None, sched=None, scaler=None,
                    epoch=0, step=0, score=None, track=None, device=None,
                    weights_only=False, elapsed_sec=0.0, epoch_complete=True,
                    saver=None, note=None, loss_fn=None):
    """Full state by default, so a run can resume exactly where it stopped.

    epoch_complete=False marks a checkpoint written part-way through an epoch, so
    resume replays that epoch instead of silently skipping the rest of its data.
    """
    payload = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
               "step": step, "score": score, "track": track,
               "elapsed_sec": elapsed_sec, "epoch_complete": epoch_complete}
    if not weights_only:
        if loss_fn is not None and hasattr(loss_fn, "state"):
            payload["loss"] = loss_fn.state()      # AUC-M's learned a / b / alpha
        if opt is not None:
            payload["optimizer"] = opt.state_dict()
        if sched is not None:
            payload["scheduler"] = sched.state_dict()
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        if device is not None:
            payload["rng"] = _rng_state(device)
    if saver is not None:
        # freeze it here, on the training thread, THEN hand it off. Serialising live
        # tensors on a worker while the next step mutates them is the way this goes
        # wrong; a host-RAM copy costs ~0.2s against ~10s of torch.save.
        return saver.write(_cpu_snapshot(payload), path, note)
    _write_checkpoint(payload, path)
    if note:
        print(note)
    return path


def _write_checkpoint(payload: dict, path: Path) -> Path:
    """torch.save via a temp file, so a kill mid-write cannot leave a half checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)                          # atomic: never leave a half-written ckpt
    return path


def _cpu_snapshot(x):
    """Deep-copy every tensor to host RAM, leaving plain python objects alone."""
    if torch.is_tensor(x):
        return x.detach().to("cpu", copy=True)
    if isinstance(x, dict):
        return {k: _cpu_snapshot(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu_snapshot(v) for v in x)
    return x


class AsyncSaver:
    """Write checkpoints on a worker thread, from a frozen CPU copy.

    Base writes last.pt + best.pt + a top-k file at every validation (~2.4 GB) and a
    1 GB periodic checkpoint every 100 steps. Inline, the GPU idles through all of it.

    Only ONE write runs at a time, and starting a new one first joins the previous:
    that keeps file operations ordered, so pruning and top-k eviction can never race
    a write. In practice it never blocks - saves are minutes apart and take seconds.
    """

    def __init__(self, enabled: bool = True):
        self.enabled, self.thread = bool(enabled), None
        self.count, self.error = 0, None

    def write(self, payload: dict, path: Path, note: str | None = None) -> Path:
        if not self.enabled:
            _write_checkpoint(payload, path)
            if note:
                print(note)
            return path
        self.join()
        import threading
        self.thread = threading.Thread(target=self._run, args=(payload, path, note),
                                       daemon=True)
        self.thread.start()
        return path

    def _run(self, payload, path, note=None):
        try:
            _write_checkpoint(payload, path)
            self.count += 1
            if note:
                # printed HERE, not at the call site: the point of the line is that the
                # file is on the volume, and with an async write the call site reaches
                # it seconds early. A claim of "saved" has to mean saved.
                print(note, flush=True)
        except Exception as e:                 # a failed write must not kill training
            self.error = f"{path.name}: {type(e).__name__}: {e}"
            print(f"  ⚠️  checkpoint write failed, {self.error}", flush=True)

    def join(self, timeout: float = 900.0) -> None:
        """Block until nothing is in flight. Call before touching checkpoint files."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout)


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
        # newest of ALL of them, not last.pt by preference: last.pt is only written at
        # a validation, so it can trail the newest step*.pt by a whole eval interval.
        # This is the preemption-safe setting - Modal restarts a preempted container on
        # the same input, and a hardcoded step number makes it replay from there forever.
        cands = list(ckpt_dir.glob("step*.pt"))
        pth = ckpt_dir / "last.pt"
        if pth.exists():
            cands.append(pth)
        return max(cands, key=lambda q: q.stat().st_mtime) if cands else None
    if isinstance(resume, int) or str(resume).isdigit():
        pth = ckpt_dir / f"step{int(resume)}.pt"
        return pth if pth.exists() else None
    return None


def load_checkpoint(resume, model, ckpt_dir: Path, opt=None, sched=None,
                    scaler=None, device=None, cfg_now: dict | None = None) -> dict:
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
    # the cosine schedule spans epochs x steps_per_epoch, so resuming under a
    # different `epochs` silently trains on a different lr curve
    was = ((ck.get("cfg") or {}).get("training") or {}).get("epochs")
    now = ((cfg_now or {}).get("training") or {}).get("epochs")
    if was is not None and now is not None and was != now:
        print(f"  ⚠️  epochs changed {was} -> {now}: the lr curve is NOT the "
              f"one this checkpoint was trained under")
    return ck


def save_predictions(pred_dir: Path, split: str, run_id: str, epoch: int, step: int,
                     score, logits, targets, study_ids, tasks) -> Path | None:
    """One .npz of raw per-study logits, so ensembling later needs no GPU at all.

    704 studies x 12 findings of float32 is 34 KB, and the 58-row panel is 3 KB, so a
    whole run's worth is under a megabyte - cheap enough to keep every validation
    rather than guessing now which checkpoints we will want to combine later.

    study_ids are stored because that is what makes the rows alignable: across
    checkpoints, and across runs. Both eval loaders are shuffle=False, so row order
    follows dataset.df, but relying on that silently would be a trap the day it changes.
    """
    try:
        pred_dir.mkdir(parents=True, exist_ok=True)
        f = pred_dir / f"{run_id}_{split}_s{step}.npz"
        np.savez_compressed(
            f, logits=np.asarray(logits, np.float32),
            targets=np.asarray(targets, np.float32),
            study_ids=np.asarray(study_ids, dtype=object),
            tasks=np.asarray(list(tasks), dtype=object),
            meta=np.asarray([run_id, split, str(epoch), str(step),
                             "" if score is None else f"{score:.6f}"], dtype=object))
        return f
    except Exception as e:                    # bookkeeping must never kill a run
        print(f"  ⚠️  could not save {split} predictions: {type(e).__name__}: {e}")
        return None


def prune_checkpoints(ckpt_dir: Path, keep_last_n: int, keep_names=(), saver=None):
    """Rolling window over the periodic step checkpoints."""
    if keep_last_n <= 0:
        return
    # deliberately does NOT wait on an in-flight write. It only ever unlinks files
    # OLDER than the newest, writes are serialised so anything older is complete, and
    # an unfinished write is a .tmp that this glob cannot see. Joining here would undo
    # the async saving entirely, since prune runs right after the periodic checkpoint.
    files = sorted(ckpt_dir.glob("step*.pt"), key=lambda q: q.stat().st_mtime)
    for f in files[:-keep_last_n]:
        if f.name not in keep_names:
            mb = f.stat().st_size / 1024**2
            try:
                f.unlink()
                print(f"  🗑️  pruned {f.name} ({mb:.0f} MB) - "
                      f"outside the keep_last_n={keep_last_n} window")
            except OSError as e:
                print(f"  [warn] could not prune {f.name}: {e}")


def update_top_k(ckpt_dir: Path, cfg, model, epoch: int, step: int, score: float,
                 track: list, k: int, weights_only: bool = True, saver=None) -> list:
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
                        step=step, score=score, weights_only=weights_only, saver=saver)
    # same reasoning as prune_checkpoints: everything in `drop` is from an earlier
    # step than the one just written, and writes are serialised, so none of it is
    # still open. The current step is only saved when it survives into `keep`.
    for r in drop:
        f = ckpt_dir / r["file"]
        if f.exists():
            try:
                f.unlink()
                print(f"  🗑️  evicted {f.name} (score {r['score']:.4f}, "
                      f"epoch {r['epoch']}) - pushed out of the top-{k}")
            except OSError as e:
                print(f"  [warn] could not evict {f.name}: {e}")
    (ckpt_dir / "top_k.json").write_text(json.dumps(keep, indent=2))
    return keep


def seed_worker(worker_id: int):
    """Give every dataloader worker its own reproducible stream."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


# =============================================================================
# eval / train
# =============================================================================
def eval_interval(cfg: dict, epoch: int) -> int:
    """Steps between validations for this epoch - 0 disables step-triggered eval.

    Cadence is driven by the GLOBAL STEP, not the epoch. The epoch only selects
    which interval applies: early epochs move fast and are not worth scoring often,
    later ones are where the best checkpoint usually shows up.
    eval_every_steps_epoch_threshold is the first epoch that uses the "after" rate.
    """
    e = cfg["evaluation"]
    thr = e.get("eval_every_steps_epoch_threshold")
    if thr:
        key = "eval_every_steps_after" if epoch >= int(thr) else "eval_every_steps_before"
        return int(e.get(key) or 0)
    return int(e.get("eval_every_steps") or 0)


@torch.no_grad()                                   # no gradients: eval only, saves memory
def evaluate(model, loader, loss_fn, cfg, device, collect: dict | None = None) -> dict:
    """Run the whole loader and return macro AUROC, per-label AUROC and mean loss.

    Pass `collect` to also keep the raw per-study outputs. They are computed either
    way - this only stops throwing them away - and they are what makes checkpoint
    ensembling free later: averaging saved logits costs no GPU at all.
    """
    model.eval()
    all_logits, all_predictions, all_targets, batch_losses = [], [], [], []
    use_amp = cfg["training"]["amp"] and device.type == "cuda"
    for images, targets in loader:
        targets = targets.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(prepare_batch(images, cfg, device))
            batch_losses.append(loss_fn(logits, targets).item())
        lg = logits.float().cpu().numpy()
        if collect is not None:
            all_logits.append(lg)
        # sigmoid turns raw logits into 0-1 probabilities
        all_predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    metrics = compute_metrics(np.concatenate(all_targets), np.concatenate(all_predictions), cfg["tasks"])
    metrics["loss"] = float(np.mean(batch_losses))
    metrics["sample_count"] = int(np.concatenate(all_targets).shape[0])
    if collect is not None:
        # logits, not probabilities: averaging in logit space is the usual way to
        # ensemble, and the probabilities are one sigmoid away whenever we want them
        collect["logits"] = np.concatenate(all_logits).astype(np.float32)
        collect["targets"] = np.concatenate(all_targets).astype(np.float32)
    return metrics


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
    try:                                   # cpu.max = "quota period"; quota/period = real cores
        with open("/sys/fs/cgroup/cpu.max") as fh:
            q, per = fh.read().split()
            if q != "max":
                cpu_rows.append(("container CPU limit", f"{int(q) / int(per):.2f} cores"))
    except Exception:
        pass
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
                elapsed: float, eta: float, tasks: list, emoji: str = "\U0001F50E",
                prev: dict | None = None, took: float | None = None):
    """One validation block: headline metrics, then every finding.

    `prev` is this set's metrics from the last time IT was scored, so the deltas
    compare like with like — val against val, audit against audit. `took` is how
    long this pass alone ran, separate from the run's total elapsed.
    """
    def d(key):
        return "" if not prev or key not in prev else f" ({m[key] - prev[key]:+.4f})"

    t_str = "" if took is None else f"  \U000023F1️ took={fmt(took)}"
    print("-" * 70)
    print(f"  {emoji} [{tag}] epoch={epoch}/{epochs} step={gstep}  n={m['sample_count']}  "
          f"loss={m['loss']:.4f}{d('loss')}{t_str}  elapsed={fmt(elapsed)}  ETA={fmt(eta)}")
    print(f"     \U0001F4CA macro AUROC={m['macro_auroc']:.4f}{d('macro_auroc')}  "
          f"AUPRC={m['macro_auprc']:.4f}{d('macro_auprc')}  "
          f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
          f"Spec={m['specificity']:.3f}")
    for t in tasks:
        v = m["per_task"][t]
        # a finding with no positives in the split has no defined AUROC/AUPRC
        au = "   n/a" if v["auroc"] != v["auroc"] else f"{v['auroc']:.4f}"
        ap = "   n/a" if v["auprc"] != v["auprc"] else f"{v['auprc']:.4f}"
        print(f"        {t:18s} AUROC={au}  AUPRC={ap}  (pos={v['pos']})")


class AsyncPersist:
    """Flush the Modal volume off the training thread.

    volume.commit() is a blocking network flush. Called inline every N steps it stalls
    the GPU for its whole duration - visible on the Modal dashboard as utilisation
    dropping to zero on exactly the commit period. The data is already written to the
    container filesystem by then; commit only publishes it, so nothing is lost by
    doing it slightly later.

    One worker at a time: if a commit is still running when the next is due, skip it
    rather than queue - the next one publishes the same files plus more. Overlapping
    commits are also what the volume API is least happy about.
    """

    def __init__(self, fn):
        self.fn, self.thread, self.skipped, self.done = fn, None, 0, 0
        self.last_error = None

    def __call__(self, blocking: bool = False):
        if self.fn is None:
            return
        if blocking:
            self.join()
            self._run()
            return
        if self.thread is not None and self.thread.is_alive():
            self.skipped += 1                  # a flush is already in flight
            return
        import threading
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.fn()
            self.done += 1
        except Exception as e:                 # a failed flush must not kill training
            self.last_error = f"{type(e).__name__}: {e}"

    def join(self, timeout: float = 300.0):
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout)


def _run_experiment(cfg: dict, model: nn.Module, out_dir: Path, persist_fn=None, log_fn=None):
    """Full training run: loaders, loss, optimizer, epoch loop, eval, checkpoints, logs."""
    persist = AsyncPersist(persist_fn)         # volume flushes, off the training thread
    saver = AsyncSaver(cfg["output"].get("async_checkpoints", True))
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
    # Trade compute for VRAM: don't keep intermediate activations, recompute them in the
    # backward pass. Roughly 30-40% slower per step, but activation memory drops by close
    # to an order of magnitude - which is the only way a 96-image study fits on a small
    # card. timm exposes this on the backbone; a plain nn.Module may not have it at all.
    if cfg["training"].get("grad_checkpointing"):
        _bb = getattr(model, "backbone", model)
        if hasattr(_bb, "set_grad_checkpointing"):
            _bb.set_grad_checkpointing(True)
            print("  🧮 gradient checkpointing ON (slower steps, far less VRAM)")
        else:
            print(f"  ⚠️  grad_checkpointing requested but {type(_bb).__name__} "
                  f"has no set_grad_checkpointing - ignored")
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
    opt = build_optimizer(model, cfg, loss_fn)
    accum = accum_steps(cfg)
    steps_per_epoch = max(1, len(loaders["train"]) // accum)
    total = steps_per_epoch * epochs
    _pesg = isinstance(loss_fn, AUCMOfficial)
    if _pesg:
        # PESG decays the lr ITSELF, stagewise, inside update_regularizer(decay_factor).
        # A cosine on top would fight it, so the schedule is a no-op here and the drops
        # come from training.optimizer.pesg.decay_epochs instead - the way libauc's own
        # CheXpert example runs it.
        sched = _NullScheduler(opt)
        warm = 0
        print("  scheduler disabled: PESG owns the lr, dropping it at epochs "
              f"{cfg['training']['optimizer'].get('pesg', {}).get('decay_epochs', [])}")
    else:
        sched = build_scheduler(opt, cfg, total)
        warm = warmup_steps(cfg, total)
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
    # Always on. There is no run for which "is every layer actually moving?" is not worth
    # knowing, and it is the one diagnostic that separates a frozen or throttled trunk from
    # a merely unlucky one - the loss curve shows them as the same thing. Cadence is
    # configurable, presence is not.
    _udc = cfg["training"].get("log_ud_ratio") or {}
    ud = UDRatio(model, _udc.get("every_steps", 200))
    print(f"  📏 update/weight ratio logging: {len(ud.params)} weight tensors, "
          f"every {ud.every} steps -> ud_ratio.csv")
    # Any gated branch in the model (CrossSliceBlock.gamma, LayerScale, ...). A gamma that
    # DECAYS is the model declining whatever the block offers, which is a real answer and
    # completely invisible in the loss - so it is reported, not inferred.
    _gates = [(n, q) for n, q in model.named_parameters() if n.endswith("gamma")]
    if _gates:
        print(f"  🚪 gates: {len(_gates)} gamma tensor(s) tracked "
              f"-> printed every {_udc.get('every_steps', 200)} steps")

    probe = ScaleProbe(model, _udc.get("every_steps", 200))
    if probe.handle is not None:
        print(f"  \U0001F4CF feature/logit scale probe on {probe.head_name!r}, "
              f"every {probe.every} steps")

    print(f"[setup] optimizer={'PESG' if _pesg else cfg['training']['optimizer']['name']}  "
          f"scheduler={'none (PESG stagewise)' if _pesg else cfg['training']['scheduler']['name']}  amp={scaler.is_enabled()}")
    print(f"[setup] steps/epoch={steps_per_epoch}  total_steps={total}  accum={accum}  "
          f"images/optimizer-step={imgs_per_batch * accum}")
    _vbs, _vnw = val_loader_settings(cfg)
    print(f"[setup] loss={cfg['training']['loss']['name']}  "
          f"train bs={train_batch_size(cfg)}/{cfg.get('dataloader', {}).get('num_workers', 4)}w  "
          f"val bs={_vbs}/{_vnw}w")

    es = cfg["training"]["early_stopping"]
    best, best_epoch, bad, gstep = -math.inf, 0, 0, 0
    start_epoch, top_k_track = 1, []
    # declared up here because the resume block below restores into them
    audit_best = [-math.inf, 0]   # [score, epoch] - a report only, never drives best.pt
    audit_bad = [0]               # audit's own patience count, shown but never acted on
    prev_eval = {}                # each set's last metrics, for the printed deltas
    ext_state = {}                # the extension in progress, so a CRASH mid-extension
                                  # resumes the SAME one instead of starting another
    # Under PESG neither the optimizer nor the scheduler state may be restored. Stage 1
    # was AdamW + cosine over TWO param groups (backbone/head); PESG has one, and its
    # no-op LambdaLR has one base_lr. Loading the old scheduler state makes base_lrs
    # length 2 against one lambda, and get_lr() dies with "zip() argument 2 is longer
    # than argument 1" on the first optimizer step. The weights are what we want from
    # this checkpoint; the optimizer state belongs to a different algorithm.
    ck = load_checkpoint(cfg.get("resume"), model, ckpt_dir,
                         None if _pesg else opt, None if _pesg else sched,
                         scaler, device, cfg_now=cfg)
    # Is this checkpoint one THIS stage wrote, or the stage-1 hand-off? Everything
    # below turns on the answer, and a workspace hand-off mid-run must land on the
    # "same stage" side or it would throw the run away and silently restart it.
    _ck_loss = (((ck.get("cfg") or {}).get("training") or {}).get("loss") or {}).get("name")
    _same_stage = bool(ck) and _ck_loss == cfg["training"]["loss"]["name"]

    # A checkpoint on the volume outlives the config that produced it. `resume: "last"` is
    # the right default (Modal preempts containers, and a hand-off must not restart the
    # run), but it means editing a setting and relaunching SILENTLY continues the old run
    # under the new label - the two halves trained under different settings and the result
    # describes neither. That is how the grad_clip fix nearly got lost: the fix was in the
    # config while the volume still held the throttled checkpoint.
    #
    # Preemption and hand-offs replay the SAME config, so they never trip this. Only an
    # actual edit does. Stage-2 hand-offs (a different loss) are already a deliberate
    # weights-only transfer, so they are exempt, and so is `extend`, whose whole job is to
    # change lr and epochs.
    if ck and _same_stage and not (cfg["training"].get("extend") or {}).get("enable"):
        _guard = [("model.name", ("model", "name")),
                  ("training.grad_clip", ("training", "grad_clip")),
                  ("training.epochs", ("training", "epochs")),
                  ("optimizer.lr", ("training", "optimizer", "lr")),
                  ("optimizer.backbone_lr_mult", ("training", "optimizer", "backbone_lr_mult")),
                  ("optimizer.layer_decay", ("training", "optimizer", "layer_decay")),
                  ("optimizer.weight_decay", ("training", "optimizer", "weight_decay")),
                  ("scheduler.name", ("training", "scheduler", "name")),
                  ("input.size", ("input", "size")),
                  ("input.n_slices", ("input", "n_slices")),
                  ("input.slots", ("input", "slots")),
                  ("target.mode", ("target", "mode")),
                  ("paths.train_csv", ("paths", "train_csv")),
                  ("paths.val_csv", ("paths", "val_csv"))]

        def _dig(d, path):
            for k in path:
                if not isinstance(d, dict) or k not in d:
                    return "<absent>"
                d = d[k]
            return d

        _ocfg = ck.get("cfg") or {}
        _diff = [(lbl, _dig(_ocfg, pth), _dig(cfg, pth)) for lbl, pth in _guard
                 if _dig(_ocfg, pth) != _dig(cfg, pth) and _dig(_ocfg, pth) != "<absent>"]
        if _diff:
            _w = max(len(l) for l, _, _ in _diff)
            _lines = "\n".join(f"      {l:<{_w}}  checkpoint={o!r}  ->  config={n!r}"
                                 for l, o, n in _diff)
            if cfg.get("resume_force"):
                print(f"\n  ⚠️  RESUMING ACROSS A CONFIG CHANGE (resume_force is set):"
                      f"\n{_lines}\n"
                      f"      The run trains part of itself under each. The result is not "
                      f"attributable to either config.\n")
            else:
                raise RuntimeError(
                    f"refusing to resume: this checkpoint (step {ck.get('step')}) was "
                    f"written under a different config.\n{_lines}\n\n"
                    f"    Continuing would train one half of the run under each setting, "
                    f"so the result would describe neither.\n"
                    f"    Pick one:\n"
                    f"      * start clean - delete the run's results/ on the volume:\n"
                    f"          modal volume rm -r {cfg['modal']['runs_volume']}"
                    f" /{cfg['experiment']['name']}/results\n"
                    f"      * continue the OLD run on a new lr curve - set "
                    f"training.extend.enable\n"
                    f"      * you really do mean it - set resume_force: true in the config")


    if _pesg and ck:
        if _same_stage:
            # Our own checkpoint: PESG's momentum buffers AND its proximal reference
            # v_ref are meaningful and must come back. v_ref especially - it is what
            # update_regularizer rewrites at the decay epoch, so dropping it would
            # restart the proximal term against the wrong anchor.
            if ck.get("optimizer"):
                try:
                    opt.load_state_dict(ck["optimizer"])
                    print("  ↩️  PESG state restored (momentum + proximal reference)")
                except Exception as e:
                    print(f"  [warn] could not restore PESG: {type(e).__name__}: {e}")
            if ck.get("loss") and hasattr(loss_fn, "load_state"):
                loss_fn.load_state(ck["loss"])
                print("  ↩️  AUC-M duals restored (a, b, alpha)")
        else:
            print(f"  ↩️  weights only: stage-1 checkpoint (loss={_ck_loss!r}), PESG "
                  f"and its schedule start fresh - the AdamW/cosine state is not portable")
    # The DAM paper's two-stage framework (sec 3.4) is explicit: the AUC stage starts
    # "for learning all layers with the last classifier layer RANDOMLY INITIALIZED", and
    # CheXpert sec 4.2.1 repeats it - "we replace the last classifier layer trained in
    # the first step by random weights". The pretrained trunk is what stage 1 is for; the
    # head is deliberately thrown away so the AUC objective shapes it from scratch.
    # Must run AFTER load_checkpoint, or it would reset weights the checkpoint then
    # overwrites - which is the bug this line replaced.
    if cfg["training"].get("reinit_head") and ck and not _same_stage:
        _done = []
        for _n in ("head", "classifier", "fc"):
            _m = getattr(model, _n, None)
            if isinstance(_m, nn.Module):
                for _sub in _m.modules():
                    if isinstance(_sub, (nn.Linear, nn.Conv2d)):
                        _sub.reset_parameters()
                        _done.append(f"{_n}.{type(_sub).__name__}")
        print(f"  🔄 head re-initialized for the AUC stage: {_done or 'NOTHING FOUND'}")
        # PESG snapshots its proximal reference v_ref at construction. That snapshot was
        # taken before the reinit, so without this the proximal term would pull the fresh
        # head back toward the discarded one - quietly undoing the reinit.
        if hasattr(opt, "model_ref"):
            try:
                _ps = [q for g in opt.param_groups for q in g["params"]]
                for _r, _q in zip(opt.model_ref, _ps):
                    _r.data = _q.data.clone()
                print("  🔄 PESG reference point refreshed onto the new head")
            except Exception as _e:
                print(f"  [warn] could not refresh the PESG reference: {type(_e).__name__}: {_e}")

    resumed_elapsed, history, resume_skip = 0.0, [], 0
    if ck:
        gstep = int(ck.get("step") or 0)
        ck_epoch = int(ck.get("epoch") or 0)
        tr = ck.get("track") or {}
        best = tr.get("best", -math.inf)
        best_epoch = tr.get("best_epoch", 0)
        bad = int(tr.get("bad", 0))               # keep counting toward patience
        top_k_track = tr.get("top_k", [])
        history = list(tr.get("history", []))     # so the final summary sees the whole run
        audit_best[:] = list(tr.get("audit_best", [-math.inf, 0]))
        audit_bad[0] = int(tr.get("audit_bad", 0))
        prev_eval.update(tr.get("prev_eval", {}))  # deltas survive a resume
        ext_state.update(tr.get("extend", {}) or {})
        resumed_elapsed = float(ck.get("elapsed_sec") or 0.0)
        if ck.get("epoch_complete", True):
            start_epoch = ck_epoch + 1
        else:
            # mid-epoch: skip the studies this epoch already trained on. Derived from
            # gstep, not a stored counter, so the skip always lands on an accumulation
            # boundary. ResumableSampler makes the epoch's order reproducible.
            start_epoch = ck_epoch
            done = gstep - (ck_epoch - 1) * steps_per_epoch
            resume_skip = max(0, done) * accum          # in micro-batches
            print(f"  ↩️  mid-epoch resume: epoch {ck_epoch} continues at "
                  f"optimizer step {done}/{steps_per_epoch} "
                  f"({resume_skip} micro-batches skipped, never re-loaded)")
        print(f"  ↩️  restored: best={best:.4f} @ epoch {best_epoch}  "
              f"patience={bad}/{es['patience']}  top_k={len(top_k_track)}  "
              f"elapsed={fmt(resumed_elapsed)}")

        # AFTER the restore, because the block above reassigns best/bad from the
        # checkpoint - doing this earlier is silently undone. A discarded head makes
        # this a materially different model: stage 1's 0.9050 is not a bar a RANDOM
        # classifier can clear inside `patience` validations, so the run would die in
        # its first epoch. The stage also starts on a clean epoch rather than finishing
        # stage 1's - best.pt was written mid-epoch and those 32 leftover steps belong
        # to the old objective.
        if cfg["training"].get("reinit_head") and not _same_stage:
            best, best_epoch, bad = -math.inf, 0, 0
            audit_best[:] = [-math.inf, 0]
            audit_bad[0] = 0
            start_epoch, resume_skip = ck_epoch + 1, 0
            print(f"  🔄 early-stopping state reset and epoch {ck_epoch} dropped: "
                  f"new head, new objective, so the old bar and the half-finished epoch "
                  f"do not carry. Starting clean at epoch {start_epoch}")

    # ---- extend: carry a FINISHED run on under a fresh lr curve ---------------
    # A plain resume restores the cosine schedule the checkpoint stopped under. Once
    # that schedule has run to its end the lr sits at min_lr_mult (1% of base), so
    # "resume" would technically continue and effectively train on nothing. `extend`
    # keeps the weights and the AdamW moments but hands the optimizer a NEW cosine
    # over the extra epochs, at a new peak lr - the cheap way to give a run that was
    # still improving at its last step more room, without paying for it from scratch.
    ext = cfg["training"].get("extend") or {}
    if ext.get("enable"):
        if not ck:
            raise RuntimeError("training.extend.enable is set but no checkpoint was "
                               "loaded - there is nothing to extend. Check `resume`.")
        extra = int(ext.get("epochs") or 0)
        if extra < 1:
            raise ValueError("training.extend.enable is set but extend.epochs < 1")

        # An extension that is ALREADY under way must not start a second one. Without
        # this, a crash at epoch 13 of an 11->16 extension would resume, read epoch 13
        # as the new baseline, and extend to 19 - on a fresh lr curve back at its peak,
        # every time it crashed. The extension's identity is stored in the checkpoint,
        # so `continuing` means "carry the SAME curve on from where it stopped".
        continuing = bool(ext_state)
        if continuing:
            base_epoch = int(ext_state["base_epoch"])
            extra = int(ext_state["epochs"])          # the config cannot move the goalposts
        else:
            base_epoch = int(ck.get("epoch") or 0)
        epochs = base_epoch + extra           # keep epoch numbering continuous in the logs
        ext_start = base_epoch * steps_per_epoch      # first step OF the extension
        done_epochs = int(ck.get("epoch") or 0)
        start_epoch = done_epochs + 1
        resume_skip = 0                       # the last epoch finished; start a clean one
        if not ck.get("epoch_complete", True):
            # crashed mid-epoch: fall back on the same skip the normal resume path uses
            start_epoch = done_epochs
            resume_skip = max(0, gstep - (done_epochs - 1) * steps_per_epoch) * accum

        ecfg = json.loads(json.dumps(cfg))    # private copy - only the schedule differs
        if ext.get("warmup_frac") is not None:
            ecfg["training"]["scheduler"]["warmup_frac"] = float(ext["warmup_frac"])
        # An extension is short, so the run's usual 1% floor spends most of it at a lr that
        # cannot move anything. Raising the floor keeps the extra epochs actually training.
        if ext.get("min_lr_mult") is not None:
            ecfg["training"]["scheduler"]["min_lr_mult"] = float(ext["min_lr_mult"])
        ext_total = steps_per_epoch * extra

        base_lr = float(cfg["training"]["optimizer"]["lr"])
        new_lr = ext_state.get("lr") if continuing else ext.get("lr")
        # `not continuing` matters. The ratio rescales initial_lr IN PLACE, and on a
        # resume the optimizer state already comes back holding the extension's lr - so
        # re-applying it multiplies the cut in a second time. A 5e-5 -> 2.5e-6 extension
        # returned at 1.25e-7 after one hand-off, 20x under its own floor, and the run
        # went flat while looking like it was training. Silent, and this repo hands off
        # constantly. On a resume the restored optimizer is already correct.
        if new_lr and not continuing:
            # scale every group by the same ratio, so whatever split build_optimizer
            # made (backbone_lr_mult, layer_decay) survives untouched
            ratio = float(new_lr) / base_lr
            for g in opt.param_groups:
                g["initial_lr"] = g.get("initial_lr", g["lr"]) * ratio
                g["lr"] = g["initial_lr"]
        if ext.get("reset_optimizer"):
            opt.state.clear()                 # throw away the AdamW moments too
        # fresh LambdaLR: last_epoch back to -1, so the new cosine starts at its peak
        sched = build_scheduler(opt, ecfg, ext_total)
        warm = warmup_steps(ecfg, ext_total)
        into = max(0, gstep - ext_start)      # steps already spent inside the extension
        for _ in range(into):
            sched.step()                      # fast-forward onto the SAME point of the curve
        total = ext_start + ext_total         # keeps ETA and the "x of total" prints honest
        if ext.get("reset_patience", True) and not continuing:
            bad, audit_bad[0] = 0, 0          # only once, at the start of the extension
        ext_state.update({"base_epoch": base_epoch, "epochs": extra, "lr": new_lr})

        _lrs = ", ".join(f"{g['lr']:.2e}" for g in opt.param_groups)
        _pat = ("reset to 0" if (ext.get("reset_patience", True) and not continuing)
                else f"kept at {bad}")   # reset only fires on a NEW extension
        _opt = "RESET" if ext.get("reset_optimizer") else "moments kept"
        _wf = ecfg["training"]["scheduler"].get("warmup_frac", 0.05)
        _banner("CONTINUING AN EXTENSION" if continuing else "EXTENDING A FINISHED RUN",
                "🔁")
        print(f"  from       : epoch {done_epochs}, step {gstep}, best {best:.4f}")
        if continuing:
            print(f"  ↩️  already {into}/{ext_total} steps into the extension that "
                  f"started at epoch {base_epoch} - same curve, NOT a new one")
        print(f"  adding     : {extra} epochs = {ext_total} steps "
              f"(epochs {start_epoch}-{epochs}, steps {gstep + 1}-{total})")
        print(f"  peak lr    : {_lrs}   (base was {base_lr:.2e}"
              + (f", now {float(new_lr):.2e})" if new_lr else ", unchanged)"))
        print(f"  warmup     : {warm} steps ({_wf:.0%} of the extension)")
        print(f"  optimizer  : {_opt}")
        print(f"  patience   : {_pat}")
    train_log = CSVLogger(out / cfg["output"]["train_log_csv"], run_id)
    val_log = CSVLogger(out / cfg["output"]["val_log_csv"], run_id)
    t0 = time.time() - resumed_elapsed            # ETA/elapsed span the whole run
    prev_loss = None

    last_val_step = -1
    score, improved = -math.inf, False
    # Chronic gradient clipping is silent and it costs real learning rate: a trunk
    # whose gnorm sits above grad_clip gets every update rescaled, so the lr in the
    # config stops being the lr the model actually sees. dinov3_convnext_base was
    # clipped on 88% of steps at grad_clip=1.0 and underfitted for it, while the
    # ImageNet trunk on the identical config was clipped on 16%. Counted here and
    # warned about below, so it can never again be found by hand after the fact.
    clip_seen, clip_hit, clip_warned = [0], [0], [False]
    clip_vals = deque(maxlen=4000)     # finite gnorms, for the distribution below

    stop_now = [False]            # raised the moment patience runs out, so the step
                                  # loop can stop AT 4/4 rather than carrying on to
                                  # 5/4 or 6/4 and only checking at the epoch boundary

    def _track():
        """Everything a resume needs that is not weights. ONE definition: the periodic
        step checkpoint used to write its own shorter version, so audit_best and the
        eval deltas were lost on every hand-off (we always resume from a step*.pt)."""
        return {"best": best, "best_epoch": best_epoch, "bad": bad,
                "top_k": top_k_track, "history": history,
                "audit_best": list(audit_best), "audit_bad": audit_bad[0],
                "prev_eval": dict(prev_eval), "extend": dict(ext_state)}

    def _validate_and_checkpoint(epoch: int, gstep: int, tag: str = "epoch"):
        """Score val (and audit), log, checkpoint, update early stopping."""
        nonlocal best, best_epoch, bad, top_k_track, last_val_step
        nonlocal score, improved
        if gstep == last_val_step:            # a step-trigger already covered this step
            return None
        last_val_step = gstep
        el = time.time() - t0
        eta = el * (total / max(gstep, 1) - 1)
        print("=" * 70)
        print(f"\U0001F50E VALIDATION ({tag}) - epoch {epoch}/{epochs}  step {gstep}  "
              f"({len(loaders['val'])} batches, {len(loaders['val'].dataset)} studies) ...")
        _t = time.time()
        _vp = {}
        vm = evaluate(model, loaders["val"], loss_fn, cfg, device, collect=_vp)
        val_secs = time.time() - _t
        _print_eval("VAL", epoch, epochs, gstep, vm, el, eta, cfg["tasks"],
                    prev=prev_eval.get("val"), took=val_secs)
        prev_eval["val"] = dict(vm)

        _ap = {}                      # stays empty when there is no audit split
        row = {"epoch": epoch, "step": gstep, "trigger": tag,
               "val_loss": round(vm["loss"], 6),
               "val_macro_auroc": round(vm["macro_auroc"], 6),
               "val_macro_auprc": round(vm["macro_auprc"], 6),
               "val_f1": round(vm["f1"], 4), "val_precision": round(vm["precision"], 4),
               "val_recall": round(vm["recall"], 4),
               "val_specificity": round(vm["specificity"], 4),
               "val_secs": round(val_secs, 1)}
        for t in cfg["tasks"]:
            row[f"val_auroc_{t}"] = round(vm["per_task"][t]["auroc"], 6)
            row[f"val_auprc_{t}"] = round(vm["per_task"][t]["auprc"], 6)

        # verdict printed with its own split, not after both — decided here, the
        # checkpoint itself is written further down
        score = row.get(es["monitor"], vm["macro_auroc"])
        improved = score > best if es["mode"] == "max" else score < best
        prev_best = best
        if improved:
            best, best_epoch, bad = score, epoch, 0
            was = "" if prev_best == -math.inf else f"  (was {prev_best:.4f}, {score - prev_best:+.4f})"
            print(f"     ⭐ new best {es['monitor']} = {score:.4f}{was} -> best.pt")
        else:
            bad += 1
            if es["enable"] and bad >= es["patience"]:
                stop_now[0] = True     # stop right here, not at the next epoch end
            print(f"     ⚠️  no improvement ({bad}/{es['patience']})  "
                  f"{es['monitor']} = {score:.4f}, best {best:.4f} @ epoch {best_epoch}")

        if "audit" in loaders:
            _t = time.time()
            _ap = {}
            am = evaluate(model, loaders["audit"], loss_fn, cfg, device, collect=_ap)
            audit_secs = time.time() - _t
            _print_eval("AUDIT-58", epoch, epochs, gstep, am, el, eta, cfg["tasks"],
                        "\U0001F9EA", prev=prev_eval.get("audit"), took=audit_secs)
            prev_eval["audit"] = dict(am)
            # the audit tracks its own best, purely as a report \u2014 it never touches
            # best.pt or the early-stopping counter
            a_score = am["macro_auroc"]
            if a_score > audit_best[0]:
                a_prev = audit_best[0]
                was = "" if a_prev == -math.inf else f"  (was {a_prev:.4f}, {a_score - a_prev:+.4f})"
                audit_best[0], audit_best[1] = a_score, epoch
                audit_bad[0] = 0
                print(f"     \u2B50 new best audit_macro_auroc = {a_score:.4f}{was}")
            else:
                audit_bad[0] += 1
                # same denominator as val so the two lines read alike, but nothing
                # ever stops on it - 58 studies are far too few to trust that way
                print(f"     \u26A0\uFE0F  no improvement ({audit_bad[0]}/{es['patience']})  audit_macro_auroc = {a_score:.4f}, "
                      f"best {audit_best[0]:.4f} @ epoch {audit_best[1]}")
            print("     \u2139\uFE0F  report only - the panel's labels, never drives best.pt")
            row["audit_macro_auroc"] = round(am["macro_auroc"], 6)
            row["audit_secs"] = round(audit_secs, 1)
            for t in cfg["tasks"]:
                row[f"audit_auroc_{t}"] = round(am["per_task"][t]["auroc"], 6)

        row["elapsed_s"] = round(el, 1)
        val_log.log(row)
        history.append(row)
        if log_fn:
            log_fn({f"val/{k}": v for k, v in row.items() if isinstance(v, (int, float))},
                   gstep)

        done = (tag == "epoch")          # mid-epoch checkpoints must replay their epoch

        # update the top-K board FIRST, so best.pt/last.pt carry the current board.
        # Saving before it left the board a step stale, and a resumed run then could
        # not evict files it did not know about (keep_top_k=2 kept 3).
        top_k_track = update_top_k(ckpt_dir, cfg, model, epoch, gstep, score, top_k_track,
                                   cfg["output"].get("keep_top_k", 0),
                                   cfg["output"].get("top_k_weights_only", True),
                                   saver=saver)

        if improved:
            save_checkpoint(ckpt_dir / cfg["output"]["best_name"], cfg, model, opt, sched,
                            scaler, epoch, gstep, score, track=_track(), device=device,
                            elapsed_sec=el, epoch_complete=done, saver=saver, loss_fn=loss_fn,
                            note=f"  \U0001F4BE best.pt written (score {score:.4f})")

        ud.dump(out / "ud_ratio.csv")         # flushed with last.pt, survives preemption
        save_checkpoint(ckpt_dir / "last.pt", cfg, model, opt, sched, scaler, epoch, gstep,
                        score, track=_track(), device=device, loss_fn=loss_fn,
                        elapsed_sec=el, epoch_complete=done, saver=saver,
                        note=("  \U0001F4BE saved last.pt (epoch %d)%s" % (
                            epoch, f"  \U0001F3C6 top-{len(top_k_track)} kept"
                            if top_k_track else "")))
        # raw logits for both splits, every validation. Ensembling top-K checkpoints
        # later then costs nothing: load, average, score - no model, no GPU.
        pred_dir = out / cfg["output"].get("predictions_dir", "predictions")
        _saved = []
        for _split, _pack in (("val", _vp), ("audit", _ap)):
            _ldr = loaders.get(_split)
            if not _pack or _ldr is None:
                continue
            _ids = _ldr.dataset.df["StudyInstanceUID"].tolist()
            _f = save_predictions(pred_dir, _split, run_id, epoch, gstep, score,
                                  _pack["logits"], _pack["targets"], _ids, cfg["tasks"])
            if _f:
                _saved.append(f"{_split}:{_pack['logits'].shape[0]}")
        if _saved:
            print("  📦 predictions saved (" + ", ".join(_saved)
                  + f") -> {pred_dir.name}/")


        summary_path = out / cfg["output"]["summary_json"]
        summary = json.dumps({
            "experiment": cfg["experiment"]["name"], "run_id": run_id,
            "epoch": epoch, "epochs": epochs,
            "best_score": best, "best_epoch": best_epoch, "monitor": es["monitor"],
            "hardware": hardware_info(cfg),
            "last": row, "elapsed": fmt(time.time() - t0),
            # always last, always null until a human fills it in after submitting.
            # carried over on every rewrite so a filled-in record survives a resume.
            "submission": carried_submission(summary_path)}, indent=2)
        summary_path.write_text(summary)
        # a per-run copy too, so the sync can carry it without clobbering the other side
        (cap_dir / f"{run_id}_summary.json").write_text(summary)
        persist()
        model.train()
        return score


    # images per optimizer step: studies per batch x accumulation x slots x slices.
    # Fixed for the run, so total images is just gstep x this.
    per_step_imgs = (train_batch_size(cfg) * accum
                     * len(cfg["input"]["slots"]) * cfg["input"]["n_slices"])
    retr_prev = (0, 0, 0, 0.0)      # last retrieval snapshot
    for epoch in range(start_epoch, epochs + 1):
        print("\n" + "=" * 70)
        print(f"\U0001F4DA EPOCH {epoch}/{epochs}")
        print("=" * 70)
        if _pesg:
            _pe = cfg["training"]["optimizer"].get("pesg", {}) or {}
            if epoch in set(_pe.get("decay_epochs", []) or []):
                # averages the accumulated weights into the reference point AND divides
                # the lr - both halves of libauc's stagewise step, not just an lr drop
                opt.update_regularizer(decay_factor=float(_pe.get("decay_factor", 10)))
        _EPOCH.value = epoch          # workers seed their augmentation off this
        skip = resume_skip if epoch == start_epoch else 0
        smp = getattr(loaders["train"], "sampler", None)
        if hasattr(smp, "set_epoch"):
            smp.set_epoch(epoch, skip * train_batch_size(cfg))   # sampler counts studies
        model.train()
        ep_t0, seen = time.time(), 0
        # two throughput numbers so a slow volume can't hide behind a fast GPU:
        #   compute = images / GPU time only, real = images / wall incl. the data wait.
        # next() is timed by hand (not via `for`) so eval/checkpoint time between
        # steps never gets miscounted as waiting on data.
        roll = deque(maxlen=20)          # (imgs, wall, data_wait) for the last 20 steps
        win_imgs, win_t0, win_data, win_first = 0, time.time(), 0.0, 0.0
        _train_it, it = iter(loaders["train"]), skip
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
            # `it % accum == 0` is the micro-batch that also triggers the optimizer step
            # below. Arming one earlier means the NEXT iteration's arm(False) wipes the
            # reading before read() sees it, which is why the probe printed nothing.
            probe.arm(probe.active(gstep + 1) and it % accum == 0)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss = loss_fn(model(prepare_batch(x, cfg, device)), y) / accum
            scaler.scale(loss).backward()
            seen += imgs_per_batch
            win_imgs += imgs_per_batch
            if it % accum == 0:
                gnorm = float("nan")
                _clip = float(cfg["training"]["grad_clip"] or 0.0)
                if _clip:
                    scaler.unscale_(opt)
                    gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), _clip))
                    if math.isfinite(gnorm):        # AMP skips a step on inf/nan
                        clip_seen[0] += 1
                        clip_hit[0] += int(gnorm > _clip)
                        clip_vals.append(gnorm)
                _ud_on = ud.active(gstep + 1)
                if _ud_on:
                    ud.before()
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                gstep += 1
                _extra = {}                # gate rows and AUC-M duals, merged into the log
                _pb = probe.read()
                if _ud_on:
                    _uo = ud.after(epoch, gstep)
                    _um = ud.summarize(_uo)
                    if _um:
                        print(_um)
                if _pb:
                    print(ScaleProbe.summarize(_pb))
                if _gates and _ud_on:
                    # Whether a spliced-in block is actually USED is the whole question, so
                    # report the gate in full: the mean says how much branch survives, min/max
                    # say whether it moved uniformly or on a handful of channels, and `open`
                    # counts channels past 1e-3. All of it also lands in train_log.csv.
                    print("  \U0001F6AA gates (gamma)")
                    _gopen = 0.0
                    for _n, _q in _gates:
                        _v = _q.detach().float()
                        _a = _v.abs()
                        _lbl = _n.rsplit(".", 2)[-2] if "." in _n else _n
                        _op = int((_a > 1e-3).sum())
                        _gopen = max(_gopen, float(_a.mean()))
                        print(f"     {_lbl:<10} |g| mean {float(_a.mean()):.5f}  "
                              f"min {float(_v.min()):+.5f}  max {float(_v.max()):+.5f}  "
                              f"std {float(_v.std()):.5f}  open {_op}/{_v.numel()}")
                        _extra[f"gate_{_lbl}_absmean"] = round(float(_a.mean()), 8)
                        _extra[f"gate_{_lbl}_max"] = round(float(_v.max()), 8)
                        _extra[f"gate_{_lbl}_min"] = round(float(_v.min()), 8)
                        _extra[f"gate_{_lbl}_open"] = _op
                    if _gopen < 1e-4:
                        print("     \u26a0\ufe0f  every gate still ~0: the blocks are "
                              "contributing nothing yet")
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
                # the number that actually decides what the run costs: every image
                # since step 0 over every second since step 0. Unlike the two above it
                # carries validation, checkpoint writes, volume commits and epoch
                # boundaries - all the things the in-loop figures step around.
                # gstep and elapsed both survive a resume, so it spans the whole run.
                overall_ips = gstep * per_step_imgs / max(time.time() - t0, 1e-9)
                mem = (torch.cuda.max_memory_allocated() / 1024**3
                       if device.type == "cuda" else 0.0)
                el = time.time() - t0
                eta = el * (total / gstep - 1)
                if _pesg and hasattr(loss_fn, "stats"):
                    # update, not assign: _extra may already hold the gate rows
                    _extra.update(loss_fn.stats(cfg["tasks"]))   # AUC-M duals a / b / alpha
                train_log.log({**_extra, **ScaleProbe.row(_pb),
                               "epoch": epoch, "epoch_step": it // accum,
                               "global_step": gstep, "loss": round(l, 6), "lr": lr,
                               "lr_min": lr_lo, "lr_max": lr_hi,
                               "gnorm": round(gnorm, 4),
                               "clip_pct": (round(100.0 * clip_hit[0] / clip_seen[0], 1)
                                            if clip_seen[0] else None),
                               "img_s": round(ips, 1),
                               "real_img_s": round(real_ips, 1),
                               "data_wait_pct": round(wait_pct, 1),
                               "overall_img_s": round(overall_ips, 1),
                               "gpu_mem_gb": round(mem, 2), "elapsed_s": round(el, 1)})
                # flushing the volume is its own concern - a crash between two
                # validations must not cost the log rows written in between
                # 200 steps is enough to tell a chronic rate from a warmup spike
                if (not clip_warned[0] and clip_seen[0] >= 200
                        and clip_hit[0] / clip_seen[0] > 0.5):
                    clip_warned[0] = True
                    _v = sorted(clip_vals)
                    _md, _mn = _v[len(_v) // 2], sum(_v) / len(_v)
                    print(
                        f"\n  ⚠️  CLIPPING {100.0 * clip_hit[0] / clip_seen[0]:.0f}% of {clip_seen[0]} steps at grad_clip={_clip:g} "
                        f"-> real step below the live lr {lr_hi:.2e}.\n"
                        f"      gnorm  med {_md:.2f}  mean {_mn:.2f}  min {_v[0]:.2f}  max {_v[-1]:.2f}  ->  raise grad_clip or lower lr\n")

                ce = int(cfg["output"].get("commit_every_steps", 0) or 0)
                if ce and gstep % ce == 0:
                    persist()                  # async: never blocks the step loop
                ev = eval_interval(cfg, epoch)
                if ev and gstep % ev == 0:
                    _validate_and_checkpoint(epoch, gstep, f"every {ev} steps")
                    if stop_now[0]:
                        break             # stop AT patience, not at the next epoch end
                every = cfg["output"].get("checkpoint_every_steps", 0)
                if every and gstep % every == 0:
                    save_checkpoint(ckpt_dir / f"step{gstep}.pt", cfg, model, opt, sched,
                                    scaler, epoch, gstep,
                                    track=_track(), loss_fn=loss_fn,
                                    device=device, elapsed_sec=el,
                                    epoch_complete=False, saver=saver,
                                    note=(f"  \U0001F4BE periodic checkpoint @ step "
                                          f"{gstep} (rolling, keep_last_n="
                                          f"{cfg['output'].get('keep_last_n', 3)})"))
                    prune_checkpoints(ckpt_dir, cfg["output"].get("keep_last_n", 3),
                                      saver=saver)
                    persist()             # flush logs to the volume too
                dbg = cfg["output"].get("save_debug_images", 0)
                if dbg and (gstep % dbg == 0 or gstep == 1):
                    save_debug_batch(prepare_batch(x, cfg, device), cfg, dbg_dir,
                                     epoch, gstep, y, prefix=f"{run_id}_")
                if gstep == 1 and warm:
                    print(f"  🔥 warmup STARTED - lr ramps from 0 to base over "
                          f"{warm} step(s) ({cfg['training']['scheduler']['warmup_frac']:.0%} "
                          f"of {total})")
                if gstep == 1 and not warm:
                    print(f"  ▶️  no warmup: PESG holds lr at {lr_hi:.3e} until "
                          f"update_regularizer drops it")
                if warm and gstep == warm:   # not elif: warm can be 1 on tiny runs
                    print(f"  ✅ warmup COMPLETE at step {warm} - base lr reached "
                          f"(backbone {lr_lo:.3e}, head {lr_hi:.3e}); cosine decay begins")
                if gstep % cfg["output"]["console_log_every"] == 0 or gstep == 1:
                    d = "" if prev_loss is None else f" ({l - prev_loss:+.4f})"
                    # one lr when every group shares it, both ends when LLRD or
                    # backbone_lr_mult splits them
                    lr_s = (f"lr={lr_hi:.3e}" if lr_lo == lr_hi
                            else f"lr[head={lr_hi:.3e} backbone={lr_lo:.3e}]")
                    print("─" * 70)
                    print(f"  Epoch {epoch}/{epochs}  epoch_step: {it // accum}/{steps_per_epoch}"
                          f"  global_step: {gstep}/{total}")
                    # one step's gnorm is noisy (ours span 0.6 to 23 on the same run), so
                    # the rolling mean is what actually says whether the trunk runs hot,
                    # and clip% says how much of it is surviving the cap
                    _gw = list(clip_vals)[-50:]
                    _gs = ""
                    if _gw:
                        _gs = (f" (avg{len(_gw)} {sum(_gw) / len(_gw):.2f}"
                               f", clip {100.0 * clip_hit[0] / max(clip_seen[0], 1):.0f}%)")
                    print(f"     loss={l:.4f}{d}  {lr_s}  gnorm={gnorm:.2f}{_gs}  "
                          f"elapsed={fmt(el)}  ETA={fmt(eta)}  gpu_mem(peak)={mem:.2f}GB")
                    print()                      # split what the model is doing from how fast
                    print(f"     throughput: {ips:.0f} img/s compute | {real_ips:.0f} img/s real "
                          f"({len(roll)}-step) | data-wait {wait_pct:.0f}%")
                    _eff = 100.0 * overall_ips / max(real_ips, 1e-9)
                    print(f"     ➤ END-TO-END: {overall_ips:.0f} img/s over the whole "
                          f"run ({_eff:.0f}% of in-loop) — {gstep * per_step_imgs / 1e6:.2f}M "
                          f"images in {fmt(time.time() - t0)}")
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

        # drop_last=false leaves a ragged tail: those micro-batches ran backward() but
        # never reached opt.step(), so without this their gradients leak into the first
        # step of the next epoch (and a resumed run, being a fresh process, would not
        # have them - the two trajectories then diverge).
        if it % accum:
            opt.zero_grad(set_to_none=True)
            print(f"  \U0001F9F9 discarded {it % accum} trailing micro-batch(es) "
                  f"(not a full accumulation window)")

        ee = int(cfg["evaluation"].get("eval_every_epochs") or 0)
        if ee and epoch % ee == 0:
            _validate_and_checkpoint(epoch, gstep, "epoch")

        if stop_now[0] or (es["enable"] and bad >= es["patience"]):
            print(f"\n🛑 early stop: {bad} validations without improvement "
                  f"(patience={es['patience']})")
            break

    if last_val_step != gstep:
        print("\n  📌 final validation (these last steps hit no trigger)")
        _validate_and_checkpoint(epoch, gstep, "final")

    train_log.close(); val_log.close()
    total_time = time.time() - t0
    best_row = next((r for r in history if r["epoch"] == best_epoch),
                    history[-1] if history else {})
    print("\n" + "=" * 70)
    print(f"\U0001F3C1 FINISHED - {cfg['experiment']['name']}")
    print("=" * 70)
    # history holds one row per VALIDATION, which is not one per epoch once
    # eval_every_steps fires more than once inside an epoch
    print(f"  \u23F1\uFE0F  total time       : {fmt(total_time)}  "
          f"({epoch} epochs, {gstep} steps, {len(history)} validations)")
    print(f"  \u2B50 best {es['monitor']:<15s}: {best:.4f}  @ epoch {best_epoch}")
    if best_row:
        print(f"  \U0001F4C9 val loss at best  : {best_row.get('val_loss')}")
        print(f"  \U0001F4CA macro AUPRC       : {best_row.get('val_macro_auprc')}")
        if "audit_macro_auroc" in best_row:
            print(f"  \U0001F9EA audit-58 AUROC    : {best_row['audit_macro_auroc']}  (report only)")
        # the only place the 12 findings are printed; every validation writes them
        # to val_log.csv but printing them each time buries the run log
        has_audit = "audit_macro_auroc" in best_row
        rows = []
        ranked = sorted(((t, best_row.get(f"val_auroc_{t}", float("nan")))
                         for t in cfg["tasks"]), key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0))
        for t, v in ranked:
            cells = [t, "  n/a" if v != v else f"{v:.4f}",
                     ("  n/a" if best_row.get(f"val_auprc_{t}", float("nan"))
                      != best_row.get(f"val_auprc_{t}", float("nan"))
                      else f"{best_row[f'val_auprc_{t}']:.4f}")]
            if has_audit:
                a = best_row.get(f"audit_auroc_{t}", float("nan"))
                cells.append("  n/a" if a != a else f"{a:.4f}")
            cells.append("\u2588" * int(max(0.0, v - 0.5) * 40) if v == v else "")
            rows.append(cells)
        heads = ["finding", "val AUROC", "val AUPRC"] + (["audit-58"] if has_audit else []) \
            + ["above chance"]
        _table(f"per-finding at the best epoch ({best_epoch})", heads, rows, "\U0001F50D")
        print("\n  \U0001F4C8 reference: keyword labeler 0.700, LLM labels 0.895 (vs the 58)")
    print(f"  \U0001F4BE checkpoints       : {ckpt_dir}")
    print(f"  \U0001F4DD logs              : {out / cfg['output']['train_log_csv']}")
    print(f"                        {out / cfg['output']['val_log_csv']}")
    print("=" * 70)

    # curves for every run, and a plotting bug must never sink a finished run
    if cfg["output"].get("make_plots", True):
        _safe("make_plots", make_plots, cfg, out,
              cfg["output"].get("plots_dir", "plots"))
        saver.join()                       # every checkpoint on disk before we flush
        if saver.error:
            print(f"  ⚠️  a checkpoint write failed: {saver.error}")
        persist(blocking=True)             # the run is over: this one has to land
        if persist.skipped:
            print(f"  ℹ️  {persist.done} volume flushes, {persist.skipped} "
                  f"skipped as redundant (one was already in flight)")
        if persist.last_error:
            print(f"  ⚠️  last volume flush error: {persist.last_error}")
    return best


SUBMISSION_TEMPLATE = {
    "submitted": None,            # UTC, "2026-08-20 16:09" — kaggle reports UTC
    "public_score": None,         # the leaderboard number
    "private_score": None,        # only after the deadline
    "runtime_seconds": None,      # what the Efficiency Prize is ranked on
    "runtime": None,              # the same thing readable: "44:32" or "1:35:34"
    "kernel": None,               # "<user>/<slug>"
    "kernel_version": None,       # which version was submitted
    "submission_ref": None,       # kaggle's id, so the entry can be found again
    "checkpoint": None,           # which checkpoint went into the dataset
    "notes": None,
}


def carried_submission(summary_path) -> dict:
    """Keep a hand-filled submission record across summary rewrites.

    The summary is rewritten at every validation, so a record typed in by hand would
    otherwise be erased by the next one — or by a resume.
    """
    try:
        old = json.loads(Path(summary_path).read_text()).get("submission")
    except Exception:
        old = None
    if not isinstance(old, dict):
        return dict(SUBMISSION_TEMPLATE)
    rec = {k: old.get(k, v) for k, v in SUBMISSION_TEMPLATE.items()}
    # fill runtime from the seconds, so only one of the two has to be typed in
    if rec.get("runtime") is None:
        rec["runtime"] = clock(rec.get("runtime_seconds"))
    return rec


def clock(sec: float | None) -> str | None:
    """Seconds -> "44:32", or "1:35:34" once it runs past an hour."""
    if sec is None:
        return None
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


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
                         "libauc",   # official AUCMLoss + PESG, training only

                         "matplotlib", "psutil", "pydicom")
            .add_local_dir(str(Path(__file__).parent), "/root/runs")
            # only the small split CSVs travel in the image; stats/ alone is 824 MB of
            # local analysis output the container never opens
            .add_local_dir(str(Path(__file__).parent.parent / "data"), "/root/data",
                           ignore=["shards", "raw", "backups", "stats", "sample_dicoms", "pretrained",
                                   "*.jsonl", "*.parquet"]))


def modal_resources(cfg: dict) -> dict:
    """gpu/timeout always; cpu and memory only when the run config asks for them."""
    m = cfg["modal"]
    kw = {"gpu": m["gpu"], "timeout": int(m.get("timeout_hours", 8) * 3600),
          # release the GPU the moment training returns. Modal's default is 60s of
          # idle-but-billed reservation, which we would pay for while this PC is still
          # syncing artifacts back in the local entrypoint. One run per app here, so
          # there is no cold start to amortise and nothing to keep warm for.
          "scaledown_window": int(m.get("scaledown_window", 2))}
    if m.get("cpu"):
        kw["cpu"] = float(m["cpu"])              # cores, a reservation not a cap
    if m.get("memory_gb"):
        kw["memory"] = int(float(m["memory_gb"]) * 1024)   # config is GB, modal wants MiB
    return kw


def _sync_module():
    """modal_scripts/sync_runs.py — lazy, because it only exists on this PC."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modal_scripts"))
    import sync_runs
    return sync_runs


def launch_run_id(cfg: dict, exp_dir) -> str:
    """Allocate this launch's runNNN, here on the launching PC.

    The volume is consulted too (a `modal volume ls`, no container), so a number
    already used remotely is never handed out again.
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


_SYNCED = False


def sync_back(cfg: dict, status: str = "", once: bool = True) -> None:
    """Pull this run's artifacts home. Fires on success, on crash, and on Ctrl+C.

    `once` guards against syncing twice when both the signal handler and the
    entrypoint's finally fire for the same interrupt.
    """
    global _SYNCED
    if once and _SYNCED:
        return
    _SYNCED = True
    try:
        sr = _sync_module()
        s = sr.session_from_env()
        if s is None:
            print("  ⚠️  no modal profile selected - skipping the artifact sync")
            return
        print(f"\n  \U0001F504 syncing artifacts back ({status}) ...")
        sr.sync_session(s, cfg["experiment"]["name"])
    except Exception as e:                     # a failed sync must never mask a training error
        print(f"  ⚠️  artifact sync failed: {type(e).__name__}: {e}")
    plot_after_run(cfg, status)                # after the sync: it needs the merged CSVs


def plot_after_run(cfg: dict, status: str = "") -> None:
    """Draw the training curves here on this PC, once the run has actually finished.

    make_plots is off on Modal because a container only holds the CSV rows it wrote
    itself - a run resumed several times would produce several partial curves. By the
    time this runs the sync has merged them all into the local CSVs, so this is the
    only place the whole run can be drawn at all.

    Only on "finished": a crashed or interrupted run gets picked up again later, and
    redrawing a half-run on every hand-off is noise.
    """
    if status != "finished":
        return
    try:
        exp = cfg["experiment"]["name"]
        results = (Path(__file__).resolve().parent / exp
                   / cfg["output"].get("run_dir", "results"))
        if not (results / cfg["output"]["train_log_csv"]).exists():
            print(f"  ⚠️  no local CSVs in {results} - skipping the plots")
            return
        make_plots(cfg, results, cfg["output"].get("plots_dir", "plots"))
    except Exception as e:                     # a bad plot must never mask a finished run
        print(f"  ⚠️  plotting failed: {type(e).__name__}: {e}")


def sync_on_interrupt(cfg: dict) -> None:
    """Sync on Ctrl+C too, by hooking the signal rather than trusting `finally`.

    `modal run` installs its own SIGINT handler and tears the local process down
    before the entrypoint's finally block gets there, so stopping a run by hand used
    to leave the CSVs and debug images sitting on the volume. The container itself
    keeps committing (`output.commit_every_steps`), so they were never lost — but
    they had to be fetched by hand afterwards.

    Modal's handler still runs after ours, so Ctrl+C keeps stopping the app as before.
    """
    import signal

    def make(previous):
        def handler(signum, frame):
            print("\n  \u23F9\uFE0F  interrupted — pulling artifacts before exit "
                  "(Ctrl+C again to skip)", flush=True)
            sync_back(cfg, "interrupted")
            if callable(previous):
                return previous(signum, frame)      # let modal stop the app as usual
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        return handler

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, make(signal.getsignal(sig)))
        except Exception:                           # not the main thread, or no such signal
            pass


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

    # update-to-weight ratio, if the run logged it. Its own CSV, not train_log, because
    # it is one row PER TENSOR per step rather than one row per step.
    def _plot_ud():
        ud_csv = results_dir / "ud_ratio.csv"
        if not ud_csv.exists():
            return
        d = pd.read_csv(ud_csv)
        if d.empty:
            return

        def _ishead(n):
            # NOT a substring test on "fc": every ConvNeXt block has mlp.fc1/mlp.fc2,
            # so that matched 100+ trunk tensors and put them all in the legend.
            return "backbone" not in n and any(k in n for k in ("head", "classifier", "fc"))

        def _depth(n):
            # natural sort: plain string order puts blocks.10 before blocks.2, which
            # scrambled the depth axis into noise.
            return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)]

        head = sorted([n for n in d["param"].unique() if _ishead(n)], key=_depth)
        trunk = sorted([n for n in d["param"].unique() if not _ishead(n)], key=_depth)
        names = trunk + head                       # shallow trunk -> deep trunk -> head
        steps = sorted(d["step"].unique())
        vc = plt.get_cmap("viridis")
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                      gridspec_kw={"height_ratios": [2, 1]})

        # a single sample draws nothing as a line, so fall back to markers
        style = dict(marker="o", ms=3, ls="none") if len(steps) < 2 else dict(lw=0.7)
        for i, n in enumerate(trunk):
            t = d[d["param"] == n].sort_values("step")
            ax.plot(t["step"], t["log10_ud"], color=vc(i / max(len(trunk) - 1, 1)),
                    alpha=0.55, **style)
        for n in head:
            t = d[d["param"] == n].sort_values("step")
            ax.plot(t["step"], t["log10_ud"], color="crimson", zorder=5,
                    label=f"HEAD  {n}", **dict(style, **({} if len(steps) < 2 else {"lw": 2.2})))
        # one proxy entry for the 100+ trunk curves, instead of 100+ legend rows
        ax.plot([], [], color=vc(0.5), lw=2, label=f"trunk ({len(trunk)} tensors)")
        ax.axhline(-3, color="k", ls="--", lw=1.2, label="-3  (from-scratch target)")
        ax.axhspan(-6, -5, color="steelblue", alpha=0.10,
                   label="-5..-6  (pretrained drift)")
        ax.set_ylabel("log10(update / weight)")
        ax.set_title(f"{name} - update-to-weight ratio")
        ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)
        _vlines(ax, bsteps)

        last = d[d["step"] == d["step"].max()].set_index("param").reindex(names)
        xs = range(len(names))
        ax2.plot(xs, last["log10_ud"].values, "-", lw=0.8, color="0.6", zorder=1)
        ax2.scatter(list(xs)[:len(trunk)], last["log10_ud"].values[:len(trunk)],
                    s=12, color="steelblue", label="trunk", zorder=3)
        ax2.scatter(list(xs)[len(trunk):], last["log10_ud"].values[len(trunk):],
                    s=45, color="crimson", marker="*", label="head", zorder=4)
        ax2.axhline(-3, color="k", ls="--", lw=1.2)
        ax2.axhspan(-6, -5, color="steelblue", alpha=0.10)
        ax2.set_xlabel("tensor, shallow trunk -> deep trunk -> head")
        ax2.set_ylabel("log10(update / weight)")
        ax2.set_title(f"profile at step {int(d['step'].max())}")
        ax2.legend(fontsize=8, loc="best"); ax2.grid(True, alpha=0.3)
        _save(fig, "ud_ratio.png")

    # AUC-M internals, if this run used that loss. alpha pinned at 0 means the dual
    # never engaged and the min-max collapsed; b - a is the margin actually opened.
    def _plot_aucm():
        if tr is None or "aucm_alpha_mean" not in tr.columns:
            return
        d = tr.dropna(subset=["aucm_alpha_mean"])
        if d.empty:
            return
        gapc = [c for c in d.columns if c.startswith("aucm_gap__")]
        alc = [c for c in d.columns if c.startswith("aucm_alpha__")]
        fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
        for r in runs:
            t = d[d["run_id"] == r] if "run_id" in d.columns else d
            if t.empty:
                continue
            c = cmap[r]
            axes[0].plot(t["global_step"], t["aucm_alpha_mean"], color=c, lw=1.4,
                         label=_lbl(r))
            if {"aucm_alpha_min", "aucm_alpha_max"} <= set(t.columns):
                axes[0].fill_between(t["global_step"], t["aucm_alpha_min"],
                                     t["aucm_alpha_max"], color=c, alpha=0.15)
            for col, ls in (("aucm_a_mean", "-"), ("aucm_b_mean", "--")):
                if col in t.columns:
                    axes[1].plot(t["global_step"], t[col], color=c, ls=ls, lw=1.4,
                                 label=_lbl(r, f" {col.split('_')[1]}"))
        axes[0].set_ylabel("alpha (dual)"); axes[0].axhline(0, color="k", lw=1)
        axes[0].set_title(f"{name} - AUC-M internals  "
                          f"(alpha stuck at 0 = objective collapsed)")
        axes[1].set_ylabel("a (pos mean) / b (neg mean)")
        vc = plt.get_cmap("tab20")
        for i, col in enumerate(sorted(gapc)):
            axes[2].plot(d["global_step"], d[col], lw=1.0, color=vc(i % 20),
                         label=col.replace("aucm_gap__", ""))
        if "aucm_gap_mean" in d.columns:
            axes[2].plot(d["global_step"], d["aucm_gap_mean"], color="k", lw=2,
                         label="mean")
        axes[2].set_ylabel("b - a  (margin opened)")
        # per-task alpha: a single finding whose dual is pinned at 0 means AUC-M has
        # given up on THAT finding, which the mean and the min/max band both hide.
        dead = []
        for i, col in enumerate(sorted(alc)):
            t = col.replace("aucm_alpha__", "")
            axes[3].plot(d["global_step"], d[col], lw=1.0, color=vc(i % 20), label=t)
            if float(d[col].abs().max()) < 1e-8:
                dead.append(t)
        axes[3].set_ylabel("alpha per finding")
        axes[3].set_xlabel("global step")
        if dead:
            axes[3].set_title("⚠️  dual never left zero for: " + ", ".join(dead))
        axes[2].set_xlabel("")
        for ax in axes:
            ax.grid(True, alpha=0.3); _vlines(ax, bsteps)
            ax.legend(fontsize=7, ncol=3, loc="best")
        _save(fig, "aucm_internals.png")

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
    _safe("plot:ud_ratio", _plot_ud)
    _safe("plot:aucm", _plot_aucm)
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
                       desc: str = "", collect: dict | None = None) -> tuple:
    """Score one dataframe with the eval transforms. -> (y_true, y_prob, mean_loss).

    Pass `collect` to also keep the raw logits. They are computed either way - this
    only stops throwing them away - and they are what ensemble.py reads as a member.
    """
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
        if collect is not None:
            collect.setdefault("_lg", []).append(logits.float().cpu().numpy())
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
        _pk = {}
        y_true, y_prob, loss = _predict_dataframe(cfg, model, df, device, loss_fn,
                                                  desc=f"eval/{label}", collect=_pk, **kw)
        ids = df.get("StudyInstanceUID", pd.Series(range(len(df)))).astype(str).values

        # the same .npz a training-time validation writes, so a run that finished
        # BEFORE prediction-saving existed can still be made into an ensemble member
        # without retraining it - re-scoring one checkpoint is minutes, not hours
        if y_true is not None and _pk.get("_lg"):
            _rid = ((ck.get("cfg", {}).get("experiment", {}) or {}).get("run_id")
                    or cfg.get("experiment", {}).get("run_id") or "eval")
            _f = save_predictions(res / cfg["output"].get("predictions_dir", "predictions"),
                                  label, str(_rid), ck.get("epoch"), ck.get("step"),
                                  ck.get("score"), np.concatenate(_pk["_lg"]),
                                  y_true, ids, tasks)
            if _f:
                print(f"  💾 {_f}")
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


# =============================================================================
# submission — Kaggle is a code competition: the notebook gets raw DICOM, never
# our .npz shards, so the DICOM -> uint8 path here must match
# modal_scripts/modal_preprocess.py byte for byte or the model sees a different
# intensity distribution than it trained on. verify_dicom_path() proves it does.
# =============================================================================
SHARD_LO_PCT, SHARD_HI_PCT = 0.1, 99.9      # the wide window baked into every shard


def _slice_sort_key(ds, frame):
    """Through-plane position: ImagePositionPatient projected onto the slice normal."""
    iop, ipp = ds.get("ImageOrientationPatient"), ds.get("ImagePositionPatient")
    if iop is not None and ipp is not None and len(iop) == 6:
        r, c = np.array(iop[:3], float), np.array(iop[3:], float)
        return float(np.dot(np.array(ipp, float), np.cross(r, c)))
    if ds.get("SliceLocation") is not None:
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0) or 0) + (frame or 0)


def dicom_series_volume(files) -> np.ndarray:
    """One series' DICOM files -> (n_slices, H, W) uint8, exactly as the shards store it."""
    import pydicom

    slices = []
    bad = 0
    for f in files:
        # one slice that will not decode should cost that slice, not the series.
        # transfer syntaxes vary across this corpus and a single odd file is common.
        try:
            ds = pydicom.dcmread(str(f), force=True)
            a = ds.pixel_array
        except Exception:
            bad += 1
            continue
        if a.ndim == 3:                                  # multi-frame: split it out
            slices.extend((ds, a[k], k) for k in range(a.shape[0]))
        else:
            slices.append((ds, a, None))
    if bad:
        print(f"  ⚠️  {bad}/{len(files)} slices would not decode", flush=True)
    if not slices:
        return np.zeros((0, 0, 0), np.uint8)

    slices.sort(key=lambda it: _slice_sort_key(it[0], it[2]))
    ref = slices[0][0]

    h, w = slices[0][1].shape[:2]
    if not all(s[1].shape[:2] == (h, w) for s in slices):
        shapes = [s[1].shape[:2] for s in slices]
        h, w = max(set(shapes), key=shapes.count)
        slices = [(d, (cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
                       if a.shape[:2] != (h, w) else a), f) for d, a, f in slices]

    vol = np.stack([s[1] for s in slices]).astype(np.float32)
    slope = float(getattr(ref, "RescaleSlope", 1) or 1)
    inter = float(getattr(ref, "RescaleIntercept", 0) or 0)
    if slope != 1 or inter != 0:
        vol = vol * slope + inter

    lo = float(np.percentile(vol, SHARD_LO_PCT))
    hi = float(np.percentile(vol, SHARD_HI_PCT))
    if hi <= lo:
        lo, hi = float(vol.min()), float(max(vol.max(), vol.min() + 1))
    return (np.clip((vol - lo) / (hi - lo), 0, 1) * 255.0 + 0.5).astype(np.uint8)


def scan_dicom_index(dicom_root, series_csv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Header-only scan -> the schema 07_series_index.csv has, plus {series: [files]}.

    Files are grouped by the SeriesInstanceUID in their header rather than by folder,
    so any directory layout works. plan_slots and the mm/px resampling both need the
    geometry, and reading one header per file is far cheaper than decoding pixels.
    """
    import pydicom

    root = Path(dicom_root)
    files_by_series: dict[str, list] = {}
    geom: dict[str, dict] = {}

    def _geom(ds, study_uid):
        ps = ds.get("PixelSpacing") or [1.0, 1.0]
        thick = float(ds.get("SliceThickness", 0) or 0)
        return {"pixel_spacing_r": float(ps[0]), "pixel_spacing_c": float(ps[1]),
                "slice_thickness": thick,
                "spacing_between": float(ds.get("SpacingBetweenSlices", 0) or 0) or thick,
                "rows": int(ds.get("Rows", 0)), "cols": int(ds.get("Columns", 0)),
                "study": study_uid}

    # fast path: the layout is <root>/<study>/<series>/<file>.dcm, so the UIDs are
    # already in the path and one header per series is enough for the geometry.
    # reading all of them costs ~2s per study, which is real money over a big test set.
    fast, by_folder = 0, set()
    for _, r in series_csv.iterrows():
        d = root / str(r["StudyInstanceUID"]) / str(r["SeriesInstanceUID"])
        if not d.is_dir():
            continue
        fs = sorted(f for f in d.iterdir() if f.is_file())
        if not fs:
            continue
        try:
            ds = pydicom.dcmread(str(fs[0]), stop_before_pixels=True, force=True)
        except Exception:
            continue
        ser = str(r["SeriesInstanceUID"])
        files_by_series[ser] = fs
        geom[ser] = _geom(ds, str(r["StudyInstanceUID"]))
        by_folder.add(ser)
        fast += 1
    if fast:
        print(f"  🗂️  {fast} series indexed by folder "
              f"({len(series_csv) - fast} need a header scan)")

    # fallback for whatever the folder pass could not place. this must stay scoped to
    # the studies actually missing: rglob over the whole root is ~240k header reads on
    # the real test set, and a single series without a folder would trigger it.
    missing = series_csv[~series_csv["SeriesInstanceUID"].astype(str).isin(by_folder)]
    if len(missing):
        print(f"  \U0001F50E {len(missing)} series not found by folder, searching "
              f"{missing['StudyInstanceUID'].nunique()} study dirs", flush=True)
        want_all = set(missing["SeriesInstanceUID"].astype(str))
        for study_uid, g in missing.groupby("StudyInstanceUID"):
            sdir = root / str(study_uid)
            if not sdir.is_dir():
                continue                                 # handled by the sweep below
            want = set(g["SeriesInstanceUID"].astype(str))
            for f in sdir.rglob("*"):
                if not f.is_file():
                    continue
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                except Exception:                        # not a DICOM, skip it
                    continue
                ser = str(getattr(ds, "SeriesInstanceUID", "") or "")
                if ser not in want:
                    continue
                files_by_series.setdefault(ser, []).append(f)
                geom.setdefault(ser, _geom(ds, str(study_uid)))
                want_all.discard(ser)

        # only now, if some series has no study folder to look in at all (a flat dump,
        # say), sweep the whole root. this is the expensive path and it says so.
        still = want_all - set(files_by_series)
        if still:
            print(f"  \u26A0\uFE0F  {len(still)} series still unplaced, sweeping the "
                  f"whole root by header — this is slow", flush=True)
            for f in root.rglob("*"):
                if not still:
                    break
                if not f.is_file():
                    continue
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                except Exception:
                    continue
                ser = str(getattr(ds, "SeriesInstanceUID", "") or "")
                if ser not in still and ser not in files_by_series:
                    continue
                if ser in still or ser in geom:
                    files_by_series.setdefault(ser, []).append(f)
                    geom.setdefault(ser, _geom(
                        ds, str(getattr(ds, "StudyInstanceUID", "") or "")))

    plane_map = {"sagittal": "sag", "coronal": "cor", "axial": "ax"}
    rows = []
    for _, r in series_csv.iterrows():
        ser = r["SeriesInstanceUID"]
        if ser not in files_by_series:
            continue
        g = geom[ser]
        plane = str(r["Anatomical_Plane"]).lower()
        pre = plane_map.get(plane, plane[:3])
        n = len(files_by_series[ser])
        rows.append({"study": r["StudyInstanceUID"], "series": ser,
                     "plane": plane, "fluid": int(r["Fluid_Sensitive"]),
                     "slot": f"{pre}_{'fluid' if int(r['Fluid_Sensitive']) else 'plain'}",
                     "n_slices": n, "rows": g["rows"], "cols": g["cols"],
                     "pixel_spacing_r": g["pixel_spacing_r"],
                     "pixel_spacing_c": g["pixel_spacing_c"],
                     "slice_thickness": g["slice_thickness"],
                     "spacing_between": g["spacing_between"],
                     "depth_mm": g["spacing_between"] * n})
    return pd.DataFrame(rows), files_by_series


def verify_dicom_path(cfg: dict, dicom_root, series_csv: pd.DataFrame,
                      shards_dir=None, limit: int = 3) -> bool:
    """Decode a few series from DICOM and assert they equal the stored shard.

    This is the check that catches a silent preprocessing drift before it costs a
    submission — a mismatch here shifts the input distribution the model sees.
    """
    shards = Path(shards_dir or cfg["paths"]["shards_dir"])
    idx, files_by_series = scan_dicom_index(dicom_root, series_csv)
    checked = ok = 0
    for _, r in idx.iterrows():
        shard = shards / f"{r['study']}.npz"
        if not shard.exists():
            continue
        with np.load(shard) as z:
            if r["series"] not in z.files:
                continue
            stored = z[r["series"]]
        got = dicom_series_volume(files_by_series[r["series"]])
        checked += 1
        same = got.shape == stored.shape and np.array_equal(got, stored)
        ok += bool(same)
        print(f"  {'✅' if same else '❌'} {r['series'][:24]}...  "
              f"dicom {got.shape} vs shard {stored.shape}"
              + ("" if same else f"  max|diff|={np.abs(got.astype(int) - stored.astype(int)).max()}"
                 if got.shape == stored.shape else "  SHAPE MISMATCH"))
        if checked >= limit:
            break
    print(f"  {ok}/{checked} series reproduced exactly from DICOM")
    return checked > 0 and ok == checked


def build_kaggle_ensemble_assets(members: list, out_dir, kaggle_user: str,
                                 dataset_slug: str, notebook_slug: str, title: str = "",
                                 blend_weights: dict | None = None):
    """Bundle SEVERAL runs into one Kaggle dataset + a notebook that averages them.

    members: [(name, cfg, exp_dir, checkpoint_path), ...] - member 0 owns the loader.

    Layout differs from the single-model bundle in one way that matters: each member
    gets its own m<i>_model.pt / m<i>_config.json / m<i>_def.py, all FLAT in the
    dataset (kaggle datasets skip subdirectories), and the notebook imports each def
    under its own module name. `from model_def import build_model` would collide -
    every run's train.py defines a class called SliceModel.
    """
    out = Path(out_dir)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    (out / "notebook").mkdir(parents=True, exist_ok=True)
    w = out / "weights"

    manifest = []
    stale = []
    for i, (name, cfg, exp_dir, ckpt) in enumerate(members):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        # A checkpoint scoring BELOW its own run's best validation is a stale copy -
        # typically pulled from the workspace where a run STARTED, when the run was handed
        # off mid-training and finished somewhere else. This is invisible without a check:
        # the ensemble sweep scores members from results/predictions/*.npz (which holds
        # every validation) while THIS ships the checkpoint, so the sweep can promise one
        # model and the submission send a worse one. convnext_large shipped 0.8780 while
        # being measured at 0.9061, across at least one real submission.
        try:
            _vl = Path(exp_dir) / cfg["output"]["run_dir"] / "val_log.csv"
            if _vl.exists() and ck.get("score") is not None:
                _best = float(pd.read_csv(_vl)["val_macro_auroc"].max())
                if float(ck["score"]) < _best - 0.002:
                    stale.append((name, ck.get("step"), float(ck["score"]), _best))
        except Exception as _e:
            print(f"  [warn] could not sanity-check {name}'s checkpoint: "
                  f"{type(_e).__name__}: {_e}")
        torch.save({"model": ck["model"], "cfg": ck.get("cfg", cfg),
                    "epoch": ck.get("epoch"), "step": ck.get("step"),
                    "score": ck.get("score")}, w / f"m{i}_model.pt")
        c = json.loads(json.dumps(cfg, default=str))
        c["model"]["pretrained"] = False           # no internet at inference
        c["model"].pop("weights", None)            # ...and no external checkpoint either:
                                                   # the trunk arrives inside m<i>_model.pt,
                                                   # so _weights_path would only crash
        (w / f"m{i}_config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
        src = (Path(exp_dir) / "train.py").read_text(encoding="utf-8").split("def run_local(")[0]
        src = "\n".join(l for l in src.splitlines()
                        if not l.startswith("cfg = sc.load_config("))
        (w / f"m{i}_def.py").write_text(src, encoding="utf-8")
        manifest.append({"i": i, "name": name, "epoch": ck.get("epoch"),
                         "step": ck.get("step"), "score": ck.get("score")})
        print(f"  \U0001F4E6 m{i} {name:30s} epoch {ck.get('epoch')} "
              f"step {ck.get('step')} score {ck.get('score')}")
    if stale:
        lines = "\n".join(
            f"    {n}: shipping step {st} @ {sc_:.4f}, but val_log's best is {b:.4f}"
            for n, st, sc_, b in stale)
        raise SystemExit(
            f"STALE CHECKPOINT(S) - refusing to build a submission that ships a model "
            f"worse than the one measured:\n{lines}\n"
            f"  The run was probably handed off between workspaces and this best.pt came "
            f"from the one where it STARTED. Find the workspace holding the later "
            f"checkpoints and pull its best.pt.")

    (w / "members.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Per-class blend weights, if the caller fitted some. Written as an explicit
    # (n_members, n_tasks) matrix in MEMBER order and TASK order, not as a dict, so the
    # notebook cannot silently reorder it. ensemble.py saves a dict keyed by name; the
    # conversion happens HERE, once, where both orders are known.
    if blend_weights:
        tasks0 = members[0][1]["tasks"]
        names = [m[0] for m in members]
        missing_t = [t for t in tasks0 if t not in blend_weights]
        if missing_t:
            raise ValueError(f"blend_weights is missing {len(missing_t)} task(s): "
                             f"{missing_t[:4]}")
        mat = []
        for name in names:
            row = []
            for t in tasks0:
                per = blend_weights[t]
                if name not in per:
                    raise ValueError(f"blend_weights[{t!r}] has no entry for {name!r} - "
                                     f"it holds {sorted(per)}")
                row.append(float(per[name]))
            mat.append(row)
        (w / "blend_weights.json").write_text(json.dumps(
            {"members": names, "tasks": tasks0, "matrix": mat}, indent=2), encoding="utf-8")
        print(f"  \u2696\ufe0f  blend weights: {len(names)} members x {len(tasks0)} findings")
        for name, row in zip(names, mat):
            print(f"     {name:32s} {' '.join(f'{v:.2f}' for v in row)}")

    # DINOv3's licence is not permissive like MRI-CORE's Apache-2.0: redistributing the
    # weights - and putting them in a Kaggle dataset IS redistribution - obliges us to
    # include the licence text and to display "Built with DINOv3" wherever the materials
    # are made available. Neither is optional, and neither was being shipped until a
    # rules audit caught it, so it is automated here rather than left to remember.
    _dino = [m[0] for m in members if "dinov3" in m[0].lower()]
    if _dino:
        lic_src = Path(__file__).resolve().parent.parent / "data" / "licenses" / "DINOV3_LICENSE.md"
        if not lic_src.exists():
            raise SystemExit(
                f"{_dino} is in this submission but the DINOv3 licence is missing at\n"
                f"  {lic_src}\n"
                f"Redistributing DINOv3 weights without the licence breaches its terms. "
                f"Fetch it from https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md")
        shutil.copy(lic_src, w / "DINOV3_LICENSE.md")
        (w / "NOTICE.md").write_text(
            "# Built with DINOv3\n\n"
            "This model bundle contains weights derived from Meta's DINOv3 "
            "(members: " + ", ".join(_dino) + ").\n\n"
            "DINOv3 is distributed under the DINOv3 License, a copy of which is included "
            "in this dataset as DINOV3_LICENSE.md. Use of the DINOv3 materials is subject "
            "to that licence.\n\n"
            "Other pretrained weights in this bundle carry their own licences and are not "
            "covered by the DINOv3 License.\n", encoding="utf-8")
        print(f"  \u2696\ufe0f  DINOv3 member(s) {_dino} -> shipped DINOV3_LICENSE.md "
              f"+ 'Built with DINOv3' NOTICE.md")

    shutil.copy(Path(__file__), w / "shared_code.py")
    wheels_src = Path(__file__).resolve().parent / "_wheels"
    for whl in (wheels_src.glob("*.whl") if wheels_src.is_dir() else []):
        shutil.copy(whl, w / whl.name)
    (w / "dataset-metadata.json").write_text(json.dumps(
        {"title": title or f"knee ensemble weights ({len(members)})",
         "id": f"{kaggle_user}/{dataset_slug}",
         "licenses": [{"name": "CC0-1.0"}]}, indent=2), encoding="utf-8")

    (out / "notebook" / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{kaggle_user}/{notebook_slug}",
        # kaggle slugifies the TITLE and 409s if it does not match the id, so derive
        # the title from the slug rather than writing a prettier one
        "title": notebook_slug.replace("-", " "),
        "code_file": "submit.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{kaggle_user}/{dataset_slug}"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
    }, indent=2), encoding="utf-8")

    # "prominently display" means the notebook, not just a file in the dataset
    _notice_cell = None
    if _dino:
        _notice_cell = (
            "# Built with DINOv3\n\n"
            "This notebook uses weights derived from Meta's **DINOv3**. DINOv3 is "
            "distributed under the DINOv3 License, included in the attached dataset as "
            "`DINOV3_LICENSE.md`.\n\n"
            "Other pretrained weights used here carry their own separate licences.")

    cells = [
        r"""import sys, glob, json, subprocess, traceback, importlib.util
from pathlib import Path
hits = sorted(glob.glob('/kaggle/input/**/shared_code.py', recursive=True))
if not hits:
    raise SystemExit('shared_code.py not found. tree: '
                     + str(sorted(glob.glob('/kaggle/input/**/*', recursive=True))[:200]))
W = Path(hits[0]).parent
sys.path.insert(0, str(W))
subs = sorted(glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True))
if not subs:
    raise SystemExit('competition data not found under /kaggle/input')
COMP = Path(subs[0]).parent
MEMBERS = json.loads((W / 'members.json').read_text())
BLEND = None
_bw = W / 'blend_weights.json'
if _bw.exists():
    BLEND = json.loads(_bw.read_text())
    print('per-class blend weights found for', len(BLEND['members']), 'members')
else:
    print('no blend_weights.json -> flat average')
print('weights:', W, '| comp:', COMP)
print('members:', [m['name'] for m in MEMBERS])""",
        r"""# the test set mixes JPEG Lossless and JPEG 2000 in with the raw series and pydicom
# reads neither on its own. no internet here, so the wheels ride along in the dataset.
wheels = sorted(glob.glob(str(W / '*.whl')))
try:
    import pylibjpeg, openjpeg, libjpeg
    print('dicom decoders already present')
except ImportError:
    if wheels:
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                            '--no-deps', '-q', *wheels], capture_output=True, text=True)
        print('installed decoders rc=', r.returncode, r.stderr[-300:])
    else:
        print('WARNING: no decoder wheels, compressed series will fail')""",
        r"""import shared_code as sc, pandas as pd, torch
series = pd.read_csv(COMP / 'test_series.csv')
sub = pd.read_csv(COMP / 'sample_submission.csv')
TASKS = [c for c in sub.columns if c != 'StudyInstanceUID']
print(len(sub), 'studies to score')

def bail(why):
    # always leave a valid file behind: 0.5 everywhere scores about 0.5, while no file
    # at all scores nothing and still costs one of the five daily attempts
    traceback.print_exc()
    print('FALLING BACK TO 0.5:', why)
    out = sub.copy()
    out[TASKS] = 0.5
    out.to_csv('submission.csv', index=False)

bail('nothing has run yet')          # overwritten the moment real work lands""",
        r"""# member 0 owns the loader geometry; the others only re-normalise the same batch
try:
    CFGS = [json.loads((W / ('m%d_config.json' % m['i'])).read_text()) for m in MEMBERS]
    idx, files_by_series = sc.scan_dicom_index(COMP / 'test_series', series)
    plan = sc.plan_slots(idx, CFGS[0])
    print(len(plan), 'studies planned of', len(sub))
except Exception:
    bail('scan/plan failed'); raise SystemExit(0)""",
        r"""# each member's model_def under its OWN module name - every run's train.py defines a
# class called SliceModel, so one shared `model_def` would hand every member the same
# architecture and silently score the wrong thing
def _load_def(i):
    spec = importlib.util.spec_from_file_location('m%d_def' % i, W / ('m%d_def.py' % i))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['m%d_def' % i] = mod
    spec.loader.exec_module(mod)
    return mod

try:
    MODELS = []
    for m, cfg in zip(MEMBERS, CFGS):
        i = m['i']
        model = _load_def(i).build_model(cfg)
        ck = torch.load(W / ('m%d_model.pt' % i), map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        MODELS.append(model.eval())
        print('m%d %-30s epoch %s score %s' % (i, m['name'], ck.get('epoch'), ck.get('score')))
except Exception:
    bail('model load failed'); raise SystemExit(0)""",
        r"""# STRESS repeats the public studies to imitate the hidden set's size. 0 = a normal run.
STRESS = 0

todo = sub[['StudyInstanceUID']]
if STRESS:
    reps = -(-STRESS // len(todo))
    todo = pd.concat([todo] * reps, ignore_index=True).head(STRESS)
    print('STRESS MODE:', len(todo), 'studies')

# ONE decode pass shared by every member. Decoding is ~9s/study against <1s of GPU,
# so looping predict_dicom per member would be ~4x9s = over the 9h cap; this is
# ~9s + N x <1s. Averaging happens in logit space, as measured offline.
import time as _time
_t0 = _time.time()
try:
    # reorder the saved matrix into MEMBERS order, and into the task order this
    # notebook actually submits in - both are checked, because a silent mismatch here
    # blends the wrong model into the wrong finding and still writes a valid-looking file
    _W = None
    if BLEND:
        _names = [m['name'] for m in MEMBERS]
        assert _names == BLEND['members'], (
            'member order differs: %s vs %s' % (_names, BLEND['members']))
        _ti = [BLEND['tasks'].index(t) for t in TASKS]
        _W = [[row[j] for j in _ti] for row in BLEND['matrix']]
        print('using per-class blend weights')
    pred = sc.predict_dicom_ensemble(CFGS, MODELS, todo, plan, idx, files_by_series,
                                     batch_size=1, weights=_W)
    # the 9h cap is enforced on the HIDDEN test set, which is larger than what runs
    # here, so the number that has to be extrapolated is s/study, not the total
    _el = _time.time() - _t0
    _rate = _el / max(len(todo), 1)
    print('INFERENCE  %.1f s for %d studies  =  %.2f s/study  |  9h cap fits ~%d studies'
          % (_el, len(todo), _rate, int(9 * 3600 / max(_rate, 1e-9))))
    if not STRESS:
        out = sub[['StudyInstanceUID']].merge(pred, on='StudyInstanceUID', how='left')
        out[TASKS] = out[TASKS].astype(float).fillna(0.5).clip(0, 1)
        assert len(out) == len(sub), '%d rows for %d studies' % (len(out), len(sub))
        out.to_csv('submission.csv', index=False)
        print(out[TASKS].describe().T[['mean', 'min', 'max']])
except Exception:
    bail('inference failed')
pd.read_csv('submission.csv').head()""",
    ]
    _cells = [{"cell_type": "code", "metadata": {}, "source": c.splitlines(True),
               "outputs": [], "execution_count": None} for c in cells]
    if _notice_cell:                      # first cell, so it is the first thing seen
        _cells.insert(0, {"cell_type": "markdown", "metadata": {},
                          "source": _notice_cell.splitlines(True)})
    nb = {"cells": _cells,
          "metadata": {"kernelspec": {"name": "python3", "language": "python",
                                      "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / "notebook" / "submit.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in w.iterdir()) / 1e9
    print(f"  \U0001F4E6 weights   -> {w}  ({size:.2f} GB)")
    print(f"  \U0001F4D3 notebook  -> {out / 'notebook'}")
    _kv("kaggle commands", [
        ("1 dataset", f"kaggle datasets create -p {w}"),
        ("2 notebook", f"kaggle kernels push -p {out / 'notebook'}"),
        ("3 status", f"kaggle kernels status {kaggle_user}/{notebook_slug}"),
    ], "\U0001F680")
    return out


def build_kaggle_assets(cfg: dict, exp_dir, out_dir, checkpoint_path,
                        kaggle_user: str, dataset_slug: str = "", notebook_slug: str = ""):
    """Write everything `kaggle kernels push` needs: weights, config, notebook, metadata.

    Two Kaggle objects are involved: a DATASET holding the weights (notebooks run
    offline, so the checkpoint cannot be downloaded at run time) and a NOTEBOOK that
    attaches both it and the competition data.
    """
    exp = cfg["experiment"]["name"]
    out = Path(out_dir)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    (out / "notebook").mkdir(parents=True, exist_ok=True)
    dataset_slug = dataset_slug or f"knee-{exp.replace('_', '-')}"
    notebook_slug = notebook_slug or f"knee-{exp.replace('_', '-')}-submit"

    # weights: strip the optimizer/scheduler state, it is dead weight for inference
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch.save({"model": ck["model"], "cfg": ck.get("cfg", cfg),
                "epoch": ck.get("epoch"), "step": ck.get("step"),
                "score": ck.get("score")}, out / "weights" / "model.pt")
    # the trunk arrives inside model.pt, so an external `weights` path (MRI-CORE,
    # RadImageNet) must NOT survive into the kernel - build_model would go looking for
    # a 435 MB file that never travelled and die before scoring a single study
    _kcfg = json.loads(json.dumps(cfg, default=str))
    _kcfg["model"]["pretrained"] = False
    _kcfg["model"].pop("weights", None)
    (out / "weights" / "config.json").write_text(json.dumps(_kcfg, indent=2))
    (out / "weights" / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"knee {exp} weights", "id": f"{kaggle_user}/{dataset_slug}",
         "licenses": [{"name": "CC0-1.0"}]}, indent=2))

    # the notebook imports shared_code from the weights dataset, so inference and
    # training run the exact same slot planning, resampling and windowing code
    shutil.copy(Path(__file__), out / "weights" / "shared_code.py")
    train_py = Path(exp_dir) / "train.py"
    if train_py.exists():
        # keep only the model definition — the kaggle kernel has no yaml configs and
        # no modal, so the module-level load_config and everything from run_local down
        # would blow up on import
        src = train_py.read_text(encoding="utf-8")
        src = src.split("def run_local(")[0]
        src = "\n".join(l for l in src.splitlines()
                        if not l.startswith("cfg = sc.load_config("))
        (out / "weights" / "model_def.py").write_text(src, encoding="utf-8")

    # the test DICOMs arrive as JPEG Lossless and JPEG 2000 as well as raw, and the
    # notebook has no internet to pip install a decoder — so ship the wheels with it
    # flat, not in a wheels/ subfolder: `kaggle datasets` skips subdirectories
    # unless you pass --dir-mode, and that flag is broken on windows in CLI 2.2.4
    wheels_src = Path(__file__).resolve().parent / "_wheels"
    if wheels_src.is_dir():
        wd = out / "weights"
        for whl in wheels_src.glob("*.whl"):
            shutil.copy(whl, wd / whl.name)
        print(f"  \U0001F4E6 wheels     -> {len(list(wd.glob('*.whl')))} dicom decoders")

    (out / "notebook" / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{kaggle_user}/{notebook_slug}",
        "title": f"knee {exp} submit",
        "code_file": "submit.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        # the competition bans P100 (sm_60, no kernels in kaggle's torch build).
        # this is the exact enum the server uses for "GPU T4 x2" in the UI.
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{kaggle_user}/{dataset_slug}"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
    }, indent=2))

    cells = [
        '''# kaggle nests the mounts (/kaggle/input/datasets/<user>/<slug>) and the rerun
# swaps in the hidden data, so nothing here may be hardcoded
import sys, glob, json, subprocess, traceback
from pathlib import Path
hits = sorted(glob.glob('/kaggle/input/**/shared_code.py', recursive=True))
if not hits:
    raise SystemExit('shared_code.py not found. tree: '
                     + str(sorted(glob.glob('/kaggle/input/**/*', recursive=True))[:200]))
W = Path(hits[0]).parent
sys.path.insert(0, str(W))
subs = sorted(glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True))
if not subs:
    raise SystemExit('competition data not found under /kaggle/input')
COMP = Path(subs[0]).parent
print('weights:', W, '| comp:', COMP)''',

        '''# the test set mixes JPEG Lossless and JPEG 2000 in with the raw series and
# pydicom reads neither on its own. no internet here, so the wheels ride along
# inside the weights dataset.
wheels = sorted(glob.glob(str(W / '*.whl')))
try:
    import pylibjpeg, openjpeg, libjpeg
    print('dicom decoders already present')
except ImportError:
    if wheels:
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                            '--no-deps', '-q', *wheels], capture_output=True, text=True)
        print('installed decoders rc=', r.returncode, r.stderr[-300:])
    else:
        print('WARNING: no decoder wheels, compressed series will fail')''',

        '''import shared_code as sc, pandas as pd, torch
cfg = json.loads((W / 'config.json').read_text())
cfg['model']['pretrained'] = False        # no internet: weights come from the ckpt
series = pd.read_csv(COMP / 'test_series.csv')
sub = pd.read_csv(COMP / 'sample_submission.csv')
TASKS = [c for c in sub.columns if c != 'StudyInstanceUID']
print(len(sub), 'studies to score')

def bail(why):
    # always leave a valid file behind: 0.5 everywhere scores about 0.5, while no
    # file at all scores nothing and still costs one of the five daily attempts
    traceback.print_exc()
    print('FALLING BACK TO 0.5:', why)
    out = sub.copy()
    out[TASKS] = 0.5
    out.to_csv('submission.csv', index=False)

bail('nothing has run yet')          # overwritten the moment real work lands''',

        '''# header scan -> which series fills which slot, plus the mm/px geometry
try:
    idx, files_by_series = sc.scan_dicom_index(COMP / 'test_series', series)
    plan = sc.plan_slots(idx, cfg)
    print(len(plan), 'studies planned of', len(sub))
except Exception:
    bail('scan/plan failed'); raise SystemExit(0)''',

        '''try:
    from model_def import build_model
    model = build_model(cfg)
    ck = torch.load(W / 'model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    print('loaded epoch', ck.get('epoch'), 'score', ck.get('score'))
except Exception:
    bail('model load failed'); raise SystemExit(0)''',

        '''# STRESS repeats the public studies to imitate the hidden set's size. the
# rerun log is never shown to us, so this is the only way to prove the pipeline
# survives ~1300 studies before spending a submission. 0 = a normal run.
STRESS = 0

todo = sub[['StudyInstanceUID']]
if STRESS:
    reps = -(-STRESS // len(todo))
    todo = pd.concat([todo] * reps, ignore_index=True).head(STRESS)
    print('STRESS MODE:', len(todo), 'studies')

# one study per batch: 4 slots x 24 slices is already 96 images, and kaggle's
# T4 has 15 GB against the 80 GB this trained on
try:
    pred = sc.predict_dicom(cfg, model, todo, plan, idx, files_by_series,
                            batch_size=1)
    if not STRESS:
        # merge rather than trust row order, and fill anything missing with 0.5,
        # so the file always has exactly the studies the sample submission asks for
        out = sub[['StudyInstanceUID']].merge(pred, on='StudyInstanceUID', how='left')
        out[TASKS] = out[TASKS].astype(float).fillna(0.5).clip(0, 1)
        assert len(out) == len(sub), f'{len(out)} rows for {len(sub)} studies'
        out.to_csv('submission.csv', index=False)
        print(out[TASKS].describe().T[['mean', 'min', 'max']])
except Exception:
    bail('inference failed')
pd.read_csv('submission.csv').head()''',

    ]
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": c.splitlines(True),
                     "outputs": [], "execution_count": None} for c in cells],
          "metadata": {"kernelspec": {"name": "python3", "language": "python",
                                      "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / "notebook" / "submit.ipynb").write_text(json.dumps(nb, indent=1))

    print(f"  📦 weights   -> {out / 'weights'}")
    print(f"  📓 notebook  -> {out / 'notebook'}")
    _kv("kaggle commands", [
        ("1 dataset", f"kaggle datasets create -p {out / 'weights'}   "
                      f"(later: version -p ... -m msg)"),
        ("2 notebook", f"kaggle kernels push -p {out / 'notebook'}"),
        ("3 status", f"kaggle kernels status {kaggle_user}/{notebook_slug}"),
        ("4 submit", f"kaggle competitions submit -c rsna-knee-abnormality-detection "
                     f"-f submission.csv -k {kaggle_user}/{notebook_slug} -v <VERSION> -m \"{exp}\""),
    ], "\U0001F680")
    return out


@torch.no_grad()
def predict_dicom_ensemble(cfgs: list, models: list, studies_df: pd.DataFrame, plan: dict,
                           index: pd.DataFrame, files_by_series: dict, device=None,
                           batch_size: int | None = None, num_workers: int | None = None,
                           report_every: int = 50, weights=None) -> pd.DataFrame:
    """Score several models in ONE pass over the DICOMs and average their logits.

    Decoding a study costs ~9s against <1s of GPU, so looping predict_dicom once per
    member would re-decode everything N times: four members is ~13h against the
    competition's 9h cap. Decoding once and running every model on the same uint8
    batch is ~9s + N x <1s, which fits.

    The members must agree on the LOADER geometry (slots, n_slices, size, mm_per_px,
    windowing) because they share the batch. They may still differ in normalisation -
    that is applied per model in prepare_batch, on GPU, from the same uint8 tensor
    (MRI-CORE wants [0,1], the ImageNet backbones want their own mean/std).

    Averaging is in LOGIT space, matching how the offline ensemble was measured.

    weights: optional per-class blend weights, shaped (n_members, n_tasks) in the SAME
    member order as `models` and the same task order as cfgs[0]["tasks"]. None means the
    flat mean. Fitted weights come from ensemble.py's search, whose held-out (out-of-fold)
    estimate beat flat by ~0.002 - the in-sample number it also reports is ~0.004 higher
    still and must not be believed.
    """
    device = device or torch.device("cuda" if usable_cuda() else "cpu")
    nw = num_workers if num_workers is not None else min(4, (os.cpu_count() or 2))
    cfg0, tasks = cfgs[0], cfgs[0]["tasks"]

    # A silently mis-shaped or mis-ordered weight matrix would blend the wrong member into
    # the wrong finding and still produce a plausible submission, so check it loudly here
    # rather than trusting the caller.
    W = None
    if weights is not None:
        W = torch.as_tensor(np.asarray(weights, dtype=np.float32))
        if W.shape != (len(models), len(tasks)):
            raise ValueError(f"weights must be (n_members, n_tasks) = "
                             f"({len(models)}, {len(tasks)}), got {tuple(W.shape)}")
        col = W.sum(dim=0)
        if not torch.all(col > 0):
            raise ValueError(f"every task needs a non-zero weight column, got sums {col}")
        W = (W / col).to(device)                  # each finding's weights sum to 1
        print(f"  \u2696\ufe0f  weighted blend over {len(models)} members x {len(tasks)} findings")
    n = len(studies_df)

    # slice_context is a LOADER setting (KneeShardDataset._load fills the RGB channels
    # with neighbours), not a prepare_batch one, so a 2.5D member cannot ride along on a
    # duplicate-channel decode pass - it would silently score the wrong pixels.
    geo = ("slots", "n_slices", "size", "mm_per_px", "window", "window_pct", "channels",
           "slice_context", "slice_context_stride")
    for i, c in enumerate(cfgs[1:], 1):
        bad = [k for k in geo if c["input"].get(k) != cfg0["input"].get(k)]
        if bad:                                    # loud: a silent mismatch would score
            raise ValueError(                      # every member on the wrong pixels
                f"member {i} disagrees on loader geometry {bad} - it cannot share a "
                f"decode pass with member 0")

    ds = KneeShardDataset(studies_df.assign(**{t: 0.0 for t in tasks}),
                          cfg0, plan, train=False, dicom=(index, files_by_series))
    bs = int(batch_size or val_loader_settings(cfg0)[0])
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=device.type == "cuda", persistent_workers=False)
    for m in models:
        m.eval().to(device)

    out, done, t0, failed = [], 0, time.time(), 0
    it = iter(loader)
    while True:
        try:
            x, _ = next(it)
        except StopIteration:
            break
        except Exception as e:
            left = n - done
            print(f"  ❌ loader died at study {done} ({type(e).__name__}: {e}); "
                  f"remaining {left} score 0.5")
            out.append(np.full((left, len(tasks)), 0.5, np.float32))
            failed += left
            break
        k = x.shape[0]
        try:
            lg = []
            for c, m in zip(cfgs, models):
                amp = c["training"]["amp"] and device.type == "cuda"
                with torch.autocast("cuda", enabled=amp):
                    lg.append(m(prepare_batch(x, c, device)).float())
            stacked = torch.stack(lg)                  # (n_members, batch, n_tasks)
            if W is None:
                avg = stacked.mean(dim=0)              # logit space, as measured offline
            else:
                # W is (n_members, n_tasks) -> unsqueeze the batch axis so each finding
                # gets its own weighting across members
                avg = (stacked * W[:, None, :]).sum(dim=0)
            out.append(torch.sigmoid(avg).cpu().numpy())
        except Exception as e:                     # a dead batch scores 0.5, run goes on
            failed += k
            print(f"  ⚠️  batch at study {done} failed ({type(e).__name__}: {e}) -> 0.5")
            out.append(np.full((k, len(tasks)), 0.5, np.float32))
        done += k
        if report_every and done % report_every < k:
            el = time.time() - t0
            print(f"  … {done}/{n} studies  {el / max(done, 1):.1f}s each  "
                  f"eta {fmt(el / max(done, 1) * (n - done))}", flush=True)
    probs = np.concatenate(out) if out else np.zeros((0, len(tasks)))
    print(f"  ✅ scored {len(probs)} studies with {len(models)} members in "
          f"{fmt(time.time() - t0)}" + (f" ({failed} fell back to 0.5)" if failed else ""))
    res = pd.DataFrame(probs, columns=tasks)
    res.insert(0, "StudyInstanceUID", studies_df["StudyInstanceUID"].values)
    return res


def usable_cuda() -> bool:
    """True only if torch actually has kernels for this GPU.

    Kaggle hands out P100s (sm_60) that their own torch build can't run — without
    this check inference dies with `no kernel image is available for execution`.
    """
    if not torch.cuda.is_available():
        return False
    try:
        cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
        arch = torch.cuda.get_arch_list()
        if arch and cap not in arch:
            print(f"  ⚠️  {torch.cuda.get_device_name(0)} is {cap}, "
                  f"torch has {arch} — falling back to CPU")
            return False
        torch.zeros(1, device="cuda").add_(1)       # a real launch, in case the list lies
        return True
    except Exception as e:
        print(f"  ⚠️  cuda unusable ({e}) — falling back to CPU")
        return False


@torch.no_grad()
def predict_dicom(cfg: dict, model, studies_df: pd.DataFrame, plan: dict,
                  index: pd.DataFrame, files_by_series: dict, device=None,
                  batch_size: int | None = None, num_workers: int | None = None,
                  report_every: int = 50) -> pd.DataFrame:
    """Score studies straight from DICOM and return the submission frame.

    Decoding DICOM costs ~9s per study against <1s of GPU, so the workers are what
    decide the wall clock here, not the model. One unreadable study must not take
    the whole rerun down with it, so every batch is guarded.
    """
    device = device or torch.device("cuda" if usable_cuda() else "cpu")
    nw = num_workers if num_workers is not None else min(4, (os.cpu_count() or 2))
    n = len(studies_df)
    print(f"  🖥️  inference device: {device} | {nw} loader workers | {n} studies")
    model = model.to(device).eval()
    ds = KneeShardDataset(studies_df.assign(**{t: 0.0 for t in cfg["tasks"]}),
                          cfg, plan, train=False,
                          dicom=(index, files_by_series))
    bs = int(batch_size or val_loader_settings(cfg)[0])
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=device.type == "cuda",
                        persistent_workers=False)
    out, done, t0, failed = [], 0, time.time(), 0
    it = iter(loader)
    while True:
        # the fetch itself has to be guarded: a worker that dies decoding a study
        # raises HERE, not inside the loop body, and would otherwise escape and
        # cost every remaining study
        try:
            x, _ = next(it)
        except StopIteration:
            break
        except Exception as e:
            left = n - done
            print(f"  ❌ loader died at study {done} ({type(e).__name__}: {e}); "
                  f"remaining {left} score 0.5")
            out.append(np.full((left, len(cfg["tasks"])), 0.5, np.float32))
            failed += left
            break
        k = x.shape[0]
        try:
            with torch.autocast("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda"):
                logits = model(prepare_batch(x, cfg, device))
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        except Exception as e:                     # a dead batch scores 0.5, the run goes on
            failed += k
            print(f"  ⚠️  batch at study {done} failed ({type(e).__name__}: {e}) -> 0.5")
            out.append(np.full((k, len(cfg["tasks"])), 0.5, np.float32))
        done += k
        if report_every and done % report_every < k:
            el = time.time() - t0
            print(f"  … {done}/{n} studies  {el / max(done, 1):.1f}s each  "
                  f"eta {fmt(el / max(done, 1) * (n - done))}", flush=True)
    probs = np.concatenate(out) if out else np.zeros((0, len(cfg["tasks"])))
    print(f"  ✅ scored {len(probs)} studies in {fmt(time.time() - t0)}"
          + (f" ({failed} fell back to 0.5)" if failed else ""))
    res = pd.DataFrame(probs, columns=cfg["tasks"])
    res.insert(0, "StudyInstanceUID", studies_df["StudyInstanceUID"].values)
    return res


def _kaggle(*args, check=True, quiet=False, cwd=None):
    """Run the kaggle CLI with this PC's CA bundle (Avast intercepts HTTPS)."""
    import subprocess

    env = dict(os.environ)
    ca = Path.home() / ".certs" / "ca-bundle.pem"
    if ca.exists():
        env["REQUESTS_CA_BUNDLE"] = env["SSL_CERT_FILE"] = str(ca)
    # the CLI writes kernel logs with the console codepage, and our prints have
    # emoji in them — without this the log download dies on a charmap error
    env["PYTHONUTF8"] = "1"                    # this one is a flag, not an encoding name
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env,
                       cwd=str(cwd) if cwd else None)
    out = (r.stdout or "") + (r.stderr or "")
    if not quiet:
        print("    " + out.strip().replace("\n", "\n    "))
    if check and r.returncode:
        raise SystemExit(f"❌ kaggle {' '.join(args)} failed ({r.returncode})")
    return out


def _wait_for_dataset(user: str, slug: str, tries: int = 20, secs: int = 30):
    """Block until kaggle finishes processing the upload.

    Push the notebook too early and the dataset simply is not mounted — the kernel
    dies with ModuleNotFoundError and you lose ten minutes working out why.
    """
    for i in range(tries):
        time.sleep(secs)
        out = _kaggle("datasets", "files", f"{user}/{slug}", quiet=True, check=False)
        if "model.pt" in out and "error" not in out.lower():
            print(f"      ✅ dataset ready after {(i + 1) * secs}s")
            return
    print("      ⚠️  dataset still processing, pushing the notebook anyway")


def submission_message(cfg: dict, exp_dir) -> str:
    """`<exp> | val 0.8940 | audit58 0.8901` - read off the finished run.

    Both numbers come from val_log.csv, deliberately. training_summary.json used to
    supply the val score, and it is a per-side "latest" file that the sync would leave
    stale - which is how convnext_small_4slot went to Kaggle labelled val 0.8491 when
    it had actually scored 0.8940. The CSV is merged across every resumption, so it is
    the one source that is always complete.

    audit58 is the panel score AT the best-val checkpoint, not the audit's own best,
    because best.pt is the checkpoint being submitted.
    """
    exp_dir = Path(exp_dir)
    name = cfg["experiment"]["name"]
    res = exp_dir / cfg.get("output", {}).get("run_dir", "results")
    try:
        log = pd.read_csv(res / "val_log.csv")
        row = log.loc[log["val_macro_auroc"].idxmax()]
        bits = [name, f"val {float(row['val_macro_auroc']):.4f}"]
        if "audit_macro_auroc" in log.columns and pd.notna(row["audit_macro_auroc"]):
            bits.append(f"audit58 {float(row['audit_macro_auroc']):.4f}")
        return " | ".join(bits)
    except Exception:
        pass
    try:                                    # no CSV: fall back to the summary
        summary = json.loads((res / "training_summary.json").read_text())
        return f"{name} | val {float(summary['best_score']):.4f}"
    except Exception:
        return name


def kaggle_publish(assets_dir, user: str, dataset_slug: str, notebook_slug: str,
                   message: str, wait: bool = True, poll_secs: int = 60,
                   submit: bool = True) -> str | None:
    """Dataset -> notebook -> wait for the run -> submit. Returns the kernel version.

    The weights go up as a dataset because a code-competition notebook has no
    internet; the notebook itself is what Kaggle scores.
    """
    assets = Path(assets_dir)
    _banner("KAGGLE PUBLISH", "\U0001F680")

    print(f"  1/4 dataset {user}/{dataset_slug}")
    # kaggle CLI 2.2.4 mangles -p on windows (weights/config.json -> weights_config.json),
    # so run it from inside the folder and hand it a bare dot
    w = assets / "weights"
    # `datasets status` answers 403 for a dataset that does not exist yet, so asking
    # is useless — try to create, and fall back to a new version if it is already there
    out = _kaggle("datasets", "create", "-p", ".", cwd=w, check=False)
    # kaggle says this three different ways depending on what already exists:
    # "already exists", a 409, a 403, or "title ... is already in use"
    low = out.lower()
    if any(k in low for k in ("already exists", "already in use", "409", "403")):
        print("      exists -> new version")
        _kaggle("datasets", "version", "-p", ".", "-m", message, cwd=w)
    elif "error" in out.lower() or "forbidden" in out.lower():
        raise SystemExit("❌ dataset upload failed")
    _wait_for_dataset(user, dataset_slug)

    print(f"\n  2/4 notebook {user}/{notebook_slug}")
    out = _kaggle("kernels", "push", "-p", str(assets / "notebook"))
    m = re.search(r"version\s+(\d+)", out, re.I)
    version = m.group(1) if m else None
    print(f"      pushed version {version or '?'}")

    if not wait:
        return version
    print("\n  3/4 waiting for the kernel to finish")
    # wall clock from "pushed" to "complete". Not the same as the kernel's own runtime
    # (kaggle queues before it starts), but it is the only figure visible from here, and
    # it upper-bounds the 9h cap. The notebook prints the real s/study inside its log.
    wait_t0 = time.time()
    while True:
        st = _kaggle("kernels", "status", f"{user}/{notebook_slug}", quiet=True)
        low = st.lower()
        if "complete" in low:
            el = time.time() - wait_t0
            print(f"      ✅ complete in {el / 60:.1f} min "
                  f"({el / 3600:.2f} h of the 9 h cap, queue included)"); break
        if "error" in low or "cancel" in low:
            print(f"      ❌ {st.strip()}")
            print("      fetch the log with: kaggle kernels output "
                  f"{user}/{notebook_slug} -p .")
            return version
        print(f"      … {st.strip()[:90]}")
        time.sleep(poll_secs)

    if not submit:
        return version
    print("\n  4/4 submitting")
    args = ["competitions", "submit", "-c", "rsna-knee-abnormality-detection",
            "-f", "submission.csv", "-k", f"{user}/{notebook_slug}", "-m", message]
    if version:
        args += ["-v", version]
    _kaggle(*args)
    return version
