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

import os

# Kaggle gives 4 vCPUs. Left alone, OpenMP and MKL each start a thread per core inside
# EVERY PyTorch intra-op call, so two per-GPU inference threads plus the DICOM decode
# workers oversubscribe the box and the second card costs more than it pays. These are
# read once, at import, so the cap has to sit above `import torch`.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import copy
import csv
import io
import json
import math
import multiprocessing as _mp
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

import logging as _logging


class _ExitWhenAppStopped(_logging.Filter):
    """Swallow modal's heartbeat tracebacks once the app is stopped, and exit the client.

    Stopping a run from the Modal dashboard leaves the local `modal run` process heartbeating
    a dead app every 10 s, and each reply - ConflictError "App state is APP_STATE_STOPPED" -
    is logged as a full traceback until Ctrl+C. That reply means the app is already gone, so
    the right end state is simply this process exiting. Anything else passes through.
    """

    _done = False

    def filter(self, record):
        exc = record.exc_info[1] if record.exc_info else None
        text = "%s %s" % (record.getMessage(), exc or "")
        if "APP_STATE_STOPPED" not in text:
            return True
        if not _ExitWhenAppStopped._done:
            _ExitWhenAppStopped._done = True
            print("\n  app stopped on Modal - closing the local client", flush=True)
            os._exit(0)                  # the remote app is already stopped; nothing to clean up
        return False


for _name in ("modal-utils", "modal-client"):
    _logging.getLogger(_name).addFilter(_ExitWhenAppStopped())

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
    # A run's own script must load ITS OWN folder. Every run starts as a copy of another, and
    # each script hardcodes EXP_NAME; a copy that keeps its parent's name silently trains the
    # PARENT (its labels, its results folder) with nothing failing. That cost a day twice
    # (15 Sep: five CV folds; 24 Sep: both MedGemma-label twins). Refuse to start instead.
    import inspect
    try:
        caller = Path(inspect.stack()[1].filename).resolve()
        if (caller.name in ("train.py", "evaluate.py", "submit.py", "plot_results.py")
                and (caller.parent / "config.yaml").exists()
                and caller.parent.parent.name == exp_dir.resolve().parent.name
                and caller.parent.name != exp_dir.name):
            raise RuntimeError(
                f"runs/{caller.parent.name}/{caller.name} loads the config of "
                f"'{exp_dir.name}', not its own folder. Set EXP_NAME = "
                f"\"{caller.parent.name}\" in that script (python checks/run_name_check.py "
                f"finds every one).")
    except RuntimeError:
        raise
    except Exception:                                   # inspection itself must never block
        pass
    shared = yaml.safe_load((exp_dir.parent / "shared_config.yaml").read_text(encoding="utf-8"))
    own = yaml.safe_load((exp_dir / config_name).read_text(encoding="utf-8")) or {}
    cfg = _deep_merge(shared, own)
    _coerce_sci(cfg)
    cfg.setdefault("experiment", {}).setdefault("name", exp_dir.name)   # folder name is the default run name
    if verbose:
        print_config(cfg)
    return cfg


def amp_dtype(cfg: dict) -> torch.dtype:
    """training.amp_dtype: "float16" (default, every run so far, with a GradScaler) | "bfloat16" (no scaler needed;
    for language-model trunks such as qwen35_2b_vlm, whose fp16 activations can overflow)."""
    return torch.bfloat16 if str(cfg["training"].get("amp_dtype", "float16")) == "bfloat16" else torch.float16


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
    _es = t.get("early_stopping", {}) or {}
    n_img = images_per_study(cfg)
    _sl = (", ".join("%s=%d" % (k, v) for k, v in i["n_slices"].items())
           if isinstance(i["n_slices"], dict) else i["n_slices"])
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
        ("images per study", f"{len(i['slots'])} slots x {_sl} slices = {n_img} images"),
        ("resolution", (f"{i['size']}->{i['resize_to']} px" if i.get("resize_to")
                        else f"{i['size']} x {i['size']} px") + f" @ {i['mm_per_px']} mm/px "
                       f"= {field:.0f} mm field"),
        ("slice sampling", f"train={i['train_slice_sampling']}  eval={i['slice_sampling']}  "
                           f"margin={i.get('slice_margin', 0)}"),
        ("missing slots", f"fallback_same_plane={i['fallback_same_plane']}"
                          + ("" if not isinstance(i['fallback_same_plane'], (list, tuple))
                             else f"  (no fallback: "
                                  f"{[q for q in i['slots'] if q not in i['fallback_same_plane']]})")),
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
        # a run with nothing held out turns this off and drops the other keys with it
        ("early stopping",
         (f"{_es.get('monitor')} ({_es.get('mode')}), patience={_es.get('patience')}"
          if _es.get('enable', True) else "off")),
    ], "\U0001F3CB\uFE0F")

    _kv("dataloader", [
        ("num_workers", dl.get("num_workers")),
        ("prefetch/persist", f"{dl.get('prefetch_factor')} / {dl.get('persistent_workers')}"),
        ("pin_memory", dl.get("pin_memory")),
        ("drop_last", dl.get("drop_last")),
    ], "\U0001F69A")

    _kv("evaluation & output", [
        ("eval cadence", _eval_cadence_str(cfg)
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
    """A console label for the active Modal workspace (a local, git-ignored helper decides).

    For the CONSOLE and for results/.run_ids/*, both of which stay on this PC - the
    launch log lands in results/train_config/, and that plus .run_ids are git-ignored.
    Never put this in a committed artifact: use runtime_side() for those.
    """
    ws = os.environ.get("MODAL_WORKSPACE") or os.environ.get("MODAL_PROFILE")
    if ws:
        return ws
    try:                                       # optional, git-ignored local lookup
        from _local_workspace import label
        return label() or runtime_side()
    except Exception:
        return runtime_side()


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
    # true/false for every slot, or a LIST naming the slots that may fall back. Per-slot
    # matters when one slot is the variable under test: cor_plain has no fallback of its
    # own (its only same-plane neighbour is cor_fluid, which is already its own slot, so
    # falling back would feed the same series twice), while the other four keep theirs.
    _fb = cfg["input"]["fallback_same_plane"]
    if isinstance(_fb, (list, tuple, set)):
        fallback_slots = set(_fb)
    else:
        fallback_slots = set(slots) if _fb else set()

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
            if cand.empty and slot in fallback_slots:
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


def plan_slot_candidates(index: pd.DataFrame, cfg: dict) -> dict[str, dict[str, list]]:
    """Every series competing for each slot, ranked exactly as plan_slots ranks them -> {study: {slot: [series...]}}.

    Element 0 is the series plan_slots picks (inference always uses it). The rest are the duplicates plan_slots
    discards: 2,136 series in 1,464 of 4,407 studies for the 4 standard slots. input.series_schedule shows them in
    training (scheduled_series)."""
    slots, rule = cfg["input"]["slots"], cfg["input"]["duplicate_rule"]
    _fb = cfg["input"]["fallback_same_plane"]
    fallback_slots = set(_fb) if isinstance(_fb, (list, tuple, set)) else (set(slots) if _fb else set())
    sort_by, asc = {
        "coverage":      (["depth_mm", "pixel_spacing_c"], [False, True]),
        "resolution":    (["pixel_spacing_c", "depth_mm"], [True, False]),
        "slices":        (["n_slices", "depth_mm"], [False, False]),
        "series_number": (["series", "depth_mm"], [True, False]),
        "random":        (["series"], [True]),
    }[rule]
    out: dict[str, dict[str, list]] = {}
    for study, g in index.groupby("study", sort=False):
        c = {}
        for slot in slots:
            cand = g[g.slot == slot]
            if cand.empty and slot in fallback_slots:
                cand = g[g.plane == PLANE_OF[slot]]
            if cand.empty:
                continue
            ranked = list(cand.sort_values(sort_by, ascending=asc)["series"])
            k = min(int(cfg["input"].get("slot_variant", 0)), len(ranked) - 1)
            c[slot] = [ranked[k]] + [s for j, s in enumerate(ranked) if j != k]   # plan_slots' pick first
        out[study] = c
    return out


def scheduled_series(cands: list, epoch: int, epochs: int, best_last: int, key: str):
    """Which of a slot's ranked series to show in this epoch (input.series_schedule).

    STRUCTURED, not random (user, 4 Oct): odd epochs and the last `best_last` epochs use the best series (the one
    inference sees); even epochs rotate through the duplicates in a fixed round-robin, so over the run each duplicate
    is seen equally often. A per-study/slot offset keeps the studies from all switching to the same rank at once.
    With 12 epochs and best_last 2: best on epochs 1,3,5,7,9,11,12; duplicates on 2,4,6,8,10."""
    n = len(cands)
    if n == 1 or epoch < 1 or epoch > epochs - best_last or epoch % 2 == 1:
        return cands[0]
    import zlib
    off = zlib.crc32(key.encode()) % (n - 1)
    return cands[1 + (epoch // 2 - 1 + off) % (n - 1)]


def pick_slices(n_available: int, n_want: int, mode: str, rng,
                margin: float = 0.0, offset: float = 0.0,
                reverse: bool = False, short_stack: str = "repeat") -> np.ndarray:
    """Pick n_want slice indices from a stack of n_available real slices.
    Never blends slices into fake ones; repeats indices if the stack is short.

    offset shifts the even grid by that fraction of ONE BIN, which is what makes sliding
    window test-time augmentation possible: offset 0, 1/3, 2/3 sample three interleaved
    grids that together cover the stack far more densely than any one of them. It only
    affects the "even" (evaluation) mode; random sampling already varies.

    reverse returns the picks medial-to-lateral instead of lateral-to-medial. For a sagittal
    series that is the laterality mirror (the slice axis IS the left-right axis), and for any
    series it is a test-time variation the model has never seen in that order.

    Both default to the old behaviour exactly.
    """
    # the first/last slices of a stack are usually past the edge of the knee
    lo = int(n_available * margin)
    hi = n_available - lo
    if hi - lo >= n_want:
        edges = np.linspace(lo, hi, n_want + 1)          # n_want equal bins across the stack
        if mode == "random" and short_stack == "pad":
            # "pad" never repeats: disjoint integer bins [ceil(e_i), ceil(e_i+1)), each holding >= 1 slice because the
            # bins are >= 1 slice wide here. (The default branch below lets neighbouring bins share their boundary slice,
            # so a stack just longer than n_want can pick the same slice twice; kept as is so old runs reproduce.)
            a_, b_ = np.ceil(edges[:-1]).astype(int), np.ceil(edges[1:]).astype(int)
            idx = np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(a_, b_)])
        elif mode == "random":
            # one random slice per bin: same coverage, different pick each epoch
            a_, b_ = np.floor(edges[:-1]).astype(int), np.ceil(edges[1:]).astype(int)
            idx = np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(a_, b_)])
        else:
            centres = (edges[:-1] + edges[1:]) / 2.0
            if offset:
                centres = centres + float(offset) * (edges[1] - edges[0])
            idx = np.clip(centres.astype(int), 0, n_available - 1)   # bin centres
    elif short_stack == "pad":
        # input.short_stack "pad" (6 Oct, user): never repeat a slice. A stack with room for n_want once the margin is
        # dropped gives n_want distinct slices spread over all of it; a shorter stack gives EVERY real slice once and
        # -1 for the rest, which _load turns into blank slices and models that mask blank images ignore.
        if n_available >= n_want:
            edges = np.linspace(0, n_available, n_want + 1)
            if mode == "random":                         # disjoint bins over the whole stack, one random slice each
                a_, b_ = np.ceil(edges[:-1]).astype(int), np.ceil(edges[1:]).astype(int)
                idx = np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(a_, b_)])
            else:
                idx = np.clip(((edges[:-1] + edges[1:]) / 2.0).astype(int), 0, n_available - 1)
        else:
            idx = np.concatenate([np.arange(n_available), -np.ones(n_want - n_available, dtype=int)])
    else:
        # too few slices: repeat rather than interpolate (17.8% of series at n=24)
        idx = np.round(np.linspace(0, n_available - 1, n_want)).astype(int)
    return idx[::-1].copy() if reverse else idx


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


def _load_attribute_targets(cfg: dict):
    """{study_id: [mask(K), value(K)]} for the auxiliary ATTRIBUTE loss, or None when off.

    training.aux_attributes: {enable, path, weight}. The csv (data_code/25_extract_report_attributes.py) holds K graded
    report attributes per study as integers or empty ("the report does not say"). Each is scaled to [0, 1] by its own
    maximum over the file and trained with binary cross-entropy against a sigmoid output; empty cells get mask 0 and
    contribute nothing. Unlike aux_report (the whole report embedding, -0.0049 on 13 Sep), these are finding-relevant
    graded facts only. The head is the run's, dropped under eval(), so inference never sees it.
    """
    ax = (cfg.get("training", {}).get("aux_attributes") or {})
    if not ax.get("enable"):
        return None
    path = resolve(ax.get("path", "25_report_attributes.csv"), cfg)
    if not Path(path).exists():
        raise SystemExit(f"[aux] aux_attributes.enable is true but {path} is missing. Build it with\n"
                         f"      python data_code/25_extract_report_attributes.py --build")
    df = pd.read_csv(path).set_index("StudyInstanceUID")
    cols = [c for c in df.columns if c not in ("model", "bad")]
    V = df[cols].astype(float)
    V = V / V.max().replace(0, 1.0)                     # each attribute to [0, 1] by its own top grade
    M = V.notna().astype(np.float32).values
    X = np.nan_to_num(V.values.astype(np.float32))
    print(f"  \U0001F9E9 aux attribute target: {len(df)} studies x {len(cols)} attributes, "
          f"{M.mean():.0%} of cells reported")
    return {s: torch.from_numpy(np.concatenate([M[i], X[i]])) for i, s in enumerate(df.index.astype(str))}


def aux_attribute_count(cfg: dict) -> int:
    """K, the number of attribute columns, so a run's model can size its head before the dataset exists."""
    ax = (cfg.get("training", {}).get("aux_attributes") or {})
    if not ax.get("enable"):
        return 0
    path = resolve(ax.get("path", "25_report_attributes.csv"), cfg)
    cols = pd.read_csv(path, nrows=1).columns
    return len([c for c in cols if c not in ("StudyInstanceUID", "model", "bad")])


def _load_report_targets(cfg: dict):
    """{study_id: standardised report vector} for the auxiliary loss, or None when off.

    WHY THIS EXISTS. The 12 labels give a study at most 12 bits of supervision. The reports
    they were derived from are ~1,000 characters and carry laterality, location, severity
    and findings with no label column at all. Encoding each report once turns that into a
    dense target an auxiliary head can be trained against, at zero inference cost: the head
    is dropped in eval mode.

    STANDARDISATION IS NOT OPTIONAL. The findings live in LOWER-variance directions of the
    embedding than report style does, so on the raw vector the auxiliary gradient would
    chase writing style. The npz carries mean/std computed on the TRAIN split only, so
    held-out reports never influence the scaling. Dead dimensions were already dropped when
    the file was written; dividing a constant output by its ~0 sd would turn rounding noise
    into a unit-variance target.
    """
    ax = (cfg.get("training", {}).get("aux_report") or {})
    if not ax.get("enable"):
        return None
    path = resolve(ax.get("path", "11_report_emb.npz"), cfg)
    if not Path(path).exists():
        raise SystemExit(
            f"[aux] aux_report.enable is true but {path} is missing. Build it with\n"
            f"      modal run runs/others/encode_reports.py\n"
            f"  then put it where the run can see it (locally data/, on Modal the runs "
            f"volume).")
    z = np.load(path, allow_pickle=True)
    E = (z["emb"].astype(np.float32) - z["mean"]) / z["std"]
    ids = [str(s) for s in z["study_ids"]]
    print(f"  \U0001F4DD aux report target: {E.shape[0]} studies x {E.shape[1]} dims "
          f"from {str(z['model'])}, standardised on the train split")
    return {s: torch.from_numpy(E[i]) for i, s in enumerate(ids)}


def slot_slices(cfg: dict, slot: str) -> int:
    """How many real slices this slot samples.

    input.n_slices is normally one integer for every slot. It may instead be a mapping
    {slot: count}, because the planes do not deserve equal sampling: the MCL is a flat band
    on the inner edge of the knee whose whole length sits in very few CORONAL planes, so
    skipping coronal planes can miss it outright, while the sagittal and axial findings are
    either already well covered or limited by WHERE we sample rather than how densely.
    """
    n = cfg["input"]["n_slices"]
    if isinstance(n, dict):
        if slot not in n:
            raise ValueError("input.n_slices has no entry for slot %r (has %s)"
                             % (slot, sorted(n)))
        return int(n[slot])
    return int(n)


def slot_groups(cfg: dict, slot: str) -> int:
    """How many IMAGES this slot becomes, i.e. forward passes it costs.

    slices_as_channels packs consecutive slices into one image's channels. It follows
    n_slices: give it a mapping too when n_slices is one. Channels per image must come out
    the same for every slot, because the backbone's stem has a single fixed in_chans.
    """
    g = cfg["input"].get("slices_as_channels")
    if isinstance(g, dict):
        if slot not in g:
            raise ValueError("input.slices_as_channels has no entry for slot %r" % slot)
        g = int(g[slot])
    else:
        g = int(g or 0)
    return g if g > 0 else slot_slices(cfg, slot)


def per_slot_input(cfg: dict) -> bool:
    """True when slices are configured per slot, which changes the item layout."""
    return isinstance(cfg["input"]["n_slices"], dict) or \
        isinstance(cfg["input"].get("slices_as_channels"), dict)


def check_slot_geometry(cfg: dict) -> int:
    """Validate a per-slot setup and return the channels-per-image every slot agrees on."""
    chans = set()
    for s in cfg["input"]["slots"]:
        n, g = slot_slices(cfg, s), slot_groups(cfg, s)
        if n % g:
            raise ValueError("slot %r: slices_as_channels=%d must divide n_slices=%d"
                             % (s, g, n))
        chans.add(n // g)
    if len(chans) != 1:
        raise ValueError("every slot must yield the same channels per image, got %s. The "
                         "backbone stem has one in_chans." % sorted(chans))
    return chans.pop()


def _mirror_slot(v, slot: str, ctx: int = 1) -> np.ndarray:
    """Mirror one slot's volume for laterality, on the axis that plane actually needs.

    v is (n_slices * ctx, H, W). Sagittal reverses the SLICE order; coronal and axial
    reverse the image width. See the measurement in KneeShardDataset.__getitem__.
    """
    plane = str(slot).split("_")[0].lower()
    if plane == "sag":
        # medial-lateral IS the slice axis here, so mirroring reverses slice ORDER
        if ctx > 1:
            # slice-major (s0c0, s0c1, s0c2, s1c0, ...): reverse whole slices, not planes
            v = v.reshape(-1, ctx, *v.shape[1:])[::-1].reshape(v.shape)
            return np.ascontiguousarray(v)
        return np.ascontiguousarray(v[::-1])
    if plane in ("cor", "ax"):
        # medial-lateral runs across the image, so mirroring reverses WIDTH
        return np.ascontiguousarray(v[:, :, ::-1])
    # No silent default. The original laterality bug WAS a width flip applied to sagittal
    # slots: it corrupted half the input and produced a confident wrong conclusion that
    # took two runs to undo. An unknown plane has an unknown medial-lateral axis, so refuse
    # rather than guess.
    raise ValueError(
        "laterality_flip: slot %r has plane %r, which has no known medial-lateral axis. "
        "Known planes: sag (reverses slice order), cor and ax (reverse width). Add the new "
        "plane to _mirror_slot and to checks/laterality_axis_check.py, which derives the "
        "axis from ImageOrientationPatient rather than assuming it." % (slot, plane))


def _sides_from_ipp(index: pd.DataFrame) -> dict:
    """{study: 'left'|'right'} from the x of ImagePositionPatient, per study.

    DICOM patient coordinates put +x toward the patient's LEFT, and a knee sits well off
    the midline, so the mean x over a study's series is solidly signed. This is the same
    rule runs/others/extract_laterality.py applied to the shard metadata to build
    data/12_laterality.csv, which agreed with 542 explicitly labelled studies 97.8% of the
    time. Studies with no usable x are simply absent, so they never get flipped.
    """
    if "ipp_x" not in index.columns:
        return {}
    df = index[["study", "ipp_x"]].dropna()
    if df.empty:
        return {}
    m = df.groupby("study")["ipp_x"].mean()
    return {s: ("left" if x > 0 else "right") for s, x in m.items()}


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
        # Per-slot slice counts. The slot volumes then have different depths, so they cannot
        # be stacked into (slots, D, H, W); __getitem__ groups each slot into its images and
        # concatenates along the IMAGE axis instead, emitting (1, total_images, C, H, W).
        # prepare_batch's existing 6-dim branch consumes that unchanged. When n_slices is a
        # plain integer nothing below changes and the old layout is produced byte for byte.
        self.per_slot = per_slot_input(cfg)
        self.slot_n = {s: slot_slices(cfg, s) for s in self.slots}
        self.slot_g = {s: slot_groups(cfg, s) for s in self.slots}
        self.slot_chans = check_slot_geometry(cfg) if self.per_slot else None
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
        # the epoch counter travels WITH the dataset. Reading the module global from a worker
        # only works when workers are forked (Linux). Windows spawns them, each re-imports this
        # file and gets its OWN _EPOCH frozen at 0, so every local epoch replayed epoch 1's
        # augmentation: gnorm fell 0.7 -> 0.13 and three local runs scored ~0.905 (28 Sep,
        # checks/epoch_seed_check.py). A Value handed over at worker start is the real one.
        self._epoch = _EPOCH
        self._retr = (_RETR_NS, _RETR_BYTES, _RETR_IMGS)   # read-time telemetry, same reason
        # 2.5D: instead of copying one slice into 3 identical channels, fill them with
        # neighbouring slices of the SAME series, so the backbone sees through-plane
        # context. Costs nothing extra - those channels already existed, just wasted.
        self.ctx = 3 if i.get("slice_context") == "neighbours" else 1
        self.ctx_stride = int(i.get("slice_context_stride", 1))
        # Laterality normalisation. Mirror one side so every study faces the same way, so
        # "medial" is always the same half of the frame. self.sides stays empty when the
        # knob is off, and the flip below is then a dict lookup that never matches.
        # test-time knobs, both no-ops by default. slice_offset slides the even sampling
        # grid within a bin; slice_reverse reverses the picked order. Neither is useful in
        # training - they exist so one checkpoint can be scored several ways at inference.
        self.slice_offset = float(i.get("slice_offset") or 0.0)
        self.slice_reverse = bool(i.get("slice_reverse"))
        # "repeat" (default, every run so far) or "pad": short stacks get blank slices, never repeated ones
        self.short_stack = str(i.get("short_stack") or "repeat")
        if self.short_stack == "pad" and i.get("slice_context") == "neighbours":
            raise ValueError("input.short_stack 'pad' is not supported with slice_context 'neighbours'")
        self.flip_side = i.get("laterality_flip")
        self.sides = {}
        if self.flip_side and self.flip_side not in ("left", "right"):
            raise ValueError("input.laterality_flip must be 'left', 'right' or null, "
                             "got %r" % (self.flip_side,))
        if self.flip_side and dicom is None:
            lat = pd.read_csv(resolve(cfg["paths"]["laterality_csv"], cfg))
            self.sides = dict(zip(lat["StudyInstanceUID"], lat["side"]))
        # (index, {series: [files]}) at submission time: the same volumes come from
        # raw DICOM instead of a shard, everything downstream is identical
        self.dicom_files = None
        # Auxiliary report-embedding target. Loaded ONLY for training splits and only when
        # asked for: at submission time there are no reports, and the auxiliary head is
        # dropped anyway, so inference must never depend on this.
        self.aux, self.aux_kind = None, None
        if train and dicom is None:
            self.aux = _load_report_targets(cfg)
            self.aux_kind = "report" if self.aux is not None else None
            att = _load_attribute_targets(cfg)
            if att is not None:
                if self.aux is not None:
                    raise SystemExit("[aux] aux_report and aux_attributes are both enabled; pick one")
                self.aux, self.aux_kind = att, "attributes"
        if dicom is not None:
            dicom_index, self.dicom_files = dicom
            self.geo = dicom_index.set_index("series")[["pixel_spacing_c", "pixel_spacing_r"]]
            if self.flip_side:
                # Submission has no 12_laterality.csv - the test studies were never in it.
                # Same rule as extract_laterality.py, from the headers build_dicom_index
                # already read: mean x of ImagePositionPatient over the study's series,
                # +x pointing to the patient's LEFT. A study with no usable x stays
                # unflipped, which is the majority orientation and so the safer default.
                self.sides = _sides_from_ipp(dicom_index)
                print("  laterality_flip=%r: %d/%d studies sided from DICOM"
                      % (self.flip_side, len(self.sides), dicom_index["study"].nunique()),
                      flush=True)
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

    def _load(self, study: str, series: str, rng, n_slices: int | None = None) -> np.ndarray:
        """One series -> (n_slices * ctx, size, size) float array, resampled and windowed."""
        _t = time.perf_counter()
        n_slices = self.n_slices if n_slices is None else n_slices
        arr = self._read_series(study, series)
        if arr is None:                                   # series missing for this study
            return np.zeros((n_slices * self.ctx, self.size, self.size), np.float32)
        idx = pick_slices(arr.shape[0], n_slices, self.mode, rng, self.margin,
                          self.slice_offset, self.slice_reverse, getattr(self, "short_stack", "repeat"))
        if self.ctx > 1:
            # k-stride, k, k+stride, clamped at the ends; flattened slice-major so a
            # later reshape gives (n_slices, ctx, H, W)
            off = np.arange(-(self.ctx // 2), self.ctx // 2 + 1) * self.ctx_stride
            idx = np.clip(np.asarray(idx)[:, None] + off[None, :],
                          0, arr.shape[0] - 1).reshape(-1)
        pad = idx < 0                                       # short_stack "pad": -1 = a blank slice
        vol = arr[np.where(pad, 0, idx)].astype(np.float32)
        if pad.any():
            vol[pad] = 0.0
        # only the volume read is timed; resample/window/augment come after
        ns, nb, ni = self._retr                          # the real shared counters (see _epoch)
        with ns.get_lock():
            ns.value += int((time.perf_counter() - _t) * 1e9)
            nb.value += int(arr.nbytes)
            ni.value += len(idx)

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
            real = vol[~pad] if (pad.any() and axis is None and (~pad).any()) else vol   # blank padding never sets the window
            lo = np.percentile(real, lo_p, axis=axis, keepdims=axis is not None)
            hi = np.percentile(real, hi_p, axis=axis, keepdims=axis is not None)
            # stretch the [lo, hi] intensity range back over the full 0-255 range
            vol = np.clip((vol - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255.0
            if pad.any():
                vol[pad] = 0.0                                  # padding stays exactly blank (a constant image)
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
        # EXTRA TYPES (2 Oct, augmentation-strength test). Each is OFF unless its p > 0, and an
        # off type draws NOTHING from rng: the same rng goes on to pick the next slot's slices,
        # so one extra draw would silently change every run trained before these existed.
        g = a.get("gamma") or {}
        if g.get("p", 0) > 0 and rng.random() < g["p"]:
            lo, hi = g["range"]                       # <1 brightens the darks, >1 darkens them
            vol = 255.0 * (np.clip(vol, 0, 255) / 255.0) ** rng.uniform(lo, hi)
        nz = a.get("noise") or {}
        if nz.get("p", 0) > 0 and rng.random() < nz["p"]:
            sd = rng.uniform(0, nz["std"]) * 255      # scanner noise, fresh per voxel
            vol = vol + rng.normal(0, sd, vol.shape).astype(np.float32)
        bl = a.get("blur") or {}
        if bl.get("p", 0) > 0 and rng.random() < bl["p"]:
            sig = rng.uniform(*bl["sigma"])           # softer images, like a lower-resolution scan
            vol = np.stack([cv2.GaussianBlur(sl, (0, 0), sig) for sl in vol])
        co = a.get("cutout") or {}
        if co.get("p", 0) > 0 and rng.random() < co["p"]:
            side = max(1, int(co["frac"] * s))        # square holes, the same through every slice
            for _ in range(int(co.get("n", 1))):
                y0, x0 = rng.integers(0, s - side + 1, 2)
                vol[:, y0:y0 + side, x0:x0 + side] = 0
        return np.clip(vol, 0, 255)

    def __getitem__(self, i: int):
        """Build one study's tensor: stack the 4 slot volumes, attach its 12 labels."""
        row = self.df.iloc[i]
        study = row["StudyInstanceUID"]                 # one study = one patient's knee exam
        # every item's randomness is a pure function of (seed, epoch, index) for train
        # and (seed, study) for val/audit - so a run is reproducible and a resumed run
        # sees the same stream it would have seen uninterrupted
        rng = np.random.default_rng([self.seed, int(self._epoch.value), i] if self.train
                                    else [self.seed, hash(study) & 0xFFFFFFFF])
        chosen = self.plan.get(study, {})               # {slot: series} picked by plan_slots
        if self.train and getattr(self, "cands", None):
            # input.series_schedule: this epoch's series per slot (the best one, or a duplicate in turn)
            _ss = self.cfg["input"]["series_schedule"]
            _ep, _E = int(self._epoch.value), int(self.cfg["training"]["epochs"])
            chosen = {sl: scheduled_series(c, _ep, _E, int(_ss.get("best_last_epochs", 2)), f"{study}|{sl}")
                      for sl, c in self.cands.get(study, {}).items()}
        # mirror this study if it is the side we normalise away. Unknown side = leave it.
        mirror = bool(self.flip_side) and self.sides.get(study) == self.flip_side

        vols = []
        for slot in self.slots:
            series = chosen.get(slot)                   # None when this study lacks that view
            n_sl = self.slot_n[slot]
            blank = np.zeros((n_sl * self.ctx, self.size, self.size), np.float32)
            if series is None:
                v = blank
            else:
                try:
                    v = self._load(study, series, rng, n_sl)
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
            # before augment, so a later warp acts on the normalised image the way it
            # would in a run that trained this way from the start.
            #
            # THE AXIS DEPENDS ON THE PLANE, and getting this wrong is not a small error.
            # Measured from ImageOrientationPatient over the local shards, the fraction of
            # the patient left-right axis lying along each image axis is:
            #     axial     horizontal 0.989   vertical 0.113   slice 0.021
            #     coronal   horizontal 0.987   vertical 0.026   slice 0.139
            #     sagittal  horizontal 0.097   vertical 0.031   slice 0.991
            # So for coronal and axial a left/right mirror IS a horizontal flip, but for
            # sagittal the medial-lateral direction is the SLICE axis and there is no
            # left-right axis in the image plane at all. Flipping a sagittal image
            # horizontally reverses ANTERIOR-POSTERIOR, which is anatomical corruption
            # rather than normalisation. Sagittal mirrors by reversing its slice order.
            # checks/laterality_axis_check.py proves this from the data.
            if mirror:
                v = _mirror_slot(v, slot, self.ctx)
            if self.aug is not None:
                v = self._augment(v, rng)             # same warp across a slice's channels
            if self.ctx > 1:
                v = v.reshape(n_sl, self.ctx, self.size, self.size)
            elif self.per_slot:
                # group this slot's slices into its own images now, since slots with
                # different depths cannot be stacked and reshaped together later
                v = v.reshape(self.slot_g[slot], self.slot_chans, self.size, self.size)
            vols.append(v)

        if self.per_slot:
            # (1, total_images, channels, H, W): one pseudo-slot holding every image, which
            # is exactly the shape prepare_batch's 6-dim branch already expects per study
            x = torch.from_numpy(np.concatenate(vols, axis=0).astype(np.uint8)[None])
        else:
            # (slots, D, H, W) normally, (slots, D, ctx, H, W) with 2.5D
            x = torch.from_numpy(np.stack(vols).astype(np.uint8))
        y = torch.tensor([float(row[t]) for t in self.tasks], dtype=torch.float32)   # 12 probs in [0,1]
        # "hard" mode turns the soft label probabilities into crisp 0/1 targets
        if self.cfg["target"]["mode"] == "hard":
            thr, ls = self.cfg["target"]["hard_threshold"], self.cfg["target"]["label_smoothing"]
            y = (y > thr).float() * (1 - ls) + ls / 2
        if self.aux is None:
            return x, y                                  # unchanged for every existing run
        if self.aux_kind == "attributes":
            # [mask(K), value(K)]; a study with no extracted attributes gets an all-zero mask
            a = self.aux.get(study)
            if a is None:
                a = torch.zeros(next(iter(self.aux.values())).shape, dtype=torch.float32)
            return x, y, a
        # a study with no report gets a zero vector and a 0 weight, so it contributes
        # nothing to the auxiliary loss instead of being trained toward the origin
        a = self.aux.get(study)
        if a is None:
            a = torch.zeros(next(iter(self.aux.values())).shape, dtype=torch.float32)
            w = torch.zeros(1)
        else:
            w = torch.ones(1)
        return x, y, torch.cat([w, a])


def model_input_size(cfg: dict) -> int:
    """The side length the BACKBONE sees, which is not always the loader's.

    input.size is the decoded image; input.resize_to, when set, is what prepare_batch
    shrinks it to on GPU. Build models against this, never against input.size directly.
    """
    return int(cfg["input"].get("resize_to") or cfg["input"]["size"])


def images_per_study(cfg: dict) -> int:
    """How many IMAGES one study becomes, i.e. how many forward passes the model runs.

    Not slots x n_slices any more. With input.slices_as_channels a slot becomes that many
    images whose channels are its slices, so a study costs slots x groups passes, not
    slots x n_slices. Getting this wrong does not crash anything: it silently multiplies
    every throughput number, and it printed MFU 364% (of a hard 100% ceiling) on the first
    slices_as_channels run before this existed.
    """
    i = cfg["input"]
    if per_slot_input(cfg):
        return sum(slot_groups(cfg, s) for s in i["slots"])
    g = int(i.get("slices_as_channels") or 0)
    return len(i["slots"]) * (g if g > 0 else int(i["n_slices"]))


def prepare_batch(x: torch.Tensor, cfg: dict, device) -> torch.Tensor:
    """uint8 (B, slots, D, H, W) -> normalized float (B*slots*D, C, H, W)."""
    # ship uint8 to the GPU (4x smaller transfer), normalize there
    c = cfg["input"]["channels"]
    # slices_as_channels: the whole SLOT becomes ONE image whose channels are its slices,
    # so a study costs 4 forward passes instead of 4 x 24 = 96. Measured at 336px that is
    # 0.039x the compute of the per-slice layout, and unlike a 3-channel bottleneck it
    # discards nothing: the backbone is built with in_chans=n_slices and timm adapts the
    # pretrained stem filters to the wider input. The cost is that cross-slice reasoning
    # must now happen inside the stem, and pooling collapses from 96 vectors to 4.
    as_ch = bool(cfg["input"].get("slices_as_channels"))
    x = x.to(device, non_blocking=True).float().div_(255.0)
    if x.dim() == 6:                               # 2.5D: channels already hold neighbours
        b, s, d, ctx, h, w = x.shape
        x = x.reshape(b * s * d, ctx, h, w)
    elif as_ch:
        # `slices_as_channels` is the number of IMAGES each slot becomes. 1 puts all 24
        # slices in one image; 2 splits them into two images of 12 channels, and so on.
        # It is not a style choice, it sets the stem's information budget: ConvNeXt-Base's
        # stem maps a 4x4 patch of in_chans to 128 outputs, so 24 channels COMPRESSES
        # 384 -> 128 (3.0x) where the usual 3 channels EXPAND 48 -> 128 (2.7x). Splitting
        # into 2 images of 12 halves that to 1.5x while keeping every slice.
        b, s, d, h, w = x.shape
        g = max(1, int(cfg["input"]["slices_as_channels"]))
        if d % g:
            raise ValueError(f"slices_as_channels={g} must divide n_slices={d}")
        x = x.reshape(b * s * g, d // g, h, w)
        c = d // g
    else:
        b, s, d, h, w = x.shape
        x = x.view(b * s * d, 1, h, w)             # every slice becomes its own 2D image
        if c > 1:
            x = x.expand(-1, c, -1, -1)            # grey copied to 3 channels for ImageNet backbones
    # Optional single downscale for a backbone whose resolution is not the loader's.
    # It sits HERE, after the flatten and before normalisation, because prepare_batch is
    # the one function training, validation and every inference path share - so a member
    # can never be trained at one size and scored at another. "area" matches the loader's
    # own interpolation, and the field of view is unchanged, so this is a pure resample.
    r = cfg["input"].get("resize_to")
    if r and x.shape[-1] != int(r):
        x = F.interpolate(x, size=(int(r), int(r)), mode="area")
    if as_ch and c != len(cfg["input"]["norm_mean"]):
        # A WIDENED stem (12 or 24 slice-channels): norm_mean/std describe three RGB
        # channels and there is no per-channel statistic to apply, so one scalar covers all
        # of them. Averaging the three is the right reduction: it is what a grey pixel would
        # have received anyway once it was copied into all three channels.
        m = float(np.mean(cfg["input"]["norm_mean"]))
        s_ = float(np.mean(cfg["input"]["norm_std"]))
        return (x - m) / s_
    # slices_as_channels landing on exactly 3 channels (grouped 2.5D) falls THROUGH to the
    # per-channel path below, because the stem is then the untouched pretrained RGB stem and
    # must be fed the statistics it was trained with. That also makes it byte-identical to
    # how convnext_small_25d normalises, which is the overlapping version of this same idea.
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
        if train and (cfg["input"].get("series_schedule") or {}).get("enable"):
            ds.cands = plan_slot_candidates(idx, cfg)            # every competing series, ranked
            _nd = sum(len(v) - 1 for st in ds.df.StudyInstanceUID for v in ds.cands.get(st, {}).values())
            print(f"  \U0001F501 series schedule ON: {_nd} extra series across the train studies take turns "
                  f"on even epochs; the best series on odd epochs and the last "
                  f"{int(cfg['input']['series_schedule'].get('best_last_epochs', 2))}", flush=True)
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


class ConfidenceWeighted(nn.Module):
    """Wrap any per-element loss and weight each CELL by how sure the labeler was.

    The targets are soft LLM probabilities, and their distance from 0.5 is the labeler's own
    confidence: 0.95 means the report said so, 0.50 means the report was silent and the prompt
    told it to fall back on a base rate. Trained plainly, a coin-flip cell produces as much
    gradient as a stated fact, so the model spends capacity reproducing guesses. In this corpus
    that is 23.6% of Synovitis cells and 24.1% of Effusion cells - our weakest and most
    ambiguous findings.

    weight = |2p - 1| ** power, so p=0.95 -> 0.90, p=0.75 -> 0.50, p=0.50 -> 0.00.

    NOTE what this does NOT do: the TARGET is untouched. A 0.95 cell is still regressed toward
    0.95. Only its share of the gradient changes. Weights are normalised by their mean so the
    effective learning rate does not silently drop as cells are discounted.

    `floor` keeps an ambiguous cell from vanishing completely - at floor=0 a p=0.5 cell is
    fully ignored, which throws away the information that the finding is plausible at all.
    """

    def __init__(self, base: nn.Module, power: float = 1.0, floor: float = 0.0):
        super().__init__()
        self.base = base
        self.power = float(power)
        self.floor = float(floor)

    def forward(self, logits, target):
        w = (2.0 * target - 1.0).abs().clamp(0.0, 1.0) ** self.power
        if self.floor > 0:
            w = self.floor + (1.0 - self.floor) * w
        # per-element loss, then confidence-weighted mean. Reduction is forced to "none"
        # here; a base loss that cannot do that is refused at build time.
        per = self.base(logits, target)
        if per.shape != target.shape:
            raise RuntimeError("confidence weighting needs a per-element loss, got %s"
                               % (tuple(per.shape),))
        return (w * per).sum() / w.sum().clamp(min=1e-6)


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

    # confidence weighting: only meaningful against SOFT targets, because hard 0/1 targets
    # make every weight exactly 1 and the wrapper becomes a no-op with extra arithmetic.
    conf = lc.get("confidence_weight") or {}
    if conf.get("enable"):
        if cfg.get("target", {}).get("mode") != "soft":
            raise SystemExit("confidence_weight needs target.mode: soft - with hard 0/1 "
                             "targets every weight is 1 and the setting does nothing")
        base = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        return ConfidenceWeighted(base, conf.get("power", 1.0),
                                  conf.get("floor", 0.0)).to(device)
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


UPLOAD_POINTER = Path.home() / ".knee_upload_modal_config"   # one line: a Modal config path


def resolve_upload_modal_config(value: str) -> str:
    """`modal_config: "local"` reads the real path from ~/.knee_upload_modal_config, a file
    outside the repo, so no workspace-specific path is ever committed. Any other value is
    used as given."""
    if value != "local":
        return value
    if not UPLOAD_POINTER.exists():
        raise SystemExit(f"upload_checkpoints.modal_config is 'local' but {UPLOAD_POINTER} "
                         f"does not exist: write the Modal config path into it")
    return UPLOAD_POINTER.read_text(encoding="utf-8").strip()


def _adamw(o: dict):
    """torch AdamW, or bitsandbytes' 8-bit AdamW when optimizer.eight_bit is set: the two
    moment buffers stored in 8 bits (block-wise quantised) instead of 32, 4x less optimizer
    memory, same update rule. Tensors under 4,096 elements stay 32-bit."""
    if o.get("eight_bit"):
        import bitsandbytes as bnb
        print("  🗜️  8-bit AdamW (bitsandbytes): moment buffers in 8 bits")
        return bnb.optim.AdamW8bit
    return torch.optim.AdamW


def chunk_backbone(model: nn.Module, n: int) -> None:
    """Run model.backbone over `n` images at a time instead of the whole (B*windows) stack.

    Training: each chunk sits under its own activation checkpoint, so only ONE chunk's
    activations are alive at once (the rest is recomputed chunk by chunk in the backward).
    Eval: plain chunks, no checkpoint. Exact for backbones that treat every image on its own
    (LayerNorm, no BatchNorm, no cross-image attention); refused otherwise. The forward is
    patched on the instance, so parameter names and checkpoints are unchanged."""
    from torch.utils.checkpoint import checkpoint
    bb = getattr(model, "backbone")
    bn = [m_ for m_ in bb.modules() if isinstance(m_, nn.modules.batchnorm._BatchNorm)]
    if bn:
        raise SystemExit(f"backbone_chunk: {len(bn)} BatchNorm layers mix images across a "
                         f"chunk boundary, so chunking would change the result")
    orig = bb.forward

    def fwd(x):
        grad = bb.training and torch.is_grad_enabled()
        if x.shape[0] <= n and not grad:
            return orig(x)
        return torch.cat([checkpoint(orig, c, use_reentrant=False) if grad else orig(c)
                          for c in x.split(n)], dim=0)
    bb.forward = fwd
    print(f"  🧩 backbone runs {n} images at a time (training: one checkpointed chunk alive)")


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
        return _adamw(o)(groups, lr=head_lr, weight_decay=o.get("weight_decay", 0.05))
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
    return _adamw(o)(groups, lr=head_lr, weight_decay=o.get("weight_decay", 0.05))


def schedule_steps(cfg: dict, total_steps: int) -> int:
    """Steps the lr CURVE spans, which is not always the steps we train for.

    scheduler.schedule_epochs stretches the cosine over a different number of epochs than
    training.epochs, so a shortened run replays a longer run's lr trajectory instead of
    annealing fully into its own end.
    """
    se = cfg["training"]["scheduler"].get("schedule_epochs")
    ep = cfg["training"].get("epochs") or 0
    if not se or not ep or int(se) == int(ep):
        return total_steps
    return max(1, int(round(total_steps * float(se) / float(ep))))


def warmup_steps(cfg: dict, total_steps: int) -> int:
    """How many steps the lr ramp lasts — same formula build_scheduler uses."""
    n = schedule_steps(cfg, total_steps)
    return max(1, int(n * cfg["training"]["scheduler"].get("warmup_frac", 0.05)))


def build_scheduler(opt, cfg: dict, total_steps: int):
    """Learning-rate schedule: ramp up over the first few %, then cosine-decay to min_lr_mult."""
    s = cfg["training"]["scheduler"]
    warm = warmup_steps(cfg, total_steps)
    lo = s.get("min_lr_mult", 0.01)
    n = schedule_steps(cfg, total_steps)
    if n != total_steps:
        print(f"  \U0001F4C9 lr curve spans {n} steps ({s['schedule_epochs']} epochs) but "
              f"training stops at {total_steps} ({cfg['training']['epochs']} epochs) - "
              f"ends at {100 * total_steps / n:.0f}% of the cosine, not at its floor")
    total_steps = n

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
            # head weight norm too: logits = head_w . features, so features and logits alone
            # cannot say WHICH side moved. dinov2_vits14_25d showed feat_l2 FALLING
            # 24.5 -> 17.9 while logit_std ROSE 0.78 -> 1.30, which can only mean the head
            # grew to compensate - and a bigger head sends bigger gradients back into the
            # trunk, eroding the features further. That loop is invisible without this.
            w = getattr(_mod, "weight", None)
            self.buf = (float(f.std()), float(f.norm(dim=-1).mean()),
                        float(o.std()), float(o.abs().max()),
                        float(w.detach().float().norm()) if w is not None else float("nan"))

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
        r = {"feat_std": round(b[0], 4), "feat_l2": round(b[1], 3),
             "logit_std": round(b[2], 4), "logit_absmax": round(b[3], 3)}
        if len(b) > 4:
            r["head_w_norm"] = round(b[4], 4)
        return r

    @staticmethod
    def summarize(b):
        if not b:
            return ""
        fs, fl, ls, lm = b[:4]
        hw = b[4] if len(b) > 4 else None
        flag = ""
        if fs > 2.5 or fs < 0.15:
            flag = ("   <-- feature scale is far from 1: the head's effective lr is not "
                    "what the config says. Consider model.feat_norm")
        head = f"   |   head |W| {hw:.2f}" if hw is not None and hw == hw else ""
        return (f"  \U0001F4CF scale  features std {fs:.3f}  L2 {fl:.1f}   |   "
                f"logits std {ls:.3f}  |max| {lm:.2f}{head}{flag}")


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


def _uncompiled(model):
    """The real module behind a torch.compile wrapper, or the model itself.

    torch.compile returns an OptimizedModule that holds the original under
    `_orig_mod`, and its state_dict keys all gain an "_orig_mod." prefix. Saving that
    prefix would make every checkpoint depend on whether compile happened to be on,
    so checkpoints are always written from the ORIGINAL module.
    """
    return getattr(model, "_orig_mod", model)


def _load_model_state(model, sd, allow_head_mismatch: bool = False):
    """Load weights whichever side of torch.compile they were saved from.

    Strips a leading "_orig_mod." (a checkpoint written by an older compiled run) and
    loads into the unwrapped module, so compiled and non-compiled runs can resume from
    each other's checkpoints in either direction.

    `allow_head_mismatch` is for SPECIALIST transfer: seed a 1-task model from a 12-task
    checkpoint by taking the backbone and dropping the head, whose shape cannot match
    ((12, 1024) into (1, 1024)). It is deliberately NOT the default - a silent partial load
    is how a half-random model trains for an hour and looks completely normal doing it.
    Every skipped and every missing tensor is printed, and anything missing OUTSIDE the head
    is a hard error.
    """
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
              for k, v in sd.items()}
    m = _uncompiled(model)
    if not allow_head_mismatch:
        return m.load_state_dict(sd)

    own = m.state_dict()
    keep, skipped = {}, []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            keep[k] = v
        else:
            skipped.append("%s %s -> %s" % (k, tuple(v.shape),
                                            tuple(own[k].shape) if k in own else "absent"))
    missing = [k for k in own if k not in keep]
    # a skipped tensor is only acceptable if it is the task head
    bad = [k for k in missing if "head" not in k.lower() and "classifier" not in k.lower()
           and "fc." not in k.lower()]
    if bad:
        raise SystemExit("transfer would leave %d NON-HEAD tensor(s) untrained: %s\n"
                         "   the checkpoint does not match this architecture"
                         % (len(bad), bad[:6]))
    print("  transfer: loaded %d tensors, fresh head (%d tensor(s))" % (len(keep), len(missing)))
    for x in skipped:
        print("      skipped %s" % x)
    for k in missing:
        print("      fresh   %s %s" % (k, tuple(own[k].shape)))
    m.load_state_dict(keep, strict=False)
    return None


def save_checkpoint(path: Path, cfg, model, opt=None, sched=None, scaler=None,
                    epoch=0, step=0, score=None, track=None, device=None,
                    weights_only=False, elapsed_sec=0.0, epoch_complete=True,
                    saver=None, note=None, loss_fn=None):
    """Full state by default, so a run can resume exactly where it stopped.

    epoch_complete=False marks a checkpoint written part-way through an epoch, so
    resume replays that epoch instead of silently skipping the rest of its data.
    """
    payload = {"model": _uncompiled(model).state_dict(), "cfg": cfg, "epoch": epoch,
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
    _after_checkpoint_write(path)              # optional off-site copy (CheckpointUploader)
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


class CheckpointUploader:
    """Copy freshly written checkpoints to a Modal volume, so a LOCAL run can keep only a few on
    disk (output.upload_checkpoints: {modal_config, volume, patterns}).

    Each file is hard-linked into results/_upload_staging first (no extra disk, and it survives
    the rolling prune deleting the original mid-upload), then `modal volume put --force` runs in a
    subprocess whose ONLY Modal config is the one given, on a single background thread, newest
    version first if the same name is queued twice. Success and failure are printed in the
    training console; a failed upload never stops training.
    """

    def __init__(self, spec: dict, run_name: str, res_dir: Path, rel_ckpt_dir: str, default_volume: str):
        import fnmatch
        import queue
        import threading
        self.spec, self.run = spec, run_name
        # the same volume and path a Modal run writes to: <runs_volume>:/<run>/results/checkpoints
        self.volume = spec.get("volume") or default_volume
        self.rel = rel_ckpt_dir
        self.patterns = spec.get("patterns", ["step*.pt", "top_step*.pt", "best.pt"])
        self.cfg_path = str(Path(os.path.expanduser(resolve_upload_modal_config(spec["modal_config"]))))
        self.stage = Path(res_dir) / "_upload_staging"
        self.stage.mkdir(parents=True, exist_ok=True)
        self._match = lambda n: any(fnmatch.fnmatch(n, p) for p in self.patterns)
        self.q, self.pending, self.lock = queue.Queue(), {}, threading.Lock()
        self.done = self.failed = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"  ☁️  checkpoint upload ON: {', '.join(self.patterns)} -> Modal volume "
              f"'{self.volume}':/{self.run}/{self.rel}/ ({self.cfg_path})", flush=True)

    def offer(self, path: Path) -> None:
        path = Path(path)
        if not self._match(path.name):
            return
        staged = self.stage / path.name
        try:
            if staged.exists():
                staged.unlink()
            os.link(path, staged)                  # same volume: instant, no extra space
        except OSError as e:
            print(f"  ⚠️  upload staging failed for {path.name}: {e}", flush=True)
            return
        with self.lock:
            fresh = path.name not in self.pending
            self.pending[path.name] = staged
        if fresh:
            self.q.put(path.name)

    def _loop(self):
        import subprocess
        import time as _t
        while True:
            name = self.q.get()
            if name is None:
                return
            with self.lock:
                staged = self.pending.pop(name, None)
            if staged is None or not staged.exists():
                continue
            size = staged.stat().st_size
            remote = f"/{self.run}/{self.rel}/{name}"
            env = dict(os.environ, MODAL_CONFIG_PATH=self.cfg_path, PYTHONUTF8="1")
            ca = Path.home() / ".certs" / "ca-bundle.pem"
            if ca.exists():
                env["REQUESTS_CA_BUNDLE"] = env["SSL_CERT_FILE"] = str(ca)
            t0 = _t.time()
            r = subprocess.run([sys.executable, "-m", "modal", "volume", "put", "--force",
                                self.volume, str(staged), remote],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", env=env)
            dt = _t.time() - t0
            if r.returncode == 0:
                self.done += 1
                print(f"  ☁️  uploaded {name} ({size / 1e9:.2f} GB) -> {self.volume}:{remote} "
                      f"in {dt:.0f}s ({size / 1e6 / max(dt, 1e-9):.0f} MB/s)", flush=True)
            else:
                self.failed += 1
                tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["?"]
                print(f"  ⚠️  upload FAILED for {name} after {dt:.0f}s: {tail[0][:200]}", flush=True)
            with self.lock:
                if name not in self.pending:
                    try:
                        staged.unlink()
                    except OSError:
                        pass

    def join(self) -> None:
        """Wait for every queued upload, then stop the thread."""
        self.q.put(None)
        self.thread.join()
        print(f"  ☁️  checkpoint uploads finished: {self.done} uploaded, {self.failed} failed", flush=True)


_UPLOADER = None                                   # set by run_experiment when configured


def _after_checkpoint_write(path: Path) -> None:
    if _UPLOADER is not None:
        _UPLOADER.offer(path)


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


def resolve_init_from(spec: str, out_dir, cfg: dict) -> Path:
    """Where another run's checkpoint lives, given "<run>/<checkpoint>" or a path.

    out_dir is THIS run's folder, so its parent is the runs root on either side: the Modal
    volume mount or the local repo. A bare run name means that run's best.pt.
    """
    p = Path(str(spec))
    if p.is_absolute() and p.exists():
        return p
    runs_root = Path(out_dir).parent
    parts = str(spec).replace("\\", "/").split("/")
    run = parts[0]
    name = parts[1] if len(parts) > 1 else cfg["output"]["best_name"]
    if not name.endswith(".pt"):
        name += ".pt"
    cand = [runs_root / run / cfg["output"]["run_dir"] / cfg["output"]["checkpoints_dir"] / name,
            runs_root / str(spec),
            Path(str(spec))]
    for c in cand:
        if c.exists():
            return c
    raise FileNotFoundError(
        "model.init_from=%r matched nothing. Looked in:\n  %s"
        % (spec, "\n  ".join(str(c) for c in cand)))


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
    _load_model_state(model, ck["model"])
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
def _eval_cadence_str(cfg: dict) -> str:
    """One line describing the tiered validation cadence, for print_config."""
    e = cfg["evaluation"]
    thr = e.get("eval_every_steps_epoch_threshold")
    if not thr:
        return f"every {e.get('eval_every_steps') or 0} steps"
    mid, mid_rate = e.get("eval_every_steps_epoch_mid_threshold"), e.get("eval_every_steps_mid")
    parts = []
    if mid and mid_rate and int(mid) < int(thr):
        parts.append(f"{e.get('eval_every_steps_before')} steps until epoch {int(mid)}")
        parts.append(f"{mid_rate} until epoch {int(thr)}")
    else:
        parts.append(f"{e.get('eval_every_steps_before')} steps until epoch {int(thr)}")
    parts.append(f"then every {e.get('eval_every_steps_after')}")
    return ", ".join(parts)


def eval_interval(cfg: dict, epoch: int) -> int:
    """Steps between validations for this epoch - 0 disables step-triggered eval.

    Cadence is driven by the GLOBAL STEP, not the epoch. The epoch only selects
    which interval applies: early epochs move fast and are not worth scoring often,
    later ones are where the best checkpoint usually shows up.
    eval_every_steps_epoch_threshold is the first epoch that uses the "after" rate.

    Three tiers, coarse to fine:
        epoch <  mid_threshold   ->  eval_every_steps_before
        epoch <  threshold       ->  eval_every_steps_mid
        epoch >= threshold       ->  eval_every_steps_after
    The middle tier is optional. Leave eval_every_steps_epoch_mid_threshold unset (or
    equal to the main threshold) and it collapses to the old two-tier behaviour, so
    every existing config keeps its exact cadence.
    """
    e = cfg["evaluation"]
    thr = e.get("eval_every_steps_epoch_threshold")
    if not thr:
        return int(e.get("eval_every_steps") or 0)
    thr = int(thr)
    mid = e.get("eval_every_steps_epoch_mid_threshold")
    mid_rate = e.get("eval_every_steps_mid")
    if mid and mid_rate and int(mid) < thr and epoch >= int(mid) and epoch < thr:
        return int(mid_rate or 0)
    key = "eval_every_steps_after" if epoch >= thr else "eval_every_steps_before"
    return int(e.get(key) or 0)


def eval_due(cfg: dict, epoch: int, gstep: int, resume_eval: list) -> str | None:
    """Should this step be validated, and what should the log call the trigger?

    Returns the trigger label, or None to skip. `resume_eval` is a one-element list used
    as a mutable flag: the one-off `evaluation.validate_on_resume` validation DISARMS the
    moment it is consumed, whichever branch consumes it, so it can never fire twice and a
    resume landing exactly on a cadence step scores once, not twice.

    Split out of the training loop so it can be tested without a GPU. The loop must not
    re-implement this - a copy is a copy that drifts.
    """
    ev = eval_interval(cfg, epoch)
    if ev and gstep % ev == 0:
        resume_eval[0] = False
        return "every %d steps" % ev
    if resume_eval[0]:
        resume_eval[0] = False
        return "first step after resume"
    return None


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
        with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype(cfg)):
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
    i = cfg["input"]
    if per_slot_input(cfg):
        return                      # no single slot depth to lay the grid out by
    slots, d = len(i["slots"]), int(i["n_slices"])
    g = int(i.get("slices_as_channels") or 0)       # images per slot; 0 = one image per slice
    per = images_per_study(cfg)                     # images for ONE study, NOT slots * d
    if x_prep.shape[0] < per:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Invert EXACTLY the normalisation prepare_batch applied, which depends on the channel
    # count: per-channel when the stem is the pretrained 3-channel one, the averaged scalar
    # when the stem was widened to hold slices. Slicing norm_mean to x_prep.shape[1] was
    # wrong for a widened stem - it yields 3 values against 12 or 24 channels and does not
    # broadcast - so this branched the same way prepare_batch does instead.
    n_ch = x_prep.shape[1]
    if n_ch == len(i["norm_mean"]):
        mean = torch.tensor(i["norm_mean"]).view(1, -1, 1, 1)
        std = torch.tensor(i["norm_std"]).view(1, -1, 1, 1)
    else:
        mean = torch.tensor(float(np.mean(i["norm_mean"]))).view(1, 1, 1, 1)
        std = torch.tensor(float(np.mean(i["norm_std"]))).view(1, 1, 1, 1)
    v = x_prep[:per].detach().float().cpu() * std + mean       # undo normalization
    v = (v.clamp(0, 1) * 255).to(torch.uint8).numpy()          # keep EVERY channel

    # Where does (slot r, slice sl) live? With one image per slice it is row r*d+sl, channel
    # 0. With slices_as_channels the slices are packed into channels, so it is image
    # r*g + sl//cpg at channel sl%cpg. Getting this wrong does not crash, it just draws the
    # wrong planes on the contact sheet - which is the one artifact we use to eyeball input.
    cpg = d // g if g else 1

    def plane(r: int, sl: int):
        return v[r * g + sl // cpg, sl % cpg] if g else v[r * d + sl, 0]

    cols = min(8, d)
    picks = np.linspace(0, d - 1, cols).astype(int)
    h, w = v.shape[-2:]                             # v is (images, channels, H, W) now
    sheet = np.zeros((slots * h, cols * w), np.uint8)
    for r in range(slots):
        for c, sl in enumerate(picks):
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = plane(r, sl)

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
                     f"{n * images_per_study(cfg):,}",
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


def set_lp_ft_phase(cfg: dict, model: nn.Module, epoch: int) -> None:
    """LP-FT: freeze the backbone for the first N epochs, then unfreeze it.

    LP-FT is Linear Probing then Fine-Tuning. Phase 1 trains ONLY the head on frozen
    features; phase 2 unfreezes everything and fine-tunes normally.

    WHY IT EXISTS HERE. Kumar et al., ICLR 2022 (arXiv 2202.10054) show that when the
    pretrained features are good and the domain shift is large, fine-tuning from a
    RANDOMLY INITIALISED head is worse than a frozen linear probe: the head is still noise
    while it back-propagates into the trunk, so the first updates destroy the very features
    that made the backbone worth choosing. Their fix is to let the head converge first.

    We hit exactly that with siglip2_vitb_512. Its FROZEN features probe at 0.7016 macro
    AUROC, the best of any backbone we own and above convnext_base's 0.6832 - yet
    fine-tuned end-to-end it stalled at 0.6325, the worst first validations in the project,
    with a training loss that did not move for 1,100 steps. Good features, ruined by the
    fine-tune. Web photos to grayscale knee MRI is about as large a shift as exists.

    Off by default: every existing run trained without it and their numbers must not move.

        training:
          lp_ft_epochs: 1      # epochs 1..N train the head alone, N+1 onward everything

    The optimizer keeps both param groups throughout; freezing is done with requires_grad,
    so the backbone's AdamW moments stay intact across the switch and the lr schedule is
    untouched. Frozen parameters also skip their backward pass, so phase 1 is cheaper.
    """
    n = int(cfg.get("training", {}).get("lp_ft_epochs") or 0)
    if n <= 0:
        return
    bb = getattr(_uncompiled(model), "backbone", None)
    if bb is None:
        print("  ⚠️  lp_ft_epochs set but the model has no .backbone - ignored")
        return
    freeze = epoch <= n
    already = not next(bb.parameters()).requires_grad
    if freeze == already and epoch != 1:
        return                                   # nothing to change this epoch
    for p in bb.parameters():
        p.requires_grad_(not freeze)
    n_bb = sum(p.numel() for p in bb.parameters()) / 1e6
    if freeze:
        print(f"  \U0001F9CA LP-FT phase 1: backbone FROZEN ({n_bb:.1f}M params), head only "
              f"- epochs 1..{n}")
    else:
        print(f"  \U0001F525 LP-FT phase 2: backbone UNFROZEN ({n_bb:.1f}M params) from "
              f"epoch {epoch}")


def check_normalization(cfg: dict, model: nn.Module) -> None:
    """Fail fast when the config's pixel normalisation contradicts the backbone's own.

    Every pretrained backbone was trained on pixels scaled a particular way, and it expects
    the same scaling at fine-tune time. Getting this wrong is SILENT: the run trains, the
    loss falls, and the model is simply worse than it should be. It cost us a whole
    mricore_vitb_mae pretraining run, which inherited ImageNet statistics while every other
    MRI-CORE run used [0, 1], and nothing anywhere complained.

    timm records the right values on each model's `pretrained_cfg`, so this compares the
    config against the backbone and refuses to start on a mismatch.

    To deviate DELIBERATELY - which is legitimate, e.g. MRI-CORE's weights are not timm's
    so timm's defaults do not describe them - set:

        input:
          norm_mismatch_ok: true   # and say WHY in a comment

    Only the backbone is inspected. A model with no timm backbone is skipped silently.
    """
    inp = cfg.get("input", {})
    want_m, want_s = inp.get("norm_mean"), inp.get("norm_std")
    if want_m is None or want_s is None:
        return
    # A run that loads its OWN checkpoint (model.weights) uses the timm entry only as an
    # empty skeleton - those pretrained weights are overwritten and never seen. Its
    # pretrained_cfg then describes a checkpoint that is not in the model, so comparing
    # against it is meaningless: the MRI-CORE runs load a trunk pretrained under ImageNet
    # statistics into vit_base_patch16_224, whose cfg says 0.5/0.5. Only the config author
    # knows what the real trunk saw, so say so and move on.
    if cfg.get("model", {}).get("weights"):
        print(f"[norm] skipped: model.weights={cfg['model']['weights']!r} replaces the timm "
              f"weights, so its pretrained_cfg does not describe this trunk. "
              f"Using config mean {list(want_m)}, std {list(want_s)}.")
        return
    # the backbone may be nested (SliceModel.backbone) or be the model itself
    for m in (getattr(_uncompiled(model), "backbone", None), _uncompiled(model)):
        pcfg = getattr(m, "pretrained_cfg", None)
        if isinstance(pcfg, dict) and pcfg.get("mean") and pcfg.get("std"):
            break
    else:
        return
    got_m, got_s = list(pcfg["mean"]), list(pcfg["std"])
    close = (len(got_m) == len(want_m[:len(got_m)])
             and all(abs(a - b) < 1e-3 for a, b in zip(got_m, want_m))
             and all(abs(a - b) < 1e-3 for a, b in zip(got_s, want_s)))
    name = cfg.get("model", {}).get("name", "?")
    if close:
        print(f"[norm] OK: config matches {name}'s pretrained_cfg "
              f"(mean {got_m}, std {got_s})")
        return
    msg = (f"normalisation mismatch for backbone {name!r}:\n"
           f"    config    mean {list(want_m)}  std {list(want_s)}\n"
           f"    backbone  mean {got_m}  std {got_s}\n"
           f"  The backbone was pretrained on the second pair. Training on the first is a "
           f"SILENT error:\n"
           f"  the run will converge and simply score worse.\n"
           f"  Fix input.norm_mean / input.norm_std, or set input.norm_mismatch_ok: true "
           f"if the deviation\n"
           f"  is deliberate (say why in a comment).")
    if inp.get("norm_mismatch_ok"):
        print(f"[norm] DELIBERATE MISMATCH, allowed by input.norm_mismatch_ok\n  {msg}")
        return
    raise SystemExit("[norm] " + msg)


def check_grad_clip(cfg: dict):
    """grad_clip below 3.0 is a configuration error here, so refuse to start.

    shared_config.yaml already defaults to 3.0, but that protects nothing: 45 run configs
    set their own value, and the way a new run is made in this project is by COPYING the
    closest existing one. Copy anything from the convnext_base_relabel era and 1.0 comes
    with it, silently.

    Why it matters: clipping rescales any update whose gradient norm exceeds the threshold.
    convnext_base_relabel clipped 100% of its steps at 1.0, so it trained at a fraction of
    the learning rate its config advertised, and every run copied from it inherited both the
    setting and the misleading comment justifying it.

    Set training.grad_clip_below_3_ok: true to override, and say why in a comment.
    """
    g = cfg.get("training", {}).get("grad_clip")
    if g is None or float(g) >= 3.0 or float(g) == 0.0:
        return
    if cfg.get("training", {}).get("grad_clip_below_3_ok"):
        print(f"  ⚠️  grad_clip {g} is below the 3.0 floor, allowed by "
              f"training.grad_clip_below_3_ok")
        return
    lines = [
        "[grad_clip] %s is below the 3.0 floor." % g,
        "  Clipping rescales every update above the threshold, so a low cap silently",
        "  throttles the effective learning rate below the one this config advertises.",
        "  convnext_base_relabel clipped 100% of its steps at 1.0, and every run copied",
        "  from it inherited the value.",
        "  Set training.grad_clip to 3.0 or higher, or set",
        "  training.grad_clip_below_3_ok: true if it is deliberate (say why in a comment).",
    ]
    raise SystemExit("\n".join(lines))


def check_aux_wiring(cfg: dict, model: nn.Module):
    """aux_report.enable is only half the wiring: the MODEL must return the tuple.

    The engine supplies the target and the loss term, but the auxiliary head belongs to the
    run's model, because the optimizer is built from model.parameters(). If a config turns
    the feature on and the model was copied from a run without a head, the engine sees plain
    logits, adds nothing, and the run trains exactly like the baseline while its config
    claims otherwise. That is a silent no-op of the worst kind: it looks like a negative
    result. Fail loudly at startup instead.
    """
    ax = (cfg.get("training", {}).get("aux_report") or {})
    if not ax.get("enable"):
        ax = (cfg.get("training", {}).get("aux_attributes") or {})   # same contract for the attribute head
    if not ax.get("enable"):
        return
    if not float(ax.get("weight", 0) or 0):
        raise SystemExit("[aux] an auxiliary target is enabled but its weight is 0 - it would do "
                         "nothing. Set a weight or disable it.")
    m = _uncompiled(model)
    was_training = m.training
    m.train()
    try:
        with torch.no_grad():
            i = cfg["input"]
            n = images_per_study(cfg)
            probe_in = torch.zeros(n, i["channels"], model_input_size(cfg),
                                   model_input_size(cfg))
            out = m(probe_in.to(next(m.parameters()).device))
    except Exception as e:
        print(f"  ⚠️  aux wiring check could not run a forward pass "
              f"({type(e).__name__}: {e}); relying on the training loop instead")
        m.train(was_training)
        return
    m.train(was_training)
    if not isinstance(out, tuple):
        raise SystemExit(
            "[aux] aux_report.enable is true but the model returns plain logits in TRAINING "
            "mode.\n"
            "  The auxiliary head lives in the run's model, not the engine. Its forward must\n"
            "  return (logits, aux_pred) when self.training, and plain logits otherwise.\n"
            "  Without that the loss term is never added and the run silently trains as the\n"
            "  baseline while the config claims report supervision.")
    print(f"  ✅ aux wiring OK: model returns (logits {tuple(out[0].shape)}, "
          f"aux {tuple(out[1].shape)}) in training mode, plain logits under eval()")


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
    global _UPLOADER
    _UPLOADER = None
    if cfg["output"].get("upload_checkpoints"):
        _o = cfg["output"]
        _UPLOADER = CheckpointUploader(
            _o["upload_checkpoints"], cfg["experiment"]["name"], Path(out_dir) / _o["run_dir"],
            f"{_o['run_dir']}/{_o['checkpoints_dir']}", cfg.get("modal", {}).get("runs_volume", "knee-runs"))
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
    check_normalization(cfg, model)
    check_aux_wiring(cfg, model)
    check_grad_clip(cfg)

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
        # channels_last is a RANK-4 layout (N, C, H, W). A 3D backbone's Conv3d weights are
        # rank 5, and torch raises "required rank 4 tensor to use channels_last format"
        # rather than skipping them, which is what killed triad_plainconv3d_320 at startup.
        # torch has a separate rank-5 layout for exactly this case.
        _has5d = any(p_.ndim == 5 for p_ in model.parameters())
        _fmt = torch.channels_last_3d if _has5d else torch.channels_last
        model = model.to(memory_format=_fmt)
        if _has5d:
            print("  🧊 channels_last_3d (the model has rank-5 conv weights)")
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
    if cfg["training"].get("backbone_chunk"):
        chunk_backbone(model, int(cfg["training"]["backbone_chunk"]))
    tot_p = sum(q.numel() for q in model.parameters())
    tr_p = sum(q.numel() for q in model.parameters() if q.requires_grad)
    _banner("MODEL", "\U0001F9E0")
    _kv("architecture", [
        ("backbone", cfg.get("model", {}).get("name", "?")),
        ("pretrained", cfg.get("model", {}).get("pretrained")),
        ("pooling", f"{cfg.get('model', {}).get('pool')} over "
                    f"{images_per_study(cfg)} images/study"),
        ("output logits", f"{num_output_logits(cfg)}  (one per finding)"),
        None,
        ("total params", f"{tot_p:,}  ({tot_p/1e6:.2f} M)"),
        ("trainable", f"{tr_p:,}  ({tr_p/1e6:.2f} M, {100*tr_p/max(tot_p,1):.1f}%)"),
        ("frozen", f"{tot_p-tr_p:,}  ({(tot_p-tr_p)/1e6:.2f} M)"),
    ], "\U0001F9F1")

    loss_fn = build_loss(cfg, train_df, device)
    opt = build_optimizer(model, cfg, loss_fn)
    # Which groups are the pretrained trunk? Structural, not by name: everything under
    # .backbone is the trunk, the rest is the freshly initialised head. Computed once,
    # because param_groups never change shape after this.
    _bb_pids = {id(p) for n, p in model.named_parameters() if n.startswith("backbone.")}
    _grp_is_bb = [any(id(p) in _bb_pids for p in g["params"]) for g in opt.param_groups]
    accum = accum_steps(cfg)
    # CEIL, not floor: the epoch's last, partial accumulation window is stepped (rescaled to
    # its real size in the loop), exactly like the partial last batch of an accum-1 run. With
    # floor, 3,638 studies at 2 x 4 gave 454 steps/epoch against 455 at batch 8, dropped 6
    # studies per epoch and shifted the lr schedule off the Modal run's (28 Sep).
    n_micro = len(loaders["train"])
    steps_per_epoch = max(1, -(-n_micro // accum))
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
    use_amp_train = cfg["training"]["amp"] and device.type == "cuda"
    # bf16 has fp32's exponent range: no loss scaling (a scaler would only add work); fp16 keeps its scaler as before
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_train and amp_dtype(cfg) == torch.float16)
    imgs_per_batch = train_batch_size(cfg) * images_per_study(cfg)
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
    # "gamma" catches LayerScale in a ViT and the spliced cross-attention blocks; "gate"
    # catches the attention branches in convnext_base_25d_{slot,token}attn, whose gate IS the
    # experiment - it starts at zero, and how far it moves is the answer to "did attention
    # earn its place". Missing it would leave those runs uninterpretable.
    _gates = [(n, q) for n, q in model.named_parameters()
              if n.endswith("gamma") or n.endswith("gate")]
    if _gates:
        print(f"  🚪 gates: {len(_gates)} tensor(s) tracked ({', '.join(sorted({n.rsplit(chr(46), 1)[-1] for n, _ in _gates}))}) "
              f"-> every {_udc.get('every_steps', 200)} steps, and into train_log.csv")

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

    # .get defaults throughout: a run with nothing held out sets enable false and has no
    # reason to carry a monitor, a mode or a patience
    es = cfg["training"].get("early_stopping", {}) or {}
    es = {"enable": es.get("enable", True), "monitor": es.get("monitor", "val_macro_auroc"),
          "mode": es.get("mode", "max"), "patience": es.get("patience", 4),
          "reset_tag": es.get("reset_tag")}       # one-time reset when the monitored metric changed meaning
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
    # WARM START. Only when this run has nothing of its own: a preempted container must
    # resume ITS progress, not jump back to the donor and lose the epochs in between.
    _init = (cfg.get("model") or {}).get("init_from")
    if _init and not ck:
        _src = resolve_init_from(_init, out_dir, cfg)
        _dck = torch.load(_src, map_location="cpu", weights_only=False)
        _sd = _dck.get("model", _dck)
        _res = _load_model_state(
            model, _sd,
            allow_head_mismatch=bool((cfg.get("model") or {}).get("init_from_allow_head")))
        _miss = list(getattr(_res, "missing_keys", []) or [])
        _unex = list(getattr(_res, "unexpected_keys", []) or [])
        print("\n  \U0001F331 WARM START from %s" % _src)
        print("       donor: step %s  epoch %s  score %s"
              % (_dck.get("global_step") or _dck.get("step"), _dck.get("epoch"),
                 _dck.get("score")))
        print("       %d tensor(s) loaded, %d missing, %d unexpected"
              % (len(_sd) - len(_unex), len(_miss), len(_unex)))
        print("       WEIGHTS ONLY: optimizer, scheduler and the step counter stay fresh, "
              "so warmup and the whole cosine schedule run again")
        if _miss or _unex:
            # A silent partial load is how a half-random model trains for an hour and looks
            # completely normal doing it.
            raise RuntimeError(
                "warm start is incomplete: %d missing %s, %d unexpected %s. Set "
                "model.init_from_allow_head if only the head should differ."
                % (len(_miss), _miss[:6], len(_unex), _unex[:6]))

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

        # early_stopping.reset_tag: the MONITORED METRIC changed meaning (e.g. val re-graded with other labels, 7 Oct:
        # round-2 teachers moved from v4- to v3-graded val). The stored best (0.95 on the old scale) could never be
        # beaten, so patience would run out on a different metric. A new tag resets best/patience/top-k ONCE; the tag
        # is saved, so later resumes of the same tag keep counting normally.
        _rt = es.get("reset_tag")
        if _rt and tr.get("es_tag") != _rt:
            print(f"  🔄 early stopping reset for metric tag {_rt!r} (was {tr.get('es_tag')!r}): best {best:.4f} and "
                  f"patience {bad} dropped, top-k list cleared")
            best, best_epoch, bad, top_k_track = -math.inf, 0, 0, []

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
    # auxiliary report-embedding loss: its weight, and a running (sum, count) so the term
    # is logged on its own rather than hidden inside the total
    _ax = (cfg.get("training", {}).get("aux_report") or {})
    aux_w = float(_ax.get("weight", 0.0) or 0.0) if _ax.get("enable") else 0.0
    # the graded-attribute variant (training.aux_attributes): masked BCE, same (logits, aux) model contract
    _axa = (cfg.get("training", {}).get("aux_attributes") or {})
    aux_attr = bool(_axa.get("enable"))
    if aux_attr:
        aux_w = float(_axa.get("weight", 0.0) or 0.0)
    # [aligned loss sum, n, shuffled loss sum, in-batch top-1 sum, n_pairs]
    aux_run = [0.0, 0, 0.0, 0.0, 0]
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
                "prev_eval": dict(prev_eval), "extend": dict(ext_state),
                "es_tag": es.get("reset_tag")}

    def _validate_and_checkpoint(epoch: int, gstep: int, tag: str = "epoch"):
        """Score val (and audit), log, checkpoint, update early stopping."""
        nonlocal best, best_epoch, bad, top_k_track, last_val_step
        nonlocal score, improved
        if gstep == last_val_step:            # a step-trigger already covered this step
            return None
        last_val_step = gstep
        el = time.time() - t0
        eta = el * (total / max(gstep, 1) - 1)

        # FULL-DATA RUNS: evaluation.enable false. Every study is in the training set, so a
        # validation score here is IN-SAMPLE - it climbs whatever the model does and reads
        # like a score when it is not one. Skip the evaluation and keep the checkpointing,
        # which is otherwise coupled to it. With nothing held out "best" can only mean
        # LATEST, so the score is the step counter: monotone, so best.pt and the top-K board
        # always hold the newest weights.
        if not cfg["evaluation"].get("enable", True):
            score, improved = float(gstep), True
            best, best_epoch, bad = score, epoch, 0
            done = (tag == "epoch")
            top_k_track = update_top_k(ckpt_dir, cfg, model, epoch, gstep, score,
                                       top_k_track, cfg["output"].get("keep_top_k", 0),
                                       cfg["output"].get("top_k_weights_only", True),
                                       saver=saver)
            save_checkpoint(ckpt_dir / cfg["output"]["best_name"], cfg, model, opt, sched,
                            scaler, epoch, gstep, score, track=_track(), device=device,
                            elapsed_sec=el, epoch_complete=done, saver=saver,
                            loss_fn=loss_fn, note="")
            ud.dump(out / "ud_ratio.csv")
            save_checkpoint(ckpt_dir / "last.pt", cfg, model, opt, sched, scaler, epoch,
                            gstep, score, track=_track(), device=device, loss_fn=loss_fn,
                            elapsed_sec=el, epoch_complete=done, saver=saver, note="")
            print("  \U0001F4BE checkpoint at epoch %d step %d (%s) - no validation, this "
                  "run trains on every study" % (epoch, gstep, tag))
            return None

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
    per_step_imgs = train_batch_size(cfg) * accum * images_per_study(cfg)
    retr_prev = (0, 0, 0, 0.0)      # last retrieval snapshot
    # counted once on a real batch, then reused - the count itself is not free
    flops_per_img = [None]
    last_print = [None, None]       # (elapsed, gstep) of the previous console line
    _peak = _peak_tflops(torch.cuda.get_device_name(0)) if device.type == "cuda" else 0.0
    # evaluation.validate_on_resume: score ONCE on the first step back, then behave normally.
    # The cadence is a modulo on the global step, so a checkpoint that lands between two
    # validations leaves a gap on resume - and changing the cadence mid-run widens it,
    # because the schedule the earlier steps followed no longer applies. This closes that
    # gap without special-casing anything downstream.
    resume_eval = [bool(ck) and bool(cfg["evaluation"].get("validate_on_resume"))]
    if resume_eval[0]:
        print("  \U0001F501 validate_on_resume: scoring once on the first step back")

    # evaluation.validate_before_training: score at step 0, before ANY gradient has been
    # applied. For a branch run this is the only way to see what the seeded weights are
    # actually worth on THIS run's inputs - every other number is confounded with training.
    # A branch that keeps the parent's slots should reproduce the parent's score exactly;
    # if it does not, the seeding or the eval path is broken, not the recipe.
    # Set it to "only" to stop right there: the probe then costs ONE validation pass and no
    # training at all, which is the whole point when the question is "what are these weights
    # worth on these inputs" rather than "can they be improved".
    _vbt = cfg["evaluation"].get("validate_before_training")
    if _vbt and not ck:
        print("\n  \U0001F9EA validate_before_training: scoring the SEEDED weights at step 0, "
              "before any gradient step")
        _validate_and_checkpoint(0, 0, "step 0, before training")
        if str(_vbt).lower() == "only":
            print("  \U0001F9EA validate_before_training: 'only' - stopping without training")
            if persist_fn:
                persist_fn()
            return {"probe": True, "history": history}
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
        set_lp_ft_phase(cfg, model, epoch)
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
                _batch = next(_train_it)
            except StopIteration:
                break
            # 2-tuple normally; 3-tuple when the auxiliary report target is enabled. Read
            # it tolerantly so no existing run changes shape.
            x, y = _batch[0], _batch[1]
            aux_t = _batch[2] if len(_batch) > 2 else None
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
            probe.arm(probe.active(gstep + 1) and (it % accum == 0 or it == n_micro))
            with torch.autocast("cuda", enabled=use_amp_train, dtype=amp_dtype(cfg)):
                # NOT `out`: that name is the run's OUTPUT DIRECTORY in this same scope,
                # captured by the validation and checkpoint closure. Shadowing it would
                # have crashed the first validation on `out / "ud_ratio.csv"`.
                _fwd = model(prepare_batch(x, cfg, device))
                # A model with an auxiliary head returns (logits, aux_pred) IN TRAINING
                # MODE ONLY, and plain logits under eval(). That is deliberate: validation,
                # _predict_dataframe, the ensemble path and the Kaggle notebook then need
                # no changes at all and cannot be broken by this feature.
                logits, aux_p = _fwd if isinstance(_fwd, tuple) else (_fwd, None)
                loss = loss_fn(logits, y)
                if aux_p is not None and aux_t is not None and aux_attr:
                    # L = BCE(logits, y) + aux_w * mean over REPORTED cells of BCE(attr_logit, grade/max).
                    # aux_t = [mask(K), value(K)]; silent cells (mask 0) contribute nothing.
                    aux_t = aux_t.to(device, non_blocking=True)
                    K = aux_t.shape[1] // 2
                    msk, tgt = aux_t[:, :K], aux_t[:, K:]
                    denom = msk.sum().clamp(min=1.0)
                    bce = F.binary_cross_entropy_with_logits(aux_p.float(), tgt.float(), reduction="none")
                    aux_loss = (bce * msk).sum() / denom
                    loss = loss + aux_w * aux_loss
                    aux_run[0] += float(aux_loss.detach())
                    aux_run[1] += 1
                    # the study-specific check: the same loss against ANOTHER study's attributes. If aligned and
                    # shuffled are equal, the head learned the average report and is teaching the trunk nothing.
                    with torch.no_grad():
                        if tgt.shape[0] > 1:
                            _mt, _mm = torch.roll(tgt, 1, dims=0), torch.roll(msk, 1, dims=0)
                            _b = F.binary_cross_entropy_with_logits(aux_p.float(), _mt.float(), reduction="none")
                            aux_run[2] += float((_b * _mm).sum() / _mm.sum().clamp(min=1.0))
                            aux_run[4] += 1
                elif aux_p is not None and aux_t is not None:
                    # Two different multipliers here, deliberately named apart:
                    #   has_report  per-study 0/1 mask, inside the auxiliary mean
                    #   aux_w       the scalar lambda from the config, applied ONCE to the
                    #               whole auxiliary term
                    # L = BCE(logits, y) + aux_w * mean_over_reported(1 - cos). Only the
                    # RATIO of the two terms is meaningful: scaling the total loss just
                    # rescales the gradient, which the learning rate already absorbs, so
                    # the main term's coefficient is fixed at 1 and aux_w is the knob.
                    aux_t = aux_t.to(device, non_blocking=True)
                    has_report, tgt = aux_t[:, :1], aux_t[:, 1:]
                    cos = F.cosine_similarity(aux_p.float(), tgt.float(), dim=1)
                    # divide by the number of REAL reports, so a batch with a missing one
                    # is not quietly scaled down toward zero
                    denom = has_report.sum().clamp(min=1.0)
                    aux_loss = ((1.0 - cos) * has_report.squeeze(1)).sum() / denom
                    loss = loss + aux_w * aux_loss
                    aux_run[0] += float(aux_loss.detach())
                    aux_run[1] += 1
                    # THE diagnostic that matters, and the one aux_cos_loss cannot give.
                    # Most knee reports resemble each other, so a head that simply predicts
                    # the DATASET MEAN vector scores a good cosine while learning nothing
                    # about the individual study - and aux_cos_loss would fall and look
                    # healthy the whole time. Compare each prediction against a MISMATCHED
                    # report from the same batch: if aligned and shuffled are equal, the
                    # head learned the average report and this whole idea is doing nothing.
                    with torch.no_grad():
                        if tgt.shape[0] > 1:
                            # NOT `roll`: that name is the 20-step throughput deque in this
                            # same scope, and shadowing it crashed the run at step 1
                            _mismatched = torch.roll(tgt, 1, dims=0)
                            cos_s = F.cosine_similarity(aux_p.float(),
                                                        _mismatched.float(), dim=1)
                            aux_run[2] += float(((1.0 - cos_s) * has_report.squeeze(1)).sum()
                                                / denom)
                            # in-batch retrieval: is a study's OWN report its nearest of
                            # the batch? 1/B is chance, 1.0 is perfect
                            sim = F.normalize(aux_p.float(), dim=1) @ \
                                F.normalize(tgt.float(), dim=1).T
                            hit = (sim.argmax(dim=1) == torch.arange(sim.shape[0],
                                                                     device=sim.device))
                            aux_run[3] += float(hit.float().mean())
                            aux_run[4] += 1
                loss = loss / accum
            scaler.scale(loss).backward()
            seen += imgs_per_batch
            win_imgs += imgs_per_batch
            if it % accum == 0 or it == n_micro:
                _r = it % accum
                if _r:
                    # partial last window: each of its _r micro-losses was divided by accum,
                    # so the summed gradient is _r/accum of the window's MEAN gradient. Scale
                    # it back: identical to one batch of those studies at accum 1.
                    for _p in model.parameters():
                        if _p.grad is not None:
                            _p.grad.mul_(accum / _r)
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
                    print("  \U0001F6AA gates   (zero = that branch contributes nothing)")
                    _gopen = 0.0
                    for _n, _q in _gates:
                        _v = _q.detach().float()
                        _a = _v.abs()
                        # label by the OWNING MODULE: "slot_attn.gate" -> "slot_attn",
                        # "blocks.4.gamma" -> "4". Without this every branch would
                        # print under one name and the CSV columns would collide.
                        _parts = _n.split(".")
                        _lbl = _parts[-2] if len(_parts) > 1 else _n
                        _op = int((_a > 1e-3).sum())
                        _gopen = max(_gopen, float(_a.mean()))
                        # std of ONE element is undefined: torch returns nan and warns every
                        # time. tokenattn's gate is a scalar (the tokens are shared by all 12
                        # findings, so there is no per-label axis), so this would fire every
                        # 200 steps forever unless guarded.
                        _sd = f"{float(_v.std()):.5f}" if _v.numel() > 1 else "   n/a"
                        print(f"     {_lbl:<10} |g| mean {float(_a.mean()):.5f}  "
                              f"min {float(_v.min()):+.5f}  max {float(_v.max()):+.5f}  "
                              f"std {_sd}  open {_op}/{_v.numel()}")
                        _extra[f"gate_{_lbl}_absmean"] = round(float(_a.mean()), 8)
                        _extra[f"gate_{_lbl}_max"] = round(float(_v.max()), 8)
                        _extra[f"gate_{_lbl}_min"] = round(float(_v.min()), 8)
                        _extra[f"gate_{_lbl}_open"] = _op
                    if _gopen < 1e-4:
                        print("     \u26a0\ufe0f  every gate still ~0: the blocks are "
                              "contributing nothing yet")
                lrs = sched.get_last_lr()
                lr, lr_lo, lr_hi = lrs[-1], min(lrs), max(lrs)
                # Named, not just min and max. With layer_decay the minimum is the EARLIEST
                # trunk layer (0.8**depth of the head rate), which is not "the backbone
                # rate" in any useful sense - the trunk's TOP layer is what is really
                # training. Report both ends of the trunk and the head separately.
                _bb = [v for v, isb in zip(lrs, _grp_is_bb) if isb]
                _hd = [v for v, isb in zip(lrs, _grp_is_bb) if not isb]
                lr_bb_hi = max(_bb) if _bb else None
                lr_bb_lo = min(_bb) if _bb else None
                lr_head = max(_hd) if _hd else lr_hi
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
                if aux_run[1] and aux_attr:
                    # aux_attr_bce   masked BCE to this study's attributes; aux_attr_shuf the same against another
                    # study's; aux_attr_gap = shuf - aligned, THE signal (near 0 = the head learned the average)
                    _extra["aux_attr_bce"] = round(aux_run[0] / aux_run[1], 5)
                    if aux_run[4]:
                        _extra["aux_attr_shuf"] = round(aux_run[2] / aux_run[4], 5)
                        _extra["aux_attr_gap"] = round(aux_run[2] / aux_run[4] - aux_run[0] / aux_run[1], 5)
                    aux_run[:] = [0.0, 0, 0.0, 0.0, 0]
                elif aux_run[1]:
                    # aux_cos_loss  1 - cosine to the CORRECT report. 1.0 is orthogonal.
                    # aux_cos_shuf  the same against a MISMATCHED report from the batch.
                    # aux_gap       shuffled minus aligned. THIS is the real signal: it is
                    #               how much of the head's skill is study-SPECIFIC rather
                    #               than the dataset-average report. A falling aux_cos_loss
                    #               with a gap near 0 means the head learned the average
                    #               and the auxiliary supervision is doing nothing.
                    # aux_top1      in-batch retrieval: does a study's own report come out
                    #               nearest? chance is 1/batch_size - and at our batch of 2
                    #               that chance is 50%, so top1 is WEAK here and aux_gap is
                    #               the metric to trust. top1 becomes informative only on a
                    #               run with a genuinely large batch.
                    _extra["aux_cos_loss"] = round(aux_run[0] / aux_run[1], 5)
                    if aux_run[4]:
                        shuf = aux_run[2] / aux_run[4]
                        _extra["aux_cos_shuf"] = round(shuf, 5)
                        _extra["aux_gap"] = round(shuf - aux_run[0] / aux_run[1], 5)
                        _extra["aux_top1"] = round(aux_run[3] / aux_run[4], 4)
                    aux_run[:] = [0.0, 0, 0.0, 0.0, 0]
                train_log.log({**_extra, **ScaleProbe.row(_pb),
                               "epoch": epoch, "epoch_step": -(-it // accum),
                               "global_step": gstep, "loss": round(l, 6), "lr": lr,
                               "lr_min": lr_lo, "lr_max": lr_hi,
                               "lr_head": lr_head,
                               "lr_backbone_max": lr_bb_hi, "lr_backbone_min": lr_bb_lo,
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
                trig = eval_due(cfg, epoch, gstep, resume_eval)
                if trig:
                    _validate_and_checkpoint(epoch, gstep, trig)
                    if stop_now[0]:
                        break             # stop AT patience, not at the next epoch end
                _sa = int(cfg["training"].get("stop_after_steps") or 0)
                if _sa and gstep >= _sa:
                    # a deliberately short run (e.g. checking a machine against a known curve):
                    # the lr schedule stays the FULL run's, so every step up to here matches it
                    print(f"\n🛑 stop_after_steps={_sa} reached at step {gstep}")
                    stop_now[0] = True
                    break
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
                    _bbs = ("backbone %.3e" % lr_bb_hi if lr_bb_hi == lr_bb_lo
                            else "backbone top %.3e, bottom %.3e" % (lr_bb_hi, lr_bb_lo)) \
                        if lr_bb_hi is not None else "no trunk group"
                    print(f"  ✅ warmup COMPLETE at step {warm} - base lr reached "
                          f"(head {lr_head:.3e}, {_bbs}); cosine decay begins")
                if gstep % cfg["output"]["console_log_every"] == 0 or gstep == 1:
                    d = "" if prev_loss is None else f" ({l - prev_loss:+.4f})"
                    # one lr when every group shares it, both ends when LLRD or
                    # backbone_lr_mult splits them
                    if lr_lo == lr_hi:
                        lr_s = f"lr={lr_hi:.3e}"
                    elif lr_bb_hi is None:
                        lr_s = f"lr[head={lr_head:.3e}]"
                    elif lr_bb_hi == lr_bb_lo:
                        lr_s = f"lr[head={lr_head:.3e} backbone={lr_bb_hi:.3e}]"
                    else:
                        lr_s = (f"lr[head={lr_head:.3e}  backbone top={lr_bb_hi:.3e} "
                                f"bottom={lr_bb_lo:.3e}]")
                    print("─" * 70)
                    print(f"  Epoch {epoch}/{epochs}  epoch_step: {-(-it // accum)}/{steps_per_epoch}"
                          f"  global_step: {gstep}/{total}")
                    # one step's gnorm is noisy (ours span 0.6 to 23 on the same run), so
                    # the rolling mean is what actually says whether the trunk runs hot,
                    # and clip% says how much of it is surviving the cap
                    _gw = list(clip_vals)[-50:]
                    _gs = ""
                    if _gw:
                        _gs = (f" (avg{len(_gw)} {sum(_gw) / len(_gw):.2f}"
                               f", clip {100.0 * clip_hit[0] / max(clip_seen[0], 1):.0f}%)")
                    # wall time since the previous console line, so the cost of a
                    # window is readable without differencing two elapsed= values
                    # nothing to difference against on this process's FIRST line, and
                    # elapsed survives a resume while last_print does not - so subtracting
                    # a zero here would report the whole prior run as one window
                    _since = ""
                    if last_print[0] is not None and gstep > last_print[1]:
                        _since = (f"  +{fmt(el - last_print[0])}"
                                  f"/{gstep - last_print[1]}st")
                    last_print[0], last_print[1] = el, gstep
                    print(f"     loss={l:.4f}{d}  {lr_s}  gnorm={gnorm:.2f}{_gs}  "
                          f"elapsed={fmt(el)}  ETA={fmt(eta)}  gpu_mem(peak)={mem:.2f}GB"
                          f"{_since}")
                    if "aux_attr_bce" in _extra:
                        _ga = _extra.get("aux_attr_gap")
                        print(f"     \U0001F9E9 aux attributes: bce={_extra['aux_attr_bce']:.4f}"
                              f"  shuffled={_extra.get('aux_attr_shuf', float('nan')):.4f}"
                              f"  gap={_ga if _ga is not None else float('nan'):+.4f}"
                              + ("  <- study-specific" if (_ga or 0) > 0.02 else "  <- near the average"))
                    if "aux_cos_loss" in _extra:
                        # gap is the one to watch: aligned falling with gap ~0 means the
                        # head learned the AVERAGE report, not this study's. top1 is
                        # in-batch retrieval against a chance of 1/batch_size.
                        _bs = int(cfg["dataloader"]["batch_size"]) or 1
                        _gp = _extra.get("aux_gap")
                        _t1 = _extra.get("aux_top1")
                        _verdict = ""
                        if _gp is not None:
                            _verdict = ("  <- learning the AVERAGE report, not the study"
                                        if _gp < 0.02 else "  <- study-specific")
                        print(f"     \U0001F4DD aux report: 1-cos={_extra['aux_cos_loss']:.4f}"
                              f"  shuffled={_extra.get('aux_cos_shuf', float('nan')):.4f}"
                              f"  gap={_gp if _gp is not None else float('nan'):+.4f}"
                              f"  top1={_t1 if _t1 is not None else float('nan'):.3f}"
                              f" (chance {1.0 / _bs:.3f}){_verdict}")
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
                    # Is the GPU actually WORKING, or just occupied? img/s says how fast
                    # this model runs; MFU says how much of the card that speed uses, so
                    # a number that looks fine can still be leaving 80% on the table.
                    if device.type == "cuda":
                        if flops_per_img[0] is None:
                            flops_per_img[0] = _measure_flops_per_image(model, x, cfg, device)
                        _sm, _mc = _gpu_busy()
                        _bits = []
                        # x3: backward costs roughly twice forward, so fwd+bwd ~ 3x fwd
                        _ach = 3.0 * flops_per_img[0] * ips / 1e12
                        if flops_per_img[0] and _peak:
                            _bits.append(f"MFU {100.0 * _ach / _peak:.0f}% "
                                         f"({_ach:.0f} of {_peak:.0f} TFLOP/s fp16/bf16)")
                        elif flops_per_img[0]:
                            _bits.append(f"{_ach:.0f} TFLOP/s (no peak figure for this card)")
                        if _sm is not None:
                            _bits.append(f"SM busy {_sm:.0f}%")
                        if _mc is not None:
                            _bits.append(f"mem ctrl busy {_mc:.0f}%")
                        if _bits:
                            print("     compute:    " + " | ".join(_bits), flush=True)
                    prev_loss = l
                    if log_fn:
                        log_fn({"train/loss": l, "train/lr": lr, "train/gnorm": gnorm}, gstep)

        # drop_last=false leaves a ragged tail: those micro-batches ran backward() but
        # never reached opt.step(), so without this their gradients leak into the first
        # step of the next epoch (and a resumed run, being a fresh process, would not
        # have them - the two trajectories then diverge).
        if it % accum and it != n_micro:           # the full epoch's tail is now stepped
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

    # curves for every run, and a plotting bug must never sink a finished run. make_plots is
    # off in shared_config for MODAL (a container holds only its own CSV rows); a LOCAL run has
    # the whole log on disk, so it always plots (local runs had no plots at all until 30 Sep)
    if cfg["output"].get("make_plots", True) or runtime_side() == "local":
        _safe("make_plots", make_plots, cfg, out,
              cfg["output"].get("plots_dir", "plots"))
    # NOT inside the plot branch: with make_plots off (every Modal run) the final blocking
    # flush used to be skipped entirely (fixed 30 Sep)
    saver.join()                           # every checkpoint on disk before we flush
    if saver.error:
        print(f"  ⚠️  a checkpoint write failed: {saver.error}")
    persist(blocking=True)                 # the run is over: this one has to land
    if persist.skipped:
        print(f"  ℹ️  {persist.done} volume flushes, {persist.skipped} "
              f"skipped as redundant (one was already in flight)")
    if persist.last_error:
        print(f"  ⚠️  last volume flush error: {persist.last_error}")
    saver.join()                           # the last checkpoint is on disk (and offered)
    if _UPLOADER is not None:
        _UPLOADER.join()                   # ... and every upload has landed
        _UPLOADER = None
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
def _measure_flops_per_image(model, x, cfg, device) -> float:
    """FLOPs for ONE image's forward pass, counted once on a real batch.

    Counted, not looked up: our models wrap a timm trunk in a per-study head, and every
    published GFLOP figure is for the bare trunk at its own resolution.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return 0.0
    was_training = model.training
    try:
        # ONE study, not the whole batch. The answer is per-image either way, and swinv2
        # already runs at 160 of the B200's 178 GB - a second forward pass over the full
        # batch would OOM the run just to print a diagnostic.
        x1 = x[:1] if x.dim() >= 5 and x.shape[0] > 1 else x
        model.eval()
        with torch.no_grad():
            # count images from what the MODEL sees, never from the loader's shape:
            # x is (B, slots, D, H, W) and prepare_batch flattens it to (B*slots*D, C, H, W),
            # so guessing here silently under-counted by the slice depth and inflated MFU 24x
            xb = prepare_batch(x1, cfg, device)
            n_img = xb.shape[0]
            fc = FlopCounterMode(display=False)
            with fc:
                model(xb)
            total = fc.get_total_flops()
            del xb
        return float(total) / max(n_img, 1)
    except Exception as e:
        # never let a diagnostic kill a training run
        print(f"  [warn] could not count FLOPs: {type(e).__name__}: {e}")
        return 0.0
    finally:
        if was_training:
            model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            # the probe's own allocation must not masquerade as a training peak
            torch.cuda.reset_peak_memory_stats()


# Dense (non-sparse) tensor-core throughput at the precision amp actually uses, bf16/fp16
# with fp32 accumulate. Vendor headline numbers usually quote 2x these via sparsity.
_PEAK_TFLOPS = {
    "B200": 2250.0, "B100": 1800.0, "H200": 989.0, "H100": 989.0,
    "A100": 312.0, "L40S": 362.0, "L4": 121.0, "A10G": 125.0,
    "V100": 125.0, "T4": 65.0,
    "4090": 165.2, "4080": 97.5, "4070": 58.0, "4060": 60.5, "3090": 71.0,
}


def _peak_tflops(name: str) -> float:
    """Peak dense bf16 TFLOP/s for this device, 0.0 when we have no figure for it."""
    for k, v in _PEAK_TFLOPS.items():
        if k.lower() in (name or "").lower():
            return v
    return 0.0


def _gpu_busy() -> tuple:
    """(sm_busy_pct, mem_ctrl_busy_pct) from NVML, (None, None) when unavailable.

    Both are TIME-based: the fraction of the sample window in which any kernel was
    resident, and in which the memory controller was reading or writing. Neither is an
    achieved-bandwidth or a FLOP figure - a kernel using 3% of the tensor cores still
    reports ~100% SM busy, which is exactly why MFU is printed beside them.
    """
    try:
        return torch.cuda.utilization(), torch.cuda.memory_usage()
    except Exception:
        return None, None


def modal_image(extra_pip=()):
    """Container image for remote runs: deps + this runs/ folder + the small CSVs.

    extra_pip: packages one run needs on top (e.g. qwen35_2b_vlm: transformers 5, peft). Installed BEFORE the local
    files are added: Modal refuses any build step after add_local_* (it would rebuild on every code change)."""
    import modal
    img = (modal.Image.debian_slim(python_version="3.11")
            .apt_install("libgl1", "libglib2.0-0")     # OpenCV needs these system libs
            .pip_install("torch", "torchvision", "timm", "numpy", "pandas",
                         "opencv-python-headless", "scikit-learn", "pyyaml", "wandb",
                         "libauc",   # official AUCMLoss + PESG, training only
                         "nvidia-ml-py",   # torch.cuda.utilization/memory_usage need it

                         "matplotlib", "psutil", "pydicom"))
    if extra_pip:                                   # only the run that asks; every other run's image is unchanged
        img = img.pip_install(*extra_pip)
    return (img
            # CODE ONLY. Without this ignore the mount is 7,200 files and 39 GB, because
            # runs/*/results/ holds 30 GB of checkpoints plus thousands of prediction .npz
            # and debug .png - none of which the container reads. Checkpoints reach it from
            # the knee-runs VOLUME, not the image, and results travel back the other way at
            # sync time. Indexing that every launch is pure wait before the GPU starts.
            .add_local_dir(str(Path(__file__).parent), "/root/runs",
                           ignore=["**/results", "**/kaggle", "**/__pycache__", "_wheels",
                                   "**/*.pt", "**/*.npz", "**/*.png", "**/*.pth"])
            # only the small split CSVs travel in the image; stats/ alone is 824 MB of
            # local analysis output the container never opens
            # _relay_stage holds up to 49 GB of shards in transit between workspaces. It
            # must never enter the image, and because relay_shards.py writes it while a
            # build may be running, leaving it in also fails the build outright with
            # "<file> was modified during build process".
            .add_local_dir(str(Path(__file__).parent.parent / "data"), "/root/data",
                           ignore=["shards", "raw", "backups", "stats", "sample_dicoms", "pretrained",
                                   "_relay_stage", "_relay_state.json",
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


def stop_running_app(cfg: dict, timeout: int = 90) -> str | None:
    """Stop this run's Modal app, so the GPU stops billing BEFORE we spend time syncing.

    THE PROBLEM THIS FIXES. sync_on_interrupt used to pull the artifacts first and only then
    hand over to Modal's own handler, which is what actually stops the app. So from Ctrl+C
    until the last file landed - tens of seconds, sometimes a minute - the container was
    still alive and still training. That is paid A100 time spent on ~60 steps that get
    thrown away, on every single interrupt.

    Order is now: stop the container, then sync. The artifacts are safe either way because
    they live on the VOLUME, not in the container, and the container commits them every
    `output.commit_every_steps` (25) regardless.

    WHAT THIS COSTS. Stopping first can lose whatever was written since the last commit: up
    to 24 rows of train_log, and up to 99 steps of checkpoint progress
    (`checkpoint_every_steps` is 100). That is a bounded, cheap loss against paying for a
    minute of idle GPU every time.

    Found by NAME - `modal.App` is created as App(cfg["experiment"]["name"]) in every run's
    train.py - so no run folder needs to change. `--json` is used because the plain table
    truncates the description to "convnext_ba...".
    """
    import subprocess                       # not a module-level import in this file

    name = cfg.get("experiment", {}).get("name")
    if not name:
        return None
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"],
                           capture_output=True, timeout=timeout, env=env)
        out = (r.stdout or b"").decode("utf-8", "replace")
        rows = json.loads(out[out.index("["):out.rindex("]") + 1])
    except Exception as e:
        print(f"     (could not list apps: {type(e).__name__}: {e})", flush=True)
        return None
    for row in rows:
        if row.get("Description") != name or row.get("Stopped at"):
            continue
        state = (row.get("State") or "").lower()
        if "ephemeral" not in state and "running" not in state:
            continue
        app_id = row.get("App ID")
        try:
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id, "-y"],
                           capture_output=True, timeout=timeout, env=env)
            return app_id
        except Exception as e:
            print(f"     (stop failed: {type(e).__name__}: {e})", flush=True)
            return None
    return None


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

    def _mute_heartbeat_noise():
        """Stop modal's heartbeat loop from spamming tracebacks at a deliberately dead app.

        We stop the app ourselves to release the GPU before syncing. The client does not know
        that, so its background heartbeat keeps calling AppHeartbeat, gets
        `ConflictError: App state is APP_STATE_STOPPED`, and logs a FULL TRACEBACK every few
        seconds for the whole sync - tens of seconds of scrollback burying the one line that
        matters. Dropping only the heartbeat-failure records leaves every other modal error
        visible.

        The filter goes on the root HANDLERS as well as the named loggers: a filter on a
        logger only sees records logged directly to it, not ones propagating up from
        children, and the record's origin has moved between modal versions.
        """
        import logging

        class _DropHeartbeat(logging.Filter):
            def filter(self, record):
                try:
                    msg = record.getMessage()
                except Exception:
                    return True
                noisy = "Loop attempt" in msg and "heartbeat" in msg
                if not noisy and record.exc_info:
                    noisy = "APP_STATE_STOPPED" in str(record.exc_info[1])
                return not noisy

        filt = _DropHeartbeat()
        for nm in ("modal-client", "modal", "modal.runner", "modal._utils.async_utils"):
            logging.getLogger(nm).addFilter(filt)
        for h in logging.getLogger().handlers:
            h.addFilter(filt)

    def make(previous):
        def handler(signum, frame):
            print("\n  \u23F9\uFE0F  interrupted — stopping the container, then pulling artifacts "
                  "(Ctrl+C again to skip)", flush=True)
            # STOP THE CONTAINER FIRST. Syncing takes tens of seconds and the GPU bills
            # for every one of them; Modal's own handler only stops the app AFTER ours
            # returns, so doing it here is the difference between paying for the sync and
            # not. See stop_running_app for what this can cost (up to 24 unlogged steps).
            _mute_heartbeat_noise()   # before the stop, or the first failures still print
            _app = stop_running_app(cfg)
            print(f"     container {'stopped: ' + _app if _app else 'not found (already down?)'}"
                  f" - GPU released, now pulling artifacts", flush=True)
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

    # 2b) the SAME macro metric, but read like a table: every validation point gets a
    #     tick carrying its exact global step and epoch, a dropline down to the axis so
    #     the eye can trace it, and its value printed on the marker.
    def _plot_macro_detailed(col, title, fname, mode="max"):
        if va is None or col not in va.columns:
            return
        # x is the POSITION of a validation, not its step: evaluations get denser late in
        # training, and on a linear step axis those points (and their labels) pile up
        # into an unreadable smear. Even spacing gives every point the same room.
        pts = []
        for r in runs:
            v = _run_slice(va, r)
            if v is None:
                continue
            eps = v["epoch"].values if "epoch" in v.columns else v["step"].values
            for st, ep in zip(v["step"].values, eps):
                pts.append((int(st), int(ep) if ep == ep else 0))
        pts = sorted(set(pts))
        if not pts:
            return
        pos = {st: i for i, (st, _) in enumerate(pts)}
        n = len(pts)
        fig, ax = plt.subplots(figsize=(max(11.0, 0.42 * n), 7.5))

        lo, hi = np.inf, -np.inf
        for r in runs:
            v = _run_slice(va, r)
            if v is not None and not np.all(np.isnan(v[col].values.astype(float))):
                y = v[col].values.astype(float)
                lo = min(lo, np.nanmin(y)); hi = max(hi, np.nanmax(y))
        if not np.isfinite(lo):
            return
        span = max(hi - lo, 0.004)
        floor = lo - span * 0.18                 # droplines stop below every marker
        ax.set_ylim(floor, hi + span * 0.30)     # headroom for the rotated value labels

        # a resumed run re-scores steps an earlier segment already scored, with slightly
        # different values; printing every one stacks the labels into an unreadable blot.
        # Each step's number is printed once, from the LAST segment that scored it.
        owner = {}
        for r in runs:
            v = _run_slice(va, r)
            if v is not None:
                for st, yv in zip(v["step"].values, v[col].values.astype(float)):
                    if yv == yv:
                        owner[int(st)] = r
        for r in runs:
            v = _run_slice(va, r)
            if v is None:
                continue
            c = cmap[r]
            x = np.array([pos[int(st)] for st in v["step"].values], dtype=float)
            y = v[col].values.astype(float)
            ax.vlines(x, floor, y, color=c, alpha=0.25, lw=1.0, zorder=1)
            ax.plot(x, y, color=c, marker="o", ms=5, lw=1.8, label=_lbl(r), zorder=3)
            best = int(np.nanargmax(y)) if mode == "max" else int(np.nanargmin(y))
            for i, (xi, yi) in enumerate(zip(x, y)):
                if yi != yi or owner.get(int(v["step"].values[i])) != r:
                    continue
                ax.annotate("%.4f" % yi, (xi, yi), textcoords="offset points",
                            xytext=(0, 8), ha="center", va="bottom",
                            fontsize=7.5, rotation=90,
                            color="crimson" if i == best else c,
                            fontweight="bold" if i == best else "normal", zorder=5)
            if not np.all(np.isnan(y)):
                ax.scatter([x[best]], [y[best]], marker="*", s=220, color="crimson",
                           zorder=6)

        ax.set_xlim(-0.7, n - 0.3)
        ax.set_xticks(range(n))
        ax.set_xticklabels(["%d\nep %d" % (st, ep) for st, ep in pts],
                           fontsize=7)
        ax.set_xlabel("global step (and epoch) at each validation")
        ax.set_ylabel(col)
        ax.set_title("%s - %s (every validation point, evenly spaced)" % (name, title))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
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

    def _plot_aux_report():
        """Is the auxiliary report head learning the STUDY, or just the average report?

        Left: the auxiliary loss against the correct report and against a mismatched one.
        The vertical distance between them is the whole result. Most knee reports look
        alike, so a head that predicts the dataset mean drives the aligned curve down while
        the two lines stay together - which is a null result dressed up as progress.

        Right: in-batch retrieval, whether a study's own report is its nearest. Chance is
        1/batch_size, drawn as the dashed line.
        """
        if tr is None or "aux_cos_loss" not in tr.columns:
            return
        d = tr[tr["aux_cos_loss"].notna()]
        if d.empty:
            return
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        ax.plot(d["global_step"], d["aux_cos_loss"], color="crimson", lw=1.6,
                label="1 - cos to the CORRECT report")
        if "aux_cos_shuf" in d.columns and d["aux_cos_shuf"].notna().any():
            ax.plot(d["global_step"], d["aux_cos_shuf"], color="0.55", lw=1.4, ls="--",
                    label="1 - cos to a MISMATCHED report")
            ax.fill_between(d["global_step"], d["aux_cos_loss"], d["aux_cos_shuf"],
                            color="crimson", alpha=0.12,
                            label="gap = study-specific skill")
        ax.set_xlabel("global step"); ax.set_ylabel("1 - cosine")
        ax.set_title("auxiliary report loss (a gap of ~0 means it learned the average)")
        ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3); _vlines(ax, bsteps)

        if "aux_top1" in d.columns and d["aux_top1"].notna().any():
            ax2.plot(d["global_step"], d["aux_top1"], color="steelblue", lw=1.6)
            bs = int(cfg["dataloader"]["batch_size"]) or 1
            ax2.axhline(1.0 / bs, color="k", ls="--", lw=1.2,
                        label=f"chance = 1/{bs}")
            ax2.set_ylim(0, 1.02)
            ax2.legend(fontsize=8, loc="best")
        ax2.set_xlabel("global step"); ax2.set_ylabel("in-batch top-1")
        ax2.set_title("does a study's own report come out nearest?")
        ax2.grid(True, alpha=0.3); _vlines(ax2, bsteps)
        _save(fig, "aux_report.png")

    _safe("plot:aux_report", _plot_aux_report)
    _safe("plot:loss", _plot_loss)
    _safe("plot:macro_auroc", _plot_macro, "val_macro_auroc", "validation macro AUROC",
          "macro_auroc.png", "max", "audit_macro_auroc")
    _safe("plot:macro_auroc_detailed", _plot_macro_detailed, "val_macro_auroc",
          "validation macro AUROC", "macro_auroc_detailed.png", "max")
    _safe("plot:audit_macro_auroc_detailed", _plot_macro_detailed, "audit_macro_auroc",
          "audit-58 macro AUROC (panel labels)", "audit_macro_auroc_detailed.png", "max")
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
    _load_model_state(model, ck["model"])
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
        with torch.autocast("cuda", enabled=amp, dtype=amp_dtype(cfg)):
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
    """One series' DICOM files -> (n_slices, H, W) uint8, exactly as the shards store it.

    With KNEE_VOLCACHE_DIR set, the result is cached on scratch disk keyed by the sorted
    file list, so every geometry group after the first reads it instead of decoding again.
    The volume is geometry-independent, which is what makes sharing it correct.
    """
    _cdir = os.environ.get("KNEE_VOLCACHE_DIR")
    _cpath = None
    if _cdir and files:
        import hashlib
        _key = hashlib.sha1("|".join(sorted(str(f) for f in files)).encode()).hexdigest()
        _cpath = Path(_cdir) / (_key + ".npy")
        if _cpath.exists():
            try:
                return np.load(_cpath)
            except Exception:
                pass                                   # a torn write: just decode again
    _vol = _dicom_series_volume_uncached(files)
    if _cpath is not None and _vol.size:
        try:
            _tmp = _cpath.with_suffix(".tmp.npy")
            np.save(_tmp, _vol)
            os.replace(_tmp, _cpath)                   # atomic: no reader sees half a file
        except Exception:
            pass
    return _vol


def _dicom_series_volume_uncached(files) -> np.ndarray:
    """The actual decode. Kept separate so the cache above can wrap it."""
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
        # x of ImagePositionPatient. DICOM patient coordinates put +x toward the patient's
        # LEFT, and a knee sits well off the midline, so its sign names the leg. Carried
        # here so input.laterality_flip works at SUBMISSION time, where there is no
        # 12_laterality.csv - the test studies are not in it.
        ipp = ds.get("ImagePositionPatient")
        ipp_x = float(ipp[0]) if ipp is not None and len(ipp) == 3 else float("nan")
        return {"pixel_spacing_r": float(ps[0]), "pixel_spacing_c": float(ps[1]),
                "slice_thickness": thick,
                "spacing_between": float(ds.get("SpacingBetweenSlices", 0) or 0) or thick,
                "rows": int(ds.get("Rows", 0)), "cols": int(ds.get("Columns", 0)),
                "ipp_x": ipp_x, "study": study_uid}

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
                     "ipp_x": g.get("ipp_x", float("nan")),
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


# PUBLIC SIDE: the prize-safe public pipeline (runs/others/ensembling/public_side/) runs in its own kernel through
# nbconvert, so it gets a clean machine and frees both GPUs on exit. Its own cells need a real IPython (each cell
# compiles on its own, inspect.getsource on cell functions), which a plain script would not give it.
_PUBLIC_SIDE_CELL = r"""# PUBLIC PIPELINE FIRST (prize-safe copy, Stage 3 RadImageNet removed), in its own kernel. Whatever happens
# here, our ensemble still runs and submission.csv falls back to it alone.
import os, time, shutil, signal
STRESS = __STRESS__
PUBLIC_W = __PUBLIC_W__
PUBLIC_CAP_H = __PUBLIC_CAP_H__
PUBLIC_OVERLAP = __PUBLIC_OVERLAP__
WORK = Path('/kaggle/working')
PUB_CSV = WORK / 'public_side.csv'
PUB_OK = False
_pub = {'status': 'not run', 'stress': STRESS, 'weight': PUBLIC_W, 'cap_h': PUBLIC_CAP_H}
_pub_map = {}
_pub_t0 = time.time()
try:
    PUB_CSV.unlink(missing_ok=True)
    import pandas as _pd
    _penv = dict(os.environ)
    _penv['RSNA_TIME_BUDGET_S'] = str(int(PUBLIC_CAP_H * 3600))
    _penv.update(__PUBLIC_ENV__)
    _t = _pd.read_csv(COMP / 'test.csv', dtype={'StudyInstanceUID': str})
    _pub_ids = _t.StudyInstanceUID.tolist()
    if STRESS:
        # their code refuses repeated study IDs, so the stress cohort gets fake IDs whose folders point at real ones
        _fake = Path('/tmp/pub_stress_root')
        shutil.rmtree(_fake, ignore_errors=True)
        (_fake / 'test_series').mkdir(parents=True)
        for _it in COMP.iterdir():
            if _it.name not in ('test.csv', 'test_series.csv', 'test_series', 'sample_submission.csv'):
                (_fake / _it.name).symlink_to(_it, target_is_directory=_it.is_dir())
        _ts = _pd.read_csv(COMP / 'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
        _ss = _pd.read_csv(COMP / 'sample_submission.csv', dtype={'StudyInstanceUID': str})
        _rt, _rs, _rss = [], [], []
        for _k in range(STRESS):
            _r = _pub_ids[_k % len(_pub_ids)]
            _f = '%s.%d' % (_r, 9000 + _k)
            _pub_map[_f] = _r
            for _src, _acc in ((_t, _rt), (_ts, _rs), (_ss, _rss)):
                _row = _src[_src.StudyInstanceUID == _r].copy()
                _row['StudyInstanceUID'] = _f
                _acc.append(_row)
            (_fake / 'test_series' / _f).symlink_to(COMP / 'test_series' / _r, target_is_directory=True)
        _pd.concat(_rt).to_csv(_fake / 'test.csv', index=False)
        _pd.concat(_rs).to_csv(_fake / 'test_series.csv', index=False)
        _pd.concat(_rss).to_csv(_fake / 'sample_submission.csv', index=False)
        _penv['RSNA_COMP_ROOT'] = str(_fake)
        _pub_ids = list(_pub_map)
        print('PUBLIC STRESS: %d fake studies over %d real ones' % (len(_pub_ids), len(set(_pub_map.values()))))
    _pnb = WORK / 'public_pipeline.ipynb'
    shutil.copy(W / 'public_pipeline.ipynb', _pnb)
    _cmd = [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', str(_pnb),
            '--output', 'public_pipeline_executed.ipynb', '--output-dir', str(WORK),
            '--ExecutePreprocessor.timeout=-1', '--ExecutePreprocessor.kernel_name=python3']
    print('PUBLIC SIDE start, cap %.2f h%s' % (PUBLIC_CAP_H, ', OVERLAPPED with our ensemble' if PUBLIC_OVERLAP else ''), flush=True)
    _pub_fh = open(WORK / 'public_log.txt', 'w')
    _proc = subprocess.Popen(_cmd, cwd=str(WORK), env=_penv, stdout=_pub_fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
except Exception as _e:
    traceback.print_exc()
    _proc = None
    _pub['status'] = 'failed: %s: %s' % (type(_e).__name__, str(_e)[:500])


def _pub_finish():
    # wait for the public kernel (until its cap), validate public_side.csv, clean up, write the receipt
    global PUB_OK
    if _pub.get('finished'):
        return
    _pub['finished'] = True
    try:
        if _proc is None:
            raise RuntimeError(_pub['status'])
        try:
            _rc = _proc.wait(timeout=max(1.0, _pub_t0 + PUBLIC_CAP_H * 3600 + 600 - time.time()))
        except subprocess.TimeoutExpired:
            # the whole process group: nbconvert AND the kernel it started, or the GPUs stay held
            os.killpg(_proc.pid, signal.SIGKILL)
            _proc.wait()
            _rc = 'timeout'
        _pub_fh.close()
        _pub['returncode'] = _rc
        if _rc != 0:
            raise RuntimeError('public notebook ended with %s' % _rc)
        import pandas as _pd
        _p = _pd.read_csv(PUB_CSV, dtype={'StudyInstanceUID': str})
        _lab = [c for c in _p.columns if c != 'StudyInstanceUID']
        _want = [c for c in _pd.read_csv(COMP / 'sample_submission.csv', nrows=1).columns if c != 'StudyInstanceUID']
        assert _lab == _want, 'public columns %s' % _lab
        assert sorted(_p.StudyInstanceUID) == sorted(_pub_ids), 'public study set differs from test.csv'
        import numpy as _npp
        assert _npp.isfinite(_p[_lab].to_numpy(float)).all(), 'public side has non-finite values'
        _pub.update(status='ok', rows=len(_p))
        PUB_OK = True
    except Exception as _e:
        traceback.print_exc()
        if not str(_pub['status']).startswith('failed'):
            _pub['status'] = 'failed: %s: %s' % (type(_e).__name__, str(_e)[:500])
        try:
            _pub['log_tail'] = (WORK / 'public_log.txt').read_text(errors='replace')[-3000:]
        except Exception:
            pass
    _pub['seconds'] = round(time.time() - _pub_t0, 1)
    _pub['overlap'] = PUBLIC_OVERLAP
    for _d in ('/tmp/rsna_pixels_v559', '/kaggle/temp/rsna_pixels_v559', '/tmp/pub_stress_root',
               str(WORK / '_coat_env'), str(WORK / '_g96_env'), str(WORK / '_d4_env'), str(WORK / 'd4_input')):
        shutil.rmtree(_d, ignore_errors=True)
    (WORK / 'public_pipeline.ipynb').unlink(missing_ok=True)
    try:
        _pub['gpu_mem_after'] = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
                                               capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    (WORK / 'public_receipt.json').write_text(json.dumps(_pub, indent=1, default=str))
    print('PUBLIC SIDE %s in %.1f min (%s rows) | GPU memory after: %s'
          % (_pub['status'][:200], _pub['seconds'] / 60, _pub.get('rows'), _pub.get('gpu_mem_after')), flush=True)


if not PUBLIC_OVERLAP:
    _pub_finish()"""

_PUBLIC_BLEND_CELL = r"""# BLEND per finding in rank space: (1 - w) x rank(ours) + w x rank(public). Ours alone if the public side failed;
# the public side alone if ours fell back to 0.5.
import numpy as _np
if not Path('submission.csv').exists():          # the public kernel's start-up deletes it; never end without one
    bail('no submission.csv before the blend')
_pub_finish()                                    # overlap: wait for the public kernel here (no-op otherwise)
_bl = {'mode': 'ours only', 'weight': PUBLIC_W, 'public_status': _pub['status'], 'overlap': PUBLIC_OVERLAP}

def _rank_blend(ours, theirs, w):
    theirs = theirs.set_index('StudyInstanceUID').reindex(ours.StudyInstanceUID.astype(str))
    assert theirs[TASKS].notna().all().all(), 'public side misses some studies'
    out = ours[['StudyInstanceUID']].copy()
    corr = {}
    for t in TASKS:
        wt = float(w[t]) if isinstance(w, dict) else float(w)
        ro = ours[t].rank(method='average', pct=True).to_numpy()
        rt = theirs[t].rank(method='average', pct=True).to_numpy()
        out[t] = (1 - wt) * ro + wt * rt
        corr[t] = round(float(_np.corrcoef(ro, rt)[0, 1]), 4) if len(ro) > 2 else None
    assert _np.isfinite(out[TASKS].to_numpy(float)).all()
    return out, corr

try:
    if PUB_OK:
        _theirs = pd.read_csv(PUB_CSV, dtype={'StudyInstanceUID': str})
        if STRESS:
            # dry run on the real public studies (first fake copy of each): proves the path, writes no submission
            _theirs['StudyInstanceUID'] = _theirs.StudyInstanceUID.map(_pub_map)
            _theirs = _theirs.drop_duplicates('StudyInstanceUID')
            _ours = pred.drop_duplicates('StudyInstanceUID').reset_index(drop=True)
            _ours['StudyInstanceUID'] = _ours.StudyInstanceUID.astype(str)
            _prev, _bl['rank_corr'] = _rank_blend(_ours, _theirs, PUBLIC_W)
            _prev.to_csv('stress_blend_preview.csv', index=False)
            _bl.update(mode='stress dry run', rows=len(_prev))
        else:
            _ours = pd.read_csv('submission.csv', dtype={'StudyInstanceUID': str})
            _ours.to_csv('submission_ours.csv', index=False)
            _neutral = float((_ours[TASKS].to_numpy(float) == 0.5).mean())
            _w = 1.0 if _neutral > 0.5 else PUBLIC_W
            _out, _bl['rank_corr'] = _rank_blend(_ours, _theirs, _w)
            assert len(_out) == len(sub) and set(_out.StudyInstanceUID) == set(sub.StudyInstanceUID.astype(str))
            _out.to_csv('submission.csv', index=False)
            _bl.update(mode='public only (ours fell back)' if _w == 1.0 else 'blended', rows=len(_out),
                       ours_neutral_share=round(_neutral, 4))
except Exception as _e:
    traceback.print_exc()
    _bl['error'] = '%s: %s' % (type(_e).__name__, str(_e)[:500])
    if not STRESS and Path('submission_ours.csv').exists():
        shutil.copy('submission_ours.csv', 'submission.csv')
        _bl['mode'] = 'ours only (blend failed)'
Path('blend_receipt.json').write_text(json.dumps(_bl, indent=1, default=str))
print('BLEND', json.dumps(_bl, default=str))
pd.read_csv('submission.csv').head()"""


def _public_cell_last(cell: str) -> str:
    """The public-side cell for ORDER "last": our ensemble has already written submission.csv, so
      - STRESS comes from the inference cell (one STRESS line per notebook, as submit_from_stress_stamp needs)
      - the cap is whatever the 9 h allows: 8.2 h minus the time already used (+10 min kill grace = 8.4 h worst case), at most the configured cap; under
        0.5 h left, the public side is skipped and ours stands
      - our models are released first, and our submission.csv is kept aside and restored after the public kernel
        (whose start-up deletes it), so the blend always finds it and a failed public side leaves ours exactly
    """
    head = "STRESS = __STRESS__\nPUBLIC_W = __PUBLIC_W__\nPUBLIC_CAP_H = __PUBLIC_CAP_H__\n"
    assert cell.count(head) == 1
    cell = cell.replace(head, (
        "# STRESS comes from the inference cell (ours ran FIRST)\n"
        "PUBLIC_W = __PUBLIC_W__\n"
        "PUBLIC_CAP_H = min(__PUBLIC_CAP_H__, (8.2 * 3600 - (time.time() - _KERNEL_T0)) / 3600)\n"
        "print('PUBLIC SIDE after ours: %.2f h used, cap %.2f h' % ((time.time() - _KERNEL_T0) / 3600, PUBLIC_CAP_H), flush=True)\n"
        "if not STRESS and Path('submission.csv').exists():\n"
        "    shutil.copy('submission.csv', 'submission_ours.csv')      # the public kernel deletes submission.csv\n"
        "for _nm in ('MODELS', 'SPECS'):\n"
        "    globals().pop(_nm, None)                                  # free our weights before the public kernel loads\n"
        "import gc as _gc\n"
        "_gc.collect()\n"
        "try:\n"
        "    import torch as _tch\n"
        "    _tch.cuda.empty_cache()\n"
        "except Exception:\n"
        "    pass\n"))
    start = "try:\n    PUB_CSV.unlink(missing_ok=True)\n"
    assert cell.count(start) == 1
    cell = cell.replace(start, "try:\n    if PUBLIC_CAP_H < 0.5:\n        raise RuntimeError('skipped: %.2f h left after our ensemble' % PUBLIC_CAP_H)\n"
                               "    PUB_CSV.unlink(missing_ok=True)\n")
    mark = "    _pub['overlap'] = PUBLIC_OVERLAP\n"
    assert cell.count(mark) == 1
    cell = cell.replace(mark, mark + "    _pub['order'] = 'last'\n")
    tail = "if not PUBLIC_OVERLAP:\n    _pub_finish()"
    assert cell.rstrip().endswith(tail)
    return cell.rstrip() + ("\nif not STRESS and Path('submission_ours.csv').exists():\n"
                            "    shutil.copy('submission_ours.csv', 'submission.csv')     # ours back in place before the blend\n")


def build_kaggle_ensemble_assets(members: list, out_dir, kaggle_user: str,
                                 dataset_slug: str, notebook_slug: str, title: str = "",
                                 blend_weights: dict | None = None, stress: int = 0,
                                 public_side: dict | None = None, infer_env: dict | None = None):
    """Bundle SEVERAL runs into one Kaggle dataset + a notebook that averages them.

    public_side: {"notebook": path, "weight": float or {task: float}, "cap_h": float,
    "datasets": [...], "models": [...], "env": {extra env vars for the public kernel}} also runs that notebook (the prize-safe public
    pipeline) first in its own kernel and rank-blends its public_side.csv with ours.

    members: [(name, cfg, exp_dir, checkpoint_path), ...] - member 0 owns the loader.

    Layout differs from the single-model bundle in one way that matters: each member
    gets its own m<i>_model.pt / m<i>_config.json / m<i>_def.py, all FLAT in the
    dataset (kaggle datasets skip subdirectories), and the notebook imports each def
    under its own module name. `from model_def import build_model` would collide -
    every run's train.py defines a class called SliceModel.
    """
    out = Path(out_dir)
    # WIPE the staging directory first. `exist_ok=True` alone leaves every file a previous,
    # larger ensemble wrote, so a 12-member build followed by an 8-member one uploads
    # m8_model.pt and m9_model.pt that belong to neither - dead weight in the dataset, and a
    # trap for anything that counts files to decide what shipped.
    import shutil as _shutil
    if out.exists():
        _shutil.rmtree(out, ignore_errors=True)
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
        # The val_log check above is BLIND for a full-data run: evaluation.enable is false,
        # so there is no val_log, and its "score" is just the step counter. Compare the
        # checkpoint's step with the last step the run actually trained instead.
        # convnext_base_full_all_knees_same_orientation was about to ship step 3850 - a copy
        # pulled mid-run during a workspace hand-off - while training finished at 4400.
        try:
            _tl = Path(exp_dir) / cfg["output"]["run_dir"] / "train_log.csv"
            if not cfg.get("evaluation", {}).get("enable", True) and _tl.exists():
                _tlc = pd.read_csv(_tl)
                # train_log names it global_step; a guessed "step" would raise, be caught
                # by the except below, and quietly disable this whole check
                _col = "global_step" if "global_step" in _tlc.columns else "step"
                _last = int(_tlc[_col].max())
                _ckst = int(ck.get("global_step") or ck.get("step") or 0)
                if _ckst < _last - 1:
                    stale.append((name, _ckst, float(_ckst), float(_last)))
        except Exception as _e:
            print(f"  [warn] could not step-check {name}'s checkpoint: "
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
        # A full checkpoint carries the optimizer state as well as the weights, so it can be
        # three times the model. Rebinding `ck` next iteration is not enough on a 16 GB
        # machine: fragmentation builds across members and the ninth dies with
        # "DefaultCPUAllocator: not enough memory" asking for a few megabytes.
        del ck
        import gc as _gc
        _gc.collect()
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
    # A STALE blend_weights.json is silent and wrong: the builder only writes this file
    # when weights are given, but the notebook loads it whenever it EXISTS, so a flat
    # submission built in a folder a weighted one used would ship the old weights.
    _bw = w / "blend_weights.json"
    if not blend_weights and _bw.exists():
        _bw.unlink()
        print("  🧹 removed a stale blend_weights.json - this build is FLAT")
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
        "dataset_sources": [f"{kaggle_user}/{dataset_slug}"]
                           + list((public_side or {}).get("datasets", [])),
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
        **({"model_sources": list(public_side["models"])}
           if public_side and public_side.get("models") else {}),
    }, indent=2), encoding="utf-8")
    if public_side:
        shutil.copy(public_side["notebook"], w / "public_pipeline.ipynb")
        print(f"  \U0001F91D public side: {Path(public_side['notebook']).name}, weight "
              f"{public_side.get('weight', 0.5)}, cap {public_side.get('cap_h', 3.0)} h, "
              f"{len(public_side.get('datasets', []))} datasets + {public_side.get('models', [])}")

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
# Kaggle sometimes mounts the PREVIOUS dataset version right after an upload (cost a 43 min
# stress run on 1 Oct). Stop in seconds instead; the publisher sees this and pushes again.
import hashlib
_eng = hashlib.sha1((W / 'shared_code.py').read_bytes()).hexdigest()
if _eng != '__ENGINE__':
    Path('engine_mismatch.txt').write_text(_eng)
    raise RuntimeError('ENGINE MISMATCH: mounted %s, expected __ENGINE__' % _eng)
print('engine', _eng)
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

# Load weights whichever side of torch.compile they were saved from. The notebook does
# not import shared_code, so this mirrors its helper: strip a leading '_orig_mod.' (left
# by a checkpoint written from a compiled model) and load into the unwrapped module.
def _load_model_state(model, sd):
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
              for k, v in sd.items()}
    return getattr(model, '_orig_mod', model).load_state_dict(sd)

# SPECS say how to rebuild a member from nothing. The second T4 is driven by a separate
# PROCESS, because threads share one interpreter lock and measured 2x slower per card, and
# a built model holding CUDA tensors cannot be pickled to a child. The child rebuilds its
# own members from these same three files. Plain data only, so it survives the spawn.
SPECS = []
try:
    MODELS = []
    for m, cfg in zip(MEMBERS, CFGS):
        i = m['i']
        SPECS.append({'i': int(i), 'dir': str(W), 'cfg': cfg})
        model = _load_def(i).build_model(cfg)
        ck = torch.load(W / ('m%d_model.pt' % i), map_location='cpu', weights_only=False)
        _load_model_state(model, ck['model'])
        MODELS.append(model.eval())
        print('m%d %-30s epoch %s score %s' % (i, m['name'], ck.get('epoch'), ck.get('score')))
except Exception:
    bail('model load failed'); raise SystemExit(0)""",
        r"""# STRESS repeats the public studies to imitate the hidden set's size, so the
# per-study rate is MARGINAL rather than dominated by one-time warm-up. It also
# suppresses submission.csv, because the repeated rows are not a real submission.
# 0 = a normal run.
STRESS = __STRESS__

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
    # _grouped, not the plain one: members that disagree on loader geometry (size,
    # mm_per_px, n_slices, slots, windowing) cannot share a decode pass, so it decodes once
    # per geometry GROUP and blends at the end. With every member on one geometry this does
    # exactly what predict_dicom_ensemble did, in one pass, so it is safe as the default.
    pred = sc.predict_dicom_ensemble_grouped(CFGS, MODELS, todo, plan, idx, files_by_series,
                                             batch_size=1, weights=_W, model_specs=SPECS)
    # the 9h cap is enforced on the HIDDEN test set, which is larger than what runs
    # here, so the number that has to be extrapolated is s/study, not the total
    _el = _time.time() - _t0
    _rate = _el / max(len(todo), 1)
    print('INFERENCE  %.1f s for %d studies  =  %.2f s/study  |  9h cap fits ~%d studies'
          % (_el, len(todo), _rate, int(9 * 3600 / max(_rate, 1e-9))))
    # timing.json is written by predict_dicom_ensemble and holds the decode/per-member
    # split. Echo it so it is in the log too, then leave the file as a kernel output for
    # runtime_budget.py to ingest.
    try:
        print('TIMING ' + open('timing.json').read().replace(chr(10), ' '))
    except Exception:
        pass
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
    if infer_env:
        # opt-in inference speed-ups (KNEE_INFER_HALF / KNEE_INFER_CL), set before anything runs so the
        # spawned card-1 children inherit them
        cells[0] = ("import os\nos.environ.update(%r)\nprint('inference env', %r)\n"
                    % (dict(infer_env), dict(infer_env))) + cells[0]
    _order_last = bool(public_side) and public_side.get("order") == "last"
    if _order_last:
        # OURS FIRST (5 Oct night): a valid submission exists after ~1.5 h whatever the public side does; the public
        # kernel then gets exactly the time left, and the blend runs if it finishes
        cells[0] = "import time as _kt\n_KERNEL_T0 = _kt.time()\n" + cells[0]
        cells.append(_public_cell_last(_PUBLIC_SIDE_CELL))
        cells.append(_PUBLIC_BLEND_CELL)
    elif public_side:
        # the public pipeline runs right after the paths cell, BEFORE our cells touch a GPU or
        # install anything, so it sees the machine it scored on; STRESS moves up with it
        cells.insert(1, _PUBLIC_SIDE_CELL)
        assert cells[-1].count("STRESS = __STRESS__\n") == 1
        cells[-1] = cells[-1].replace("STRESS = __STRESS__\n", "# STRESS is set in the public-side cell\n")
        # an overlapped public kernel deletes submission.csv at start-up, and a stress run never rewrites it: this
        # read crashed the 14-member overlapped stress (5 Oct). The blend cell guarantees the file instead.
        assert cells[-1].rstrip().endswith("pd.read_csv('submission.csv').head()")
        cells[-1] = cells[-1].rstrip()[:-len("pd.read_csv('submission.csv').head()")] + "print('inference cell done')"
        cells.append(_PUBLIC_BLEND_CELL)
    if public_side:
        cells = [c.replace("__PUBLIC_W__", repr(public_side.get("weight", 0.5)))
                  .replace("__PUBLIC_CAP_H__", repr(float(public_side.get("cap_h", 3.0))))
                  .replace("__PUBLIC_ENV__", repr(dict(public_side.get("env", {}))))
                  .replace("__PUBLIC_OVERLAP__", repr(bool(public_side.get("overlap", False)))) for c in cells]
    _eng = _code_sha(w / "shared_code.py")
    cells = [c.replace("__STRESS__", str(int(stress))).replace("__ENGINE__", _eng) for c in cells]
    _cells = [{"cell_type": "code", "metadata": {}, "source": c.splitlines(True),
               "outputs": [], "execution_count": None} for c in cells]
    if public_side:
        _cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": (
            "# Combined: our ensemble + the public pipeline (prize-safe)\n\n"
            "The public side is a copy of `yamadan96/rsna-knee-d4-public0946` (code by Mattia Angeli, Jiwei Liu, "
            "dreaddevelopment / Roman Tamrazov, Pilkwang and others) with Stage 3 removed (RadImageNet is "
            "non-commercial). Its weights are CC0 datasets; DINOv2-small (Meta) is Apache-2.0. It runs first in "
            "its own kernel, then our ensemble, then a per-finding rank blend of the two.\n").splitlines(True)})
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
                        kaggle_user: str, dataset_slug: str = "", notebook_slug: str = "",
                        kaggle_name: str = ""):
    """Write everything `kaggle kernels push` needs: weights, config, notebook, metadata.

    Two Kaggle objects are involved: a DATASET holding the weights (notebooks run
    offline, so the checkpoint cannot be downloaded at run time) and a NOTEBOOK that
    attaches both it and the competition data.
    """
    exp = cfg["experiment"]["name"]
    out = Path(out_dir)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    (out / "notebook").mkdir(parents=True, exist_ok=True)
    # Kaggle names (titles, slugs) are capped at 50 characters. A long run name cut to fit
    # can land on ANOTHER run's dataset and notebook (the _v3_labels copy of a 44-character
    # run cut back to exactly its parent's names, 28 Sep) and push a new version of them.
    # So a run whose names would be cut must pass a short, unique kaggle_name instead.
    kname = kaggle_name or exp
    if not kaggle_name and any(exp not in _kaggle_title(exp, s) for s in ("weights", "submit")):
        raise SystemExit(f"❌ '{exp}' is too long for Kaggle's 50-character names and would be "
                         f"cut to a name another run may own: pass kaggle_name (KAGGLE_NAME in "
                         f"submit.py), short and unique")
    dataset_slug = dataset_slug or f"knee-{kname.replace('_', '-')}"
    notebook_slug = notebook_slug or f"knee-{kname.replace('_', '-')}-submit"
    # Kaggle takes the slug from the title, so a title trimmed to the 50-character cap
    # moves the notebook. Follow it rather than fight it: the id becomes the title's slug.
    _nb_title = _kaggle_title(kname, "submit")
    notebook_slug = _kaggle_slug(_nb_title)

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
        {"title": _kaggle_title(kname, "weights"), "id": f"{kaggle_user}/{dataset_slug}",
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
        "title": _nb_title,
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
    _sd = ck['model']
    # a checkpoint written from a torch.compile-wrapped model carries an '_orig_mod.'
    # prefix on every key; strip it so either kind of checkpoint loads
    if any(k.startswith('_orig_mod.') for k in _sd):
        _sd = {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
               for k, v in _sd.items()}
    model.load_state_dict(_sd)
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


# ---------------------------------------------------------------- runtime budget --
# The competition caps a submission at 9 hours on the hidden test set, and the ensemble
# is now large enough that the cap, not the score, decides what can ship. Guessing the
# cost from training throughput was wrong by a factor of ten once already, so every run
# measures itself and writes the numbers down.
#
# The cost has exactly two parts and they scale differently:
#   FIXED    model load, DICOM index build - paid once, invisible on a big test set
#   DECODE   per STUDY, paid once however many members ride along
#   GPU      per study PER MEMBER - the only part that grows when a member is added
# Lumping decode into a single "seconds per study" makes every member look far more
# expensive than it is, which is what makes an 8-member ensemble look unaffordable.
TIMING_JSON = "timing.json"


def _timing_report(timing: dict, n_studies: int) -> str:
    """The per-stage breakdown, as text for inference_log.txt."""
    n = max(n_studies, 1)
    lines = ["", "runtime breakdown (seconds per study)",
             f"  decode  {timing['decode_s'] / n:8.3f}   paid once per study, "
             f"however many members"]
    for name, sec in zip(timing["members"], timing["member_s"]):
        lines.append(f"  gpu     {sec / n:8.3f}   {name}")
    # with the members split across cards the cards run at once, so the batch costs the
    # SLOWEST card, not the sum of both
    gpu = float(timing.get("gpu_wall_s") or sum(timing["member_s"])) / n
    lines.append(f"  ---")
    lines.append(f"  total   {timing['decode_s'] / n + gpu:8.3f}   "
                 f"({timing['decode_s'] / n / max(timing['decode_s'] / n + gpu, 1e-9) * 100:.0f}% decode)")
    return chr(10).join(lines)


def _write_timing_json(timing: dict, n_studies: int, device) -> None:
    """Machine-readable timings, so runtime_budget.py can predict the next ensemble.

    Written as a kernel OUTPUT file: the rerun's stdout is not downloadable, but the
    commit run's outputs are, and the per-study rates measured there are the ones that
    extrapolate.
    """
    n = max(n_studies, 1)
    gpu_name = ""
    try:
        if device is not None and getattr(device, "type", "") == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    payload = {
        "n_studies": int(n_studies),
        "gpu": gpu_name,
        "decode_s_per_study": timing["decode_s"] / n,
        "members": [{"name": nm, "gpu_s_per_study": sec / n}
                    for nm, sec in zip(timing["members"], timing["member_s"])],
        "total_s_per_study": (timing["decode_s"] + sum(timing["member_s"])) / n,
    }
    # >1 when members disagreed on loader geometry and the ensemble needed a decode pass per
    # group. decode_s_per_study above is the TOTAL across passes, so the budget stays honest.
    if timing.get("decode_passes"):
        payload["decode_passes"] = int(timing["decode_passes"])
    Path(TIMING_JSON).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"  ⏱️  timings -> {TIMING_JSON}")



def usable_gpus(cfg: dict | None = None) -> int:
    """How many CUDA devices we may actually use for inference.

    Kaggle's "GPU T4 x2" gives two. Everything in this project used one until this existed.
    A config can pin it with inference.max_gpus, which is what the tests use to force the
    single-device path on a two-card machine.
    """
    if not usable_cuda():
        return 0
    # OFF BY DEFAULT until the split path is fixed. The first free kernel run with two
    # cards produced predictions compressed into [0.309, 0.707] where the same ensemble on
    # one card gives [0.052, 0.974], and five of nine members reported 0.000 GPU seconds.
    # Both symptoms say members are being dropped or blended as zeros. Opt in explicitly
    # with inference.max_gpus once checks/two_gpu_split_check.py covers the failure.
    n = torch.cuda.device_count()
    cap = (cfg or {}).get("inference", {}).get("max_gpus", 1)
    return max(1, min(n, int(cap or 1)))


def tune_inference_backend(models=None, verbose: bool = True) -> None:
    """Free throughput at submission time. No accuracy change, no risk.

    cudnn.benchmark: inference shapes here are FIXED per member (the window count and
    resolution never vary inside a run), which is exactly the case benchmark is for - it
    profiles each convolution once and keeps the fastest algorithm. It is off by default.

    channels_last: on a T4 the fp16 tensor cores want NHWC. torch converts NCHW on the fly
    otherwise, every layer, every batch. Rank-4 only - a Conv3d weight is rank 5 and torch
    raises rather than skipping, which is how a 3D backbone once died at startup.

    TF32 is deliberately NOT touched: it is Ampere and newer, and Kaggle's T4 is Turing.
    """
    torch.backends.cudnn.benchmark = True
    # Cap the BLAS thread pools. Our decode runs in its own worker processes and torch
    # spawns its own; left uncapped, OpenBLAS and MKL each grab every core and the three
    # fight each other. The public 0.941 notebook caps these at 4 for the same reason.
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "4")
    moved = 0
    for m in models or []:
        try:
            if all(p.ndim != 5 for p in m.parameters()):
                m.to(memory_format=torch.channels_last)
                moved += 1
        except Exception:
            pass
    # once per process: with shared_decode this runs per chunk per group, and ~200
    # identical lines would bury the ones that matter
    if verbose and not _BANNERS.get("backend"):
        _BANNERS["backend"] = True
        print("  ⚡ inference backend: cudnn.benchmark=True, channels_last on %d/%d "
              "models" % (moved, len(models or [])), flush=True)


def _member_costs(cfgs: list) -> list:
    """A rough GPU cost per member, used only to balance the two cards.

    Measured seconds per study from the eight-member kernel run where we have them; for
    anything else, the window count times a size factor, which is what actually drives the
    forward pass. Being wrong here costs balance, never correctness.
    """
    known = {"convnext_base_relabel": 3.432, "mricore_vitb_cross_attention": 1.387,
             "mricore_vitb_mae_cross_attention": 1.396, "dinov3_convnext_base": 1.507,
             "convnext_base_25d_18slices": 2.005,
             "convnext_base_25d_all_knees_same_orientation": 2.005,
             "swinv2_base_in22k": 4.211, "convnext_base_25d_poolattn": 0.185,
             "convnext_base_25d_perfinding_slots_and_pooling": 2.053}
    out = []
    for c in cfgs:
        nm = c["experiment"]["name"]
        if nm in known:
            out.append(known[nm])
            continue
        try:
            w = images_per_study(c)
            px = float(c["input"]["size"]) ** 2
            out.append(max(0.05, w * px / (24 * 336 * 336) * 2.0))
        except Exception:
            out.append(1.0)
    return out


# ==========================================================================================
# TWO CARDS, TWO PROCESSES.
#
# The threaded member split was correct and measured twice as slow PER CARD: cuda:0 went
# 7.751 -> 16.618 s per study on its own members and cuda:1 8.624 -> 19.536, so two threads
# at half speed each gained precisely nothing. That is the Global Interpreter Lock. Issuing
# a CUDA kernel is Python work that holds the lock, and our members are many small kernels
# over 96 images per study, so the second thread can only issue while the first is not.
#
# A separate PROCESS has its own interpreter and its own lock, so the two cards really do
# issue at the same time. The cost is that a model cannot be handed to a child already built
# (a CUDA tensor does not cross a process boundary), so the child REBUILDS its members from
# the same three files the bundle already ships: m<i>_def.py, m<i>_config.json, m<i>_model.pt.
# That happens once, at start-up, and is amortised over every study.
#
# The batch itself crosses as raw uint8 bytes on a pipe rather than through shared memory:
# /dev/shm is only 64 MB in some containers and a batch is ~100 MB, so shared memory is a
# silent failure waiting to happen. A pipe copy costs about 0.1 s per study against the
# ~16 s of GPU it unblocks.
# ==========================================================================================

def _is_meta(m) -> bool:
    """True for a member whose weights were released (moved to the storage-less meta device)."""
    try:
        return m is not None and next(m.parameters()).device.type == "meta"
    except StopIteration:
        return False


class _HalfInfer(nn.Module):
    """Pure fp16 inference for one member: fp16 weights AND input, autocast off inside.

    Measured on the heavy 96 x 512 px ConvNeXt member (infer_speed_bench.py): 2.03x faster than
    autocast with fp32 weights, max logit change 0.0015. fp16 weights UNDER autocast gained
    nothing (0.92x), which is why autocast is switched off here rather than left to the caller.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m.half()

    def forward(self, x, *a, **k):
        with torch.autocast("cuda", enabled=False):
            out = self.m(x.half(), *a, **k)
        return out.float() if torch.is_tensor(out) else out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["m"], name)


def _infer_layout(model):
    """Opt-in inference speed-ups for ONE member, applied identically in the parent and in every
    child process (children build their own members, so a parent-only tweak misses them).
      KNEE_INFER_CL=1    channels_last for rank-4 models
      KNEE_INFER_HALF=1  pure fp16 (_HalfInfer)
    Both default off: the submitted ensembles so far ran without them."""
    if os.environ.get("KNEE_INFER_CL") == "1":
        try:
            if all(p.ndim != 5 for p in model.parameters()):
                model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    if os.environ.get("KNEE_INFER_HALF") == "1" and not isinstance(model, _HalfInfer):
        model = _HalfInfer(model)
    return model


def build_member_from_bundle(spec: dict):
    """Rebuild ONE member inside a worker process, from the bundle's own files.

    spec is plain data so it survives pickling to a spawned child: {"i", "dir", "cfg"}.
    Every run's train.py defines a class called SliceModel, so each member's model_def is
    imported under its OWN module name - one shared name would hand every member the same
    architecture and silently score the wrong thing.
    """
    import importlib.util
    d, i = Path(spec["dir"]), int(spec["i"])
    ms = importlib.util.spec_from_file_location("m%d_def" % i, d / ("m%d_def.py" % i))
    mod = importlib.util.module_from_spec(ms)
    sys.modules["m%d_def" % i] = mod
    ms.loader.exec_module(mod)
    model = mod.build_model(spec["cfg"])
    ck = torch.load(d / ("m%d_model.pt" % i), map_location="cpu", weights_only=False)
    sd = ck["model"]
    if any(k.startswith("_orig_mod.") for k in sd):      # written from a compiled model
        sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
              for k, v in sd.items()}
    getattr(model, "_orig_mod", model).load_state_dict(sd)
    return _infer_layout(model.eval())


def _proc_worker(dev_str: str, specs: list, task_q, res_q):
    """The child. Builds its members once, then scores whatever batches arrive.

    Every failure is REPORTED, never swallowed: a silent worker death used to leave zero
    logits behind, which sigmoid to exactly 0.5, and the run still announced "9/9 members".
    """
    import traceback
    try:
        dev = torch.device(dev_str)
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        models, cfgs = {}, {}
        for sp in specs:
            # sp["i"] names the FILES (m<i>_def.py); sp["local"] is this member's slot in
            # the calling group's stack, and that is what the parent scatters by.
            loc = int(sp["local"])
            models[loc] = build_member_from_bundle(sp).to(dev)
            cfgs[loc] = sp["cfg"]
        res_q.put(("ready", sorted(models)))
    except Exception as e:
        res_q.put(("error", "worker %s could not build its members: %r\n%s"
                   % (dev_str, e, traceback.format_exc())))
        return

    while True:
        msg = task_q.get()
        if msg is None:
            return
        if msg == ("offload",):
            # Between geometry groups: park this child's weights in host memory so the next
            # group's members have the card. Moving them back is exact, and costs ~0.1 s/GB.
            try:
                for i in models:
                    models[i].to("cpu")
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                # what this child still holds on its card, for the log and the check
                res_q.put(("ok", torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0))
            except Exception as e:
                res_q.put(("error", "worker %s could not offload: %r\n%s"
                           % (dev_str, e, traceback.format_exc())))
                return
            continue
        try:
            for i in models:                 # back on the card if a previous offload parked them
                if next(models[i].parameters()).device != dev:
                    models[i].to(dev)
            shape, dtype, buf = msg
            x = torch.from_numpy(
                np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape).copy())
            out, times = {}, {}
            with torch.no_grad():            # its own: no_grad is thread AND process local
                for i in sorted(models):
                    c = cfgs[i]
                    amp = bool(c["training"]["amp"]) and dev.type == "cuda"
                    t0 = time.perf_counter()
                    with torch.autocast("cuda", enabled=amp):
                        xb = prepare_batch(x, c, dev)
                        o = models[i](xb).float()
                    del xb
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    times[i] = time.perf_counter() - t0
                    out[i] = o.cpu().numpy()
            del x
            if dev.type == "cuda":
                # Hand the allocator's cached blocks back to the driver. Several children
                # share one card across geometry groups, and a child that keeps 11.5 GiB
                # reserved leaves the next one 124 MiB to work with - which is exactly how
                # knee-nine-full-medimageinsight died.
                torch.cuda.empty_cache()
            res_q.put(("ok", (out, times)))
        except Exception as e:
            res_q.put(("error", "worker %s failed on a batch: %r\n%s"
                       % (dev_str, e, traceback.format_exc())))
            return


class _CardProcess:
    """One spawned child owning one card. Built once, reused for every batch."""

    def __init__(self, dev_str: str, specs: list, build_timeout: float = 1800.0):
        ctx = _mp.get_context("spawn")       # fork after CUDA init is undefined behaviour
        self.dev_str, self.n, self.specs = dev_str, len(specs), list(specs)
        self.task_q, self.res_q = ctx.Queue(), ctx.Queue()
        self.p = ctx.Process(target=_proc_worker,
                             args=(dev_str, specs, self.task_q, self.res_q), daemon=True)
        self.p.start()
        tag, payload = self.res_q.get(timeout=build_timeout)
        if tag != "ready":
            self.close()
            raise RuntimeError(payload)
        self.members = payload

    def submit(self, x_np) -> None:
        self.task_q.put((tuple(x_np.shape), x_np.dtype.str, x_np.tobytes()))

    def offload(self, timeout: float = 600.0) -> None:
        """Move this child's weights to host memory and wait until the card is free.
        Returns the bytes the child still has allocated on its card afterwards."""
        self.task_q.put(("offload",))
        return self.result(timeout=timeout)

    def result(self, timeout: float = 1800.0):
        """Wait in short slices, and give up the moment the child is gone.

        A single get(timeout=3600) meant a dead worker cost a FULL HOUR per batch before
        anyone noticed: two studies took two hours that way, and both ended up as 0.5.
        """
        import queue as _q
        waited = 0.0
        while waited < timeout:
            try:
                tag, payload = self.res_q.get(timeout=5.0)
            except _q.Empty:
                waited += 5.0
                if not self.p.is_alive():
                    raise RuntimeError(
                        "worker %s died without reporting (exit code %r)"
                        % (self.dev_str, self.p.exitcode))
                continue
            if tag != "ok":
                raise RuntimeError(payload)
            return payload
        raise RuntimeError("worker %s produced nothing in %.0f s" % (self.dev_str, timeout))

    def close(self) -> None:
        try:
            self.task_q.put(None)
            self.p.join(30)
        except Exception:
            pass
        if self.p.is_alive():
            self.p.terminate()
            self.p.join(10)


_CARD_PROCS: dict = {}


def _get_card_processes(devices: list, assign: list, model_specs: list):
    """One long-lived child per extra card, cached across chunks and geometry groups.

    predict_dicom_ensemble is called once per group per chunk, and rebuilding a card's
    members every time would cost far more than the split saves.
    """
    cards = []
    for di in range(1, len(devices)):
        # model_specs is in the CALLER's order, so i is the local slot and spec["i"] the
        # bundle number that names the files.
        specs = [dict(model_specs[i], local=i) for i in assign[di]]
        if not specs:
            # A one-member geometry group gives the second card nothing. Spawning an empty
            # child still costs a CUDA context (a few hundred MB) on a card that other
            # groups' children are sharing, so don't.
            cards.append(None)
            continue
        key = (str(devices[di]),
               tuple(sorted((str(sp.get("dir")), int(sp["i"]), int(sp["local"]))
                            for sp in specs)))
        cp = _CARD_PROCS.get(key)
        if cp is None or not cp.p.is_alive():
            cp = _CardProcess(str(devices[di]), specs)
            _CARD_PROCS[key] = cp
        cards.append(cp)
    return cards


def _child_resident_gb() -> float:
    """Weights (GB, from the bundle's m<i>_model.pt sizes) that all live children keep on their
    cards. Each geometry group has its own child, and they all share the second card."""
    tot = 0
    for cp in _CARD_PROCS.values():
        for sp in getattr(cp, "specs", []) or []:
            try:
                tot += (Path(sp["dir"]) / ("m%d_model.pt" % int(sp["i"]))).stat().st_size
            except (OSError, KeyError, TypeError, ValueError):
                pass
    return tot / 1e9


def _trim_host_memory() -> None:
    """Give freed host memory back to the system. Moving weights between the card and host
    memory every chunk leaves glibc's heap fragmented, and resident memory then creeps up a
    few hundred MB per chunk (1 Oct stress run: 20.8, 23.1, 23.5, 23.7 GB). Linux only."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def shutdown_card_processes() -> None:
    """Stop every worker. Safe to call twice, and called at the end of a grouped run."""
    for cp in list(_CARD_PROCS.values()):
        try:
            if cp is None:
                continue
            cp.close()
        except Exception:
            pass
    _CARD_PROCS.clear()


import atexit as _atexit                     # a stranded child would hold the kernel open
_atexit.register(shutdown_card_processes)


def _assign_members(cfgs: list, n_dev: int) -> tuple:
    """Greedy longest-processing-time: returns (assign, load).

    assign[d] lists the member indices device d owns, load[d] their predicted seconds.
    Balancing by measured cost matters because the members are wildly uneven -
    swinv2_base_in22k is 4.237 s per study against convnext_base_25d_poolattn's 0.195, so
    round-robin would idle most of the second card. Getting this wrong costs balance only,
    never a number.
    """
    costs = _member_costs(cfgs)
    load = [0.0] * n_dev
    assign = [[] for _ in range(n_dev)]
    for i in sorted(range(len(cfgs)), key=lambda j: -costs[j]):     # heaviest first
        d = min(range(n_dev), key=lambda k: load[k])
        assign[d].append(i)
        load[d] += costs[i]
    return assign, load


def _shard_models(models: list, cfgs: list, devices: list) -> tuple:
    """Assign each member to ONE device. Returns (placed, assign).

    placed[i] is member i already on its device; assign[d] lists the member indices that
    device d owns. Every model exists once, so two cards cost the same total memory as one
    and each holds about half - the opposite of replicating, which is what ran the T4 out.
    """
    if len(devices) == 1:
        return [m.eval().to(devices[0]) for m in models], [list(range(len(models)))]
    assign, load = _assign_members(cfgs, len(devices))
    placed = [None] * len(models)
    for d, idxs in enumerate(assign):
        for i in idxs:
            placed[i] = models[i].eval().to(devices[d])
    if _BANNERS.get("shard"):        # once, not once per chunk per group
        return placed, assign
    _BANNERS["shard"] = True
    print("  \u26a1 %d GPUs, members split by cost: %s"
          % (len(devices), " | ".join(
              "%s=%.1fs" % (str(devices[d]), load[d]) for d in range(len(devices)))),
          flush=True)
    for d, idxs in enumerate(assign):
        print("       %s: %s" % (devices[d],
                                 ", ".join(cfgs[i]["experiment"]["name"][:34] for i in idxs)),
              flush=True)
    return placed, assign


_BANNERS: dict = {}


@torch.no_grad()
def predict_dicom_ensemble(cfgs: list, models: list, studies_df: pd.DataFrame, plan: dict,
                           index: pd.DataFrame, files_by_series: dict, device=None,
                           batch_size: int | None = None, num_workers: int | None = None,
                           report_every: int = 50, weights=None,
                           return_member_logits: bool = False, model_specs=None,
                           tasks_out: list | None = None):
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
    # tasks_out: the caller's column list. The grouped path passes the full 12, so a geometry group made ONLY of
    # partial-head members (a specialist with its own views) still returns 12 columns: NaN where it has no opinion.
    cfg0, tasks = cfgs[0], list(tasks_out) if tasks_out is not None else cfgs[0]["tasks"]
    _inf0 = cfg0.get("inference", {}) or {}
    # Studies per forward. Larger amortises kernel-launch overhead, which is CPU work done
    # under the GIL: at two studies the loop is launch-bound, and a second GPU thread
    # launching at the same time just queues behind the first.
    if batch_size is None and _inf0.get("batch_size"):
        batch_size = int(_inf0["batch_size"])
    if num_workers is None and _inf0.get("num_workers") is not None:
        num_workers = int(_inf0["num_workers"])
    nw = num_workers if num_workers is not None else min(4, (os.cpu_count() or 2))

    # Which of the head task list each member actually predicts. A task-group branch
    # (convnext_base_trauma / _wear / _fluid) carries only its own group, so its logits
    # have to land in the right columns instead of being assumed to be all 12.
    COLS, PRESENT = [], torch.zeros(len(models), len(tasks))
    for i, c in enumerate(cfgs):
        mt = list(c["tasks"])
        unknown = [t for t in mt if t not in tasks]
        if unknown:
            raise ValueError(f"member {i} ({c['experiment']['name']}) predicts "
                             f"{unknown}, which member 0 does not - it cannot join")
        idx = [tasks.index(t) for t in mt]
        COLS.append(torch.as_tensor(idx, device=device, dtype=torch.long))
        PRESENT[i, idx] = 1.0
    if tasks_out is None and not torch.all(PRESENT.sum(dim=0) > 0):
        gap = [t for j, t in enumerate(tasks) if PRESENT[:, j].sum() == 0]
        raise ValueError(f"no member predicts {gap} - the ensemble cannot score them")
    partial = [c["experiment"]["name"] for i, c in enumerate(cfgs)
               if len(c["tasks"]) != len(tasks)]
    if partial:
        print(f"  \U0001f9e9 partial-head members, averaged only on their own findings: "
              f"{', '.join(partial)}")

    # A silently mis-shaped or mis-ordered weight matrix would blend the wrong member into
    # the wrong finding and still produce a plausible submission, so check it loudly here
    # rather than trusting the caller.
    W = None
    if weights is not None:
        W = torch.as_tensor(np.asarray(weights, dtype=np.float32))
        if W.shape != (len(models), len(tasks)):
            raise ValueError(f"weights must be (n_members, n_tasks) = "
                             f"({len(models)}, {len(tasks)}), got {tuple(W.shape)}")
        W = W * PRESENT                           # a member cannot vote on what it lacks
        col = W.sum(dim=0)
        if not torch.all(col > 0):
            raise ValueError(f"every task needs a non-zero weight column, got sums {col}")
        W = (W / col).to(device)                  # each finding's weights sum to 1
        print(f"  \u2696\ufe0f  weighted blend over {len(models)} members x {len(tasks)} findings")
    n = len(studies_df)

    # slice_context is a LOADER setting (KneeShardDataset._load fills the RGB channels
    # with neighbours), not a prepare_batch one, so a 2.5D member cannot ride along on a
    # duplicate-channel decode pass - it would silently score the wrong pixels.
    # resize_to is deliberately ABSENT: it is applied per member inside prepare_batch, from
    # each member's own cfg, so members may disagree on it and still share one decode pass.
    # That is the whole point of the setting.
    # ONE definition, shared with predict_dicom_ensemble_grouped, so the guard here and the
    # grouping there can never drift apart and disagree about what a decode pass requires.
    for i, c in enumerate(cfgs[1:], 1):
        bad = [k for k in LOADER_GEOMETRY_KEYS if c["input"].get(k) != cfg0["input"].get(k)]
        if bad:                                    # loud: a silent mismatch would score
            raise ValueError(                      # every member on the wrong pixels
                f"member {i} disagrees on loader geometry {bad} - it cannot share a "
                f"decode pass with member 0. Use predict_dicom_ensemble_grouped, which "
                f"decodes once per geometry group and blends at the end.")

    ds = KneeShardDataset(studies_df.assign(**{t: 0.0 for t in tasks}),
                          cfg0, plan, train=False, dicom=(index, files_by_series))
    bs = int(batch_size or val_loader_settings(cfg0)[0])
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=device.type == "cuda", persistent_workers=False)
    # BOTH of Kaggle's T4s. The MEMBERS are split across the cards, not the batch: each
    # model lives on exactly one card, so the total weight memory equals the single-card
    # path and each card carries about half of it. Splitting the BATCH instead needed every
    # model on every card, which left a 15 GiB T4 with 256 MiB free before any work.
    _ngpu = usable_gpus(cfg0) if device.type == "cuda" else 1
    devices = ([torch.device("cuda", i) for i in range(_ngpu)]
               if device.type == "cuda" and _ngpu > 1 else [device])
    if device.type == "cuda" and (cfg0.get("inference", {}) or {}).get("tune_backend", False):
        tune_inference_backend(models)
    if device.type == "cuda":                      # opt-in KNEE_INFER_* (children apply it in build_member_from_bundle)
        models = [None if m is None else _infer_layout(m) for m in models]

    # PROCESSES, not threads, when the caller can tell a child how to rebuild a member.
    # Threads share one Global Interpreter Lock and our members are launch-bound, so the
    # threaded split measured 2x slower per card. A child has its own lock.
    _mode = str((cfg0.get("inference", {}) or {}).get("split_mode", "process")).lower()
    cards, assign, _proc_mode = [], None, False
    if len(devices) > 1 and _mode == "process" and model_specs:
        try:
            assign, _load = _assign_members(cfgs, len(devices))
            cards = _get_card_processes(devices, assign, model_specs)
            _proc_mode = True
            # only THIS process's members go on card 0; the children own the rest and hold
            # their own copies, so placing them here would waste a card's memory twice.
            _placed = [None] * len(models)
            for _i in assign[0]:
                _placed[_i] = models[_i].eval().to(devices[0])
            # FREE THIS PROCESS'S COPIES OF THE CHILDREN'S MEMBERS. The notebook builds every
            # member here on the CPU, and each child then rebuilds its own from the bundle, so
            # these copies were never used again yet sat in host RAM for the whole run: ~4.3 GB
            # in the ten + 5 MII folds, which helped push a 33.7 GB machine to 32.7 GB until
            # the kernel killed a data loader and its studies fell back to 0.5 (1 Oct). A card
            # that dies rebuilds its members from the bundle (_run_here), never from these.
            for _d in range(1, len(assign)):
                for _i in assign[_d]:
                    if not _is_meta(models[_i]):
                        models[_i].to_empty(device="meta")
            models = _placed
            if not _BANNERS.get("shard"):
                _BANNERS["shard"] = True
                print("  \u26a1 %d cards, one PROCESS each, members split by cost: %s"
                      % (len(devices), " | ".join("%s=%.1fs" % (devices[d], _load[d])
                                                  for d in range(len(devices)))), flush=True)
                for d, idxs in enumerate(assign):
                    print("       %s: %s" % (devices[d], ", ".join(
                        cfgs[i]["experiment"]["name"][:34] for i in idxs)), flush=True)
        except Exception as _e:
            # A card we cannot start is a speed loss; a submission we cannot write is the
            # whole run. Fall back loudly and carry on with one card.
            print("  \u26a0\ufe0f  could not start a worker process (%r) - one card only"
                  % (_e,), flush=True)
            shutdown_card_processes()
            cards, assign, devices, _proc_mode = [], None, [device], False
    if assign is None:
      # a member whose copy was freed for a child (to_empty on "meta") is rebuilt from the
      # bundle before this process has to score it itself
      if model_specs and any(_is_meta(m) for m in models if m is not None):
          models = [build_member_from_bundle(dict(model_specs[_i], local=_i)) if _is_meta(_m)
                    else _m for _i, _m in enumerate(models)]
      try:
        models, assign = _shard_models(models, cfgs, devices)
      except torch.cuda.OutOfMemoryError:
        # Losing the second card costs speed; losing the submission costs everything.
        print("  \u26a0\ufe0f  out of memory placing %d members on %d cards - one card only"
              % (len(models), len(devices)), flush=True)
        torch.cuda.empty_cache()
        devices = [device]
        models, assign = _shard_models(models, cfgs, devices)
    dev_of = {i: devices[d] for d, idxs in enumerate(assign) for i in idxs}
    # WHAT IS ACTUALLY RESIDENT, before a single forward. Three runs blamed "activations"
    # on a 15 GiB T4 reporting 14.19 GiB ALLOCATED (live, not merely cached) while only
    # ~3.2 GB of weights existed - a gap of ~11 GB nobody had measured. Print it rather
    # than infer it.
    if device.type == "cuda" and not _BANNERS.get("mem"):
        _BANNERS["mem"] = True
        # None where a CHILD PROCESS owns the member: it holds its own copy on its own
        # card, so there is nothing resident here to weigh.
        _wt = sum(p.numel() * p.element_size()
                  for m in models if m is not None for p in m.parameters())
        for _d in range(torch.cuda.device_count()):
            print("  📊 cuda:%d  allocated %.2f GB  reserved %.2f GB  of %.1f GB"
                  % (_d, torch.cuda.memory_allocated(_d) / 1e9,
                     torch.cuda.memory_reserved(_d) / 1e9,
                     torch.cuda.get_device_properties(_d).total_memory / 1e9), flush=True)
        _here = sum(1 for m in models if m is not None)
        print("  📊 this group's %d of %d member(s) are resident here and weigh %.2f GB"
              % (_here, len(models), _wt / 1e9), flush=True)

    out, done, t0, failed = [], 0, time.time(), 0
    notes = []

    def _ram_note(tag: str) -> str:
        # host RAM and the cards' free memory, so a group that dies on Kaggle names the
        # resource it ran out of in inference_log.txt (stdout is not downloadable)
        try:
            import psutil
            vm = psutil.virtual_memory()
            r = "RAM used %.1f / %.1f GB" % ((vm.total - vm.available) / 1e9, vm.total / 1e9)
        except Exception:
            r = "RAM unknown"
        if device.type == "cuda":
            try:
                r += "; " + ", ".join("cuda:%d free %.1f GB" % (d, torch.cuda.mem_get_info(d)[0] / 1e9)
                                      for d in range(torch.cuda.device_count()))
            except Exception:
                pass
        return "%s [%s]: %s" % (tag, ", ".join(c["experiment"]["name"][:30] for c in cfgs), r)
    notes.append(_ram_note("group start"))
    # Runtime accounting, because the 9h cap is the binding constraint as the ensemble
    # grows. Two costs behave completely differently and must never be lumped together:
    # DECODE is paid once per study no matter how many members ride along, while GPU is
    # paid per member per study. Predicting the cost of adding a member needs them apart.
    timing = {"decode_s": 0.0, "gpu_s": 0.0,
              "member_s": [0.0] * len(models),
              "members": [c["experiment"]["name"] for c in cfgs],
              "n_gpus": len(devices)}
    # one accumulator per device; combined as a MEAN at the end, because the devices run
    # concurrently and it is the wall clock that the 9 h cap measures
    _dev_member_s = [[0.0] * len(models) for _ in devices]
    # A card that fails is retired for the REST of the run, and its members are rebuilt in
    # this process. Retrying a card that has already run itself out of memory just buys the
    # same failure once per batch.
    _dead_cards: set = set()
    _fallback_models: dict = {}

    def _neutral(k: int) -> np.ndarray:
        """What a failed batch of k studies contributes: a no-opinion score.

        In probability space that is 0.5; in LOGIT space the same thing is 0.0, and getting
        this wrong would quietly push every failed study towards the positive class, since
        sigmoid(0.5) = 0.62 rather than 0.5.
        """
        if return_member_logits:
            return np.zeros((len(models), k, len(tasks)), np.float32)
        return np.full((k, len(tasks)), 0.5, np.float32)
    it = iter(loader)
    while True:
        _tdec = time.perf_counter()
        try:
            x, _ = next(it)
        except StopIteration:
            break
        except Exception as e:
            left = n - done
            print(f"  ❌ loader died at study {done} ({type(e).__name__}: {e}); "
                  f"remaining {left} score 0.5")
            # this path used to be the one failure that reached no log file
            notes.append("LOADER DIED at study %d (%s: %s) -> %d studies 0.5; %s"
                         % (done, type(e).__name__, str(e)[:300], left, _ram_note("at death")))
            out.append(_neutral(left))
            failed += left
            break
        timing["decode_s"] += time.perf_counter() - _tdec
        k = x.shape[0]

        def _score_batch_proc():
            """Fire the children FIRST, then do this process's own members while they run.

            The children are busy the whole time we are, which is the entire point: their
            kernel launches go through their own interpreter lock, not ours. Everything
            lands in a CPU stack because the members finish in different processes.
            """
            st_cpu = torch.full((len(models), x.shape[0], len(tasks)), float("nan"))
            xnp = x.numpy() if x.is_contiguous() else x.contiguous().numpy()
            _sub = time.perf_counter()
            for _di, cp in enumerate(cards, start=1):
                if cp is not None and _di not in _dead_cards:
                    cp.submit(xnp)
            _ship = time.perf_counter() - _sub

            dev0 = devices[0]
            for _mi in assign[0]:
                c, m = cfgs[_mi], models[_mi]
                amp = c["training"]["amp"] and dev0.type == "cuda"
                _tm = time.perf_counter()
                with torch.autocast("cuda", enabled=amp):
                    xb = prepare_batch(x, c, dev0)
                    o = m(xb).float()
                del xb
                if dev0.type == "cuda":
                    torch.cuda.synchronize(dev0)
                _dev_member_s[0][_mi] += time.perf_counter() - _tm
                st_cpu[_mi][:, COLS[_mi].cpu()] = o.cpu()

            def _run_here(idxs, di):
                """Score these members in THIS process, rebuilding them if we must.

                The parent holds None for a member a child owns, so losing a card means
                rebuilding from its spec. That costs a model load once; emitting 0.5 costs
                the submission.
                """
                for _mi in idxs:
                    m = _fallback_models.get(_mi)
                    if m is None:
                        m = build_member_from_bundle(
                            dict(model_specs[_mi], local=_mi)).to(devices[0])
                        _fallback_models[_mi] = m
                    c = cfgs[_mi]
                    amp = c["training"]["amp"] and devices[0].type == "cuda"
                    _t2 = time.perf_counter()
                    with torch.autocast("cuda", enabled=amp):
                        xb = prepare_batch(x, c, devices[0])
                        o = m(xb).float()
                    del xb
                    if devices[0].type == "cuda":
                        torch.cuda.synchronize(devices[0])
                    _dev_member_s[di][_mi] += time.perf_counter() - _t2
                    st_cpu[_mi][:, COLS[_mi].cpu()] = o.cpu()

            # .result() RAISES what the child reported. A worker that dies quietly used to
            # leave zero logits, which sigmoid to exactly 0.5, while the run still announced
            # "9/9 members" and shipped a submission squeezed into [0.309, 0.707].
            for di, cp in enumerate(cards, start=1):
                if cp is None:
                    continue                 # nothing was assigned to this card
                if di in _dead_cards:
                    _run_here(assign[di], di)
                    continue
                try:
                    got, times = cp.result()
                except Exception as _e:
                    print("  \u26a0\ufe0f  card %s failed (%s) - retiring it and scoring "
                          "its %d member(s) on %s for the rest of the run"
                          % (devices[di], str(_e).split(chr(10))[0][:150],
                             len(assign[di]), devices[0]), flush=True)
                    _dead_cards.add(di)
                    try:
                        cp.close()
                    except Exception:
                        pass
                    if devices[0].type == "cuda":
                        torch.cuda.empty_cache()
                    _run_here(assign[di], di)
                    continue
                for _mi, arr in got.items():
                    st_cpu[_mi][:, COLS[_mi].cpu()] = torch.from_numpy(arr)
                    _dev_member_s[di][_mi] += times[_mi]
            timing["ship_s"] = timing.get("ship_s", 0.0) + _ship

            _dead = [i for i in range(st_cpu.shape[0])
                     if not torch.isfinite(st_cpu[i]).any()
                     or float(st_cpu[i].abs().max()) == 0.0]
            if _dead:
                raise RuntimeError(
                    "two-card process split: member(s) %s produced no usable logits - "
                    "refusing to blend them as 0.5"
                    % [cfgs[i]["experiment"]["name"] for i in _dead])
            if return_member_logits:
                return st_cpu.numpy()
            if W is None:
                avg = torch.nanmean(st_cpu, dim=0)
            else:
                avg = torch.nan_to_num(st_cpu, nan=0.0).mul(W.cpu()[:, None, :]).sum(dim=0)
            return torch.sigmoid(avg).numpy()

        def _score_batch_split():
            """Every card runs ITS OWN members on the whole batch, at the same time.

            Each member is resident on exactly one card, so this is the same arithmetic as
            the single-card path with the members reordered - and reordering cannot change
            a blend that is a mean. The stack is assembled on the CPU because the members
            finish on different devices.
            """
            from concurrent.futures import ThreadPoolExecutor
            st_cpu = torch.full((len(models), x.shape[0], len(tasks)), float("nan"))
            # 0 = no micro-batching, the whole batch in one forward. Micro-batching was
            # added to chase an out-of-memory that turned out to be the thread-local
            # no_grad above; forcing one study per forward starves the GPU instead -
            # convnext_base_25d_poolattn went from 0.185 to 5.403 s per study at MICRO=1.
            MICRO = int((cfg0.get("inference", {}) or {}).get("micro_studies", 0) or 0)
            MICRO = MICRO if MICRO > 0 else x.shape[0]

            @torch.no_grad()
            def _one(di):
                # torch.no_grad() IS THREAD-LOCAL. predict_dicom_ensemble carries the
                # decorator, but a ThreadPoolExecutor worker does not inherit it, so autograd
                # built a graph and kept every intermediate alive: 14.10 GiB allocated on a
                # 14.56 GiB T4 for ONE study. The single-card path never used threads, which
                # is precisely why it never saw this.
                dev = devices[di]
                if dev.type == "cuda":
                    torch.cuda.set_device(dev)
                for _mi in assign[di]:
                    c, m = cfgs[_mi], models[_mi]
                    amp = c["training"]["amp"] and dev.type == "cuda"
                    _tm = time.perf_counter()
                    # MICRO-BATCH. A 512 px member sees 96 windows per study, so a whole
                    # batch of studies in one forward is already a 15 GiB T4's worth of
                    # activations - which is what ran the cards out, not the weights.
                    # Studies are scored independently and pooling never crosses a study,
                    # so slicing here cannot change a number.
                    outs = []
                    for _s0 in range(0, x.shape[0], MICRO):
                        with torch.autocast("cuda", enabled=amp):
                            xb = prepare_batch(x[_s0:_s0 + MICRO], c, dev)
                            outs.append(m(xb).float().cpu())
                        del xb
                    o = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    _dev_member_s[di][_mi] += time.perf_counter() - _tm
                    # COLS was built on cuda:0; index the CPU stack with a CPU copy
                    st_cpu[_mi][:, COLS[_mi].cpu()] = o

            # ThreadPoolExecutor, NOT threading.Thread: .result() re-raises here. A bare
            # Thread swallows its exception and the outer handler then substitutes zeros,
            # which sigmoid to 0.5 - that is how the first two-GPU run shipped a submission
            # squeezed into [0.309, 0.707] while reporting "9/9 members".
            with ThreadPoolExecutor(max_workers=len(devices)) as ex:
                for f in [ex.submit(_one, i) for i in range(len(devices))]:
                    f.result()
            _dead = [i for i in range(st_cpu.shape[0])
                     if not torch.isfinite(st_cpu[i]).any()
                     or float(st_cpu[i].abs().max()) == 0.0]
            if _dead:
                raise RuntimeError(
                    "two-GPU split: member(s) %s produced no usable logits - refusing to "
                    "blend them as 0.5"
                    % [cfgs[i]["experiment"]["name"] for i in _dead])
            if return_member_logits:
                return st_cpu.numpy()
            if W is None:
                avg = torch.nanmean(st_cpu, dim=0)
            else:
                avg = torch.nan_to_num(st_cpu, nan=0.0).mul(W.cpu()[:, None, :]).sum(dim=0)
            return torch.sigmoid(avg).numpy()

        def _score_batch():
            """One batch through every member, freeing each input as we go. With five
            members resident the allocator fragments, and study 2 can die where study 1
            was fine."""
            # NaN means "this member has no opinion on this finding". The task-group
            # branches predict only their own group (4, 6 and 2 of the 12), so their
            # output is narrower than the head task list and has to be scattered into
            # the right columns rather than stacked as-is.
            # In PROCESS mode every member this process owns is already on devices[0],
            # so a group that gave the second card nothing falls through to the plain
            # single-card path below. It must NOT reach _score_batch_split, which is the
            # threaded path and is 2x slower per card.
            if any(c is not None for c in cards):
                return _score_batch_proc()
            if len(devices) > 1 and not _proc_mode:
                return _score_batch_split()
            stacked = torch.full((len(models), x.shape[0], len(tasks)),
                                 float("nan"), device=device)
            for _mi, (c, m) in enumerate(zip(cfgs, models)):
                amp = c["training"]["amp"] and device.type == "cuda"
                _tm = time.perf_counter()
                with torch.autocast("cuda", enabled=amp):
                    xb = prepare_batch(x, c, device)
                    o = m(xb).float()
                del xb                                 # the (96, 3, 512, 512) input
                if device.type == "cuda":
                    torch.cuda.synchronize()           # else we time the launch, not the work
                _el = time.perf_counter() - _tm
                timing["member_s"][_mi] += _el
                # With more than one device configured, the fold at the end rebuilds
                # member_s from the per-device accumulators, so a member scored on
                # THIS path has to be recorded there too or it reports 0.000 s.
                if len(devices) > 1:
                    _dev_member_s[0][_mi] += _el
                stacked[_mi][:, COLS[_mi]] = o
            if return_member_logits:
                # Raw per-member LOGITS, un-blended, for predict_dicom_ensemble_grouped:
                # members that disagree on loader geometry cannot share this decode pass, so
                # the blend has to wait until every geometry group has been scored. NaN still
                # means "this member has no opinion on this finding".
                r = stacked.cpu().numpy()             # (n_members, k, n_tasks)
                del stacked
                return r
            if W is None:
                # nanmean, so each finding is averaged over exactly the members that
                # predict it. With every member full-width this is the plain mean.
                avg = torch.nanmean(stacked, dim=0)    # logit space, as measured offline
            else:
                # W is (n_members, n_tasks) -> unsqueeze the batch axis so each finding
                # gets its own weighting across members. Absent members are already
                # zero-weighted and the remaining column renormalised, above.
                avg = torch.nan_to_num(stacked, nan=0.0).mul(W[:, None, :]).sum(dim=0)
            r = torch.sigmoid(avg).cpu().numpy()
            del stacked, avg
            return r

        _tgpu = time.perf_counter()
        try:
            try:
                out.append(_score_batch())
            except torch.cuda.OutOfMemoryError:
                # the memory is usually there, just spread across blocks the allocator
                # is still holding - so free them and try the same batch once more
                torch.cuda.empty_cache()
                print(f"  ♻️  OOM at study {done}, retrying on a cleared cache",
                      flush=True)
                out.append(_score_batch())
        except Exception as e:                     # a dead batch scores 0.5, run goes on
            failed += k
            msg = f"batch at study {done} failed ({type(e).__name__}: {e}) -> 0.5"
            print(f"  ⚠️  {msg}", flush=True)
            notes.append(msg)
            out.append(_neutral(k))
        timing["gpu_s"] += time.perf_counter() - _tgpu
        if device.type == "cuda":
            torch.cuda.empty_cache()               # next study starts on a clean heap
        done += k
        if report_every and done % report_every < k:
            el = time.time() - t0
            print(f"  … {done}/{n} studies  {el / max(done, 1):.1f}s each  "
                  f"eta {fmt(el / max(done, 1) * (n - done))}", flush=True)
    # OFFLOAD THIS GROUP'S WEIGHTS before the next group runs. Every member used to stay on
    # its card once loaded, so the cards ended up holding EVERY group's weights at once: in
    # the ten + 5 MII folds, card 0 carried 7.17 GB of weights when the 480 px member's group
    # came up, against 4.28 GB in the ten, and that member's batches ran out of memory and
    # fell back to zeros. Parking weights in host memory between groups is exact (the same
    # tensors move back), so at any moment a card holds only the group being scored.
    if (device.type == "cuda"
            and (cfg0.get("inference", {}) or {}).get("offload_between_groups", True)):
        for _i, _m in enumerate(models):
            if _m is not None:
                _m.to("cpu")
        # The CHILDREN keep their weights on their own card by default. The member that ran
        # out of memory (the 480 px MII, a one-member group) always lands on card 0, which
        # this process drives, and offloading the children too parked another ~4 GB in host
        # RAM: the 1 Oct stress run reached 32.7 of 33.7 GB and the kernel killed a loader.
        # "auto" (the default) offloads them only once the weights every child keeps on the
        # second card pass card_weight_budget_gb, so a bigger ensemble cannot fill that card
        # either, while one that fits keeps its weights resident and costs no host RAM.
        _oc = (cfg0.get("inference", {}) or {}).get("offload_children", "auto")
        if _oc == "auto":
            _budget = float((cfg0.get("inference", {}) or {}).get("card_weight_budget_gb", 6.0))
            _oc = _child_resident_gb() > _budget
        if _oc:
            for _di, cp in enumerate(cards, start=1):
                if cp is not None and _di not in _dead_cards:
                    try:
                        cp.offload()
                    except Exception as _e:
                        notes.append("card %s could not offload (%s)" % (devices[_di], str(_e)[:150]))
        for _m in _fallback_models.values():
            _m.to("cpu")
        torch.cuda.empty_cache()
        _trim_host_memory()
    notes.append(_ram_note("group end"))
    timing["notes"] = list(notes)
    # member-logit batches are (n_members, k, n_tasks), so studies live on axis 1
    if return_member_logits:
        probs = (np.concatenate(out, axis=1) if out
                 else np.zeros((len(models), 0, len(tasks)), np.float32))
        n_scored = probs.shape[1]
    else:
        probs = np.concatenate(out) if out else np.zeros((0, len(tasks)))
        n_scored = len(probs)
    if device.type == "cuda":
        torch.cuda.empty_cache()          # the next geometry group's children share this card
    if len(devices) > 1:
        # each member ran on exactly one card, so its own time is already true. What is NOT
        # the sum is the wall clock: the cards ran at the same time, so a batch costs the
        # slowest card, and that is the number the 9 h cap turns on.
        for _i in range(len(models)):
            timing["member_s"][_i] = sum(d[_i] for d in _dev_member_s)
        timing["gpu_wall_s"] = max(sum(d[i] for i in assign[di])
                                   for di, d in enumerate(_dev_member_s))
    summary = (f"scored {n_scored} studies with {len(models)} members in "
               f"{fmt(time.time() - t0)}"
               + (f" ({failed} fell back to 0.5)" if failed else ""))
    print(f"  ✅ {summary}")
    # In member-logit mode this is ONE GROUP of a grouped ensemble, not the whole run. Each
    # group would overwrite the previous group's files, leaving a log that claims the
    # ensemble had only the last group's members - which is exactly what happened on the
    # first grouped test run: "scored 3 studies with 1 members". Hand the timing back
    # instead and let predict_dicom_ensemble_grouped write one combined report.
    if return_member_logits:
        return probs, timing
    # the kernel's stdout is not downloadable, so leave the diagnosis in a file that
    # `kaggle kernels output` can fetch - otherwise a Kaggle-only failure is invisible
    try:
        Path("inference_log.txt").write_text(summary + chr(10) + chr(10).join(notes)
                                             + chr(10) + _timing_report(timing, n_scored),
                                             encoding="utf-8")
    except Exception:
        pass
    try:
        _write_timing_json(timing, n_scored, device)
    except Exception as e:
        print(f"  (timing not recorded: {type(e).__name__}: {e})")
    res = pd.DataFrame(probs, columns=tasks)
    res.insert(0, "StudyInstanceUID", studies_df["StudyInstanceUID"].values)
    return res


# laterality_flip IS a decode setting: it mirrors the pixels, per plane, inside the
# dataset. Leaving it out let a member trained on unmirrored studies be scored on
# mirrored ones - same shapes, no error, wrong input for every left knee. It cost
# the 0.933 submission one of its eight members.
LOADER_GEOMETRY_KEYS = ("slots", "n_slices", "size", "mm_per_px", "window", "window_pct",
                        "channels", "slice_context", "slice_context_stride",
                        "laterality_flip")


def loader_geometry(cfg: dict) -> tuple:
    """The settings that decide what PIXELS a member is handed.

    Members sharing all of these can share one decode pass. resize_to is deliberately
    absent: it is applied per member inside prepare_batch from that member's own cfg, so
    members may disagree on it and still ride the same batch. slices_as_channels is absent
    for the same reason - it is a reshape of an already-decoded tensor, not a decode setting.
    """
    i = cfg["input"]
    return tuple(repr(i.get(k)) for k in LOADER_GEOMETRY_KEYS)


@torch.no_grad()
def predict_dicom_ensemble_grouped(cfgs: list, models: list, studies_df: pd.DataFrame,
                                   plan: dict, index: pd.DataFrame, files_by_series: dict,
                                   device=None, batch_size: int | None = None,
                                   num_workers: int | None = None, report_every: int = 50,
                                   weights=None, model_specs=None) -> pd.DataFrame:
    """Score an ensemble whose members do NOT all agree on loader geometry.

    predict_dicom_ensemble decodes each study once and hands the same uint8 batch to every
    member, which is what makes a six-member ensemble affordable - but it therefore requires
    every member to want the same pixels, and refuses to start otherwise. That blocks an
    otherwise-free win: convnext_base_25d_group3 is 336 px where the rest are 512, and
    swapping it in for convnext_large is +0.0003 val while replacing a 3h34m 196M-parameter
    model with a 51-minute one.

    So: group the members by geometry, decode once PER GROUP, and blend at the end. Cost is
    one extra decode per extra group - measured at 0.188 s/study on a T4 against
    convnext_large's 3.488 s/study of GPU, so the swap above still nets ~3.0 s/study FASTER
    despite paying for a second pass.

    Every group still goes through predict_dicom_ensemble, so its geometry guard remains in
    force within a group; this only removes the requirement that there be exactly one group.
    With a single group the behaviour is identical to calling it directly.
    """
    device = device or torch.device("cuda" if usable_cuda() else "cpu")
    tasks = cfgs[0]["tasks"]
    n = len(studies_df)

    groups: dict = {}
    for i, c in enumerate(cfgs):
        groups.setdefault(loader_geometry(c), []).append(i)

    print(f"\n\U0001f5c2️  {len(models)} members in {len(groups)} geometry group(s) "
          f"= {len(groups)} decode pass(es) over {n} studies")
    for gi, (key, idx) in enumerate(groups.items()):
        c0 = cfgs[idx[0]]["input"]
        print(f"   group {gi}: size={c0['size']} mm_per_px={c0['mm_per_px']} "
              f"n_slices={c0['n_slices']} slots={len(c0['slots'])} -> "
              f"{', '.join(cfgs[i]['experiment']['name'] for i in idx)}")

    # (n_members, n_studies, n_tasks) LOGITS, filled group by group, then blended ONCE. NaN
    # is the initial value so a member that somehow never scores cannot silently vote 0.
    allm = np.full((len(models), n, len(tasks)), np.nan, np.float32)
    timing = {"decode_s": 0.0, "gpu_s": 0.0, "member_s": [0.0] * len(models),
              "members": [c["experiment"]["name"] for c in cfgs],
              "decode_passes": len(groups),
              "n_gpus": usable_gpus(cfgs[0]) if device.type == "cuda" else 0}
    notes_g = []
    t0 = time.time()
    # Resolve each group's member order ONCE: chunking would otherwise redo it per chunk.
    group_plan = []
    for gi, (key, idx) in enumerate(groups.items()):
        sub_cfgs = [cfgs[i] for i in idx]
        sub_models = [models[i] for i in idx]
        # Every member must agree on the TASK list for the columns to line up across groups.
        # predict_dicom_ensemble scatters into cfgs[0]["tasks"] of ITS OWN subset, so a group
        # led by a partial-head member would return a narrower array than this one expects.
        if list(sub_cfgs[0]["tasks"]) != list(tasks):
            order = sorted(range(len(idx)), key=lambda j: len(sub_cfgs[j]["tasks"]),
                           reverse=True)
            idx = [idx[j] for j in order]
            sub_cfgs = [cfgs[i] for i in idx]
            sub_models = [models[i] for i in idx]
            # a group of only partial-head members is fine: tasks_out below aligns its columns to the full list
        group_plan.append((gi, idx, sub_cfgs, sub_models))

    # DECODE ONCE. Every group re-read and re-decompressed every DICOM, although
    # dicom_series_volume's output is geometry-independent. With shared_decode the studies
    # are walked in chunks, every group scores a chunk before the next is decoded, and the
    # raw volumes are shared through a scratch-disk cache that holds one chunk at a time.
    _inf = cfgs[0].get("inference", {}) or {}
    shared = bool(_inf.get("shared_decode")) and len(group_plan) > 1
    chunk = max(1, int(_inf.get("decode_chunk", 32))) if shared else max(n, 1)
    cache_dir = None
    if shared:
        import tempfile
        cache_dir = Path(tempfile.mkdtemp(prefix="knee_volcache_"))
        os.environ["KNEE_VOLCACHE_DIR"] = str(cache_dir)
        print(f"  \u26a1 shared decode: {len(group_plan)} groups share one decode, "
              f"{chunk} studies at a time, cache {cache_dir}", flush=True)
    try:
        for c0 in range(0, n, chunk):
            cdf = studies_df.iloc[c0:c0 + chunk]
            k = len(cdf)
            for gi, idx, sub_cfgs, sub_models in group_plan:
                if c0 == 0:
                    print(f"\n\u25b6 decode pass {gi + 1}/{len(groups)}", flush=True)
                # idx maps this group's local slots back to the caller's member order,
                # so the specs must be reordered with exactly the same map as the models.
                sub_specs = ([model_specs[i] for i in idx] if model_specs else None)
                lg, gtiming = predict_dicom_ensemble(
                    sub_cfgs, sub_models, cdf, plan, index, files_by_series, device=device,
                    model_specs=sub_specs,
                    batch_size=batch_size, num_workers=num_workers,
                    report_every=report_every, return_member_logits=True, tasks_out=list(tasks))
                # everything ADDS across chunks; assigning would keep only the last chunk
                timing["decode_s"] += gtiming["decode_s"]
                timing["gpu_s"] += gtiming["gpu_s"]
                for slot, i in enumerate(idx):
                    timing["member_s"][i] += gtiming["member_s"][slot]
                # a group's own failures (out of memory, a dead card) used to be printed and
                # lost: the kernel's stdout is not downloadable, so they go in the log file
                for _nt in gtiming.get("notes", []):
                    notes_g.append("group %d chunk at %d: %s" % (gi, c0, _nt))
                if lg.shape[1] != k:
                    # A short group used to be zero-padded in silence. Zeros are logits, so
                    # they sigmoid to 0.5 and look like real predictions.
                    miss = k - lg.shape[1]
                    msg = ("group %d chunk at %d returned %d of %d studies; %d padded with "
                           "a NO-OPINION logit" % (gi, c0, lg.shape[1], k, miss))
                    print("  \u26a0\ufe0f  " + msg, flush=True)
                    notes_g.append(msg)
                    pad = np.zeros((lg.shape[0], miss, len(tasks)), np.float32)
                    lg = np.concatenate([lg, pad], axis=1)
                for slot, i in enumerate(idx):
                    allm[i, c0:c0 + k] = lg[slot]
            if cache_dir is not None:
                for f in cache_dir.glob("*.npy"):      # keep scratch disk at one chunk
                    try:
                        f.unlink()
                    except OSError:
                        pass
    finally:
        # every child holds a card and its own copy of that card's weights, so a stranded
        # one would keep the kernel alive past the end of the run
        shutdown_card_processes()
        if cache_dir is not None:
            os.environ.pop("KNEE_VOLCACHE_DIR", None)
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)

    # blend exactly as the single-pass path does: nanmean, or per-class weights
    A = torch.as_tensor(allm)
    if weights is None:
        avg = torch.nanmean(A, dim=0)
    else:
        W = torch.as_tensor(np.asarray(weights, dtype=np.float32))
        if W.shape != (len(models), len(tasks)):
            raise ValueError(f"weights must be ({len(models)}, {len(tasks)}), "
                             f"got {tuple(W.shape)}")
        present = (~torch.isnan(A)).any(dim=1).float()      # (n_members, n_tasks)
        W = W * present
        col = W.sum(dim=0)
        if not torch.all(col > 0):
            raise ValueError(f"every task needs a non-zero weight column, got sums {col}")
        W = W / col
        avg = torch.nan_to_num(A, nan=0.0).mul(W[:, None, :]).sum(dim=0)
    probs = torch.sigmoid(avg).numpy()

    # ONE combined report for the whole ensemble. Written here and nowhere else: the
    # per-group calls deliberately do not write, or the last group would overwrite the rest
    # and the log would under-report the ensemble.
    scored = [cfgs[i]["experiment"]["name"] for i in range(len(models))
              if not np.isnan(allm[i]).all()]
    summary = (f"scored {n} studies with {len(scored)}/{len(models)} members in "
               f"{fmt(time.time() - t0)} over {len(groups)} decode pass(es)")
    print(f"  ✅ {summary}")
    if len(scored) != len(models):
        missing = [c["experiment"]["name"] for i, c in enumerate(cfgs)
                   if np.isnan(allm[i]).all()]
        print(f"  ⚠️  MEMBERS THAT NEVER SCORED: {', '.join(missing)}")
    try:
        _diag = [summary, f"members: {', '.join(scored)}",
                 f"gpus used: {timing.get('n_gpus')}",
                 # which engine actually ran: Kaggle can start a kernel on the PREVIOUS
                 # dataset version right after an upload, and that is invisible otherwise
                 f"code sha1 {_code_sha(__file__)}",
                 # how many chunks the studies were scored in: the reload-after-offload path
                 # only runs from the second chunk on, and 3 public studies never get there
                 f"decode chunks {-(-n // chunk) if n else 0} of up to {chunk} studies"]
        # a member whose logits are all-zero or all-NaN contributes a flat 0.5 and drags
        # the whole blend towards the middle. It must be named in the log, not inferred
        # later from a suspiciously narrow submission.csv.
        _flat = [c["experiment"]["name"] for i, c in enumerate(cfgs)
                 if np.isnan(allm[i]).all() or float(np.nanmax(np.abs(allm[i])) or 0) == 0.0]
        if _flat:
            _diag.append("MEMBERS CONTRIBUTING NOTHING: " + ", ".join(_flat))
        if notes_g:
            _diag.extend(notes_g)
        _rng = np.nanmin(allm), np.nanmax(allm)
        _diag.append("member logit range %.3f .. %.3f (a healthy ensemble spans several "
                     "units; near zero means members are voting 0.5)" % _rng)
        Path("inference_log.txt").write_text(
            chr(10).join(_diag) + chr(10) + _timing_report(timing, n), encoding="utf-8")
    except Exception:
        pass
    try:
        _write_timing_json(timing, n, device)
    except Exception as e:
        print(f"  (timing not recorded: {type(e).__name__}: {e})")

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


def _wait_for_dataset(user: str, slug: str, tries: int = 40, secs: int = 30,
                      expect_models: int = 0, local_dir=None):
    """Block until kaggle finishes processing the upload, and check EVERY member landed.

    Push the notebook too early and the dataset is not mounted — the kernel dies with
    ModuleNotFoundError and you lose ten minutes working out why.

    `expect_models` matters more than the wait. This used to accept the dataset as soon as
    the string "model.pt" appeared anywhere in the listing, which is true once m0 lands. A
    real submission was lost that way: kaggle reported "Upload successful: m2_model.pt
    (357MB)", silently dropped it during ingest, and the check passed on m0 - so the notebook
    ran, failed on a missing file, and the whole kernel run was wasted. Name every file we
    expect and refuse to continue until all of them are listed.
    """
    want = ["m%d_model.pt" % i for i in range(expect_models)] or ["model.pt"]
    missing = list(want)
    for i in range(tries):
        time.sleep(secs)
        # --page-size 100: the listing PAGINATES at 20 rows by default, and this dataset has
        # ~30 files (N models + N configs + N defs + licences + wheels + shared_code.py). With
        # the default page, m5/m6/m7_model.pt sort onto page 2 and simply are not returned -
        # which reads exactly like kaggle dropped them. It cost two false alarms and a wasted
        # kernel run before the cause was pagination rather than ingest.
        out = _kaggle("datasets", "files", f"{user}/{slug}", "--page-size", "100",
                      quiet=True, check=False)
        if "error" in out.lower():
            continue
        missing = [w for w in want if w not in out]
        if not missing and local_dir is not None:
            # Same names are not enough: the OLD version lists them too (1 Oct, stress run 3
            # ran the previous engine). Every file's size must match what was just uploaded.
            try:
                have = kaggle_dataset_files(f"{user}/{slug}")
            except SystemExit:
                continue
            loc = {f.name: f.stat().st_size for f in Path(local_dir).iterdir()
                   if f.is_file() and f.name != "dataset-metadata.json"}
            missing = sorted(k for k in set(loc) | set(have) if loc.get(k) != have.get(k))
        if not missing:
            print(f"      ✅ dataset ready after {(i + 1) * secs}s "
                  f"({len(want)} model file(s) confirmed"
                  f"{', every file size matches the upload' if local_dir is not None else ''})")
            return
    raise SystemExit(
        "❌ kaggle never listed %d of the %d model files after %ds: %s\n"
        "   The upload reported success and ingest dropped them. Re-push the dataset "
        "before running the notebook - continuing would waste the kernel run."
        % (len(missing), len(want), tries * secs, ", ".join(missing)))


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


def _kaggle_slug(title: str) -> str:
    """The slug Kaggle will actually derive from a kernel title.

    It slugifies the TITLE and ignores the id you asked for. So compute the slug the same
    way it does and use that as the id, rather than discovering the mismatch afterwards.
    """
    out = "".join(c if c.isalnum() else "-" for c in title.lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def _kaggle_title(exp: str, suffix: str) -> str:
    """Kaggle rejects a dataset or kernel title outside 6-50 characters.

    `convnext_base_25d_all_knees_same_orientation` alone is 44, so "knee <exp> weights"
    is 56 and the create fails. Drop the decoration before touching the name itself, so
    the title stays readable and still identifies the run.
    """
    for t in ("knee %s %s" % (exp, suffix), "%s %s" % (exp, suffix)):
        if 6 <= len(t) <= 50:
            return t
    # keep the suffix even when the name has to be cut: the dataset and the notebook are
    # built from the same exp, so dropping it would give them identical titles
    return ("%s %s" % (exp[:50 - len(suffix) - 1], suffix))[:50]


def bundle_fingerprint(assets_dir) -> str:
    """One hash for everything that decides what an ensemble notebook computes: the engine
    code, the notebook (with its STRESS line normalised away, so a STRESS build and the real
    build of the same bundle match), and every member's def, config and weight file size."""
    import hashlib
    a = Path(assets_dir)
    h = hashlib.sha1()
    h.update((a / "weights" / "shared_code.py").read_bytes())
    nb = (a / "notebook" / "submit.ipynb").read_text(encoding="utf-8")
    nb = re.sub(r"STRESS = \d+", "STRESS = N", nb)
    nb = re.sub(r'PUBLIC_W = .*?\\n"', r'PUBLIC_W = N\\n"', nb)   # the public-side blend weight is a submit-time knob
    h.update(nb.encode("utf-8"))
    for f in sorted((a / "weights").glob("public_*")):
        h.update(f.name.encode() + f.read_bytes())
    for f in sorted((a / "weights").glob("m*_*")):
        if f.name.endswith("_model.pt"):
            h.update(("%s %d" % (f.name, f.stat().st_size)).encode())
        else:
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


STRESS_PASSES = Path(__file__).resolve().parent / "others" / "ensembling" / "stress_passes"
HIDDEN_TEST_STUDIES = 1300         # the data page: "about 1300 studies in the test set". The old 946 was solved from
                                   # one timed rerun and under-projected the 5 Oct combined runs badly
PUBLIC_REAL_DATA_FACTOR = 1.3      # a stress run repeats 3 cached public studies; real varied studies run slower
HIDDEN_RUNTIME_LIMIT_H = 8.0       # margin under the 9 h cap: Kaggle's machines vary


def _code_sha(path) -> str:
    import hashlib
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _check_public_side(d: Path, stress: int) -> tuple:
    """(problems, projected hidden-set hours, overlapped?) from a kernel's public_receipt.json / blend_receipt.json."""
    problems, hours = [], 0.0
    rp, bp = d / "public_receipt.json", d / "blend_receipt.json"
    if not rp.exists():
        return ["no public_receipt.json: the public-side cell never finished"], hours, False
    r = json.loads(rp.read_text(encoding="utf-8"))
    if r.get("status") != "ok":
        problems.append("public side %s | log tail: %s" % (str(r.get("status"))[:300],
                                                            str(r.get("log_tail", ""))[-600:]))
    if stress:
        if r.get("rows") != stress:
            problems.append("public side scored %s studies, the stress run asked for %d" % (r.get("rows"), stress))
        hours = float(r.get("seconds", 0)) / max(stress, 1) * HIDDEN_TEST_STUDIES / 3600 * PUBLIC_REAL_DATA_FACTOR
    b = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else {}
    want = "stress dry run" if stress else "blended"
    if b.get("mode") != want:
        problems.append("blend mode %r, expected %r (%s)" % (b.get("mode"), want, b.get("error", "")))
    else:
        print("      public side ok in %.1f min, blend %s, rank corr ours vs public %s"
              % (float(r.get("seconds", 0)) / 60, b.get("mode"), b.get("rank_corr")))
    return problems, hours, ("last" if r.get("order") == "last" else bool(r.get("overlap")))


def verify_stress_output(user: str, notebook_slug: str, n_members: int, n_studies: int,
                         code_sha: str, public: bool = False) -> list:
    """Reasons a STRESS commit run does NOT clear its bundle for submission. Empty = cleared.

    The normal commit run scores 3 public studies in ONE decode chunk, so anything that only
    happens between chunks never runs before the hidden 946-study rerun. That is how the ten +
    5 MII folds ensemble scored 0.500 on 1 Oct: the reload path after an offload was never
    exercised. A STRESS run repeats the public studies across several chunks; this reads what
    it wrote and demands that every chunk, every group and every member actually scored."""
    import tempfile
    problems, d = [], Path(tempfile.mkdtemp(prefix="knee_stress_"))
    _kaggle("kernels", "output", "%s/%s" % (user, notebook_slug), "-p", str(d),
            quiet=True, check=False)
    log = d / "inference_log.txt"
    if not log.exists():
        problems.append("no inference_log.txt: inference died before writing it")
    else:
        txt = log.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"scored (\d+) studies with (\d+)/(\d+) members", txt)
        if not m:
            problems.append("the log has no 'scored N studies with k/k members' line")
        else:
            if int(m.group(1)) != n_studies:
                problems.append("scored %s studies, the stress run asked for %d" % (m.group(1), n_studies))
            if int(m.group(2)) != n_members or int(m.group(3)) != n_members:
                problems.append("scored %s/%s members, expected %d" % (m.group(2), m.group(3), n_members))
        for bad, why in (("CONTRIBUTING NOTHING", "a member contributed nothing"),
                         ("NEVER SCORED", "a member never scored"),
                         ("LOADER DIED", "a data loader died"),
                         ("failed (", "batches failed"), ("-> 0.5", "studies fell back to 0.5"),
                         ("padded with", "a group returned short and was padded"),
                         ("could not offload", "a card could not offload")):
            if bad in txt:
                problems.append("%s: %s" % (why, [l for l in txt.splitlines() if bad in l][0][:200]))
        idle = [nm for sec, nm in re.findall(r"gpu\s+(\d+\.\d+)\s+(\S+)", txt) if float(sec) == 0.0]
        if idle:
            problems.append("member(s) took 0.000 s: %s" % ", ".join(idle))
        cm = re.search(r"code sha1 ([0-9a-f]{40})", txt)
        if not cm:
            problems.append("the log does not name its engine code: an old dataset version ran")
        elif cm.group(1) != code_sha:
            problems.append("the kernel ran engine code %s, not this bundle's %s (Kaggle used a "
                            "previous dataset version)" % (cm.group(1)[:10], code_sha[:10]))
        # HOST RAM HEADROOM. The hidden rerun has ~30 chunks where a stress run has 4, so a
        # run that already brushes the ceiling here will cross it there.
        ram = [(float(a), float(b)) for a, b in re.findall(r"RAM used ([\d.]+) / ([\d.]+) GB", txt)]
        if not ram:
            problems.append("the log records no host RAM figures")
        else:
            peak = max(a / b for a, b in ram)
            if peak > 0.85:
                problems.append("host RAM peaked at %.0f%% of the machine; the limit is 85%% "
                                "(the 946-study rerun runs ~8x more chunks)" % (peak * 100))
            # RAM GROWTH projected to the hidden set: resident memory creeps between chunks,
            # so the last chunk-to-chunk rise, carried across the hidden set's chunk count,
            # must still leave 10% of the machine free.
            ends = [(int(c), float(a), float(b)) for c, a, b in re.findall(
                r"chunk at (\d+): group end \[[^\]]*\]: RAM used ([\d.]+) / ([\d.]+) GB", txt)]
            per_chunk = {}
            for c, a, b in ends:                      # the last group's figure in each chunk
                per_chunk[c] = (a, b)
            seq = [per_chunk[c] for c in sorted(per_chunk)]
            if len(seq) >= 3:
                rise = max(0.0, seq[-1][0] - seq[-2][0])
                more = max(0, -(-HIDDEN_TEST_STUDIES // max(1, n_studies // len(seq))) - len(seq))
                proj = seq[-1][0] + rise * more
                if proj / seq[-1][1] > 0.90:
                    problems.append("host RAM is still rising %.1f GB per chunk: over the hidden set's "
                                    "%d more chunks it would reach %.1f of %.1f GB (limit 90%%)"
                                    % (rise, more, proj, seq[-1][1]))
        pub_h, overlap = 0.0, False
        if public:
            pp, pub_h, overlap = _check_public_side(d, n_studies)
            problems += pp
        # WALL CLOCK projected to the hidden set, against the 9 h cap with a margin. The public
        # side's stress time includes its one-off model loading, so its projection runs high.
        wm = re.search(r"scored \d+ studies with \d+/\d+ members in ((?:\d+h)?(?:\d+m)?(?:\d+s)?)", txt)
        if wm:
            secs = sum(int(v) * {"h": 3600, "m": 60, "s": 1}[u]
                       for v, u in re.findall(r"(\d+)([hms])", wm.group(1)))
            ours_h = (secs / max(n_studies, 1)) * HIDDEN_TEST_STUDIES / 3600
            if overlap == "last":
                # ours first, then the public side capped to what is left of 8.2 h: the run cannot pass the cap,
                # so the gate is that ours leaves the public side real room; whether it finishes is reported
                room = 8.2 - (ours_h + 0.25)
                print("      ours first: %.2f h, public side projected %.2f h into %.2f h of room -> %s"
                      % (ours_h + 0.25, pub_h, room, "BLENDS" if pub_h <= room else "likely CUT (ours alone)"))
                if room < 4.0:
                    problems.append("ours takes %.1f h, leaving the public side only %.1f h" % (ours_h + 0.25, room))
                proj_h = 0.0
            else:
                # overlapped: both rates were measured WHILE sharing the machine, so the longer side bounds the run
                proj_h = (max(ours_h, pub_h) if overlap else ours_h + pub_h) + 0.25
                print("      projected hidden-set runtime %.2f h (ours %.2f h, public side %.2f h, %s)"
                      % (proj_h, ours_h, pub_h, "overlapped" if overlap else "sequential"))
            if proj_h > HIDDEN_RUNTIME_LIMIT_H:
                problems.append("projected %.1f h on the %d hidden studies (%.1f s/study + 15 min "
                                "fixed + public side %.1f h) - over the %.1f h safety limit under the 9 h cap"
                                % (proj_h, HIDDEN_TEST_STUDIES, secs / max(n_studies, 1), pub_h,
                                   HIDDEN_RUNTIME_LIMIT_H))
        ch = re.search(r"decode chunks (\d+)", txt)
        if not ch or int(ch.group(1)) < 2:
            problems.append("the run used %s decode chunk(s); a stress run must cross at least 2"
                            % (ch.group(1) if ch else "an unknown number of"))
    import shutil as _sh
    _sh.rmtree(d, ignore_errors=True)
    return problems


def verify_kernel_output(user: str, notebook_slug: str, n_members: int,
                         expect_rows: int | None = None, code_sha: str | None = None,
                         public: bool = False) -> list:
    """Return the reasons this kernel's output must NOT be submitted. Empty means good.

    Reads what the notebook itself wrote: inference_log.txt for the member accounting and
    submission.csv for the predictions. A kernel can finish happily and still have produced
    a file that wastes an attempt.
    """
    import tempfile

    problems, d = [], Path(tempfile.mkdtemp(prefix="knee_kout_"))
    out = _kaggle("kernels", "output", "%s/%s" % (user, notebook_slug), "-p", str(d),
                  quiet=True, check=False)
    log = d / "inference_log.txt"
    sub = d / "submission.csv"
    missing_log = not log.exists()
    if missing_log:
        # NOT fatal on its own. The ENSEMBLE notebook writes this file, so a missing one
        # there means inference died and bail() wrote 0.5. The per-run SOLO notebook never
        # writes it at all, and treating that as failure blocked a perfectly good
        # coatnet_rmlp2_384 submission whose predictions spanned 0.0285 to 0.9938.
        # submission.csv is the ground truth; the log only adds per-member detail.
        print("      note: no inference_log.txt (normal for a solo run) - judging on "
              "submission.csv alone", flush=True)
    else:
        txt = log.read_text(encoding="utf-8", errors="replace")
        if "CONTRIBUTING NOTHING" in txt:
            line = [l for l in txt.splitlines() if "CONTRIBUTING NOTHING" in l]
            problems.append(line[0].strip() if line else "members contributed nothing")
        import re as _re
        m = _re.search(r"with (\d+)/(\d+) members", txt)
        if m and (int(m.group(1)) != n_members or int(m.group(2)) != n_members):
            problems.append("scored %s/%s members, expected %d/%d"
                            % (m.group(1), m.group(2), n_members, n_members))
        # A member at exactly 0.000 s never ran. That is how medimageinsight_davit_huge_336
        # shipped: present in the member list, absent from the arithmetic.
        idle = [nm for sec, nm in _re.findall(r"gpu\s+(\d+\.\d+)\s+(\S+)", txt)
                if float(sec) == 0.0]
        if idle:
            problems.append("member(s) took 0.000 s, so they never ran: %s" % ", ".join(idle))
        if "failed (" in txt or "-> 0.5" in txt:
            problems.append("the log reports failed batches that fell back to 0.5")
        if "LOADER DIED" in txt:
            problems.append("a data loader died and its studies fell back to 0.5")
        if code_sha:
            cm = _re.search(r"code sha1 ([0-9a-f]{40})", txt)
            if not cm or cm.group(1) != code_sha:
                problems.append("the kernel did not run this bundle's engine code (%s): Kaggle "
                                "used a previous dataset version" % (cm.group(1)[:10] if cm else "unnamed"))
    if public:                       # an ours-only fallback would just resubmit the ensemble alone
        problems += _check_public_side(d, 0)[0]
    if not sub.exists():
        problems.append("no submission.csv in the kernel output%s"
                        % (" and no inference_log.txt either" if missing_log else ""))
    else:
        import pandas as _pd
        df = _pd.read_csv(sub)
        cols = [c for c in df.columns if c != "StudyInstanceUID"]
        if expect_rows is not None and len(df) != expect_rows:
            problems.append("submission.csv has %d rows, expected %d" % (len(df), expect_rows))
        v = df[cols].to_numpy(dtype="float64")
        if public and (d / "submission_ours.csv").exists():
            # a rank blend of 3 studies lands on exactly 0.5 legitimately: judge OUR pre-blend file
            _o = _pd.read_csv(d / "submission_ours.csv")
            v = _o[[c for c in _o.columns if c != "StudyInstanceUID"]].to_numpy(dtype="float64")
        neutral = float((v == 0.5).mean())
        if neutral > 0.02:
            problems.append("%.1f%% of the predictions are exactly 0.5" % (neutral * 100))
        if float(v.max()) - float(v.min()) < 0.2:
            problems.append("predictions span only %.3f to %.3f, which is not a real blend"
                            % (float(v.min()), float(v.max())))
    import shutil as _sh
    _sh.rmtree(d, ignore_errors=True)
    return problems


def _kernel_engine_mismatch(user: str, notebook_slug: str) -> str | None:
    """The engine sha a finished kernel refused (it wrote engine_mismatch.txt), else None."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="knee_engine_"))
    _kaggle("kernels", "output", "%s/%s" % (user, notebook_slug), "-p", str(d),
            quiet=True, check=False)
    f = d / "engine_mismatch.txt"
    if f.exists():
        return f.read_text(errors="replace").strip() or "?"
    for p in d.glob("*.log"):                  # the file can be missing from a failed run
        m = re.search(r"ENGINE MISMATCH: mounted ([0-9a-f]+)", p.read_text(errors="replace"))
        if m:
            return m.group(1)
    return None


def _push_and_wait(user, notebook_slug, nb_dir, wait, poll_secs, engine_guard=False,
                   retries: int = 4, retry_wait_s: int = 180):
    """Push the notebook and wait. Returns (version, ok); ok False = error or cancel.

    With engine_guard the notebook stops in seconds when Kaggle mounted a PREVIOUS dataset
    version (it checks the engine sha it was built with). Then wait and push again, up to
    `retries` times, instead of losing a whole run to the old code."""
    version = None
    for attempt in range(1, retries + 1):
        print(f"\n  2/4 notebook {user}/{notebook_slug}"
              + (f"  (attempt {attempt}/{retries})" if attempt > 1 else ""))
        out = _kaggle("kernels", "push", "-p", str(nb_dir))
        m = re.search(r"version\s+(\d+)", out, re.I)
        version = m.group(1) if m else None
        print(f"      pushed version {version or '?'}")
        if not wait:
            return version, True
        print("\n  3/4 waiting for the kernel to finish")
        # wall clock from "pushed" to "complete". Not the same as the kernel's own runtime
        # (kaggle queues before it starts), but it is the only figure visible from here, and
        # it upper-bounds the 9h cap. The notebook prints the real s/study inside its log.
        wait_t0 = time.time()
        while True:
            st = _kaggle("kernels", "status", f"{user}/{notebook_slug}", quiet=True, check=False)
            low = st.lower()
            if "kernelworkerstatus" not in low:
                # a failed status CALL (network, API hiccup) is not a finished kernel: retry, never crash.
                # One such failure killed the 14-member stress waiter after 74 min (5 Oct).
                print(f"      … status call failed, retrying: {st.strip()[:90]}")
                time.sleep(poll_secs)
                continue
            if "complete" in low or "error" in low or "cancel" in low:
                break
            print(f"      … {st.strip()[:90]}")
            time.sleep(poll_secs)
        el = time.time() - wait_t0
        bad_engine = _kernel_engine_mismatch(user, notebook_slug) if engine_guard else None
        if bad_engine:
            print(f"      ⚠️ Kaggle mounted an OLD dataset version (engine {bad_engine[:10]}); "
                  f"the kernel stopped after {el / 60:.1f} min")
            if attempt < retries:
                print(f"      waiting {retry_wait_s}s, then pushing again")
                time.sleep(retry_wait_s)
                continue
            print("      ❌ still the old version after %d pushes: re-upload the dataset" % retries)
            return version, False
        if "complete" in low:
            print(f"      ✅ complete in {el / 60:.1f} min "
                  f"({el / 3600:.2f} h of the 9 h cap, queue included)")
            return version, True
        print(f"      ❌ {st.strip()}")
        print("      fetch the log with: kaggle kernels output "
              f"{user}/{notebook_slug} -p .")
        return version, False
    return version, False


def _stress_verify_and_stamp(assets, user, dataset_slug, notebook_slug, n_members, stress,
                             code_sha, public, fp, version) -> bool:
    """Check a finished STRESS commit run and, if it passed, write the stamp that clears this bundle."""
    assets = Path(assets)
    w = assets / "weights"
    stamp = STRESS_PASSES / ("%s.json" % fp)
    print("\n  4/4 STRESS run: checking every chunk, group and member")
    bad = verify_stress_output(user, notebook_slug, n_members, stress, code_sha, public=public)
    if bad:
        print("      ❌ STRESS RUN FAILED - this bundle may NOT be submitted:")
        for b in bad:
            print("         - %s" % b)
        return False
    STRESS_PASSES.mkdir(parents=True, exist_ok=True)
    # Keep what a later submission needs WITHOUT the weights: the notebook folder and the
    # exact file list (name -> bytes) of the dataset version this run used. A bundle can be
    # bigger than this PC's free disk; submit_from_stress_stamp reuses the dataset already
    # on Kaggle after proving it is byte-for-byte this list.
    nb_keep = STRESS_PASSES / ("%s_notebook" % fp)
    if nb_keep.exists():
        shutil.rmtree(nb_keep)
    shutil.copytree(assets / "notebook", nb_keep)
    manifest = {f.name: f.stat().st_size for f in sorted(w.iterdir()) if f.is_file()
                and f.name != "dataset-metadata.json"}
    stamp.write_text(json.dumps({"fingerprint": fp, "engine_sha1": code_sha,
                                 "notebook": "%s/%s" % (user, notebook_slug),
                                 "dataset": "%s/%s" % (user, dataset_slug),
                                 "version": version, "n_members": n_members,
                                 "stress_studies": stress, "manifest": manifest, "public": public,
                                 "passed": datetime.now().isoformat(timespec="seconds")},
                                indent=1), encoding="utf-8")
    print("      ✅ STRESS RUN PASSED: %d studies, every member scored in every chunk." % stress)
    print("      this exact bundle may now be submitted (stamp %s)" % stamp.name)
    return True


def resume_stress(assets_dir, user: str, dataset_slug: str, notebook_slug: str, stress: int,
                  version: str | None = None, poll_secs: int = 60) -> bool:
    """Finish a STRESS publish whose local waiter died (e.g. a failed status call) while the kernel ran on:
    wait for that kernel, then verify and stamp exactly as kaggle_publish would. The bundle in assets_dir
    must still be the one that was uploaded (its fingerprint and file list go into the stamp)."""
    assets = Path(assets_dir)
    n_members = len(list((assets / "weights").glob("m*_model.pt")))
    fp = bundle_fingerprint(assets)
    code_sha = _code_sha(assets / "weights" / "shared_code.py")
    public = (assets / "weights" / "public_pipeline.ipynb").exists()
    _banner("RESUME STRESS %s" % fp[:12], "\U0001F501")
    while True:
        st = _kaggle("kernels", "status", f"{user}/{notebook_slug}", quiet=True, check=False).lower()
        if "kernelworkerstatus.complete" in st:
            break
        if "kernelworkerstatus.error" in st or "kernelworkerstatus.cancel" in st:
            print("      ❌ kernel ended: %s" % st.strip()[:120])
            return False
        print("      … %s" % st.strip()[:90], flush=True)
        time.sleep(poll_secs)
    return _stress_verify_and_stamp(assets, user, dataset_slug, notebook_slug, n_members, stress,
                                    code_sha, public, fp, version)


def kaggle_publish(assets_dir, user: str, dataset_slug: str, notebook_slug: str,
                   message: str, wait: bool = True, poll_secs: int = 60,
                   submit: bool = True, n_members: int | None = None,
                   stress: int = 0, allow_unstressed: bool = False) -> str | None:
    """Dataset -> notebook -> wait for the run -> submit. Returns the kernel version.

    allow_unstressed: the USER's explicit override of the stress gate for one submission (5 Oct night, combined
    ours-first notebook); the commit run's own output checks still run before the attempt is spent.

    The weights go up as a dataset because a code-competition notebook has no
    internet; the notebook itself is what Kaggle scores.
    """
    assets = Path(assets_dir)
    if n_members is None:                      # count what we are about to ship
        n_members = len(list((Path(assets_dir) / "weights").glob("m*_model.pt")))
    _banner("KAGGLE PUBLISH", "\U0001F680")
    try:                                       # the assets are the source of truth here too
        _did = str(json.loads((assets / "weights" / "dataset-metadata.json").read_text())
                   .get("id", "")).split("/")[-1]
        if _did and _did != dataset_slug:
            print("      dataset slug from the assets: %s (was %s)" % (_did, dataset_slug))
            dataset_slug = _did
    except (OSError, ValueError):
        pass

    # THE STRESS GATE. An ensemble is submitted only if a STRESS commit run of this EXACT
    # bundle (same engine code, notebook, member defs, configs and weight files) has already
    # passed verify_stress_output. The plain commit run scores 3 studies in one chunk and
    # cannot see anything that happens between chunks: the ten + 5 MII folds ensemble passed
    # it and then scored 0.500 on the hidden set (1 Oct). Checked BEFORE anything uploads.
    ensemble = n_members > 1
    fp = bundle_fingerprint(assets) if ensemble else None
    code_sha = _code_sha(assets / "weights" / "shared_code.py") if ensemble else None
    stamp = (STRESS_PASSES / ("%s.json" % fp)) if ensemble else None
    public = (assets / "weights" / "public_pipeline.ipynb").exists()
    if ensemble:
        print("      bundle fingerprint %s  engine %s%s" % (fp[:12], code_sha[:12],
                                                          "  + PUBLIC SIDE" if public else ""))
    if submit and ensemble and not stress and not stamp.exists() and allow_unstressed:
        print("      ⚠️ no stress stamp: submitting on the user's explicit override (SKIP_STRESS_GATE)")
    elif submit and ensemble and not stress and not stamp.exists():
        raise SystemExit(
            "❌ REFUSING TO SUBMIT: no passing STRESS run for this exact bundle (%s). "
            "Build it again with STRESS = 128 and SUBMIT = False, let it pass, then submit."
            % fp[:12])
    if submit and stress:
        raise SystemExit("❌ a STRESS build writes no submission.csv: set SUBMIT = False")

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
    elif not any(k in low for k in ("successfully", "being created", "is being created")):
        # anything else is a rejection. It used to fall through to _wait_for_dataset and
        # burn its full 600s timeout before reporting "kaggle never listed model.pt",
        # which hides the real message (e.g. "title must be between 6 and 50 characters").
        raise SystemExit("❌ dataset upload failed: %s" % out.strip()[-500:])   # the END holds the reason (e.g. quota)
    _wait_for_dataset(user, dataset_slug, expect_models=len(list(w.glob("m*_model.pt"))),
                      local_dir=w if ensemble else None)

    # the assets are the source of truth: build_kaggle_assets may have had to change the
    # slug to keep it consistent with a title Kaggle would otherwise have re-slugified
    try:
        _meta = json.loads((assets / "notebook" / "kernel-metadata.json").read_text())
        _id = str(_meta.get("id", "")).split("/")[-1]
        if _id and _id != notebook_slug:
            print("      notebook slug from the assets: %s (was %s)" % (_id, notebook_slug))
            notebook_slug = _id
    except Exception:
        pass
    version, ok = _push_and_wait(user, notebook_slug, assets / "notebook", wait, poll_secs,
                                 engine_guard=ensemble)
    if not ok or not wait:
        return version

    if stress and ensemble:
        _stress_verify_and_stamp(assets, user, dataset_slug, notebook_slug, n_members, stress,
                                 code_sha, public, fp, version)
        return version
    if not submit:
        return version

    # "Complete" is not "correct". knee-nine-full-medimageinsight finished, was submitted,
    # and only then turned out to have run two of its nine members not at all and written
    # 0.5 for two of three studies. The kernel writes down everything needed to catch that,
    # so read it before spending an attempt.
    print("\n  4/4 checking the kernel's own output before submitting")
    bad = verify_kernel_output(user, notebook_slug, n_members=n_members, code_sha=code_sha,
                               public=public)
    if bad:
        print("      \u274c NOT SUBMITTING - the kernel completed but its output is bad:")
        for b in bad:
            print("         - %s" % b)
        print("      no attempt was spent. Fix it and push again.")
        return version
    print("      \u2705 output looks sound: every member ran, no 0.5 fallbacks")
    print("\n  4/4 submitting")
    args = ["competitions", "submit", "-c", "rsna-knee-abnormality-detection",
            "-f", "submission.csv", "-k", f"{user}/{notebook_slug}", "-m", message]
    if version:
        args += ["-v", version]
    _kaggle(*args)
    return version


def kaggle_dataset_files(user_slug: str) -> dict:
    """{file name: bytes} of a Kaggle dataset's CURRENT version (follows every page)."""
    import csv as _csv
    import io as _io
    out, token = {}, None
    for _ in range(50):
        args = ["datasets", "files", user_slug, "--csv", "--page-size", "200"]
        if token:
            args += ["--page-token", token]
        txt = _kaggle(*args, quiet=True)
        m = re.search(r"Next Page Token = (\S+)", txt)
        token = m.group(1) if m else None
        body = txt[txt.find("name,size"):] if "name,size" in txt else ""
        for row in _csv.DictReader(_io.StringIO(body)):
            try:
                out[row["name"]] = int(row["size"])
            except (KeyError, ValueError, TypeError):
                pass
        if not token:
            break
    return out


def submit_from_stress_stamp(fingerprint: str, message: str, poll_secs: int = 60,
                             submit: bool = True, public_w=None,
                             notebook_slug: str | None = None) -> str | None:
    """Submit an ensemble that passed its STRESS run, WITHOUT rebuilding it or holding its
    weights on this PC: the dataset version that STRESS run used is already on Kaggle.

    Refuses unless (1) a stamp exists for this fingerprint, (2) the Kaggle dataset's current
    version is byte-for-byte the stamped file list, then pushes the stamped notebook with
    STRESS = 0, and finally (3) demands the kernel ran the stamped engine (code sha1) and passed
    the usual output checks before spending an attempt.

    public_w: for a bundle with the public side, the blend weight to submit with (float or
    {task: float}); the fingerprint ignores it, so every weight rides on the one stress pass."""
    stamps = [p for p in STRESS_PASSES.glob("*.json") if p.stem.startswith(fingerprint)]
    if len(stamps) != 1:
        raise SystemExit("❌ no unique stress stamp for %r (found %d)" % (fingerprint, len(stamps)))
    st = json.loads(stamps[0].read_text(encoding="utf-8"))
    fp = st["fingerprint"]
    nb_saved = STRESS_PASSES / ("%s_notebook" % fp)
    if "manifest" not in st or "dataset" not in st or not nb_saved.exists():
        raise SystemExit("❌ stamp %s predates notebook/manifest keeping: rebuild and "
                         "run a new STRESS build" % fp[:12])
    _banner("SUBMIT FROM STRESS STAMP %s" % fp[:12], "\U0001F680")
    print("  1/4 the Kaggle dataset must be exactly the stress-tested one")
    have = kaggle_dataset_files(st["dataset"])
    want = st["manifest"]
    diff = sorted(set(have) ^ set(want)) + sorted(k for k in set(have) & set(want) if have[k] != want[k])
    if diff:
        raise SystemExit("❌ %s is not the stress-tested version: %d file(s) differ, e.g. %s"
                         % (st["dataset"], len(diff), diff[:5]))
    print("      ✅ %d files, every size matches the stamp" % len(want))
    import tempfile
    nb = Path(tempfile.mkdtemp(prefix="knee_stamp_nb_")) / "notebook"
    shutil.copytree(nb_saved, nb)
    ipynb = nb / "submit.ipynb"
    src = ipynb.read_text(encoding="utf-8")
    new, k = re.subn(r"STRESS = \d+", "STRESS = 0", src)
    if k != 1:
        raise SystemExit("❌ expected exactly one STRESS line in the saved notebook, found %d" % k)
    if "ENGINE MISMATCH" not in new or st["engine_sha1"] not in new:
        raise SystemExit("❌ the saved notebook has no engine guard for %s: rebuild and run a "
                         "new STRESS build" % st["engine_sha1"][:10])
    public = bool(st.get("public"))
    if public_w is not None:
        if not public:
            raise SystemExit("❌ public_w given, but stamp %s has no public side" % fp[:12])
        # on the PARSED notebook: a dict repr ("Baker's") brings quotes the raw JSON text cannot take
        _nbj, k = json.loads(new), 0
        for _c in _nbj["cells"]:
            for _j, _line in enumerate(_c.get("source", [])):
                if _line.startswith("PUBLIC_W = "):
                    _c["source"][_j] = "PUBLIC_W = %r\n" % (public_w,)
                    k += 1
        if k != 1:
            raise SystemExit("❌ expected exactly one PUBLIC_W line in the saved notebook, found %d" % k)
        new = json.dumps(_nbj, indent=1)
        print("      (PUBLIC_W = %r)" % (public_w,))
    ipynb.write_text(new, encoding="utf-8")
    user, _stamped_slug = st["notebook"].split("/", 1)
    if notebook_slug and notebook_slug != _stamped_slug:
        # the SAME stress-tested notebook pushed as a fresh kernel: set A's kernel kept erroring at start with an
        # empty log (5 Oct) while a probe with identical sources ran, so the kernel's own state was the suspect
        meta = json.loads((nb / "kernel-metadata.json").read_text(encoding="utf-8"))
        meta.update(id="%s/%s" % (user, notebook_slug), title=notebook_slug.replace("-", " "))
        meta.pop("id_no", None)
        (nb / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("      (pushed as a new kernel: %s)" % notebook_slug)
    notebook_slug = notebook_slug or _stamped_slug
    print("      (STRESS = 0)")
    version, ok = _push_and_wait(user, notebook_slug, nb, True, poll_secs, engine_guard=True)
    if not ok:
        return version
    print("\n  4/4 checking the kernel's own output before submitting")
    bad = verify_kernel_output(user, notebook_slug, n_members=int(st["n_members"]),
                               code_sha=st["engine_sha1"], public=public)
    if bad:
        print("      ❌ NOT SUBMITTING:")
        for b in bad:
            print("         - %s" % b)
        return version
    print("      ✅ output looks sound, and the kernel ran the stress-tested engine")
    if not submit:
        return version
    args = ["competitions", "submit", "-c", "rsna-knee-abnormality-detection",
            "-f", "submission.csv", "-k", "%s/%s" % (user, notebook_slug), "-m", message]
    if version:
        args += ["-v", version]
    _kaggle(*args)
    return version
