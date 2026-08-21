"""
sync_runs.py — keep run artifacts identical on the Modal volume and on this PC.
==============================================================================

Training happens on Modal, across ~16 workspaces, and writes to the `knee-runs`
volume under `<experiment>/results/`. The container cannot push anything to this
machine, so the pull has to happen here, right after the remote call returns —
whether it returned a score or raised.

What moves
----------
    train_log.csv, val_log.csv   merged (union of rows), written back BOTH ways
    train_config/                whole files, copy-if-missing, both ways
    debugged_images/             whole files, copy-if-missing, both ways
    training_summary.json        copy-if-missing only (it is a per-side "latest")
    checkpoints/                 never touched — far too big for this disk

Why nothing collides
--------------------
Every run carries a sequential `run_id` (run001, run002, ...), allocated on this
PC before launch and passed into the container. It is the first column of both
CSVs and it prefixes every file the run writes -- `debugged_images/run001_e1_s800.png`,
`train_config/run001_<timestamp>.txt`. So "copy the file if the other side lacks
it" is always safe, and merging is idempotent and order-independent.

Usage
-----
    python modal_scripts/sync_runs.py --session 2
    python modal_scripts/sync_runs.py --session 2 5 9   # several workspaces
    python modal_scripts/sync_runs.py --session 2 --experiment convnext_tiny_4slot
    python modal_scripts/sync_runs.py --all-sessions --direction down
    python modal_scripts/sync_runs.py --session 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import _env, workspaces  # noqa: E402  — workspace selection lives there

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
VOL = "knee-runs"
RESULTS = "results"

# never synced: too big, or purely local bookkeeping
# plots are drawn locally from the merged CSVs (make_plots is off on Modal), so a
# remote plots/ is at best stale and at worst a partial curve from one workspace
# train_config is the user's own console capture, plots are drawn locally from the
# merged CSVs - neither belongs on a volume, and syncing them was what bred those
# run001-ws1-ws2-ws3_... names as each hand-off re-prefixed an already-prefixed file
SKIP_DIRS = {"checkpoints", "plots", "train_config",
             "__pycache__", ".run_ids", ".ipynb_checkpoints"}
# row-level merge, with the columns that identify a row
MERGE_CSVS = {"train_log.csv": ("run_id", "global_step"),
              "val_log.csv": ("run_id", "step", "trigger")}
# per-side "latest" files: copy when one side lacks it, otherwise leave alone
LATEST_ONLY = {"training_summary.json"}
RUN_ID_RE = re.compile(r"run\d+")

if os.name == "nt":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass


# =============================================================================
# pretty printing
# =============================================================================
def _w(s: str) -> int:
    # emoji take two terminal cells, so plain len() skews every border to their right
    return sum(0 if ord(c) == 0xFE0F else 2 if ord(c) >= 0x1F000 else 1 for c in str(s))


def _pad(s: str, w: int) -> str:
    return str(s) + " " * max(0, w - _w(s))


def _table(title: str, headers, rows, emoji: str = ""):
    if not rows:
        return
    cols = [[str(h)] + [str(r[i]) for r in rows] for i, h in enumerate(headers)]
    wid = [min(60, max(_w(c) for c in col)) for col in cols]
    print(f"  {emoji} {title}".rstrip())
    print("┌" + "┬".join("─" * (w + 2) for w in wid) + "┐")
    print("│ " + " │ ".join(_pad(h, w) for h, w in zip(headers, wid)) + " │")
    print("├" + "┼".join("─" * (w + 2) for w in wid) + "┤")
    for r in rows:
        print("│ " + " │ ".join(_pad(c, w) for c, w in zip(r, wid)) + " │")
    print("└" + "┴".join("─" * (w + 2) for w in wid) + "┘")


# =============================================================================
# modal CLI
# =============================================================================
def _modal(session: int, *args, timeout: int = 900):
    """(returncode, stdout+stderr). `modal` is not on PATH, so go through -m."""
    env = dict(_env(session))
    # modal draws box characters; piped on Windows it dies on cp1252 unless forced
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        r = subprocess.run([sys.executable, "-m", "modal", *args],
                           env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:                       # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


# the modal CLI costs ~2.4s of process startup and auth per invocation, so a 40-file
# sync spent well over a minute doing nothing but spawning interpreters. This runs the
# whole transfer inside ONE subprocess via the Python SDK. Still a subprocess and not an
# import, because the workspace is chosen by MODAL_CONFIG_PATH, which modal reads once
# at import time - we cannot switch workspaces inside this process.
_BATCH_SRC = r"""
import json, os, sys, pathlib
import modal

vol_name, ops = json.loads(sys.stdin.read())
vol = modal.Volume.from_name(vol_name)
done, failed = [], []

puts = [o for o in ops if o[0] == "put"]
if puts:
    try:
        with vol.batch_upload(force=True) as b:
            for _, local, rel in puts:
                b.put_file(local, "/" + rel)
        done += [o[2] for o in puts]
    except Exception as e:
        failed += [[o[2], f"{type(e).__name__}: {e}"] for o in puts]

for kind, rel, dest in [o for o in ops if o[0] == "get"]:
    try:
        d = pathlib.Path(dest)
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_suffix(d.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in vol.read_file(rel):
                fh.write(chunk)
        os.replace(tmp, d)
        done.append(rel)
    except Exception as e:
        failed.append([rel, f"{type(e).__name__}: {e}"])

print("__BATCH__" + json.dumps({"done": done, "failed": failed}))
"""


def batch(session: int, ops: list, vol: str = VOL, timeout: int = 3600):
    """ops: [("put", local, rel)] and [("get", rel, dest)] -> (done, failed).

    One process for the whole list instead of one per file.
    """
    if not ops:
        return [], []
    env = dict(_env(session))
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    payload = json.dumps([vol, [list(o) for o in ops]])
    try:
        r = subprocess.run([sys.executable, "-c", _BATCH_SRC], input=payload, env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], [[o[2] if o[0] == "put" else o[1], "timeout"] for o in ops]
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if line.startswith("__BATCH__"):
            d = json.loads(line[len("__BATCH__"):])
            return d["done"], d["failed"]
    return [], [[o[2] if o[0] == "put" else o[1], out.strip()[-200:] or "no output"]
                for o in ops]


def _json_list(text: str):
    i = text.find("[")
    if i < 0:
        return None
    try:
        return json.loads(text[i:])
    except json.JSONDecodeError:
        return None


def ensure_volume(session: int, vol: str = VOL) -> bool:
    """True if the volume exists (creating it if needed) and the workspace answers."""
    rc, out = _modal(session, "volume", "list", "--json", timeout=180)
    vols = _json_list(out)
    if vols is None:
        return False
    if any(v.get("Name") == vol for v in vols):
        return True
    rc, out = _modal(session, "volume", "create", vol, timeout=180)
    return rc == 0 or "already exists" in out


def remote_ls(session: int, path: str, vol: str = VOL):
    """One directory. [] when it does not exist, None when the call itself failed."""
    rc, out = _modal(session, "volume", "ls", vol, path or "/", "--json", timeout=300)
    rows = _json_list(out)
    if rows is not None:
        return rows
    if "No such file" in out or "not found" in out.lower():
        return []
    return None


def remote_walk(session: int, base: str, vol: str = VOL) -> dict[str, int] | None:
    """{path relative to `base`: size in bytes} for every file under it, recursively."""
    files, stack, failed = {}, [base.strip("/")], False
    while stack:
        cur = stack.pop()
        rows = remote_ls(session, cur, vol)
        if rows is None:
            failed = True
            continue
        for r in rows:
            name = str(r.get("Filename", "")).strip("/")
            leaf = name.rsplit("/", 1)[-1]
            if leaf in SKIP_DIRS or leaf.startswith("."):
                continue
            if str(r.get("Type", "")).lower() == "dir":
                stack.append(name)
            else:
                rel = name[len(base.strip("/")):].strip("/")
                files[rel] = _bytes(r.get("Size", "0"))
    return None if (failed and not files) else files


_SIZE = re.compile(r"([\d.]+)\s*([KMGT]?i?B)", re.I)
_MULT = {"b": 1, "kb": 1e3, "kib": 1024, "mb": 1e6, "mib": 1024**2,
         "gb": 1e9, "gib": 1024**3, "tb": 1e12, "tib": 1024**4}


def _bytes(s) -> int:
    m = _SIZE.search(str(s))
    return int(float(m.group(1)) * _MULT.get(m.group(2).lower(), 1)) if m else 0


def remote_experiments(session: int, vol: str = VOL) -> list[str]:
    rows = remote_ls(session, "/", vol) or []
    return sorted(str(r["Filename"]).strip("/") for r in rows
                  if str(r.get("Type", "")).lower() == "dir")


def remote_run_ids(session: int, exp: str, vol: str = VOL) -> set[str]:
    """runNNN ids already present on the volume — the launcher needs these."""
    ids = set()
    for sub in ("train_config", "debugged_images"):
        for r in remote_ls(session, f"{exp}/{RESULTS}/{sub}", vol) or []:
            m = RUN_ID_RE.match(str(r.get("Filename", "")).rsplit("/", 1)[-1])
            if m:
                ids.add(m.group(0))
    return ids


def push(session: int, local: Path, rel: str, vol: str = VOL, force: bool = False):
    args = ["volume", "put", vol, str(local), "/" + rel] + (["--force"] if force else [])
    return _modal(session, *args, timeout=1800)


def pull(session: int, rel: str, dest: Path, vol: str = VOL):
    """Download one file to exactly `dest`. Never overwrites: writes via a temp dir."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        rc, out = _modal(session, "volume", "get", vol, "/" + rel, str(Path(td) / "f"),
                         timeout=1800)
        got = Path(td) / "f"
        if not got.exists():                      # some versions land the basename instead
            cand = [q for q in Path(td).rglob("*") if q.is_file()]
            got = cand[0] if cand else got
        if rc != 0 or not got.exists():
            return rc or 1, out
        shutil.move(str(got), str(dest))
    return 0, ""


# =============================================================================
# local side
# =============================================================================
def local_walk(base: Path) -> dict[str, int]:
    files = {}
    if not base.is_dir():
        return files
    for d, subs, names in os.walk(base):
        subs[:] = [s for s in subs if s not in SKIP_DIRS and not s.startswith(".")]
        for n in names:
            if n.startswith("."):
                continue
            p = Path(d) / n
            files[p.relative_to(base).as_posix()] = p.stat().st_size
    return files


def local_experiments() -> list[str]:
    return sorted(p.name for p in RUNS_DIR.glob("*")
                  if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith("."))


def atomic_write(df: pd.DataFrame, path: Path):
    """Temp file + replace, so a killed sync can never leave a half-written log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp, path)


# =============================================================================
# CSV merging
# =============================================================================
def read_csv(path: Path):
    """(frame, note). frame=None means unreadable — that side must never be trusted.

    Everything is read as text: a merge should move rows around, never reformat
    the numbers inside them.
    """
    try:
        if not path.exists():
            return pd.DataFrame(), "missing"      # absent is fine; None means CORRUPT
        if path.stat().st_size == 0:
            return pd.DataFrame(), "empty"
        df = pd.read_csv(path, dtype=str, keep_default_na=False, engine="python",
                         on_bad_lines="skip")
        return df, f"{len(df)} rows"
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "header only"
    except Exception as e:                        # noqa: BLE001
        return None, f"CORRUPT ({type(e).__name__})"


def _keycols(df: pd.DataFrame, keys) -> list:
    k = [c for c in keys if c in df.columns]
    return k or list(df.columns)


def canon(df: pd.DataFrame, keys, cols=None) -> pd.DataFrame:
    """Align columns, drop duplicate keys (richest row wins), sort by run then step."""
    df = df.copy()
    if cols is not None:
        cols = ["run_id"] + [c for c in cols if c != "run_id"]   # run_id always leads
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[list(cols)]
    df = df.fillna("")
    if "run_id" in df.columns:
        # pre-run_id logs still have to merge; they all land in one "legacy" bucket
        df["run_id"] = df["run_id"].replace("", "legacy")
    if df.empty:
        return df
    step0 = next((c for c in ("global_step", "step") if c in df.columns), None)
    if step0:
        # a killed process leaves half a line behind; a real row always has a step
        df = df[pd.to_numeric(df[step0], errors="coerce").notna()]
    rank = (df != "").sum(axis=1)
    df = (df.assign(_rank=rank).sort_values("_rank", ascending=False, kind="stable")
            .drop_duplicates(subset=_keycols(df, keys), keep="first")
            .drop(columns="_rank"))
    step = next((c for c in ("global_step", "step") if c in df.columns), None)
    by = [c for c in ("run_id",) if c in df.columns]
    if step:
        df = df.assign(_n=pd.to_numeric(df[step], errors="coerce").fillna(-1))
        by.append("_n")
    if by:
        df = df.sort_values(by, kind="stable")
    return df.drop(columns=[c for c in ("_n",) if c in df.columns]).reset_index(drop=True)


def conflicting_runs(ldf: pd.DataFrame, rdf: pd.DataFrame, keys) -> set[str]:
    """run_ids whose rows genuinely disagree between sides. Should never happen."""
    if ldf.empty or rdf.empty or "run_id" not in ldf.columns or "run_id" not in rdf.columns:
        return set()
    k = [c for c in keys if c in ldf.columns and c in rdf.columns]
    shared = [c for c in ldf.columns if c in rdf.columns]
    if not k:
        return set()
    m = ldf[shared].merge(rdf[shared], on=k, suffixes=("__l", "__r"))
    bad = set()
    for c in shared:
        if c in k:
            continue
        d = m[m[f"{c}__l"].astype(str) != m[f"{c}__r"].astype(str)]
        bad |= set(d["run_id"].astype(str))
    return bad


def merge_csv(session, exp, name, keys, lbase: Path, rbase: str, remote_has: bool,
              direction: str, dry: bool, ws: str, vol: str) -> list:
    """Union both sides on the key columns and write the result back to both."""
    lpath, rel = lbase / name, f"{rbase}/{name}"
    ldf, lnote = read_csv(lpath)
    rdf, rnote = (pd.DataFrame(), "missing")
    tmp_r = None
    if remote_has:
        tmp_r = Path(tempfile.mkdtemp()) / name
        rc, out = pull(session, rel, tmp_r, vol)
        rdf, rnote = read_csv(tmp_r) if rc == 0 else (None, f"download failed ({out[:40]})")

    if ldf is None and rdf is None:
        return [[name, "skip", f"local {lnote} / remote {rnote}"]]
    if ldf is None:                               # local unreadable: park it, remote wins
        print(f"    ⚠️  {name}: local is {lnote} — backed up, the good side wins")
        if not dry and lpath.exists():
            lpath.replace(lpath.with_suffix(f".corrupt-{int(time.time())}.bak"))
        ldf = pd.DataFrame()
    if rdf is None:
        print(f"    ⚠️  {name}: remote is {rnote} — leaving the remote file untouched")
        rdf, remote_broken = pd.DataFrame(), True
    else:
        remote_broken = False

    notes = []
    clash = conflicting_runs(ldf, rdf, keys)
    if clash:
        # same run_id, different rows: never blend them, give the remote copy its own id
        print(f"    🚨 {name}: run ids {sorted(clash)} disagree between sides — "
              f"remote rows re-tagged with -{ws}")
        rdf = rdf.copy()
        rdf["run_id"] = rdf["run_id"].where(~rdf["run_id"].isin(clash),
                                            rdf["run_id"] + f"-{ws}")
        notes.append(f"disambiguated {sorted(clash)}")

    cols = list(ldf.columns) + [c for c in rdf.columns if c not in ldf.columns]
    merged = canon(pd.concat([ldf, rdf], ignore_index=True, sort=False), keys, cols)
    lcan, rcan = canon(ldf, keys, cols), canon(rdf, keys, cols)

    acts = []
    if not merged.equals(lcan) and direction in ("both", "down"):
        acts.append([name, f"⬇ merge  {len(lcan)}→{len(merged)}", ", ".join(notes) or lnote])
        if not dry:
            atomic_write(merged, lpath)
    if (not merged.equals(rcan)) and not remote_broken and direction in ("both", "up"):
        acts.append([name, f"⬆ merge  {len(rcan)}→{len(merged)}", ", ".join(notes) or rnote])
        if not dry:
            out_tmp = Path(tempfile.mkdtemp()) / name
            atomic_write(merged, out_tmp)
            rc, out = push(session, out_tmp, rel, vol, force=True)
            if rc != 0:
                print(f"    ⚠️  upload of {name} failed: {out.strip()[:120]}")
    if not acts:
        acts.append([name, "= in sync", f"{len(merged)} rows"])
    return acts


# =============================================================================
# one experiment
# =============================================================================
def _disambiguate(rel: str, ws: str) -> str:
    """debugged_images/run004_e1_s800.png -> debugged_images/run004-<ws>_e1_s800.png"""
    head, _, leaf = rel.rpartition("/")
    m = RUN_ID_RE.match(leaf)
    leaf = f"{m.group(0)}-{ws}{leaf[m.end():]}" if m else f"{ws}-{leaf}"
    return f"{head}/{leaf}" if head else leaf


def sync_experiment(session: int, exp: str, direction="both", dry=False,
                    vol: str = VOL, ws: str = "") -> dict:
    lbase, rbase = RUNS_DIR / exp / RESULTS, f"{exp}/{RESULTS}"
    remote = remote_walk(session, rbase, vol)
    if remote is None:
        print(f"  ⚠️  {exp}: could not list the volume — skipped")
        return {"exp": exp, "ok": False, "up": 0, "down": 0}
    local = local_walk(lbase)
    if not remote and not local:
        print(f"  · {exp}: nothing on either side")
        return {"exp": exp, "ok": True, "up": 0, "down": 0}

    print(f"\n  📁 {exp}   local {len(local)} file(s) · remote {len(remote)} file(s)")
    rows, up, down = [], 0, 0
    ops = []                       # every transfer, sent as ONE batch at the end

    for name, keys in MERGE_CSVS.items():
        if name in local or name in remote:
            rows += merge_csv(session, exp, name, keys, lbase, rbase, name in remote,
                              direction, dry, ws, vol)

    for rel in sorted(set(local) | set(remote)):
        if rel in MERGE_CSVS:
            continue
        in_l, in_r = rel in local, rel in remote
        if in_l and in_r:
            if local[rel] == remote[rel] or rel in LATEST_ONLY:
                continue                          # same file, or a per-side "latest"
            # namespacing makes this impossible; if it fires, keep both copies
            dest = lbase / _disambiguate(rel, ws)
            print(f"    🚨 {rel}: both sides differ ({local[rel]} vs {remote[rel]} B) — "
                  f"remote copy saved as {dest.relative_to(lbase).as_posix()}")
            rows.append([rel, "🚨 conflict", "remote copy kept under a new name"])
            if not dry and not dest.exists():
                pull(session, f"{rbase}/{rel}", dest, vol)
            continue
        if in_l and direction in ("both", "up"):
            rows.append([rel, "⬆ upload", f"{local[rel]:,} B"])
            up += 1
            ops.append(("put", str(lbase / rel), f"{rbase}/{rel}"))
        elif in_r and direction in ("both", "down"):
            rows.append([rel, "⬇ download", f"{remote[rel]:,} B"])
            down += 1
            ops.append(("get", f"{rbase}/{rel}", str(lbase / rel)))

    if ops and not dry:
        t0 = time.time()
        _done, _failed = batch(session, ops, vol)
        print(f"    ⚡ {len(_done)}/{len(ops)} file(s) in one call, "
              f"{time.time() - t0:.1f}s")
        for _rel, _err in _failed:
            print(f"    ⚠️  {_rel} failed: {_err[:120]}")

    _table(f"{exp} — {'DRY RUN, nothing written' if dry else 'done'}",
           ["file", "action", "detail"], rows, "🔄")
    if not rows:
        print("    ✔ already in sync")
    return {"exp": exp, "ok": True, "up": up, "down": down}


def sync_session(session: int, experiment: str | None = None, direction="both",
                 dry=False, vol: str = VOL) -> list[dict]:
    """Every experiment in one workspace. Never raises — a dead workspace is reported."""
    names = workspaces()
    ws = names.get(session, f"session{session}")
    print(f"\n{'=' * 78}\n☁️  session {session} · {ws} · direction={direction}"
          f"{'  (dry run)' if dry else ''}\n{'=' * 78}")
    try:
        if not ensure_volume(session, vol):
            print(f"  ✖ workspace unreachable (no credit, bad token, or network) — skipped")
            return [{"exp": "-", "ok": False, "up": 0, "down": 0}]
        exps = ([experiment] if experiment
                else sorted(set(remote_experiments(session, vol)) | set(local_experiments())))
        return [sync_experiment(session, e, direction, dry, vol, ws) for e in exps]
    except Exception as e:                        # noqa: BLE001 — one bad workspace, not all
        print(f"  ✖ session {session} failed: {type(e).__name__}: {e}")
        return [{"exp": "-", "ok": False, "up": 0, "down": 0}]


def session_from_env() -> int | None:
    m = re.search(r"modal_session(\d+)", os.environ.get("MODAL_CONFIG_PATH", ""))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description="sync run artifacts between Modal and this PC")
    ap.add_argument("--session", type=int, nargs="+", default=None,
                    help="one or more session numbers")
    ap.add_argument("--all-sessions", action="store_true")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--direction", choices=["both", "down", "up"], default="both")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--volume", default=VOL)
    args = ap.parse_args()

    sessions = (sorted(workspaces()) if args.all_sessions
                else (args.session or [session_from_env()]))
    if sessions == [None]:
        ap.error("give --session N, --all-sessions, or set MODAL_CONFIG_PATH")

    res = []
    for s in sessions:
        res += sync_session(s, args.experiment, args.direction, args.dry_run, args.volume)
    ok = sum(1 for r in res if r["ok"])
    print(f"\n🏁 {ok}/{len(res)} experiment syncs ok · "
          f"⬆ {sum(r['up'] for r in res)} uploaded · ⬇ {sum(r['down'] for r in res)} downloaded")


if __name__ == "__main__":
    main()
