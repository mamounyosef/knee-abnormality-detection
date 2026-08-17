"""
pipeline.py — run and watch the data pipeline across every Modal workspace.
===========================================================================

Modal volumes are per-workspace with no cross-workspace copy, so each workspace
that will train has to build its own data:

    1. download   competition zip -> `knee-data`     (~1.5 h, ~247 GiB)
    2. preprocess zip -> per-study shards on `knee-shards` (~14 min, ~137 GiB)

Progress is read with `modal volume ls --json`, which is a control-plane call:
no container is started, so polling costs nothing. (An earlier verification pass
that opened every shard inside a container cost $0.60 and told us little more
than a file count would have.)

Usage
-----
    python modal_scripts/pipeline.py                       # one-shot status table
    python modal_scripts/pipeline.py --watch               # live, refreshes
    python modal_scripts/pipeline.py --launch 4 5 6        # start those sessions
    python modal_scripts/pipeline.py --launch-all --watch  # start everything, then watch

Writes data/stats/pipeline_status.md on every refresh, so the table is also
readable from the editor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_TXT = ROOT / "modal_setup_commands.txt"
STATUS_MD = ROOT / "data" / "stats" / "pipeline_status.md"

DATA_VOL = "knee-data"
SHARD_VOL = "knee-shards"
EXPECTED_STUDIES = 4407
EXPECTED_ZIP_GIB = 246.8

CA_BUNDLE = Path.home() / ".certs" / "ca-bundle.pem"


# -----------------------------------------------------------------------------
# workspace map
# -----------------------------------------------------------------------------
def workspaces() -> dict[int, str]:
    """session number -> workspace name, parsed from modal_setup_commands.txt."""
    out, name = {}, None
    for line in SETUP_TXT.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("#"):
            name = line.lstrip("# ").rstrip(":").strip()
        m = re.search(r"\.modal_session(\d+)\.toml", line)
        if m and name:
            out[int(m.group(1))] = name
    return dict(sorted(out.items()))


def _env(session: int) -> dict:
    env = dict(os.environ)
    env["MODAL_CONFIG_PATH"] = str(Path.home() / f".modal_session{session}.toml")
    if CA_BUNDLE.exists():                       # Avast intercepts TLS on this PC
        env["REQUESTS_CA_BUNDLE"] = str(CA_BUNDLE)
        env["SSL_CERT_FILE"] = str(CA_BUNDLE)
    return env


def _modal(session: int, *args, timeout=180):
    try:
        r = subprocess.run([sys.executable, "-m", "modal", *args],
                           env=_env(session), capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout
    except Exception:                            # noqa: BLE001
        return ""


def _json_or_none(text: str):
    i = text.find("[")
    if i < 0:
        return None
    try:
        return json.loads(text[i:])
    except json.JSONDecodeError:
        return None


_SIZE = re.compile(r"([\d.]+)\s*([KMGT]?i?B)", re.I)
_MULT = {"b": 1, "kb": 1e3, "kib": 1024, "mb": 1e6, "mib": 1024**2,
         "gb": 1e9, "gib": 1024**3, "tb": 1e12, "tib": 1024**4}


def _to_gib(s: str) -> float:
    m = _SIZE.search(str(s))
    if not m:
        return 0.0
    return float(m.group(1)) * _MULT.get(m.group(2).lower(), 1) / 1024**3


# -----------------------------------------------------------------------------
# status
# -----------------------------------------------------------------------------
def status(session: int) -> dict:
    st = {"session": session, "zip_gib": 0.0, "shards": 0, "shard_gib": 0.0,
          "state": "no volumes", "reachable": False}

    vols = _json_or_none(_modal(session, "volume", "list", "--json"))
    if vols is None:
        st["state"] = "unreachable"
        return st
    st["reachable"] = True
    names = {v.get("Name") for v in vols}

    if DATA_VOL in names:
        rows = _json_or_none(_modal(session, "volume", "ls", DATA_VOL, "--json")) or []
        for r in rows:
            if str(r.get("Filename", "")).endswith(".zip"):
                st["zip_gib"] = _to_gib(r.get("Size", "0"))

    if SHARD_VOL in names:
        rows = _json_or_none(_modal(session, "volume", "ls", SHARD_VOL, "--json",
                                    timeout=300)) or []
        npz = [r for r in rows if str(r.get("Filename", "")).endswith(".npz")]
        st["shards"] = len(npz)
        st["shard_gib"] = sum(_to_gib(r.get("Size", "0")) for r in npz)

    if st["shards"] >= EXPECTED_STUDIES:
        st["state"] = "READY"
    elif st["shards"] > 0:
        st["state"] = "preprocessing"
    elif st["zip_gib"] >= EXPECTED_ZIP_GIB - 1:
        st["state"] = "zip ready"
    elif st["zip_gib"] > 0:
        st["state"] = "downloading"
    else:
        st["state"] = "empty"
    return st


def render(rows: list[dict], names: dict[int, str]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = [f"# Modal pipeline status — {now}", "",
           "| # | workspace | zip (GiB) | shards | shard GiB | progress | state |",
           "|---|---|---|---|---|---|---|"]
    ready = 0
    for s in rows:
        name = names.get(s["session"], "?")
        if s["state"] == "READY":
            ready += 1
            bar = "█" * 20
            pct = "100%"
        elif s["shards"]:
            f = s["shards"] / EXPECTED_STUDIES
            bar = "█" * int(f * 20) + "·" * (20 - int(f * 20))
            pct = f"{f:.0%}"
        elif s["zip_gib"]:
            f = min(s["zip_gib"] / EXPECTED_ZIP_GIB, 1.0)
            bar = "▒" * int(f * 20) + "·" * (20 - int(f * 20))
            pct = f"{f:.0%} zip"
        else:
            bar = "·" * 20
            pct = "—"
        out.append(f"| {s['session']} | {name} | {s['zip_gib']:.1f} | "
                   f"{s['shards']} | {s['shard_gib']:.1f} | `{bar}` {pct} | {s['state']} |")
    total_gib = sum(s["zip_gib"] + s["shard_gib"] for s in rows)
    out += ["", f"**{ready}/{len(rows)} workspaces ready** · "
                f"total storage across workspaces: {total_gib:.0f} GiB",
            "", "zip ▒ · shards █ · target 4,407 studies (~137 GiB) per workspace"]
    return "\n".join(out)


# -----------------------------------------------------------------------------
# launching
# -----------------------------------------------------------------------------
def launch(session: int, st: dict) -> str:
    """Start whichever stage this workspace still needs, detached."""
    if st["state"] == "READY":
        return "already ready"
    script = ("modal_scripts/modal_preprocess.py"
              if st["zip_gib"] >= EXPECTED_ZIP_GIB - 1
              else "modal_scripts/modal_download.py")
    stage = "preprocess" if script.endswith("preprocess.py") else "download"
    subprocess.Popen([sys.executable, "-m", "modal", "run", "--detach", script],
                     env=_env(session), cwd=str(ROOT),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"launched {stage}"


def delete_zip(session: int) -> str:
    """Free the ~247 GiB archive once shards exist.

    Safe: the shards keep every series and every slice, and workspace 1 retains a
    full copy of the zip as the archive. Re-downloading is ~1.5 h unattended.
    """
    out = _modal(session, "run", "modal_scripts/modal_verify_shards.py::delete_archive",
                 "--confirm", timeout=1800)
    return "zip deleted" if "removed" in out else "delete failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, nargs="*", default=None,
                    help="which sessions to show (default: all in the setup file)")
    ap.add_argument("--launch", type=int, nargs="*", default=None,
                    help="start the next needed stage for these sessions")
    ap.add_argument("--launch-all", action="store_true")
    ap.add_argument("--watch", action="store_true", help="refresh until all are ready")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--auto-delete-zip", action="store_true",
                    help="delete the 247 GiB zip once a workspace reaches 4,407 shards")
    ap.add_argument("--keep-zip", type=int, nargs="*", default=[1],
                    help="sessions that keep their zip as the archive (default: 1)")
    ap.add_argument("--batch", type=int, default=0,
                    help="with --launch-all: only run this many workspaces at a time")
    args = ap.parse_args()

    names = workspaces()
    sessions = args.sessions or sorted(names)

    pending = []
    if args.launch is not None or args.launch_all:
        pending = sorted(names) if args.launch_all else (args.launch or [])

    while True:
        rows = []
        for s in sessions:
            print(f"  polling session {s} ...", end="\r", flush=True)
            rows.append(status(s))
        by_session = {r["session"]: r for r in rows}

        # keep `--batch` workspaces in flight at once: Kaggle throttles many
        # simultaneous downloads from one account
        if pending:
            active = sum(1 for r in rows
                         if r["state"] in ("downloading", "preprocessing"))
            cap = args.batch or len(pending)
            while pending and active < cap:
                s = pending.pop(0)
                st = by_session.get(s) or status(s)
                if st["state"] == "READY":
                    continue
                print(f"[{s}] {names.get(s,'?')}: {launch(s, st)}")
                active += 1

        if args.auto_delete_zip:
            for r in rows:
                if (r["state"] == "READY" and r["zip_gib"] > 1
                        and r["session"] not in args.keep_zip):
                    print(f"[{r['session']}] {delete_zip(r['session'])}")

        table = render(rows, names)

        STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
        STATUS_MD.write_text(table + "\n", encoding="utf-8")

        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(table)
        print(f"\nwritten: {STATUS_MD}")

        if not args.watch:
            break
        if all(r["state"] == "READY" for r in rows) and not pending:
            print("\nall workspaces ready")
            break
        print(f"refreshing in {args.interval}s — Ctrl+C to stop")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
