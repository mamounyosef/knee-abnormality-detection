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

# Windows consoles default to a codepage that mangles the box-drawing characters
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        subprocess.run(["chcp", "65001"], shell=True, capture_output=True)
    except Exception:                            # noqa: BLE001 — cosmetic only
        pass


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
        # modal prints box-drawing/progress characters; the Windows default
        # cp1252 decode raises inside subprocess's reader thread and leaves
        # stdout as None, so decode explicitly and never return None
        r = subprocess.run([sys.executable, "-m", "modal", *args],
                           env=_env(session), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return r.stdout or ""
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
def live_apps(session: int) -> int:
    """Running modal apps in this workspace.

    The volume tells you what has landed, never whether anything is still
    running: a detached job can die and leave its partial zip behind, which
    reads exactly like a download in progress. This is the liveness signal.
    """
    apps = _json_or_none(_modal(session, "app", "list", "--json"))
    if not apps:
        return 0
    return sum(1 for a in apps
               if "stopped" not in str(a.get("State", "")).lower())


def status(session: int, check_apps: bool = True) -> dict:
    st = {"session": session, "zip_gib": 0.0, "shards": 0, "shard_gib": 0.0,
          "state": "no volumes", "reachable": False, "apps": 0}

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

    # each app-list call spawns a modal CLI (~1 min), so only ask where the
    # answer can change a decision: a workspace holding no data that this run
    # never launched has nothing running by definition
    if (check_apps or st["zip_gib"] or st["shards"]) and st["shards"] < EXPECTED_STUDIES:
        st["apps"] = live_apps(session)

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


STATE_ICON = {"READY": "✔", "preprocessing": "▶", "downloading": "▼",
              "zip ready": "◆", "empty": "·", "unreachable": "✖",
              "no volumes": "·"}


def _cells(rows: list[dict], names: dict[int, str]) -> list[list[str]]:
    body = []
    for s in rows:
        if s["state"] == "READY":
            frac, mark = 1.0, "█"
        elif s["shards"]:
            frac, mark = s["shards"] / EXPECTED_STUDIES, "█"
        elif s["zip_gib"]:
            frac, mark = min(s["zip_gib"] / EXPECTED_ZIP_GIB, 1.0), "▒"
        else:
            frac, mark = 0.0, " "
        filled = int(round(frac * 18))
        bar = mark * filled + "·" * (18 - filled)
        stage = ("shards" if s["shards"] else "zip" if s["zip_gib"] else "—")
        body.append([
            str(s["session"]),
            names.get(s["session"], "?"),
            f"{STATE_ICON.get(s['state'], '?')} {s['state']}",
            f"{s['zip_gib']:6.1f}" if s["zip_gib"] else "     —",
            f"{s['shards']:>5}/{EXPECTED_STUDIES}" if s["shards"] else "        —",
            f"{s['shard_gib']:6.1f}" if s["shard_gib"] else "     —",
            f"{bar} {frac:>4.0%} {stage}",
        ])
    return body


HEADERS = ["#", "workspace", "state", "zip GiB", "shards", "shard GiB", "progress"]


def render_box(rows: list[dict], names: dict[int, str]) -> str:
    """A closed, aligned table for the terminal."""
    body = _cells(rows, names)
    widths = [max(len(str(r[i])) for r in [HEADERS] + body) for i in range(len(HEADERS))]

    def line(l, m, r):
        return l + m.join("─" * (w + 2) for w in widths) + r

    def row(cells, sep="│"):
        return sep + sep.join(f" {c:<{w}} " for c, w in zip(cells, widths)) + sep

    ready = sum(1 for s in rows if s["state"] == "READY")
    total_gib = sum(s["zip_gib"] + s["shard_gib"] for s in rows)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out = [line("┌", "┬", "┐"), row(HEADERS), line("├", "┼", "┤")]
    out += [row(c) for c in body]
    out.append(line("└", "┴", "┘"))
    out.append("")
    out.append(f"  {ready}/{len(rows)} workspaces ready   ·   "
               f"{total_gib:,.0f} GiB stored   ·   {now}")
    out.append("  ✔ ready   ▶ preprocessing   ▼ downloading   ◆ zip ready   · empty")
    return "\n".join(out)


def render_md(rows: list[dict], names: dict[int, str]) -> str:
    """Markdown version for data/stats/pipeline_status.md."""
    body = _cells(rows, names)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ready = sum(1 for s in rows if s["state"] == "READY")
    total_gib = sum(s["zip_gib"] + s["shard_gib"] for s in rows)
    out = [f"# Modal pipeline status", "", f"_{now}_", "",
           "```", render_box(rows, names), "```", "",
           f"**{ready}/{len(rows)} ready** · {total_gib:,.0f} GiB stored · "
           f"target 4,407 studies (~137 GiB) per workspace"]
    return "\n".join(out)


# -----------------------------------------------------------------------------
# launching
# -----------------------------------------------------------------------------
LAUNCH_LOG = ROOT / "data" / "stats" / "pipeline_launches"


def launch(session: int, st: dict) -> str:
    """Start whichever stage this workspace still needs, detached."""
    if st["state"] == "READY":
        return "already ready"
    script = ("modal_scripts/modal_preprocess.py"
              if st["zip_gib"] >= EXPECTED_ZIP_GIB - 1
              else "modal_scripts/modal_download.py")
    stage = "preprocess" if script.endswith("preprocess.py") else "download"
    # keep the child's output: a launch that dies instantly (no credit, transient
    # modal error) is otherwise invisible and silently burns a batch slot
    LAUNCH_LOG.mkdir(parents=True, exist_ok=True)
    log = open(LAUNCH_LOG / f"session{session}_{stage}.log", "ab")
    subprocess.Popen([sys.executable, "-m", "modal", "run", "--detach", script],
                     env=_env(session), cwd=str(ROOT),
                     stdout=log, stderr=subprocess.STDOUT)
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
    ap.add_argument("--skip", type=int, nargs="*", default=[1],
                    help="sessions never launched (default: 1, out of credit)")
    ap.add_argument("--stall", type=int, default=1200,
                    help="seconds without new bytes/shards before a workspace is "
                         "treated as dead and relaunched (both stages resume)")
    ap.add_argument("--warmup", type=int, default=900,
                    help="seconds a just-launched session counts as active before "
                         "its first bytes land on the volume")
    args = ap.parse_args()

    names = workspaces()
    sessions = args.sessions or sorted(names)

    pending = []
    if args.launch is not None or args.launch_all:
        pending = sorted(names) if args.launch_all else (args.launch or [])
    pending = [s for s in pending if s not in args.skip]
    if args.skip:
        print(f"skipping sessions {sorted(args.skip)} (not launched)")

    launched_at: dict[int, float] = {}   # session -> monotonic time of launch
    progress: dict[int, tuple[float, float]] = {}   # session -> (metric, t_moved)

    while True:
        rows = []
        for s in sessions:
            print(f"  polling session {s} ...", end="\r", flush=True)
            rows.append(status(s, check_apps=s in launched_at))
        by_session = {r["session"]: r for r in rows}

        # keep `--batch` workspaces in flight at once: Kaggle throttles many
        # simultaneous downloads from one account
        if pending:
            now = time.monotonic()

            # bytes/shards on the volume are the only honest liveness signal: a
            # detached modal job can die without warning and leaves its partial
            # data behind, so "has data" never means "is running" — only "moved
            # since the last poll" does.
            for r in rows:
                metric = r["shards"] or r["zip_gib"]
                prev = progress.get(r["session"])
                if prev is None or metric > prev[0]:
                    progress[r["session"]] = (metric, now)

            def moving(s: int) -> bool:
                m, t = progress.get(s, (0.0, now))
                return m > 0 and now - t < args.stall

            alive = {r["session"] for r in rows if r["apps"] > 0}
            # a session launched moments ago has no app registered yet — hold
            # its slot for --warmup seconds so the cap is not overshot
            warming = {s for s, t in launched_at.items()
                       if now - t < args.warmup and s not in alive}
            working = (alive & set(pending)) | warming
            active = len(working)

            for s in alive & set(pending):
                if not moving(s):
                    print(f"[{s}] {names.get(s,'?')}: app alive but no new data "
                          f"for {args.stall // 60} min")

            cap = args.batch or len(pending)
            for s in list(pending):
                if active >= cap:
                    break
                st = by_session.get(s) or status(s)
                if st["state"] == "READY":
                    pending.remove(s)
                    continue
                if s in working:
                    continue          # genuinely in flight
                # partial data but no live app: the job died, resume it
                note = (" (resuming — no live app)"
                        if st["state"] in ("downloading", "preprocessing") else "")
                print(f"[{s}] {names.get(s,'?')}: {launch(s, st)}{note}")
                launched_at[s] = now
                progress[s] = (st["shards"] or st["zip_gib"], now)
                active += 1

        if args.auto_delete_zip:
            for r in rows:
                if (r["state"] == "READY" and r["zip_gib"] > 1
                        and r["session"] not in args.keep_zip):
                    print(f"[{r['session']}] {delete_zip(r['session'])}")

        STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
        STATUS_MD.write_text(render_md(rows, names) + "\n", encoding="utf-8")

        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(render_box(rows, names))
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
