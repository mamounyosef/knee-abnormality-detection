r"""relay_shards.py - move a shard volume between two Modal workspaces, in batches.

WHY. Session 4 holds all 4,407 preprocessed shards (146.6 GB) but is out of credit; session
104 has 28.69$ of free credit but no data, because Kaggle's bulk endpoint has been returning
`404 no gcs url` since 18 Sep. Modal volumes are workspace-scoped and there is no
volume-to-volume copy across workspaces, so the bytes have to pass through this PC.

146.6 GB does not fit in 80 GB of free disk, so this works in BATCHES: pull a batch from the
source workspace, push it to the destination, delete the local copy, repeat. Only one batch
is ever on disk.

RESUMABLE. Progress is written to a state file after every file, so an interrupted run picks
up where it stopped rather than starting over. Files already present at the destination with
the right size are skipped, which also makes a second full pass cheap as a verification.

Download and upload run as SEPARATE PROCESSES, because a Modal client binds to one profile
when it is created and these are two different workspaces.

    python modal_scripts/relay_shards.py plan  --src 4 --dst 104
    python modal_scripts/relay_shards.py run   --src 4 --dst 104 --batch-gb 49
    python modal_scripts/relay_shards.py verify --src 4 --dst 104

Add --limit N to rehearse on a handful of files before committing to the whole move.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "data" / "_relay_stage"
STATE = ROOT / "data" / "_relay_state.json"
VOLUME = "knee-shards"


def cfg_path(session: str) -> str:
    return str(Path.home() / (".modal_session%s.toml" % session))


def child_env(session: str) -> dict:
    """Env for a subprocess pinned to one workspace, with the TLS fix this PC needs."""
    e = dict(os.environ)
    e["MODAL_CONFIG_PATH"] = cfg_path(session)
    ca = str(Path.home() / ".certs" / "ca-bundle.pem")
    if Path(ca).exists():
        e["REQUESTS_CA_BUNDLE"] = ca
        e["SSL_CERT_FILE"] = ca
    return e


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"done": [], "listing": None}


def save_state(st: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st), encoding="utf-8")


# ---------------------------------------------------------------- worker mode
# These two run INSIDE a child process that is already pinned to one workspace.

def _worker_list(volume: str):
    import modal
    vol = modal.Volume.from_name(volume)
    out = []
    for e in vol.iterdir("/", recursive=True):
        size = getattr(e, "size", 0) or 0
        # directories report as type 2 in the SDK; keep files only
        if size or str(e.path).endswith(".npz"):
            out.append({"path": str(e.path).lstrip("/"), "size": int(size)})
    print("__JSON__" + json.dumps(out))


def _worker_pull(volume: str, paths: list, dest: Path):
    import modal
    vol = modal.Volume.from_name(volume)
    dest.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths, 1):
        local = dest / Path(p).name
        if local.exists():
            continue
        tmp = local.with_suffix(local.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in vol.read_file(p):
                fh.write(chunk)
        tmp.rename(local)
        if i % 25 == 0 or i == len(paths):
            print("    pulled %d/%d" % (i, len(paths)), flush=True)


def _worker_push(volume: str, src: Path, remote_prefix: str):
    import modal
    vol = modal.Volume.from_name(volume, create_if_missing=True)
    files = sorted(p for p in src.glob("*.npz"))
    if not files:
        print("    nothing to push")
        return
    with vol.batch_upload(force=True) as b:
        for f in files:
            b.put_file(str(f), "%s/%s" % (remote_prefix.rstrip("/"), f.name)
                       if remote_prefix.strip("/") else "/" + f.name)
    print("    pushed %d files" % len(files), flush=True)


def _worker_sizes(volume: str):
    import modal
    vol = modal.Volume.from_name(volume, create_if_missing=True)
    out = {}
    try:
        for e in vol.iterdir("/", recursive=True):
            s = getattr(e, "size", 0) or 0
            if s:
                out[str(e.path).lstrip("/")] = int(s)
    except Exception:
        pass
    print("__JSON__" + json.dumps(out))


# ---------------------------------------------------------------- driver mode

def run_worker(session: str, fn: str, *args) -> str:
    cmd = [sys.executable, __file__, "_worker", fn, VOLUME] + [str(a) for a in args]
    r = subprocess.run(cmd, env=child_env(session), capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + "\n" + r.stderr[-4000:] + "\n")
        raise SystemExit("worker %s failed on session %s" % (fn, session))
    for line in r.stdout.splitlines():
        if line.startswith("__JSON__"):
            return line[len("__JSON__"):]
        print(line)
    return ""


def cmd_plan(a):
    listing = json.loads(run_worker(a.src, "list"))
    total = sum(x["size"] for x in listing)
    print("  source session %s / %s: %d files, %.2f GB"
          % (a.src, VOLUME, len(listing), total / 1e9))
    have = json.loads(run_worker(a.dst, "sizes") or "{}")
    todo = [x for x in listing if have.get(x["path"]) != x["size"]]
    print("  destination session %s already has %d files; %d still to move (%.2f GB)"
          % (a.dst, len(have), len(todo), sum(x["size"] for x in todo) / 1e9))
    nb = max(1, int(sum(x["size"] for x in todo) / (a.batch_gb * 1e9) + 0.999))
    print("  at --batch-gb %.0f that is %d batches; peak local disk ~%.0f GB"
          % (a.batch_gb, nb, a.batch_gb))
    free = shutil.disk_usage(ROOT.drive + "\\").free / 1e9
    print("  free on %s: %.0f GB  ->  %s"
          % (ROOT.drive, free, "OK" if free > a.batch_gb * 1.2 else "TOO TIGHT, lower --batch-gb"))
    return listing, todo


def cmd_run(a):
    listing, todo = cmd_plan(a)
    if a.limit:
        todo = todo[:a.limit]
        print("\n  --limit %d: rehearsing on %d files only\n" % (a.limit, len(todo)))
    if not todo:
        print("\n  nothing to do - destination already matches source")
        return
    st = load_state()
    done = set(st.get("done", []))
    batch, batch_bytes, n_batch, t0 = [], 0, 0, time.time()

    def flush():
        nonlocal batch, batch_bytes, n_batch
        if not batch:
            return
        n_batch += 1
        print("\n  batch %d: %d files, %.1f GB" % (n_batch, len(batch), batch_bytes / 1e9),
              flush=True)
        if STAGE.exists():
            shutil.rmtree(STAGE)
        STAGE.mkdir(parents=True, exist_ok=True)
        paths_file = STAGE / "_paths.json"
        paths_file.write_text(json.dumps([x["path"] for x in batch]), encoding="utf-8")
        print("    pulling from session %s ..." % a.src, flush=True)
        run_worker(a.src, "pull", str(paths_file), str(STAGE))
        paths_file.unlink(missing_ok=True)
        print("    pushing to session %s ..." % a.dst, flush=True)
        run_worker(a.dst, "push", str(STAGE), a.remote_prefix)
        for x in batch:
            done.add(x["path"])
        st["done"] = sorted(done)
        save_state(st)
        shutil.rmtree(STAGE, ignore_errors=True)
        el = time.time() - t0
        print("    batch %d done, %d/%d files total, %.0f min elapsed"
              % (n_batch, len(done), len(listing), el / 60), flush=True)
        batch, batch_bytes = [], 0

    for x in todo:
        if x["path"] in done:
            continue
        batch.append(x)
        batch_bytes += x["size"]
        if batch_bytes >= a.batch_gb * 1e9:
            flush()
    flush()
    print("\n  relay complete: %d files moved" % len(done))


def cmd_verify(a):
    listing = json.loads(run_worker(a.src, "list"))
    have = json.loads(run_worker(a.dst, "sizes") or "{}")
    missing = [x for x in listing if x["path"] not in have]
    wrong = [x for x in listing if x["path"] in have and have[x["path"]] != x["size"]]
    print("  source %d files / destination %d files" % (len(listing), len(have)))
    print("  missing at destination: %d" % len(missing))
    print("  size mismatch:          %d" % len(wrong))
    for x in (missing + wrong)[:10]:
        print("    %s" % x["path"])
    print("  %s" % ("VERIFIED - destination matches source"
                    if not missing and not wrong else "NOT complete, re-run `run`"))
    return 1 if (missing or wrong) else 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        fn, vol = sys.argv[2], sys.argv[3]
        if fn == "list":
            return _worker_list(vol)
        if fn == "sizes":
            return _worker_sizes(vol)
        if fn == "pull":
            paths = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
            return _worker_pull(vol, paths, Path(sys.argv[5]))
        if fn == "push":
            return _worker_push(vol, Path(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else "")
        raise SystemExit("unknown worker %s" % fn)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "run", "verify"])
    ap.add_argument("--src", required=True, help="source session number, e.g. 4")
    ap.add_argument("--dst", required=True, help="destination session number, e.g. 104")
    ap.add_argument("--batch-gb", type=float, default=49.0,
                    help="GB per batch; only this much is ever on local disk")
    ap.add_argument("--remote-prefix", default="",
                    help="destination path prefix inside the volume (default: root)")
    ap.add_argument("--limit", type=int, default=0,
                    help="rehearse on the first N files only")
    a = ap.parse_args()
    if a.action == "plan":
        cmd_plan(a)
    elif a.action == "run":
        cmd_run(a)
    else:
        raise SystemExit(cmd_verify(a))


if __name__ == "__main__":
    main()
