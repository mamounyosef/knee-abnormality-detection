r"""relay_fanout.py - seed SEVERAL Modal workspaces from one source, downloading once.

relay_shards.py moves a volume from one workspace to ONE other: pull a batch here, push it
there, delete, repeat. Running it five times to seed five workspaces would pull the same
146.6 GB five times, which is 733 GB of download for 733 GB of upload.

This pulls each batch ONCE and fans it out to every destination, so the download is paid a
single time. Uploads run concurrently but capped, because saturating the uplink with five
parallel pushes makes the machine unusable and does not go faster.

    python modal_scripts/relay_fanout.py plan   --src 4 --dst 106 107 108 109 110
    python modal_scripts/relay_fanout.py run    --src 4 --dst 106 107 108 109 110 --fanout 2
    python modal_scripts/relay_fanout.py verify --src 4 --dst 106 107 108 109 110

RESUMABLE, per destination. State is keyed "<src>-><dst>" exactly as relay_shards.py keys
it, because a global done-list is what once made a relay report "complete" while moving
nothing. A destination that is behind simply receives the batches it is missing.

VERIFIED, per destination. `run` ends by asking every destination what it actually holds and
exits non-zero if any of them is short. Bookkeeping is never the completion signal.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relay_shards as rs  # noqa: E402

ROOT = rs.ROOT
STAGE = ROOT / "data" / "_relay_stage"
VOLUME = rs.VOLUME


def worker_cmd(fn, *args):
    return [sys.executable, str(Path(rs.__file__)), "_worker", fn, VOLUME] + [str(a) for a in args]


def start_push(session, stage, remote_prefix):
    """Non-blocking push. run_worker() blocks, and the whole point here is concurrency."""
    return subprocess.Popen(worker_cmd("push", str(stage), remote_prefix),
                            env=rs.child_env(session), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def sizes(session):
    return json.loads(rs.run_worker(session, "sizes") or "{}")


def cmd_plan(a):
    listing = json.loads(rs.run_worker(a.src, "list"))
    total = sum(x["size"] for x in listing)
    print("  source session %s / %s: %d files, %.2f GB"
          % (a.src, VOLUME, len(listing), total / 1e9))
    todo_by_dst, union = {}, {}
    for d in a.dst:
        have = sizes(d)
        todo = [x for x in listing if have.get(x["path"]) != x["size"]]
        todo_by_dst[d] = {x["path"] for x in todo}
        for x in todo:
            union[x["path"]] = x
        print("  destination %-5s has %5d files, needs %5d (%.2f GB)"
              % (d, len(have), len(todo), sum(x["size"] for x in todo) / 1e9))
    todo = sorted(union.values(), key=lambda x: x["path"])
    gb = sum(x["size"] for x in todo) / 1e9
    nb = max(1, int(gb / a.batch_gb + 0.999))
    print("\n  files needed by at least one destination: %d (%.2f GB)" % (len(todo), gb))
    print("  download is paid ONCE: %.2f GB down, %.2f GB up across %d destinations"
          % (gb, gb * len(a.dst), len(a.dst)))
    print("  at --batch-gb %.0f that is %d batches; peak local disk ~%.0f GB"
          % (a.batch_gb, nb, a.batch_gb))
    free = shutil.disk_usage(ROOT.drive + "\\").free / 1e9
    print("  free on %s: %.0f GB  ->  %s"
          % (ROOT.drive, free,
             "OK" if free > a.batch_gb * 1.3 else "TOO TIGHT, lower --batch-gb"))
    return listing, todo, todo_by_dst


def cmd_run(a):
    listing, todo, todo_by_dst = cmd_plan(a)
    if a.limit:
        todo = todo[:a.limit]
        print("\n  --limit %d: rehearsing on %d files only\n" % (a.limit, len(todo)))
    if not todo:
        print("\n  nothing to do - every destination already matches the source")
        return 0

    state = {d: rs.load_state(a.src, d) for d in a.dst}
    done = {d: set(state[d].get("done", [])) for d in a.dst}
    batch, batch_bytes, n_batch, t0 = [], 0, 0, time.time()

    def flush():
        nonlocal batch, batch_bytes, n_batch
        if not batch:
            return
        n_batch += 1
        names = {x["path"] for x in batch}
        need = [d for d in a.dst if (todo_by_dst[d] & names) - done[d]]
        if not need:
            batch, batch_bytes = [], 0
            return
        print("\n  batch %d: %d files, %.1f GB  ->  destinations %s"
              % (n_batch, len(batch), batch_bytes / 1e9, ", ".join(need)), flush=True)

        if STAGE.exists():
            shutil.rmtree(STAGE)
        STAGE.mkdir(parents=True, exist_ok=True)
        pf = STAGE / "_paths.json"
        pf.write_text(json.dumps([x["path"] for x in batch]), encoding="utf-8")
        print("    pulling once from session %s ..." % a.src, flush=True)
        rs.run_worker(a.src, "pull", str(pf), str(STAGE))
        pf.unlink(missing_ok=True)

        # push to at most --fanout destinations at a time
        queue, running, failed = list(need), {}, []
        while queue or running:
            while queue and len(running) < a.fanout:
                d = queue.pop(0)
                print("    pushing to session %s ..." % d, flush=True)
                running[d] = start_push(d, STAGE, a.remote_prefix)
            for d, p in list(running.items()):
                if p.poll() is None:
                    continue
                out = p.stdout.read() if p.stdout else ""
                if p.returncode != 0:
                    failed.append(d)
                    sys.stderr.write("    push to %s FAILED:\n%s\n" % (d, out[-2000:]))
                else:
                    print("    session %s: %s" % (d, out.strip().splitlines()[-1]
                                                  if out.strip() else "pushed"), flush=True)
                    done[d] |= names
                    state[d]["done"] = sorted(done[d])
                    rs.save_state(a.src, d, state[d])
                del running[d]
            time.sleep(1)
        shutil.rmtree(STAGE, ignore_errors=True)
        if failed:
            raise SystemExit("  push failed for %s - fix and re-run `run`; other "
                             "destinations kept their progress" % ", ".join(failed))
        el = time.time() - t0
        print("    batch %d done, %.0f min elapsed, per-destination totals: %s"
              % (n_batch, el / 60,
                 ", ".join("%s:%d" % (d, len(done[d])) for d in a.dst)), flush=True)
        batch, batch_bytes = [], 0

    for x in todo:
        batch.append(x)
        batch_bytes += x["size"]
        if batch_bytes >= a.batch_gb * 1e9:
            flush()
    flush()

    if a.limit:
        print("\n  rehearsal done")
        return 0
    print("\n  confirming against every destination ...")
    return cmd_verify(a)


def cmd_verify(a):
    listing = json.loads(rs.run_worker(a.src, "list"))
    bad = 0
    for d in a.dst:
        have = sizes(d)
        missing = [x for x in listing if x["path"] not in have]
        wrong = [x for x in listing if x["path"] in have and have[x["path"]] != x["size"]]
        ok = not missing and not wrong
        bad += 0 if ok else 1
        print("  session %-5s %5d/%d files  missing %4d  wrong size %4d  %s"
              % (d, len(have), len(listing), len(missing), len(wrong),
                 "VERIFIED" if ok else "NOT COMPLETE"))
    if bad:
        print("  %d destination(s) incomplete - re-run `run`" % bad)
    else:
        print("  ALL %d destinations match the source" % len(a.dst))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "run", "verify"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True, nargs="+",
                    help="one or more destination session numbers")
    ap.add_argument("--batch-gb", type=float, default=40.0)
    ap.add_argument("--fanout", type=int, default=2,
                    help="how many uploads run at once (default 2; more saturates the "
                         "uplink and makes the machine unusable without going faster)")
    ap.add_argument("--remote-prefix", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.src in a.dst:
        raise SystemExit("source and destination must differ")
    if a.action == "plan":
        cmd_plan(a)
        return 0
    return cmd_run(a) if a.action == "run" else cmd_verify(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
