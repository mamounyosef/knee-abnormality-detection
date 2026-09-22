"""find_run_across_workspaces.py - which workspace holds this run's checkpoints?

A run that crossed a workspace hand-off leaves its peak checkpoint on whichever volume it
finished on, while the local copy can be an earlier, worse one. That is not cosmetic: every
submission carrying convnext_large shipped a model 0.028 below the score it was chosen for,
because best.pt locally was step 1200 while the run actually peaked at step 2600 elsewhere.
coatnet_rmlp2_384 is in the same state now - local best.pt scores 0.896367 where its own
val_log peaked at 0.8974.

The run's train_config only records "modal-container", so the workspace is not recoverable
from the logs and the volumes have to be asked directly.

    python modal_scripts/find_run_across_workspaces.py coatnet_rmlp2_384
    python modal_scripts/find_run_across_workspaces.py coatnet_rmlp2_384 --sessions 1-23,102-110
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HOME = Path.home()


def sessions_available():
    out = []
    for p in HOME.glob(".modal_session*.toml"):
        m = re.search(r"\.modal_session(\d+)\.toml$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def parse_spec(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def modal(session, *args, timeout=180):
    env = dict(os.environ)
    env["MODAL_CONFIG_PATH"] = str(HOME / (".modal_session%d.toml" % session))
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    ca = HOME / ".certs" / "ca-bundle.pem"
    if ca.exists():
        env["REQUESTS_CA_BUNDLE"] = env["SSL_CERT_FILE"] = str(ca)
    try:
        r = subprocess.run([sys.executable, "-m", "modal", *args], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    return (r.stdout or "") + (r.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sessions", default=None,
                    help="e.g. 1-23,102-110; default is every local session toml")
    ap.add_argument("--volume", default="knee-runs")
    args = ap.parse_args()

    ses = parse_spec(args.sessions) if args.sessions else sessions_available()
    ses = [s for s in ses if (HOME / (".modal_session%d.toml" % s)).exists()]
    print("scanning %d workspace(s) for %s/results/checkpoints\n" % (len(ses), args.run))

    hits = []
    for s in ses:
        out = modal(s, "volume", "ls", args.volume,
                    "%s/results/checkpoints" % args.run)
        files = [l.strip() for l in out.splitlines()
                 if ".pt" in l or "top_k.json" in l]
        if not files:
            print("  session%-4s -" % s)
            continue
        steps = sorted({int(m.group(1)) for f in files
                        for m in [re.search(r"step(\d+)\.pt", f)] if m})
        print("  session%-4s %d file(s), steps %s%s"
              % (s, len(files), steps[:4],
                 " .. %d" % steps[-1] if len(steps) > 4 else ""))
        hits.append((s, steps, files))

    if not hits:
        print("\nnot found on any scanned workspace")
        return 1
    print("\nfound on: %s" % ", ".join("session%d" % s for s, _, _ in hits))
    best = max(hits, key=lambda h: (max(h[1]) if h[1] else 0))
    print("highest step is on session%d (step %s)"
          % (best[0], max(best[1]) if best[1] else "?"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
