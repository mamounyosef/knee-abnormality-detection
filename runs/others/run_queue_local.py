"""run_queue_local.py - train several runs on THIS PC one after another, unattended and robustly.

For each run, in order:
  * already finished (in the queue's state file, or its log says it early-stopped / reached the
    last epoch and the process exited cleanly)  -> skip it
  * otherwise launch `python runs/<run>/train.py`. The engine's default `resume: "last"` makes a
    relaunch continue from the newest checkpoint, so a crash costs at most the steps since it.
  * the process exits 0 AND (early stop OR all epochs done) -> mark it finished, go to the next run
  * it crashes, exits 0 without finishing, or prints nothing for STALL_MIN minutes (a hang) ->
    kill its whole process tree, wait, relaunch (resumes). Up to MAX_ATTEMPTS per run.
  * three crashes in a row inside FAST_FAIL_MIN minutes each = a crash that resuming cannot fix
    (a broken checkpoint, a config error): give up on THIS run, record why, go to the next one
  * before every launch: at least MIN_FREE_GB free on the run's drive, else wait and re-check

train.py's output is streamed LIVE to this terminal, exactly as a plain `python train.py` prints
it, and also saved to runs/<run>/results/queue_attempt<N>.log. Queue events are printed between
them and appended to runs/others/run_queue_local.log.
Stopping the queue (Ctrl+C, closing the window, a kill) always stops its train.py too (a Windows
Job Object), and it refuses to start a run that another process is already training.
It only ever starts and stops the processes it launched itself.

    python runs/others/run_queue_local.py convnext_base_25d_18slices_v3_flip_f0 convnext_base_25d_18slices_v3_flip_f1 ...
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
STATE = Path(os.environ.get("QUEUE_STATE", RUNS / "others" / "run_queue_local_state.json"))
EVENTS = Path(os.environ.get("QUEUE_EVENTS", RUNS / "others" / "run_queue_local.log"))
MAX_ATTEMPTS = int(os.environ.get("QUEUE_MAX_ATTEMPTS", 12))
STALL_MIN = float(os.environ.get("QUEUE_STALL_MIN", 45))          # a local validation of ~890 studies takes ~8 min; nothing printed for 45 = a hang
FAST_FAIL_MIN = 5
MIN_FREE_GB = float(os.environ.get("QUEUE_MIN_FREE_GB", 4.0))
WAIT_AFTER_CRASH_S = float(os.environ.get("QUEUE_WAIT_S", 90))  # let the GPU driver release memory


def say(msg):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(EVENTS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)


def finished_by_files(run):
    """True when the run's own logs show it ended: early stop, or the final epoch saved."""
    res = RUNS / run / "results"
    for f in sorted((res / "train_config").glob("*.txt")) + sorted(res.glob("queue_attempt*.log")):
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "early stop:" in txt:
            return "early stop"
    try:
        s = json.loads((res / "training_summary.json").read_text(encoding="utf-8"))
        if int(s.get("epoch") or 0) >= int(s.get("epochs") or 10 ** 9):
            return "all %s epochs" % s.get("epochs")
    except (OSError, ValueError):
        pass
    return None


def kill_tree(p):
    if p.poll() is None:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            pass


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def _kill_on_exit_job():
    """A Windows Job Object that kills every process in it when THIS process ends, however it
    ends (Ctrl+C, closing the window, a kill, a crash). Without it, stopping the queue left
    train.py training on the GPU (1 Oct), and a restarted queue would have trained a second
    copy of the same run on top of it."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount",
                    "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC), ("IoInfo", _IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = _EXT()
    info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError("SetInformationJobObject failed: %d" % ctypes.get_last_error())
    return k32, job


_JOB = _kill_on_exit_job()


def _attach(p):
    if _JOB is None:
        return
    k32, job = _JOB
    import ctypes
    h = k32.OpenProcess(0x1F0FFF, False, p.pid)             # PROCESS_ALL_ACCESS
    ok = k32.AssignProcessToJobObject(job, h)
    k32.CloseHandle(h)
    if not ok:
        say("  warning: could not tie train.py to the queue (error %d); stop it by hand if you stop the queue"
            % ctypes.get_last_error())


def already_training(run):
    """PIDs of any OTHER python process already running this run's train.py."""
    import psutil
    target = str(RUNS / run / "train.py").lower()
    out = []
    for pr in psutil.process_iter(["pid", "cmdline"]):
        try:
            if pr.info["pid"] != os.getpid() and any(target == str(c).lower() for c in (pr.info["cmdline"] or [])):
                out.append(pr.info["pid"])
        except (psutil.Error, TypeError):
            pass
    return out


def one_attempt(run, n):
    """Run train.py once, its output streamed live to this terminal AND to the attempt log.
    Returns (exit code, or None if it was killed as hung; minutes; log path)."""
    import threading
    log = RUNS / run / "results" / f"queue_attempt{n}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    ca = Path.home() / ".certs" / "ca-bundle.pem"
    if ca.exists():
        env.setdefault("REQUESTS_CA_BUNDLE", str(ca))
        env.setdefault("SSL_CERT_FILE", str(ca))
    t0 = time.time()
    last = [time.time()]
    out = sys.stdout.buffer
    with open(log, "wb") as fh:
        p = subprocess.Popen([sys.executable, str(RUNS / run / "train.py")], cwd=str(ROOT),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0)
        _attach(p)

        def pump():
            # raw bytes, so progress lines, emoji and carriage returns arrive exactly as a
            # plain `python train.py` would print them
            while True:
                chunk = p.stdout.read1(65536) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
                if not chunk:
                    break
                fh.write(chunk)
                fh.flush()
                try:
                    out.write(chunk)
                    out.flush()
                except Exception:
                    pass
                last[0] = time.time()

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        last_beat = time.time()
        try:
            while p.poll() is None:
                time.sleep(5)
                if time.time() - last[0] > STALL_MIN * 60:
                    say(f"  {run}: no output for {STALL_MIN} min - killing it as hung")
                    kill_tree(p)
                    th.join(10)
                    return None, (time.time() - t0) / 60, log
                if time.time() - last_beat > 600:          # a queue heartbeat every 10 minutes
                    last_beat = time.time()
                    step = ""
                    try:
                        s = json.loads((RUNS / run / "results" / "training_summary.json").read_text(encoding="utf-8"))
                        step = " | epoch %s/%s, last validated step %s, best %.4f" % (
                            s.get("epoch"), s.get("epochs"), (s.get("last") or {}).get("step"), s.get("best_score") or 0)
                    except Exception:
                        pass
                    with open(EVENTS, "a", encoding="utf-8") as ev:
                        ev.write(f"[{datetime.now():%m-%d %H:%M:%S}]   {run}: attempt {n} running "
                                 f"{(time.time() - t0) / 60:.0f} min{step}\n")
        except KeyboardInterrupt:
            say(f"  Ctrl+C: stopping {run} and the queue")
            kill_tree(p)
            raise
        th.join(30)
    return p.returncode, (time.time() - t0) / 60, log


def main():
    runs = sys.argv[1:]
    if not runs:
        raise SystemExit(__doc__)
    for r in runs:
        if not (RUNS / r / "train.py").exists():
            raise SystemExit(f"no such run: {r}")
    st = load_state()
    say(f"QUEUE START: {len(runs)} run(s): {', '.join(runs)}")
    for run in runs:
        if st.get(run, {}).get("status") == "finished":
            say(f"{run}: already finished ({st[run].get('why')}) - skipping")
            continue
        why = finished_by_files(run)
        if why:
            st[run] = {"status": "finished", "why": why + " (found on start)"}
            save_state(st)
            say(f"{run}: its logs say it already ended ({why}) - skipping")
            continue
        fast_fails, attempt = 0, 0
        while True:
            attempt += 1
            if attempt > MAX_ATTEMPTS:
                st[run] = {"status": "gave up", "why": f"{MAX_ATTEMPTS} attempts"}
                save_state(st)
                say(f"{run}: GAVE UP after {MAX_ATTEMPTS} attempts - moving on")
                break
            busy = already_training(run)
            if busy:
                st[run] = {"status": "gave up", "why": f"already being trained by PID {busy}"}
                save_state(st)
                say(f"{run}: another process (PID {busy}) is ALREADY training it - not starting a second copy. "
                    f"Stop that one first. Moving on")
                break
            while free_gb(RUNS / run) < MIN_FREE_GB:
                say(f"{run}: only {free_gb(RUNS / run):.1f} GB free on its drive (need {MIN_FREE_GB}) - waiting 10 min")
                time.sleep(600)
            say(f"{run}: attempt {attempt} starting ({'fresh' if attempt == 1 and not (RUNS / run / 'results' / 'checkpoints').exists() else 'resumes from its last checkpoint'})")
            code, mins, log = one_attempt(run, attempt)
            why = finished_by_files(run)
            if code == 0 and why:
                st[run] = {"status": "finished", "why": why, "attempts": attempt}
                save_state(st)
                say(f"{run}: FINISHED ({why}) after {attempt} attempt(s), last one {mins:.0f} min")
                break
            tail = ""
            try:
                tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1][:160]
            except Exception:
                pass
            kind = "hung" if code is None else ("exited 0 without finishing" if code == 0 else f"crashed (exit {code})")
            say(f"{run}: attempt {attempt} {kind} after {mins:.0f} min | last line: {tail}")
            fast_fails = fast_fails + 1 if mins < FAST_FAIL_MIN else 0
            if fast_fails >= 3:
                st[run] = {"status": "gave up", "why": f"3 fast crashes in a row; see {log.name}"}
                save_state(st)
                say(f"{run}: GAVE UP - 3 crashes within {FAST_FAIL_MIN} min each; resuming cannot fix it. Moving on")
                break
            time.sleep(WAIT_AFTER_CRASH_S)
    say("QUEUE DONE: " + ", ".join(f"{r}={st.get(r, {}).get('status', '?')}" for r in runs))


if __name__ == "__main__":
    main()
