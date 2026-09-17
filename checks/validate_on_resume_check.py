r"""validate_on_resume_check.py - does the one-off resume validation fire, once, and only
when asked?

The validation cadence is a modulo on the global step (`gstep % ev == 0`). A rolling
checkpoint is saved on its own schedule, so it routinely lands BETWEEN two validations:
dinov3_mcl_slice stopped at step 1100 with its last validation at 1000. Resuming there,
the next score would not arrive until step 1200, and if the cadence changed in the
meantime the gap is against a schedule that no longer applies.

`evaluation.validate_on_resume` closes that: score once on the first step back, then behave
exactly as before. The risks worth testing are that it silently does nothing, that it
re-arms and validates every step, or that it turns itself on for runs that never asked.

No GPU, no data, no Modal.

    python checks/validate_on_resume_check.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


def simulate(cfg, resumed, first_step, last_step, epoch):
    """Replay the loop's trigger logic over a step range, returning the steps it scores."""
    resume_eval = [bool(resumed) and bool(cfg["evaluation"].get("validate_on_resume"))]
    hits = []
    for gstep in range(first_step, last_step + 1):
        ev = sc.eval_interval(cfg, epoch)
        due = bool(ev and gstep % ev == 0)
        if due or resume_eval[0]:
            resume_eval[0] = False
            hits.append((gstep, "cadence" if due else "resume"))
    return hits


# --- 1. the engine really contains this logic, not just this test -------------------
print("\n1. the trigger lives in the engine")
src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
for label, needle in [
        ("the flag is read from the config", 'cfg["evaluation"].get("validate_on_resume")'),
        ("it only arms when a checkpoint was loaded", "bool(ck) and bool("),
        ("the loop checks it alongside the cadence", "if due or resume_eval[0]:"),
        ("it disarms immediately", "resume_eval[0] = False"),
        ("the trigger is labelled in the log", '"first step after resume"')]:
    check(label, needle in src)

# --- 2. the two MCL arms ask for it, and nothing else does --------------------------
print("\n2. who has it switched on")
for a in ("dinov3_mcl_slice", "dinov3_mcl_fresh"):
    on = sc.load_config(ROOT / "runs" / a, verbose=False)["evaluation"]["validate_on_resume"]
    check("%s has it ON" % a, on is True, str(on))
others = []
for d in sorted((ROOT / "runs").iterdir()):
    if not (d / "config.yaml").exists() or d.name.startswith("dinov3_mcl_"):
        continue
    if sc.load_config(d, verbose=False)["evaluation"].get("validate_on_resume"):
        others.append(d.name)
check("every other run leaves it OFF", not others, str(others))
check("the shared default is off",
      sc.load_config(ROOT / "runs" / "convnext_base_relabel",
                     verbose=False)["evaluation"]["validate_on_resume"] is False)

# --- 3. behaviour, against the real situation these two runs are in -----------------
print("\n3. the actual resume, epoch 3 at a 100-step cadence")
cfg = sc.load_config(ROOT / "runs" / "dinov3_mcl_slice", verbose=False)
check("epoch 3 validates every 100 steps", sc.eval_interval(cfg, 3) == 100,
      str(sc.eval_interval(cfg, 3)))

# dinov3_mcl_slice: checkpoint at 1100, last validation at 1000
hits = simulate(cfg, True, 1101, 1250, 3)
check("slice scores immediately at 1101, not at 1200",
      hits[0] == (1101, "resume"), str(hits[:3]))
check("slice then resumes the normal cadence at 1200",
      [h for h in hits if h[1] == "cadence"][0][0] == 1200, str(hits[:3]))
check("the resume trigger fires exactly ONCE",
      sum(1 for h in hits if h[1] == "resume") == 1,
      "%d times" % sum(1 for h in hits if h[1] == "resume"))

# dinov3_mcl_fresh: checkpoint at 1300, last validation at 1200
hits = simulate(cfg, True, 1301, 1450, 3)
check("fresh scores immediately at 1301, not at 1400",
      hits[0] == (1301, "resume"), str(hits[:3]))

# --- 4. it changes nothing when it should not ---------------------------------------
print("\n4. no effect where it does not belong")
hits = simulate(cfg, False, 1101, 1250, 3)          # a fresh start, not a resume
check("a NON-resumed start is untouched", hits and hits[0] == (1200, "cadence"),
      str(hits[:2]))
off = sc.load_config(ROOT / "runs" / "convnext_base_relabel", verbose=False)
h2 = simulate(off, True, 1101, 1250, 3)
check("a run with the flag off is untouched on resume",
      all(t == "cadence" for _, t in h2), str(h2[:2]))
# and a checkpoint that happens to land ON a validation step must not double-score
hits = simulate(cfg, True, 1200, 1250, 3)
check("no double validation when the resume lands on a cadence step",
      len([h for h in hits if h[0] == 1200]) == 1, str(hits[:3]))

print("\n" + "=" * 72)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("validate_on_resume fires once, only on a resume, only where it is switched on")
