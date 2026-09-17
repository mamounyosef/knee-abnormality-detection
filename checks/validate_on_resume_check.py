r"""validate_on_resume_check.py - does the one-off resume validation fire, once, and only
when asked?

The validation cadence is a modulo on the global step (`gstep % ev == 0`). A rolling
checkpoint is saved on its own schedule, so it routinely lands BETWEEN two validations:
dinov3_mcl_slice stopped at step 1100 with its last validation at 1000, and
dinov3_mcl_fresh at 1300 with its last at 1200. Resuming, neither would be scored until
1200 and 1400, under a cadence that changed after those steps were trained.

`evaluation.validate_on_resume` closes that gap: score once on the first step back, then
behave exactly as before.

Two layers of test, because either alone is weak:

  PART A calls `shared_code.eval_due` directly - the SAME function the training loop
         calls. It does not re-implement the rule, so it cannot pass against an engine
         that has drifted.
  PART B trains a real model on the real local shards through the real `run_experiment`,
         stops it at a step that is deliberately NOT a validation step, resumes, and
         reads the resulting val_log off disk. That is the only way to prove the flag
         survives checkpoint save/load and actually reaches the loop.

Part B needs the local shards in data/shards and takes a couple of minutes on CPU.
Pass --fast to skip it.

    python checks/validate_on_resume_check.py
"""

import argparse
import csv
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


def run_steps(cfg, resumed, first, last, epoch):
    """Drive the REAL eval_due over a step range and collect what it decides."""
    flag = [bool(resumed) and bool(cfg["evaluation"].get("validate_on_resume"))]
    out = []
    for g in range(first, last + 1):
        t = sc.eval_due(cfg, epoch, g, flag)
        if t:
            out.append((g, "resume" if "resume" in t else "cadence"))
    return out, flag


# =====================================================================================
# PART A - the real decision function
# =====================================================================================
print("\nA1. the loop delegates instead of re-implementing")
src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
check("eval_due exists", "def eval_due(" in src)
check("the training loop CALLS it", "trig = eval_due(cfg, epoch, gstep, resume_eval)" in src)
# the modulo rule must exist in exactly ONE place, inside eval_due. A second copy in the
# loop is how the two silently drift apart, which is the bug this file mostly guards.
_body = src[src.index("def eval_due("):]
_body = _body[:_body.index("\n@torch.no_grad()")]
check("the modulo rule exists exactly once, inside eval_due",
      src.count("gstep % ev == 0") == 1 and "gstep % ev == 0" in _body,
      "%d copies repo-wide" % src.count("gstep % ev == 0"))
check("the flag arms only when a checkpoint was loaded",
      'resume_eval = [bool(ck) and bool(cfg["evaluation"].get("validate_on_resume"))]' in src)

print("\nA2. who has it switched on")
for a in ("dinov3_mcl_slice", "dinov3_mcl_fresh"):
    on = sc.load_config(ROOT / "runs" / a, verbose=False)["evaluation"]["validate_on_resume"]
    check("%s has it ON" % a, on is True, str(on))
others = [d.name for d in sorted((ROOT / "runs").iterdir())
          if (d / "config.yaml").exists() and not d.name.startswith("dinov3_mcl_")
          and sc.load_config(d, verbose=False)["evaluation"].get("validate_on_resume")]
check("every other run leaves it OFF", not others, str(others))
check("the shared default is off",
      sc.load_config(ROOT / "runs" / "convnext_base_relabel",
                     verbose=False)["evaluation"]["validate_on_resume"] is False)

print("\nA3. the situation these two runs are actually in")
cfg = sc.load_config(ROOT / "runs" / "dinov3_mcl_slice", verbose=False)
check("epochs 1-2 validate every 200", sc.eval_interval(cfg, 1) == 200
      and sc.eval_interval(cfg, 2) == 200)
check("epoch 3 onward validates every 100",
      all(sc.eval_interval(cfg, e) == 100 for e in (3, 4, 5, 6)))

hits, flag = run_steps(cfg, True, 1101, 1250, 3)        # slice: ckpt 1100, last val 1000
check("slice scores at 1101, not at 1200", hits[0] == (1101, "resume"), str(hits[:3]))
check("slice then keeps the 100-step cadence",
      [h[0] for h in hits if h[1] == "cadence"][:2] == [1200], str(hits[:3]))
check("the resume trigger fires exactly ONCE",
      sum(1 for h in hits if h[1] == "resume") == 1)
check("the flag is left disarmed", flag == [False], str(flag))

hits, _ = run_steps(cfg, True, 1301, 1450, 3)           # fresh: ckpt 1300, last val 1200
check("fresh scores at 1301, not at 1400", hits[0] == (1301, "resume"), str(hits[:3]))

print("\nA4. it changes nothing where it does not belong")
hits, _ = run_steps(cfg, False, 1101, 1250, 3)
check("a fresh start (no checkpoint) is untouched", hits[0] == (1200, "cadence"),
      str(hits[:2]))
off = sc.load_config(ROOT / "runs" / "convnext_base_relabel", verbose=False)
hits, _ = run_steps(off, True, 1101, 1600, 3)
check("a run with the flag off is untouched on resume",
      all(t == "cadence" for _, t in hits), str(hits[:2]))
hits, _ = run_steps(cfg, True, 1200, 1350, 3)
check("landing exactly ON a cadence step does not double-score",
      len([h for h in hits if h[0] == 1200]) == 1 and hits[0] == (1200, "cadence"),
      str(hits[:3]))
# a cadence of 0 disables step-triggered eval entirely; the resume one-off must still work
zero = sc.load_config(ROOT / "runs" / "dinov3_mcl_slice", verbose=False)
zero["evaluation"] = dict(zero["evaluation"], eval_every_steps_after=0,
                          eval_every_steps_before=0, eval_every_steps_epoch_threshold=1)
hits, _ = run_steps(zero, True, 1101, 1400, 3)
check("with the cadence disabled it still fires once and stops",
      hits == [(1101, "resume")], str(hits))

print("\nA5. the real config change does not trip the resume guard")
# _run_experiment refuses to resume when a GUARDED key moved, because half the run would
# then be trained under each setting. Everything we changed for these two arms lives under
# `evaluation` and `dataloader`, neither of which is guarded - so the resume is clean and
# no resume_force is needed. If that ever stops being true, this check says so.
_g = src[src.index("_guard = ["):]
_g = _g[:_g.index("]\n")]
guarded = re.findall(r'\("([a-z_.]+)",', _g)
check("the guard watches the things that change the maths",
      {"training.epochs", "optimizer.lr", "input.n_slices", "input.slots",
       "model.name"} <= set(guarded), str(sorted(guarded)))
leaked = [g for g in guarded if g.startswith(("evaluation.", "dataloader."))]
check("it does NOT watch evaluation or dataloader", not leaked, str(leaked))
for arm in ("dinov3_mcl_slice", "dinov3_mcl_fresh"):
    c = sc.load_config(ROOT / "runs" / arm, verbose=False)
    check("%s does not set resume_force" % arm, not c.get("resume_force"),
          str(c.get("resume_force")))
check("so the cadence edit resumes without forcing anything",
      not leaked and not sc.load_config(ROOT / "runs" / "dinov3_mcl_slice",
                                        verbose=False).get("resume_force"))

# =====================================================================================
# PART B - a real run, stopped off-cadence and resumed
# =====================================================================================
ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true", help="skip the end-to-end run")
args = ap.parse_args()

if not args.fast:
    print("\nB. end to end: train, stop with the checkpoint PAST the last validation,")
    print("   resume, and read the val_log off disk")
    import pandas as pd
    import torch.nn as nn

    have = sorted(q.stem for q in (ROOT / "data" / "shards").glob("*.npz"))
    check("local shards are present to train on", len(have) >= 3, "%d found" % len(have))

    if len(have) >= 3:
        tmp = Path(tempfile.mkdtemp(prefix="vor_"))
        try:
            # 6 training rows at batch 1, accumulation 1 = 6 optimizer steps per epoch.
            # Validating every 4 puts epoch 1's only validation at step 4 while the
            # epoch-end checkpoint lands at step 6 - the exact gap the flag exists for.
            EVERY, SPE = 4, 6
            base = sc.load_config(ROOT / "runs" / "dinov3_mcl_slice", verbose=False)

            def cfg_for(name, epochs, resume, on):
                c = dict(base)
                c["experiment"] = dict(base["experiment"], name=name)
                c["model"] = dict(base["model"], name="resnet18", pretrained=False,
                                  feat_norm=False)
                c["input"] = dict(base["input"], size=32, n_slices=2, mm_per_px=None,
                                  slots=["cor_fluid", "ax_fluid"])
                c["dataloader"] = dict(base["dataloader"], batch_size=1, accum_steps=1,
                                       val_batch_size=1, num_workers=0, val_num_workers=0,
                                       persistent_workers=False, prefetch_factor=None,
                                       pin_memory=False, drop_last=False)
                c["training"] = dict(base["training"], epochs=epochs, grad_clip=3.0)
                c["training"]["early_stopping"] = dict(
                    base["training"]["early_stopping"], enable=False)
                c["training"]["extend"] = dict(base["training"]["extend"], enable=False)
                c["evaluation"] = dict(
                    base["evaluation"], eval_every_steps_before=EVERY,
                    eval_every_steps_mid=EVERY, eval_every_steps_after=EVERY,
                    eval_every_steps_epoch_mid_threshold=None,
                    eval_every_steps_epoch_threshold=1, validate_on_resume=on)
                c["evaluation"]["audit"] = dict(base["evaluation"]["audit"], enable=False)
                c["output"] = dict(base["output"], checkpoint_every_steps=0,
                                   save_debug_images=0, keep_last_n=5)
                c["paths"] = dict(base["paths"], train_csv=str(tmp / "tr.csv"),
                                  val_csv=str(tmp / "va.csv"))
                c["wandb"] = dict(base.get("wandb", {}) or {}, enabled=False)
                c["resume"] = resume
                # the harness uses `epochs` to control run length and the resume
                # guard watches that key, so the TEST has to force past it. The
                # real runs do not: see A5, no guarded key changed there.
                c["resume_force"] = True
                return c

            tr = pd.read_csv(ROOT / "data" / "05_train_qwen24t-med.csv")
            tr = tr[tr["StudyInstanceUID"].isin(have)]
            va = pd.read_csv(ROOT / "data" / "05_val_qwen24t-med.csv")
            va = va[va["StudyInstanceUID"].isin(have)]
            # repeat the local studies up to SPE rows: this tests WHEN the loop validates,
            # not what the model learns
            pd.concat([tr] * 4).head(SPE).to_csv(tmp / "tr.csv", index=False)
            va.head(2).to_csv(tmp / "va.csv", index=False)

            def build(c):
                import timm

                class Tiny(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.backbone = timm.create_model("resnet18", pretrained=False,
                                                          num_classes=0)
                        self.per_study = len(c["input"]["slots"]) * c["input"]["n_slices"]
                        self.head = nn.Linear(self.backbone.num_features,
                                              sc.num_output_logits(c))

                    def forward(self, x):
                        f = self.backbone(x)
                        f = f.view(-1, self.per_study, f.shape[-1])
                        return self.head(f.mean(dim=1))

                return Tiny()

            def steps_in(d):
                f = d / "results" / "val_log.csv"
                if not f.exists():
                    return []
                return [int(r["step"]) for r in csv.DictReader(open(f, encoding="utf-8"))
                        if r.get("step")]

            def leg(d, c):
                sc.set_seed(42, False)
                sc.run_experiment(c, build(c), d)

            # ---- the arm with the flag ON -------------------------------------------
            on = tmp / "on"
            leg(on, cfg_for("vor_on", 1, None, True))
            a1 = steps_in(on)
            # step EVERY is the cadence; step SPE is the end-of-TRAINING validation the
            # engine always runs. The real arms never got that one - they were killed by
            # a credit limit mid-epoch, which is why dinov3_mcl_slice sits at checkpoint
            # 1100 with its last validation at 1000. The mechanism under test is the same
            # either way: does the resume step get scored when it otherwise would not.
            check("leg 1 validated on the cadence at step %d" % EVERY, a1[:1] == [EVERY],
                  "steps %s (step %d is the end-of-training validation)" % (a1, SPE))
            check("leg 1 did NOT validate at the resume step %d" % (SPE + 1),
                  (SPE + 1) not in a1, "steps %s" % a1)

            leg(on, cfg_for("vor_on", 2, "last", True))
            a2 = [s for s in steps_in(on) if s not in a1]
            check("resumed run scored IMMEDIATELY at step %d" % (SPE + 1),
                  (SPE + 1) in a2, "new validations at %s" % a2)
            check("and still kept the cadence at step %d" % (2 * EVERY),
                  2 * EVERY in a2, "new validations at %s" % a2)
            check("it did NOT validate every step", len(a2) <= 3,
                  "new validations at %s" % a2)

            # ---- the control, flag OFF ----------------------------------------------
            off = tmp / "off"
            leg(off, cfg_for("vor_off", 1, None, False))
            b1 = steps_in(off)
            leg(off, cfg_for("vor_off", 2, "last", False))
            b2 = [s for s in steps_in(off) if s not in b1]
            check("control with the flag OFF did NOT score at step %d" % (SPE + 1),
                  (SPE + 1) not in b2, "new validations at %s" % b2)
            check("control still validated on its own cadence", bool(b2),
                  "new validations at %s" % b2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

print("\n" + "=" * 72)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("validate_on_resume fires once, only on a resume, only where it is switched on")
