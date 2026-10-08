"""sam_check.py - training.sam (SAM / ASAM, 8 Oct) on a tiny real run_experiment, CPU only, local shards.

  1 rho 0      -> the weights after 3 steps EQUAL the SAM-off run (the second pass, zero_grad and the exact
                  restore change nothing when the perturbation is 0: the plumbing is a no-op)
  2 rho 0.05   -> weights differ from SAM off, by a small amount; banner printed; sam_gnorm / sam_loss2 /
                  sam_n logged, finite; perturbed loss >= the clean loss on average (it is the ascent point)
  3 adaptive   -> ASAM runs, logs, differs from plain SAM
  4 guard      -> accum_steps 2 with SAM refuses to start
  5 the real SAM run config (convnext_base_25d_18slices_v5_sam) loads: sam on, rho 0.05, accum 1, no aux

    python checks/sam_check.py
"""
import contextlib
import copy
import io
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

torch.cuda.is_available = lambda: False      # CPU only: the GPU stays untouched
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

fails, n_all = [], [0]


def check(name, ok, detail=""):
    n_all[0] += 1
    print("  %-66s %s%s" % (name, "ok" if ok else "FAIL", "  " + str(detail) if detail else ""))
    if not ok:
        fails.append(name)


have = {q.stem for q in (ROOT / "data/shards").glob("*.npz")}
BASE_RUN = "convnext_base_25d_18slices_v5_aug_base"
base = sc.load_config(ROOT / "runs" / BASE_RUN, verbose=False)
tmp = Path(tempfile.mkdtemp(prefix="samchk_"))
import timm  # noqa: E402

try:
    trn = pd.read_csv(ROOT / "data" / base["paths"]["train_csv"])
    trn = trn[trn.StudyInstanceUID.isin(have)].head(6)
    va = pd.read_csv(ROOT / "data" / base["paths"]["val_csv"])
    va = va[va.StudyInstanceUID.isin(have)].head(4)
    trn.to_csv(tmp / "tr.csv", index=False)
    va.to_csv(tmp / "va.csv", index=False)

    def cfg_for(tag, sam, accum=1):
        c = copy.deepcopy(base)
        c["dataloader"].update(num_workers=0, val_num_workers=0, persistent_workers=False, prefetch_factor=None,
                               pin_memory=False, batch_size=2, accum_steps=accum, val_batch_size=2, drop_last=False)
        c["experiment"] = dict(base["experiment"], name=f"sam_{tag}")
        c["model"] = dict(base["model"], name="resnet18", pretrained=False)
        c["input"] = dict(base["input"], size=32, n_slices=3, slices_as_channels=1, mm_per_px=None)
        c["training"] = dict(base["training"], epochs=1, sam=sam)
        c["training"]["early_stopping"] = dict(base["training"]["early_stopping"], enable=False)
        c["training"]["extend"] = dict(base["training"].get("extend") or {}, enable=False)
        c["evaluation"] = dict(base["evaluation"], eval_every_steps_before=3, eval_every_steps_mid=3,
                               eval_every_steps_after=3, eval_every_steps_epoch_mid_threshold=None,
                               eval_every_steps_epoch_threshold=1, validate_before_training=False)
        c["evaluation"]["audit"] = dict(base["evaluation"].get("audit") or {}, enable=False)
        c["output"] = dict(base["output"], checkpoint_every_steps=0, save_debug_images=0, keep_last_n=1,
                           keep_top_k=0, upload_checkpoints=None)
        c["paths"] = dict(base["paths"], train_csv=str(tmp / "tr.csv"), val_csv=str(tmp / "va.csv"))
        c["wandb"] = {"enabled": False}
        c["resume"] = None
        return c

    def build(c):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0, in_chans=3)
                self.per = len(c["input"]["slots"]) * int(c["input"]["n_slices"]) // 3
                self.head = nn.Linear(self.backbone.num_features, sc.num_output_logits(c))

            def forward(self, x):
                f = self.backbone(x)
                return self.head(f.view(-1, self.per, f.shape[-1]).mean(1))
        return Tiny()

    def run(tag, sam, accum=1):
        c = cfg_for(tag, sam, accum)
        d = tmp / tag
        sc.set_seed(42, True)
        buf = io.StringIO()
        err = None
        with contextlib.redirect_stdout(buf):
            try:
                sc.run_experiment(c, build(c), d)
            except AssertionError as e:
                err = str(e)
        if err:
            return None, None, buf.getvalue(), err
        ck = torch.load(d / "results/checkpoints/last.pt", map_location="cpu", weights_only=False)
        w = ck.get("model") or ck.get("model_state") or ck.get("state_dict")
        return w, pd.read_csv(d / "results/train_log.csv"), buf.getvalue(), None

    def maxdiff(a, b):
        # weights only: the test resnet18's BatchNorm running stats see the 2nd SAM forward too (ConvNeXt has none)
        return max(float((a[k].float() - b[k].float()).abs().max()) for k in a
                   if a[k].is_floating_point() and not k.endswith(("running_mean", "running_var")))

    w_off, tl_off, o_off, e_off = run("off", None)
    if e_off:
        print("OFF RUN FAILED:", e_off, o_off[-3000:])
    w_r0, tl_r0, o_r0, _ = run("rho0", {"enable": True, "rho": 0.0})
    w_s, tl_s, o_s, _ = run("sam", {"enable": True, "rho": 0.05})
    w_a, tl_a, o_a, _ = run("asam", {"enable": True, "rho": 0.5, "adaptive": True})
    _, _, o_g, err_g = run("guard", {"enable": True, "rho": 0.05}, accum=2)
    # 7: extend the finished SAM run by 1 epoch (resume last), as convnext_base_25d_18slices_v5_sam does for real
    ce = cfg_for("sam", {"enable": True, "rho": 0.05})
    ce["resume"] = "last"
    ce["training"]["extend"] = {"enable": True, "epochs": 1, "lr": 8.0e-6, "warmup_frac": 0.02, "min_lr_mult": 0.10,
                                "reset_optimizer": False, "reset_patience": True}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sc.run_experiment(ce, build(ce), tmp / "sam")
    o_e = buf.getvalue()
    tl_e = pd.read_csv(tmp / "sam/results/train_log.csv")

    print("SAM plumbing")
    check("1 SAM off: no banner, no sam columns", "SAM" not in o_off and "sam_gnorm" not in tl_off.columns)
    d0 = maxdiff(w_off, w_r0)
    check("1 rho 0 weights == SAM off weights after 3 steps", d0 < 1e-6, f"max |dw| {d0:.2e}")
    ds = maxdiff(w_off, w_s)
    check("2 rho 0.05 weights differ from SAM off", ds > 1e-7, f"max |dw| {ds:.2e}")
    check("2 banner printed", "SAM rho=0.05" in o_s)
    for nm, tl in (("2 SAM", tl_s), ("3 ASAM", tl_a)):
        cols = {"sam_gnorm", "sam_loss2", "sam_n"} <= set(tl.columns)
        check(f"{nm}: sam_gnorm / sam_loss2 / sam_n logged", cols)
        if cols:
            r = tl.dropna(subset=["sam_gnorm"])
            check(f"{nm}: one SAM pass per step, values finite",
                  len(r) >= 3 and (r.sam_n == 1).all() and r[["sam_gnorm", "sam_loss2"]].map(
                      lambda v: pd.notna(v) and abs(v) < 1e9).all().all(), f"{len(r)} rows")
    lc = next((c for c in ("loss", "train_loss") if c in tl_s.columns), None)
    if lc:
        sharp = (tl_s.sam_loss2 - tl_s[lc]).mean()
        check("2 perturbed loss >= clean loss on average (ascent point)", sharp > -1e-4, f"mean {sharp:+.5f}")
    check("3 ASAM banner printed", "SAM (adaptive) rho=0.5" in o_a)
    da = maxdiff(w_s, w_a)
    check("3 ASAM weights differ from SAM", da > 1e-7, f"max |dw| {da:.2e}")
    check("4 accum 2 + SAM refuses to start", err_g is not None and "accum_steps 1" in err_g, err_g)
    n0 = int(tl_s.global_step.max())
    n1 = int(tl_e.global_step.max())
    re = tl_e[tl_e.global_step > n0]
    check("7 extend + SAM: resumed and trained past the finished run", n1 > n0 and "adding" in o_e, f"steps {n0} -> {n1}")
    check("7 extend + SAM: SAM still on in the extension (sam_n logged)",
          "sam_n" in re.columns and re.sam_n.notna().all() and len(re) >= 1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("real run config")
p = ROOT / "runs/convnext_base_25d_18slices_v5_sam"
if p.exists():
    c = sc.load_config(p, verbose=False)
    s = c["training"].get("sam") or {}
    check("5 sam on, rho 0.05, not adaptive", s.get("enable") and s.get("rho") == 0.05 and not s.get("adaptive"))
    check("5 accum 1, no aux losses", sc.accum_steps(c) == 1 and not (c["training"].get("aux_report") or {}).get("enable")
          and not (c["training"].get("aux_attributes") or {}).get("enable"))
    rest = sc.load_config(ROOT / "runs" / BASE_RUN, verbose=False)
    same = all(c[k] == rest[k] for k in ("paths", "input", "model", "dataloader", "reproducibility"))
    tr_a = {k: v for k, v in c["training"].items() if k not in ("sam", "extend")}
    tr_b = {k: v for k, v in rest["training"].items() if k not in ("sam", "extend")}
    check("5 everything else identical to aug_base (paths/input/model/loader/seed/training)", same and tr_a == tr_b)
else:
    check("5 run folder exists", False, str(p))

print("\n" + ("ALL OK (%d/%d)" % (n_all[0], n_all[0]) if not fails else "FAILED: %s" % ", ".join(fails)))
