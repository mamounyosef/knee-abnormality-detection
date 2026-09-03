"""
resume_guard_check.py — the resume guard refuses a config change, and only that.

`resume: "last"` is deliberate: Modal preempts containers, and a workspace hand-off must
continue the run rather than restart it. The cost is that a checkpoint outlives the config
that wrote it, so editing a setting and relaunching would SILENTLY continue the old run -
half of it trained under each setting. That is what nearly happened when grad_clip was
raised to 3.0 while the throttled checkpoint was still on the volume.

This rehearses the guard on the real cases it has to get right:

    fires   - grad_clip / lr / model / epochs / csv changed
    silent  - identical config (preemption, workspace hand-off)
    silent  - extend.enable, whose whole job is to change lr and epochs
    silent  - a stage-2 hand-off, already a deliberate weights-only transfer
    warns   - resume_force, which continues anyway but says so

    python runs/others/resume_guard_check.py
"""

import copy
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

BASE = {
    "experiment": {"name": "guard_check"},
    "modal": {"runs_volume": "knee-runs"},
    "model": {"name": "convnext_base.dinov3_lvd1689m"},
    "input": {"size": 512, "n_slices": 24, "slots": ["sag_plain", "sag_fluid"]},
    "target": {"mode": "soft"},
    "paths": {"train_csv": "05_train_qwen24t-med.csv", "val_csv": "05_val_qwen24t-med.csv"},
    "training": {
        "grad_clip": 3.0,
        "epochs": 12,
        "optimizer": {"lr": 5.0e-5, "backbone_lr_mult": 0.5,
                      "layer_decay": None, "weight_decay": 0.05},
        "scheduler": {"name": "cosine"},
        "loss": {"name": "bce"},
        "extend": {"enable": False},
    },
}


def guard(ck_cfg, cfg, step=1400):
    """The engine's block, lifted verbatim so the test exercises the real logic."""
    import re
    src = (PKG / "shared_code.py").read_text(encoding="utf-8")
    start = src.index("    if ck and _same_stage and not (cfg[\"training\"]")
    end = src.index("\n    if _pesg and ck:", start)
    body = src[start:end]
    # the engine reads these from enclosing scope; supply them
    ck = {"cfg": ck_cfg, "step": step}
    _same_stage = ck_cfg.get("training", {}).get("loss", {}).get("name") == \
        cfg["training"]["loss"]["name"]
    ns = {"ck": ck, "_same_stage": _same_stage, "cfg": cfg}
    exec(re.sub(r"^    ", "", body, flags=re.M), ns)


def check(label, mutate, expect, force=False):
    cfg = copy.deepcopy(BASE)
    ck_cfg = copy.deepcopy(BASE)
    mutate(cfg, ck_cfg)
    if force:
        cfg["resume_force"] = True
    try:
        guard(ck_cfg, cfg)
        got = "allowed"
    except RuntimeError:
        got = "refused"
    ok = got == expect
    print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} expect={expect:8s} got={got}")
    return ok


def main():
    print("resume guard\n" + "-" * 78)
    r = []
    # the case that started this: the fix is in the config, the volume is not
    r.append(check("grad_clip 1.0 -> 3.0", lambda c, k: k["training"].__setitem__("grad_clip", 1.0), "refused"))
    r.append(check("lr changed", lambda c, k: k["training"]["optimizer"].__setitem__("lr", 1e-4), "refused"))
    r.append(check("backbone_lr_mult changed", lambda c, k: k["training"]["optimizer"].__setitem__("backbone_lr_mult", 1.0), "refused"))
    r.append(check("backbone swapped", lambda c, k: k["model"].__setitem__("name", "convnext_base.fb_in22k_ft_in1k"), "refused"))
    r.append(check("epochs changed", lambda c, k: k["training"].__setitem__("epochs", 20), "refused"))
    r.append(check("train csv swapped", lambda c, k: k["paths"].__setitem__("train_csv", "05_train.csv"), "refused"))
    r.append(check("input size changed", lambda c, k: k["input"].__setitem__("size", 672), "refused"))
    r.append(check("slots changed", lambda c, k: k["input"].__setitem__("slots", ["sag_plain"]), "refused"))

    # the cases that MUST stay silent, or the guard breaks normal operation
    r.append(check("identical config (preemption / hand-off)", lambda c, k: None, "allowed"))
    r.append(check("extend: lr and epochs change on purpose",
                   lambda c, k: (c["training"]["extend"].__setitem__("enable", True),
                                 c["training"]["optimizer"].__setitem__("lr", 9e-6),
                                 c["training"].__setitem__("epochs", 20)), "allowed"))
    r.append(check("stage-2 hand-off (loss differs -> weights only)",
                   lambda c, k: (k["training"]["loss"].__setitem__("name", "bce"),
                                 c["training"]["loss"].__setitem__("name", "aucm_official"),
                                 c["training"]["optimizer"].__setitem__("lr", 1e-3)), "allowed"))
    r.append(check("key absent from an older checkpoint's cfg",
                   lambda c, k: k["training"].pop("grad_clip"), "allowed"))
    r.append(check("resume_force overrides", lambda c, k: k["training"].__setitem__("grad_clip", 1.0),
                   "allowed", force=True))

    print("-" * 78)
    print(f"  {sum(r)}/{len(r)} passed")
    return 0 if all(r) else 1


if __name__ == "__main__":
    raise SystemExit(main())
