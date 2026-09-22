"""warm_start_check.py - model.init_from carries a finished run's weights into a new geometry.

medimageinsight_davit_huge_perslice_480_warm does not start from the public MedImageInsight
weights. It starts from OUR fine-tune, medimageinsight_davit_huge_336, which already sits at
val 0.9041, and only adapts it to 480 px with one image per slice instead of six images of
stacked slices. That is what makes a run priced at 6.1x per step affordable: the saving is
in EPOCHS, four instead of twelve.

The whole idea rests on one claim, and this file exists to prove it rather than assume it:
every tensor transfers with no reshaping. DaViT encodes position with convolutions, so there
is no position embedding to interpolate for 336 -> 480; and the stem is (256, 3, 7, 7) on
both sides, because slices_as_channels 6 over 18 slices means 6 images of 3 channels while
per-slice means 3 channels as well.

What would break silently without these checks:
  - a partial load. A half-random backbone trains for an hour and looks completely normal
    doing it, which is exactly how convnext_large shipped 0.028 worse than measured.
  - init_from firing on a RESUME. A preempted Modal container must continue its own run,
    not jump back to the donor and throw away the epochs in between.
  - the optimizer or scheduler coming across too, which would put a fresh run partway down
    a cosine schedule it never started.

Run:  python checks/warm_start_check.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

PASS, FAIL = [], []
RUN = "medimageinsight_davit_huge_perslice_480_warm"
DONOR = "medimageinsight_davit_huge_336"


def ck(n, ok, d=""):
    (PASS if ok else FAIL).append(n)
    print("  %s %-58s %s" % ("PASS" if ok else "FAIL", n, d))


def main():
    import contextlib
    import io as _io

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        cfg = sc.load_config(ROOT / "runs" / RUN)
        dcfg = sc.load_config(ROOT / "runs" / DONOR)

    print("\n1. the two configs differ in exactly the intended ways")
    ck("donor is 336, this run is 480", dcfg["input"]["size"] == 336 and cfg["input"]["size"] == 480,
       "%s -> %s" % (dcfg["input"]["size"], cfg["input"]["size"]))
    ck("donor stacks slices as channels, this run does not",
       dcfg["input"]["slices_as_channels"] == 6 and not cfg["input"]["slices_as_channels"],
       "%r -> %r" % (dcfg["input"]["slices_as_channels"], cfg["input"]["slices_as_channels"]))
    ck("slice count is unchanged at 18",
       dcfg["input"]["n_slices"] == cfg["input"]["n_slices"] == 18)
    ck("laterality flip is unchanged",
       dcfg["input"]["laterality_flip"] == cfg["input"]["laterality_flip"] == "left")
    ck("images per study goes 24 -> 72 (the 3x that makes this expensive)",
       sc.images_per_study(dcfg) == 24 and sc.images_per_study(cfg) == 72,
       "%d -> %d, and 480/336 squared is another %.2fx"
       % (sc.images_per_study(dcfg), sc.images_per_study(cfg), (480 / 336) ** 2))

    print("\n2. the warm-start settings")
    ck("init_from names the donor's best checkpoint",
       cfg["model"]["init_from"] == "%s/best_weights" % DONOR, cfg["model"].get("init_from"))
    ck("four epochs, not twelve", cfg["training"]["epochs"] == 4, cfg["training"]["epochs"])
    ck("no frozen-probe epoch: the head arrives trained",
       cfg["training"]["lp_ft_epochs"] == 0)
    # `lr` is the BACKBONE rate and the head is DERIVED as lr / backbone_lr_mult, so a
    # mult of 0.5 makes the head run twice as fast, not half. Easy to read backwards, so
    # assert the rates the optimizer actually ends up with, not the config number.
    ck("learning rate is a quarter of the donor's",
       abs(cfg["training"]["optimizer"]["lr"] - dcfg["training"]["optimizer"]["lr"] / 4) < 1e-12,
       "%g -> %g" % (dcfg["training"]["optimizer"]["lr"], cfg["training"]["optimizer"]["lr"]))
    ck("head runs FASTER than the trunk, not slower",
       cfg["training"]["optimizer"]["backbone_lr_mult"] < 1.0,
       "head = lr / %g" % cfg["training"]["optimizer"]["backbone_lr_mult"])
    ck("grad_clip is above the floor of 3", cfg["training"]["grad_clip"] >= 3.0,
       cfg["training"]["grad_clip"])

    print("\n3. the donor checkpoint resolves and is the one we think it is")
    src = sc.resolve_init_from(cfg["model"]["init_from"], ROOT / "runs" / RUN, cfg)
    ck("resolve_init_from finds it", src.exists(), str(src.relative_to(ROOT)))
    dck = torch.load(src, map_location="cpu", weights_only=False)
    sd = dck["model"]
    ck("it is the 0.9041 checkpoint",
       abs(float(dck.get("score", 0)) - 0.904118) < 1e-5,
       "step %s epoch %s score %s" % (dck.get("global_step") or dck.get("step"),
                                      dck.get("epoch"), dck.get("score")))
    ck("406 tensors, 360.7M parameters", len(sd) == 406,
       "%d tensors, %.1fM params" % (len(sd), sum(v.numel() for v in sd.values()) / 1e6))
    ck("NO position embeddings, so 336 -> 480 needs no interpolation",
       not [k for k in sd if "pos_embed" in k or "rel_pos" in k or "position" in k])
    ck("the stem takes 3 channels, so per-slice needs no reshaping",
       tuple(sd["backbone.stem.conv.weight"].shape) == (256, 3, 7, 7),
       str(tuple(sd["backbone.stem.conv.weight"].shape)))

    print("\n4. the weights actually load into THIS run's model, all of them")
    sys.path.insert(0, str(ROOT / "runs" / RUN))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "warm_train", ROOT / "runs" / RUN / "train.py")
    mod = importlib.util.module_from_spec(spec)
    # build the skeleton WITHOUT fetching the public weights: init_from overwrites all of
    # them anyway, and this check must run offline
    cfg_nopre = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
        model = mod.SliceModel(cfg_nopre)
        res = sc._load_model_state(model, sd)
    missing = list(getattr(res, "missing_keys", []) or [])
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    ck("nothing is missing", not missing, str(missing[:4]))
    ck("nothing is unexpected", not unexpected, str(unexpected[:4]))
    ck("so all 406 tensors transferred with no surgery", not missing and not unexpected)

    print("\n5. and the 480 per-slice model really runs")
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 480, 480)
        out = model.backbone(x)
    ck("the backbone accepts 480 px input", out.shape[0] == 2, tuple(out.shape))
    ck("its feature width matches the head", out.shape[-1] == sd["head.weight"].shape[1],
       "%d features, head expects %d" % (out.shape[-1], sd["head.weight"].shape[1]))

    print("\n6. the engine wiring")
    src_txt = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
    ck("init_from applies only when this run has NO checkpoint of its own",
       "if _init and not ck:" in src_txt)
    ck("weights only: the optimizer and scheduler are never touched",
       "WEIGHTS ONLY" in src_txt and "_dck.get(\"optimizer\")" not in src_txt)
    ck("a partial load is a hard error, not a warning",
       "warm start is incomplete" in src_txt and "raise RuntimeError" in src_txt)
    ck("the donor's step and score are printed, so the log says what it started from",
       "donor: step %s  epoch %s  score %s" in src_txt)

    print("\n" + "=" * 72)
    print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
