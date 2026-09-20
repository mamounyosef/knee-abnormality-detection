r"""external_runs_check.py - audit the four runs on backbones new to this project.

These runs differ from every previous one in a way that makes silent failure easy: the trunk
is not the ConvNeXt we have used sixty times, two of them load weights from a raw file rather
than from timm, and each was pretrained under its own normalisation. Any of those can be
wrong without raising, and the run would still train and still produce a plausible curve.

So this checks the things that fail quietly:

  1. NORMALISATION. Each backbone's own pretrained_cfg is read from timm and compared against
     what the run will actually feed it. Feeding ImageNet statistics to a trunk pretrained on
     (0.5, 0.5, 0.5) shifts every input by about a third of a standard deviation, which is
     enough to cost real accuracy and is invisible in the logs.
  2. THE WEIGHTS ACTUALLY LOAD. Building the model must fill every backbone tensor; a
     strict=False miss leaves a randomly initialised trunk that trains perfectly happily.
  3. RESOLUTION PLUMBING. A ViT's position tables must be built at OUR resolution, not the
     backbone's pretraining resolution.
  4. The settings meant to protect the pretrained weights are present: lp_ft_epochs,
     layer_decay, grad_clip above the floor, patience.
  5. Effective batch is still 8 studies, so step counts stay comparable to everything else.

    python checks/external_runs_check.py
"""

import importlib.util
import sys
from pathlib import Path

import torch
import timm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

RUNS = {
    "medsam_vit_b_336": dict(timm="samvit_base_patch16", size=336, ext="medsam",
                             feat=256, ld=0.75, gpu="H100"),
    "medsam_vit_b_512": dict(timm="samvit_base_patch16", size=512, ext="medsam",
                             feat=256, ld=0.75, gpu="H200"),
    "beitv2_base_336": dict(timm="beitv2_base_patch16_224.in1k_ft_in22k_in1k", size=336,
                            ext=None, feat=768, ld=0.75, gpu="H100"),
    "barlowtwins_resnet50_336": dict(timm="resnet50", size=336, ext="barlowtwins",
                                     feat=2048, ld=None, gpu="H100"),
}

OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def load_train(run):
    d = ROOT / "runs" / run
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("ext_" + run, d / "train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    for run, want in RUNS.items():
        print("\n=== %s" % run)
        cfg = sc.load_config(ROOT / "runs" / run, verbose=False)

        check("loads its own config", cfg["experiment"]["name"] == run,
              cfg["experiment"]["name"])
        check("backbone is %s" % want["timm"], cfg["model"]["name"] == want["timm"],
              cfg["model"]["name"])
        check("resolution %d" % want["size"], cfg["input"]["size"] == want["size"],
              str(cfg["input"]["size"]))

        # ---- 1. NORMALISATION against the backbone's own pretraining ------------------
        from timm.models import get_pretrained_cfg
        try:
            pc = get_pretrained_cfg(cfg["model"]["name"])
            d = pc.to_dict() if hasattr(pc, "to_dict") else vars(pc)
            pm, ps = tuple(d.get("mean")), tuple(d.get("std"))
        except Exception:
            pm = ps = None
        cm, cs = tuple(cfg["input"]["norm_mean"]), tuple(cfg["input"]["norm_std"])
        if pm:
            check("normalisation matches the backbone's pretraining",
                  max(abs(a - b) for a, b in zip(pm, cm)) < 1e-6
                  and max(abs(a - b) for a, b in zip(ps, cs)) < 1e-6,
                  "cfg %s/%s vs pretrained %s/%s" % (cm, cs, pm, ps))
        else:
            check("normalisation: backbone cfg readable", False, "could not read pretrained_cfg")

        # ---- 4. the guards that protect the trunk -------------------------------------
        t = cfg["training"]
        check("lp_ft_epochs is 1 (head first, then the trunk)",
              t.get("lp_ft_epochs") == 1, str(t.get("lp_ft_epochs")))
        check("grad_clip 5.0, above the floor of 3",
              float(t["grad_clip"]) >= 5.0, str(t["grad_clip"]))
        check("early stopping patience 7",
              t["early_stopping"]["patience"] == 7, str(t["early_stopping"]["patience"]))
        ld = t["optimizer"].get("layer_decay")
        check("layer_decay %s" % want["ld"], (ld or None) == want["ld"], str(ld))
        head_lr = t["optimizer"]["lr"] / t["optimizer"]["backbone_lr_mult"]
        check("head lr 1.0e-4 (lr / backbone_lr_mult)", abs(head_lr - 1e-4) < 1e-9,
              "%.2e" % head_lr)
        if want["ld"]:
            bottom = head_lr * want["ld"] ** 12
            check("with layer_decay the trunk bottom sees ~%.1e" % bottom, bottom < 1e-5,
                  "%.2e against the head's %.2e" % (bottom, head_lr))

        # ---- 5. effective batch unchanged ---------------------------------------------
        dl = cfg["dataloader"]
        check("effective batch is 8 studies",
              dl["batch_size"] * dl["accum_steps"] == 8,
              "%d x %d" % (dl["batch_size"], dl["accum_steps"]))
        check("gpu is %s" % want["gpu"], cfg["modal"]["gpu"] == want["gpu"],
              cfg["modal"]["gpu"])

        # ---- 2 + 3. build it for real --------------------------------------------------
        mod = load_train(run)
        small = sc.load_config(ROOT / "runs" / run, verbose=False)
        try:
            model = mod.build_model(small)
        except Exception as e:
            check("model builds and loads its weights", False, "%s: %s" % (type(e).__name__, e))
            continue
        check("model builds and loads its weights", True)
        check("feature dim %d" % want["feat"],
              model.backbone.num_features == want["feat"],
              str(model.backbone.num_features))

        # the trunk must NOT be at random init
        torch.manual_seed(0)
        rnd = timm.create_model(want["timm"], pretrained=False, num_classes=0,
                                **(dict(img_size=want["size"]) if "resnet" not in want["timm"]
                                   else {}))
        a, b = model.backbone.state_dict(), rnd.state_dict()
        shared = [k for k in a if k in b and a[k].shape == b[k].shape
                  and a[k].dtype.is_floating_point]
        same = sum(1 for k in shared if torch.equal(a[k], b[k]))
        check("trunk is pretrained, not random init",
              same < len(shared) * 0.02, "%d of %d tensors equal a fresh init"
              % (same, len(shared)))

        # position tables built at OUR resolution
        if "samvit" in want["timm"]:
            pe = a.get("pos_embed")
            g = want["size"] // 16
            check("pos_embed built at %dx%d, not 64x64" % (g, g),
                  pe is not None and tuple(pe.shape)[1:3] == (g, g),
                  str(tuple(pe.shape)) if pe is not None else "absent")

        model.eval()
        n_img = sc.images_per_study(small)
        with torch.no_grad():
            out = model(torch.zeros(n_img, 3, want["size"], want["size"]))
        check("forward: %d windows -> (1, 12), finite" % n_img,
              tuple(out.shape) == (1, 12) and bool(torch.isfinite(out).all()),
              str(tuple(out.shape)))

    print("\n" + "=" * 70)
    print("  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
