"""
frozen_probe.py - do a backbone's FROZEN features carry knee signal at all?

WHY THIS EXISTS
siglip2_vitb_512 stalled at 0.6237 / 0.6325 on its first two validations, the worst start
of any run in the project, with a training loss that did not move for 1,100 steps. Two
explanations fit that equally well from the training curves alone:

  A. the features are fine, the trunk just is not ADAPTING at 5e-5
  B. the features do not separate knees at all, so the head has nothing to work with

A cosine-similarity test suggested B, but that test is not trustworthy for a contrastive
model: CLIP-family embeddings are known to occupy a narrow cone, so ANY two images score a
high cosine (the "cone effect"), and a high number is not evidence of collapse.

A linear probe on FROZEN features settles it without that confound. It is scale-invariant,
rotation-invariant, and asks exactly the question that matters: is the knee signal linearly
available in this representation? A backbone whose frozen probe lands near chance cannot be
rescued by a learning rate; one that probes well but trains badly is an optimisation
problem.

Every backbone sees the SAME studies, slices and slots, so the numbers are comparable.

    modal run runs/others/frozen_probe.py
    modal run runs/others/frozen_probe.py --models "a,b" --n-studies 400
"""

import sys
from pathlib import Path


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402

# (timm name, norm mean, norm std, note). The normalisation is each checkpoint's OWN, read
# from its pretrained_cfg - feeding a backbone the wrong statistics is the exact bug the
# engine's check_normalization now guards against, and it would invalidate this comparison.
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
ZERO_ONE = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
DEFAULT_MODELS = [
    ("vit_base_patch16_siglip_512.v2_webli", *HALF, "SigLIP 2, MAP pool - the suspect"),
    ("vit_base_patch16_siglip_gap_512.v2_webli", *HALF, "SigLIP 2, avg pool - same trunk"),
    ("convnext_base.fb_in22k_ft_in1k", *IMAGENET, "our best backbone, reference"),
    ("convnext_base.dinov3_lvd1689m", *IMAGENET, "DINOv3, self-supervised on 1.7B photos"),
    ("vit_base_patch16_224.augreg_in21k", *HALF, "plain ViT-B, shape control"),
    # MRI-CORE is not a timm checkpoint: it is a DINO teacher whose keys need rewriting,
    # so it is loaded through mricore_vitb_relabel's own loader. Its normalisation is
    # [0,1], NOT ImageNet - the trunk was trained on min-max scaled slices.
    ("MRI-CORE:vit_base_patch16_224", *ZERO_ONE, "MRI-CORE, DINO on 6M MRI slices"),
]

EXP_DIR = PKG_ROOT / "mricore_vitb_relabel"      # borrow a ViT config: 4 slots x 24 slices


def _mri_core(skeleton, cfg):
    """MRI-CORE's DINO teacher, loaded into a timm ViT skeleton.

    Reuses mricore_vitb_relabel's loader rather than reimplementing the key rewriting
    (strip 'backbone.', unchunk the block indices, resample pos_embed), so this probe
    measures exactly the trunk those runs trained from.
    """
    import importlib.util
    import timm

    exp = PKG_ROOT / "mricore_vitb_relabel"
    spec = importlib.util.spec_from_file_location("mricore_train", exp / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    size = cfg["input"]["size"]
    m = timm.create_model(skeleton, pretrained=False, num_classes=0, img_size=size)
    mcfg = sc.load_config(exp, verbose=False)
    mcfg = sc.remote_cfg(mcfg) if Path("/root/runs").exists() else mcfg
    mod._load_mri_core(m, mod._weights_path(mcfg), size // m.patch_embed.patch_size[0])
    return m


def _features(m, batches, mu, sd, per_study, device):
    """(n_studies, dim) frozen study embeddings: encode every slice, mean over slices."""
    import torch

    feats = []
    with torch.no_grad():
        for x in batches:
            # mirror prepare_batch exactly: uint8 -> [0,1] -> this model's own statistics
            v = x.to(device).float().div_(255.0)
            b, s, d, h, w = v.shape
            v = v.view(b * s * d, 1, h, w).expand(-1, 3, -1, -1)
            v = (v - mu) / sd
            out = []
            for i in range(0, v.shape[0], 96):
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    out.append(m(v[i:i + 96]).float())
            feats.append(torch.cat(out).view(b, per_study, -1).mean(1).cpu())
    return torch.cat(feats).numpy()


def _probe(cfg, models, n_studies, batch_size):
    """Extract frozen features for every model, then linear-probe each one."""
    import numpy as np
    import timm
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = cfg["tasks"]

    # ONE loader, shared by every backbone, so all of them see identical pixels. The
    # dataset returns uint8; each model's own normalisation is applied per model below.
    cfg = dict(cfg)
    cfg["dataloader"] = dict(cfg["dataloader"], val_batch_size=batch_size, val_num_workers=8)
    cfg["augmentation"] = dict(cfg["augmentation"], enable=False, eval_enable=False)
    loaders = sc.make_loaders(cfg)
    loader = loaders["val"]

    batches, ys, seen = [], [], 0
    for x, y in loader:
        batches.append(x)
        ys.append(y)
        seen += y.shape[0]
        if seen >= n_studies:
            break
    Y = torch.cat(ys).numpy()
    print(f"[probe] {Y.shape[0]} studies x {len(tasks)} findings, "
          f"{sum(b.shape[0] for b in batches)} batches held in memory", flush=True)

    per_study = len(cfg["input"]["slots"]) * cfg["input"]["n_slices"]
    results = {}
    for name, mean, std, note in models:
        # one backbone failing to build must not cost the whole comparison: SwinV2 in
        # particular refuses input sizes its window partitioning does not divide
        try:
            if name.startswith("MRI-CORE:"):
                m = _mri_core(name.split(":", 1)[1], cfg)
            else:
                try:
                    m = timm.create_model(name, pretrained=True, num_classes=0,
                                          img_size=cfg["input"]["size"])
                except TypeError:               # convnext and friends take no img_size
                    m = timm.create_model(name, pretrained=True, num_classes=0)
        except Exception as e:
            print(f"[probe] SKIPPED {name}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            continue
        m = m.eval().to(device)
        mu = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        sd = torch.tensor(std, device=device).view(1, 3, 1, 1)

        try:
            F = _features(m, batches, mu, sd, per_study, device)
        except Exception as e:
            print(f"[probe] SKIPPED {name}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            del m
            torch.cuda.empty_cache()
            continue
        del m
        torch.cuda.empty_cache()

        # standardise per dimension: a linear probe should not be handed a scale advantage
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
        aucs = []
        for j, t in enumerate(tasks):
            yb = (Y[:, j] > 0.5).astype(int)
            if yb.sum() < 8 or (1 - yb).sum() < 8:
                continue
            oof = np.zeros(len(yb))
            for tr, te in StratifiedKFold(4, shuffle=True, random_state=0).split(F, yb):
                clf = LogisticRegression(max_iter=2000, C=0.1)
                clf.fit(F[tr], yb[tr])
                oof[te] = clf.predict_proba(F[te])[:, 1]
            aucs.append(roc_auc_score(yb, oof))
        macro = float(np.mean(aucs))
        results[name] = macro
        print(f"[probe] {macro:.4f}  {name:<52} dim={F.shape[1]:<5} {note}", flush=True)

    print("\n" + "=" * 78)
    print("FROZEN LINEAR PROBE, macro AUROC over %d studies (cross-validated)" % Y.shape[0])
    for name, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print("  %.4f  %s" % (v, name))
    print("=" * 78)
    print("Reading it: a backbone near 0.5-0.6 has no linearly available knee signal and a")
    print("learning rate will not fix that. One close to the reference has usable features,")
    print("so a stalled run is an OPTIMISATION problem, not a representation problem.")
    return results


try:
    import modal

    cfg = sc.load_config(EXP_DIR, verbose=False)
    app = modal.App("frozen-probe")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={cfg["modal"]["shards_mount"]: _shards,
                           cfg["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(cfg))
    def probe_remote(models: str, n_studies: int, batch_size: int):
        rcfg = sc.remote_cfg(cfg)
        chosen = DEFAULT_MODELS
        if models:
            want = [s.strip() for s in models.split(",") if s.strip()]
            chosen = [m for m in DEFAULT_MODELS if m[0] in want]
            missing = [w for w in want if not any(m[0] == w for m in chosen)]
            chosen += [(w, *HALF, "requested") for w in missing]
        _probe(rcfg, chosen, n_studies, batch_size)

    @app.local_entrypoint()
    def modal_main(models: str = "", n_studies: int = 300, batch_size: int = 4):
        probe_remote.remote(models, n_studies, batch_size)

except ImportError:
    pass
