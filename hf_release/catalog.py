"""catalog.py - every released model (where its weights come from, its scores, its story) and the card writer."""

K14 = "mamounhussam/knee-ten-mii-5folds"     # the 14-model ensemble as submitted (public 0.942 / private 0.928)
K4 = "mamounhussam/knee-raptor4-folds"       # the 4-model ensemble + the 5 per-finding-attention folds
GITHUB = "https://github.com/mamounyosef/knee-abnormality-detection"
REPO = "mamounyosef/knee-abnormality-detection"
REPO_DINO = "mamounyosef/knee-abnormality-detection-dinov3"

ENS14 = ("14-model ensemble", "flat mean of the 14 members' logits", 0.942, 0.928)
ENS4 = ("4-model ensemble", "weighted logit mean: 0.42 / 0.18 / 0.20 / 0.20", 0.942, 0.927)

LABELS = {
    "v2": "**v2** report labels: each radiology report translated to English, then scored 0-1 per finding by "
          "Qwen3.8-Max (an LLM). The 58 expert-labelled studies keep their panel labels.",
    "v2_full": "**v2** report labels on **all 4,400 labelled studies** (train + validation + the 58 audit "
               "studies with their expert labels). There is no held-out set, so this model has no honest "
               "validation score; it trained for a fixed schedule and the last checkpoint is used.",
    "v3": "**v3** labels: v2 averaged 50/50 (in logit space) with MedGemma image opinions on Effusion, "
          "Synovitis, Baker's cyst, Lateral Meniscus and Lateral OA.",
    "v4": "**v4** (\"w50\") labels: v3 blended 50/50 with out-of-fold predictions of 5-fold ConvNeXt and "
          "5-fold MedImageInsight teacher models.",
}

BACKBONES = {
    "convnext_base.fb_in22k_ft_in1k": dict(
        text="ConvNeXt-Base (Meta), ImageNet-22k pretrained and ImageNet-1k fine-tuned, via `timm`",
        base="timm/convnext_base.fb_in22k_ft_in1k", licence="Apache 2.0"),
    "convnext_base.dinov3_lvd1689m": dict(
        text="ConvNeXt-Base distilled from DINOv3 (Meta, self-supervised on 1.7B images, LVD-1689M), via `timm`",
        base="timm/convnext_base.dinov3_lvd1689m", licence="DINOv3 License"),
    "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k": dict(
        text="SwinV2-Base (Microsoft), ImageNet-22k pretrained and ImageNet-1k fine-tuned, via `timm`, "
             "run at 512 px with 16-token windows",
        base="timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k", licence="MIT"),
    "vit_base_patch16_224": dict(
        text="[MRI-CORE](https://github.com/mazurowski-lab/mri_foundation) ViT-B/16 (Duke, self-supervised on "
             "6M MRI slices from 110k volumes), loaded into a `timm` ViT at 512 px",
        base=None, licence="Apache 2.0"),
    "davit_huge_fl": dict(
        text="MedImageInsight DaViT-Huge image encoder (Microsoft, trained on ~3.8M medical images with their "
             "clinical labels), loaded into a `timm` DaViT",
        base=None, licence="MIT"),
}

SLOT_NAMES = {"sag_plain": "sagittal plain", "sag_fluid": "sagittal fluid-sensitive",
              "cor_plain": "coronal plain", "cor_fluid": "coronal fluid-sensitive",
              "ax_plain": "axial plain", "ax_fluid": "axial fluid-sensitive"}

ARCH = {
    "slice": "**Per-slice model.** Every slice of every series is its own image (the grey value copied into "
             "3 channels). The backbone turns each image into a feature vector, the vectors of one study are "
             "{pool} into one, and a linear head gives the 12 logits.",
    "25d": "**2.5D model.** Each series' slices are split into groups of {ch} consecutive slices, and each group "
           "becomes one image with those slices as its {ch} channels, so the backbone sees a little depth. The "
           "image features of one study are {pool} into one vector, and a linear head gives the 12 logits.",
    "mricore": "**Per-slice ViT with cross-slice attention.** Every slice is its own image. Inside the ViT, 3 extra "
               "attention blocks (after blocks 3, 6 and 9) let each token position attend across all {n} slices "
               "of the study, so slices can share information before pooling. The slice features are averaged, "
               "normalised, and a linear head gives the 12 logits.",
    "perfinding": "**2.5D model with per-finding views and pooling.** 6 series, each cut into 3-slice images. Each "
                  "finding reads only the views where it is visible (for example ACL: sagittal only; PF OA: axial "
                  "and sagittal plain) and pools them its own way (ACL and MCL: mean of the top-2 windows; the "
                  "rest: mean), then has its own normalisation and its own 1-output head.",
    "per_finding_attn": "**Per-finding attention over a bag of windows.** The {n} slices are joined into one stack "
                        "and read as overlapping 3-slice windows (all 94 at inference, 12 random ones in training). "
                        "For each finding, a small network scores every window, a softmax turns the scores into "
                        "weights, and the weighted mean of the window features is read out by that finding's own "
                        "linear layer.",
}


def _m(run, src, step, family, labels, about, **kw):
    return dict(run=run, src=src, step=step, family=family, labels=labels, about=about, **kw)


def _mii_fold(k, step):
    return _m(f"medimageinsight_davit_huge_336_v3_f{k}", (K14, f"m{9 + k}"), step, "25d", "v3",
              f"MedImageInsight DaViT-Huge, 2.5D at 336 px, fold {k} of 5 (v3 labels). One of the 5 folds that "
              "are in the 14-model ensemble; their out-of-fold predictions are also one of the two teachers "
              "behind the v4 labels.",
              ensembles=[ENS14], group=("5 folds together", 0.936, 0.921))


def _pfa_fold(k, member, step):
    return _m(f"convnext_base_v5_96slices_per_finding_attn_f{k}", (K4, member), step, "per_finding_attn", "v4",
              f"Fold {k} of 5 of the per-finding-attention ConvNeXt (v4 labels, round-2 folds). Their "
              "out-of-fold predictions are the teacher behind the final w50r10 labels.",
              group=("5 folds together", 0.937, 0.921))


def _teacher(k):
    return _m(f"convnext_base_25d_18slices_f{k}", ("backup", f"runs/convnext_base_25d_18slices_f{k}/results/checkpoints/best.pt"),
              None, "25d", "v2",
              f"2.5D ConvNeXt-Base, fold {k} of 5 (v2 labels). A label teacher: its out-of-fold predictions "
              "are one of the two teachers behind the v4 labels.",
              group=("5 folds together", 0.918, 0.907))


MODELS = [
    _m("convnext_base_v5_96slices_per_finding_attn", (K4, "m0"), 9000, "per_finding_attn", "v4",
       "The best single model: ConvNeXt-Base over 96 slices with a per-finding attention head.",
       ensembles=[ENS4], solo=(0.939, 0.922)),
    _m("medimageinsight_davit_huge_v5_96slices_per_finding_attn_sam", (K4, "m1"), 5300, "per_finding_attn", "v4",
       "The per-finding-attention recipe on MedImageInsight DaViT-Huge, trained with the SAM optimiser.",
       ensembles=[ENS4], solo=(0.937, 0.925)),
    _m("dinov3_convnext_base_v5", (K4, "m2"), 5800, "slice", "v4",
       "DINOv3 ConvNeXt-Base, per-slice at 512 px, on v4 labels.", ensembles=[ENS4]),
    _m("swinv2_base_in22k_v5_warm", (K4, "m3"), 2000, "slice", "v4",
       "SwinV2-Base, per-slice at 512 px. Warm-started from `swinv2_base_in22k` and fine-tuned for 6 epochs on "
       "v4 labels (97% of the data; the held-out 121 studies are its validation set).",
       ensembles=[ENS4], solo=(0.934, 0.920)),
    _m("convnext_base_full_all_knees_same_orientation", (K14, "m0"), 4400, "slice", "v2_full",
       "ConvNeXt-Base, per-slice at 512 px, trained on all labelled data.", ensembles=[ENS14]),
    _m("mricore_vitb_cross_attention", (K14, "m1"), 5040, "mricore", "v2",
       "MRI-CORE ViT-B/16 (DINOv2-style weights) with cross-slice attention.", ensembles=[ENS14]),
    _m("mricore_vitb_mae_cross_attention_fulldata_flip", (K14, "m2"), 6050, "mricore", "v2_full",
       "MRI-CORE ViT-B/16 (MAE weights) with cross-slice attention, trained on all labelled data.",
       ensembles=[ENS14]),
    _m("dinov3_convnext_base", (K14, "m3"), 5000, "slice", "v2",
       "DINOv3 ConvNeXt-Base, per-slice at 512 px, on v2 labels.", ensembles=[ENS14]),
    _m("convnext_base_25d_all_knees_same_orientation_fulldata", (K14, "m4"), 6600, "25d", "v2_full",
       "ConvNeXt-Base, 2.5D at 336 px, trained on all labelled data.", ensembles=[ENS14]),
    _m("swinv2_base_in22k", (K14, "m5"), 4500, "slice", "v2",
       "SwinV2-Base, per-slice at 512 px, on v2 labels.", ensembles=[ENS14]),
    _m("convnext_base_25d_poolattn_fulldata_flip", (K14, "m6"), 3850, "25d", "v2_full",
       "ConvNeXt-Base, 2.5D at 336 px with attention pooling, trained on all labelled data.", ensembles=[ENS14]),
    _m("convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip", (K14, "m7"), 3300, "perfinding", "v2_full",
       "ConvNeXt-Base, 2.5D over 6 series, where each finding reads its own views and pools them its own way; "
       "trained on all labelled data.", ensembles=[ENS14]),
    _m("medimageinsight_davit_huge_perslice_480_warm", (K14, "m8"), 600, "slice", "v2",
       "MedImageInsight DaViT-Huge, per-slice at its native 480 px. Warm-started from our 2.5D 336 px "
       "MedImageInsight fine-tune.", ensembles=[ENS14]),
] + [_mii_fold(k, s) for k, s in enumerate([2700, 3500, 3400, 2700, 3200])] \
  + [_pfa_fold(k, f"m{4 + k}", s) for k, s in enumerate([7000, 6960, 6250, 6928, 5250])] \
  + [_teacher(k) for k in range(5)]


def repo_of(m):
    return REPO_DINO if m.get("dinov3") else REPO


for _x in MODELS:
    _x["dinov3"] = "dinov3" in _x["run"]


# ---------------------------------------------------------------- the card

def _input_lines(cfg):
    i = cfg["input"]
    slots = i["slots"]
    n = i["n_slices"]
    g = i.get("slices_as_channels")
    per = lambda d, s: int(d[s]) if isinstance(d, dict) else (int(d) if d else 0)
    total = sum(per(n, s) for s in slots)
    rows = []
    for s in slots:
        ns, gs = per(n, s), per(g, s)
        rows.append(f"{SLOT_NAMES.get(s, s)} ({ns})")
    out = [f"- **Series ({len(slots)}):** " + ", ".join(rows) + f"; {total} slices per study. "
           "Fluid-sensitive series are fat-suppressed, so fluid and swelling look bright.",
           f"- **Resolution:** each slice resampled to {i['mm_per_px']} mm per pixel, {i['size']} x {i['size']} px "
           f"({i['mm_per_px'] * i['size']:.0f} mm field of view). Slices are evenly spaced real slices (no "
           f"interpolation between slices) from the middle {100 - 2 * 100 * i['slice_margin']:.0f}% of each series.",
           f"- **Intensity:** windowed per volume to the {i['window_pct'][0]:g}-{i['window_pct'][1]:g} percentile, "
           f"stored as 0-255, then normalised with mean {i['norm_mean']} and std {i['norm_std']}.",
           "- **Orientation:** left knees are mirrored so every knee looks like a right knee."
           if i.get("laterality_flip") else "- **Orientation:** knees are used as acquired (no left/right mirroring)."]
    return out, total


def _images_per_study(cfg):
    i = cfg["input"]
    slots, n, g = i["slots"], i["n_slices"], i.get("slices_as_channels")
    per = lambda d, s: int(d[s]) if isinstance(d, dict) else (int(d) if d else 0)
    if i.get("window_bag"):
        return int(i["window_bag"]["eval_k"]), 3
    imgs = sum(per(g, s) or per(n, s) for s in slots)
    ch = (per(n, slots[0]) // per(g, slots[0])) if g else 3
    return imgs, ch


def _training_lines(m, cfg):
    t = cfg["training"]
    o = t["optimizer"]
    head_lr = o["lr"] / o["backbone_lr_mult"] if o.get("backbone_lr_mult") else o["lr"]
    early = bool(t.get("early_stopping", {}).get("enable")) and m["labels"] != "v2_full"
    lines = [f"- **Labels:** {LABELS[m['labels']]}",
             f"- **Optimiser:** AdamW, weight decay {o['weight_decay']}, learning rate {o['lr']:g} for the "
             f"backbone and {head_lr:g} for the head" + (f", layer decay {o['layer_decay']}" if o.get("layer_decay") else "")
             + f"; {t['scheduler']['name']} schedule with {t['scheduler']['warmup_frac']:.0%} warmup; gradient "
             f"clipping at {t['grad_clip']}.",
             f"- **Loss:** binary cross-entropy on soft (0-1) targets.",
             f"- **Schedule:** {'up to ' if early else ''}{t['epochs']} epochs" +
             (f", plus a {t['extend']['epochs']}-epoch low-learning-rate extension" if t.get("extend", {}).get("enable") else "")
             + (", early stopping on validation macro AUROC" if early else "")
             + "."]
    if t.get("sam", {}).get("enable"):
        lines.append(f"- **SAM** (sharpness-aware minimisation) with rho {t['sam']['rho']}.")
    if "dataloader" in cfg:
        d = cfg["dataloader"]
        lines.append(f"- **Batch:** {d['batch_size'] * d.get('accum_steps', 1)} studies per optimiser step. "
                     "Mixed precision (PyTorch AMP).")
    aug = cfg.get("augmentation", {})
    used = [k for k, v in aug.items() if isinstance(v, dict) and v.get("p", 0) > 0]
    if aug.get("enable") and used:
        lines.append("- **Augmentation:** " + ", ".join(k.replace("_", " ") for k in used) + ".")
    return lines


def card_for(m, cfg, metrics, hist_name):
    run = m["run"]
    bb = BACKBONES[cfg["model"]["name"]]
    imgs, ch = _images_per_study(cfg)
    inp, total = _input_lines(cfg)
    pool = {"attn": "combined by learned attention weights", "max": "max-pooled"}.get(cfg["model"].get("pool"), "averaged")
    arch = ARCH[m["family"]].format(pool=pool, ch=ch, n=total)
    repo = repo_of(m)
    lic = "DINOv3 License" if m["dinov3"] else "CC BY-NC 4.0"

    meta = ["---", f"license: {'other' if m['dinov3'] else 'cc-by-nc-4.0'}"]
    if m["dinov3"]:
        meta += ["license_name: dinov3-license",
                 f"license_link: https://huggingface.co/{REPO_DINO}/blob/main/{run}/LICENSE_DINOv3.md"]
    meta += ["library_name: pytorch", "pipeline_tag: image-classification"]
    if bb["base"]:
        meta += ["base_model:", f"  - {bb['base']}"]
    meta += ["tags:", "  - medical-imaging", "  - mri", "  - knee", "  - multi-label-classification", "---", ""]

    rows = []
    if "solo" in m:
        rows.append(f"| This model alone, Kaggle public / private leaderboard | {m['solo'][0]:.3f} / {m['solo'][1]:.3f} |")
    if "group" in m:
        g = m["group"]
        rows.append(f"| {g[0]}, Kaggle public / private leaderboard | {g[1]:.3f} / {g[2]:.3f} |")
    for e in m.get("ensembles", []):
        rows.append(f"| As part of the {e[0]}, Kaggle public / private leaderboard | {e[2]:.3f} / {e[3]:.3f} |")
    if metrics.get("val_macro_auroc"):
        rows.append(f"| Validation macro AUROC at this checkpoint | {metrics['val_macro_auroc']} |")
    if metrics.get("audit58_macro_auroc"):
        rows.append(f"| Audit-58 macro AUROC (58 studies, expert panel labels) | {metrics['audit58_macro_auroc']} |")
    if m["labels"] == "v2_full":
        rows.append("| Validation | none: trained on every labelled study |")

    ens = ""
    if m.get("ensembles"):
        ens = "\n**Ensembles:** " + "; ".join(f"member of the {e[0]} ({e[1]})" for e in m["ensembles"]) + \
              f". See the [repo front page](https://huggingface.co/{REPO}) for the full member lists.\n"

    use = f'''```python
from huggingface_hub import snapshot_download
import sys

folder = snapshot_download("{repo}", allow_patterns="{run}/*") + "/{run}"
sys.path.insert(0, folder)
from modeling import load_model

model = load_model(folder)   # eval mode
# input: (n_studies * {imgs}, {ch}, {cfg['input']['size']}, {cfg['input']['size']}) float tensor, {imgs} images per study,
# in the series order above; output: (n_studies, 12) logits, sigmoid -> probabilities
```'''

    licence_lines = [
        f"- **Weights:** {lic}."
        + (" These weights are a fine-tune of DINOv3, so they are distributed under the DINOv3 License "
           "(copy in `LICENSE_DINOv3.md`). **Built with DINOv3.**" if m["dinov3"] else
           " Free for non-commercial use with credit."),
        "- **Training data:** RSNA Knee Abnormality Detection competition data, licensed for research use only "
        "([RSNA MIRA terms](https://www.rsna.org/mira-license)). No images, labels or reports are included here.",
        f"- **Backbone:** {bb['text']} ({bb['licence']}).",
    ]
    if cfg["model"]["name"] == "vit_base_patch16_224":
        licence_lines.append("- MRI-CORE: if you use this model, please also cite the MRI-CORE paper (Dong et al., "
                             "see the [MRI-CORE repository](https://github.com/mazurowski-lab/mri_foundation)).")
    if m["family"] == "per_finding_attn":
        licence_lines.append("- The input layout and attention head follow a public Kaggle notebook by dreaddevelopment.")

    files = [("model.safetensors", "weights"), ("config.json", "model, input and training settings"),
             ("modeling.py", "standalone model code (needs `torch`, a recent `timm` (tested with 1.0.27), `safetensors`)"),
             ("metrics.json", "step, scores, per-finding AUROC")]
    files.append((hist_name, "every validation during training" if hist_name.startswith("val")
                  else "the training loss log (no validation: full-data run)"))
    if m["dinov3"]:
        files.append(("LICENSE_DINOv3.md", "the DINOv3 License"))

    return "\n".join(meta) + f"""# {run}

Part of my solution to the [RSNA Knee Abnormality Detection]
(https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle competition
([code on GitHub]({GITHUB})). Given one knee MRI study, it scores 12 abnormalities.

{m['about']}

| | Score |
| --- | --- |
""".replace("]\n(", "](") + "\n".join(rows) + f"""

Scores are macro AUROC (area under the ROC curve, averaged over the 12 findings; 1.0 is perfect, 0.5 is chance).
{ens}
## What it predicts

12 logits, in this order (sigmoid for probabilities): ACL tear, MCL tear, Medial Meniscus tear, Lateral Meniscus
tear, Medial OA, Lateral OA, PF OA (patellofemoral osteoarthritis), Effusion, Synovitis, Baker's cyst, Contusion
(bone bruise), Fracture.

## Architecture

- **Backbone:** {bb['text']}.
- {arch}
- **Images per study:** {imgs}, each {ch} x {cfg['input']['size']} x {cfg['input']['size']}.

## Input

""" + "\n".join(inp) + f"""

The DICOM-to-tensor preprocessing is in the GitHub repository (`runs/shared_code.py`).

## Training

""" + "\n".join(_training_lines(m, cfg)) + f"""
- **Checkpoint:** step {metrics['step']}{f", epoch {metrics['epoch']}" if metrics.get('epoch') else ''}.

## How to use

{use}

## Files

| File | What |
| --- | --- |
""" + "\n".join(f"| `{a}` | {b} |" for a, b in files) + """

## Limitations

- **Research use only. Not a medical device and not for clinical use.**
- Trained and evaluated on one competition dataset; performance on other scanners, protocols or populations is
  unknown.
- Most training labels come from radiology reports, not from images, so they carry report-level noise.

## Licence and credits

""" + "\n".join(licence_lines) + "\n- Thanks to RSNA and Kaggle for the dataset.\n"
