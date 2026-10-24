"""upload_all.py - write both repos' front pages + ensemble recipes, then upload every exported model.

    python hf_release/export_all.py     # first: builds hf_release/out/<run>/ for every model
    python hf_release/upload_all.py     # then: uploads them

Two repos, because the DINOv3 fine-tunes must be distributed under the DINOv3 License:
    mamounyosef/knee-abnormality-detection          everything else, CC BY-NC 4.0
    mamounyosef/knee-abnormality-detection-dinov3   the 2 DINOv3 models, DINOv3 License
Unchanged files are skipped by Hugging Face (matched by hash), so re-running only uploads what changed.
"""
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
from catalog import ENS4, ENS14, GITHUB, MODELS, REPO, REPO_DINO, repo_of   # noqa: E402

FRONT = OUT / "_front"           # staging for the two front pages and the ensemble recipes


def metrics(run):
    return json.loads((OUT / run / "metrics.json").read_text(encoding="utf-8"))


def link(m):
    return f"[`{m['run']}`](https://huggingface.co/{repo_of(m)}/tree/main/{m['run']})"


def row(m):
    x = metrics(m["run"])
    val = x.get("val_macro_auroc", "-") if m["labels"] != "v2_full" else "full data"
    audit = x.get("audit58_macro_auroc", "-") if m["labels"] != "v2_full" else "full data"
    solo = f"{m['solo'][0]:.3f} / {m['solo'][1]:.3f}" if "solo" in m else "-"
    return f"| {link(m)} | {m['about'].split('. ')[0].rstrip('.')} | {m['labels'].replace('_full', ' (all data)')} | {val} | {audit} | {solo} |"


HEAD = "| Model | What | Labels | Val | Audit-58 | Alone: public / private |\n| --- | --- | --- | --- | --- | --- |"


def ensembles():
    """The two submitted ensembles as JSON recipes (members, weights, how they are combined)."""
    e4 = [m for m in MODELS if ENS4 in m.get("ensembles", [])]
    e14 = [m for m in MODELS if ENS14 in m.get("ensembles", [])]
    w4 = {"convnext_base_v5_96slices_per_finding_attn": 0.42,
          "medimageinsight_davit_huge_v5_96slices_per_finding_attn_sam": 0.18,
          "dinov3_convnext_base_v5": 0.20, "swinv2_base_in22k_v5_warm": 0.20}
    out = {
        "ensemble_4.json": {"name": ENS4[0], "combine": "weighted mean of logits, same weights for all 12 findings",
                            "kaggle_public": ENS4[2], "kaggle_private": ENS4[3],
                            "members": [{"model": m["run"], "repo": repo_of(m), "weight": w4[m["run"]]} for m in e4]},
        "ensemble_14.json": {"name": ENS14[0], "combine": "plain mean of logits",
                             "kaggle_public": ENS14[2], "kaggle_private": ENS14[3],
                             "members": [{"model": m["run"], "repo": repo_of(m), "weight": round(1 / 14, 6)} for m in e14]},
    }
    return out, e4, e14


def front_main(e4, e14):
    main = [m for m in MODELS if not m["dinov3"]]
    groups = [
        ("4-model ensemble (public 0.942 / private 0.927)", e4,
         "Weighted logit mean 0.42 / 0.18 / 0.20 / 0.20 (`ensembles/ensemble_4.json`)."),
        ("14-model ensemble (public 0.942 / private 0.928, the best score)", e14,
         "Plain logit mean of all 14 (`ensembles/ensemble_14.json`)."),
        ("PerFindingAttn-96 folds (public 0.937 / private 0.921 together)",
         [m for m in main if "96slices_per_finding_attn_f" in m["run"]],
         "5-fold cross-validation of the best recipe. Their out-of-fold predictions are the teacher behind the "
         "final (w50r10) labels."),
        ("ConvNeXt label teachers (public 0.918 / private 0.907 together)",
         [m for m in main if m["run"].startswith("convnext_base_25d_18slices_f")],
         "5-fold 2.5D ConvNeXt models on v2 labels. Their out-of-fold predictions, with the MedImageInsight folds, "
         "built the v4 labels."),
    ]
    sections = []
    for title, ms, note in groups:
        sections.append(f"### {title}\n\n{note}\n\n{HEAD}\n" + "\n".join(row(m) for m in ms))
    dino_note = (f"The 2 DINOv3 models (`dinov3_convnext_base`, `dinov3_convnext_base_v5`) are in "
                 f"[{REPO_DINO}](https://huggingface.co/{REPO_DINO}), because DINOv3 fine-tunes must be "
                 "distributed under the DINOv3 License.")
    return f"""---
license: cc-by-nc-4.0
language:
  - en
library_name: pytorch
pipeline_tag: image-classification
base_model:
  - timm/convnext_base.fb_in22k_ft_in1k
  - timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k
metrics:
  - roc_auc
tags:
  - medical-imaging
  - mri
  - knee
  - multi-label-classification
  - kaggle
  - rsna
model-index:
  - name: {e4[0]['run']}
    results:
      - task:
          type: image-classification
          name: Multi-label knee MRI abnormality detection
        dataset:
          type: rsna-knee-abnormality-detection
          name: RSNA Knee Abnormality Detection (Kaggle private test set)
        metrics:
          - type: roc_auc
            name: Macro AUROC (private leaderboard)
            value: {e4[0]['solo'][1]}
      - task:
          type: image-classification
          name: Multi-label knee MRI abnormality detection
        dataset:
          type: rsna-knee-abnormality-detection
          name: RSNA Knee Abnormality Detection (Kaggle public test set)
        metrics:
          - type: roc_auc
            name: Macro AUROC (public leaderboard)
            value: {e4[0]['solo'][0]}
---

# Knee Abnormality Detection (RSNA, Kaggle)

Trained models from my solution to the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle
competition. Each model takes one knee MRI study and scores 12 abnormalities: ACL and MCL tears, medial and lateral
meniscus tears, medial, lateral and patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, bone
contusion and fracture. Code, training recipes and the full write-up:
[GitHub]({GITHUB}).

**Best results** (macro AUROC on Kaggle's hidden test set): the 14-model ensemble scored **0.942 public /
0.928 private**; the 4-model ensemble 0.942 / 0.927; the best single model
(`convnext_base_v5_96slices_per_finding_attn`) 0.939 / 0.922.

## How it was trained, in short

- Only 58 of 4,407 training studies have expert image labels, so training labels were built from each study's
  radiology report (9 languages, translated, then scored by an LLM), refined with MedGemma image opinions and
  then with out-of-fold predictions of earlier image models ("teachers"). Label versions:
  **v2** report labels; **v3** = v2 + MedGemma on 5 findings; **v4** = v3 blended 50/50 with teacher predictions.
- Models read several MRI series (sagittal, coronal and axial; plain and fluid-sensitive) as 2D images per slice,
  2.5D images of 3 slices, or bags of 3-slice windows, and pool them into 12 scores.
- "All data" models were retrained on every labelled study, so they have no validation score.

## Models

Each folder holds one model: `model.safetensors`, `config.json`, a standalone `modeling.py`, `metrics.json`, its
training log, and a model card with every detail. Val = validation macro AUROC on the held-out split (report
labels); Audit-58 = macro AUROC on the 58 expert-labelled studies.

{dino_note}

""" + "\n\n".join(sections) + f"""

## Using a model

```python
from huggingface_hub import snapshot_download
import sys

name = "convnext_base_v5_96slices_per_finding_attn"
folder = snapshot_download("{REPO}", allow_patterns=f"{{name}}/*") + "/" + name
sys.path.insert(0, folder)
from modeling import load_model

model = load_model(folder)   # each model card gives its exact input shape
```

The DICOM-to-tensor preprocessing lives in the GitHub repository (`runs/shared_code.py`).

## Licence and intended use

- Weights: **CC BY-NC 4.0** (non-commercial use, credit required). Backbones keep their own licences, listed in
  each model card: ConvNeXt (Apache 2.0), SwinV2 (MIT), MRI-CORE (Apache 2.0), MedImageInsight (MIT).
- Trained on RSNA competition data, licensed for research use only
  ([RSNA MIRA terms](https://www.rsna.org/mira-license)). No competition images, labels or reports are included.
- **Research use only. Not a medical device and not for clinical use.**
- Thanks to RSNA and Kaggle for the dataset, and to the authors of the pretrained backbones.
"""


def front_dino():
    ms = [m for m in MODELS if m["dinov3"]]
    return f"""---
license: other
license_name: dinov3-license
license_link: https://huggingface.co/{REPO_DINO}/blob/main/LICENSE.md
language:
  - en
library_name: pytorch
pipeline_tag: image-classification
base_model:
  - timm/convnext_base.dinov3_lvd1689m
metrics:
  - roc_auc
tags:
  - medical-imaging
  - mri
  - knee
  - multi-label-classification
  - kaggle
  - rsna
  - dinov3
---

# Knee Abnormality Detection: DINOv3 models (RSNA, Kaggle)

**Built with DINOv3.** The two DINOv3 ConvNeXt-Base fine-tunes from my solution to the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle
competition. They are kept apart from [the main model repo](https://huggingface.co/{REPO}) because fine-tuned
DINOv3 weights must be distributed under the DINOv3 License (copy in `LICENSE.md`).

{HEAD}
""" + "\n".join(row(m) for m in ms) + f"""

- `dinov3_convnext_base` is a member of the 14-model ensemble (public 0.942 / private 0.928).
- `dinov3_convnext_base_v5` is a member of the 4-model ensemble (public 0.942 / private 0.927).

Each folder has the weights, config, a standalone `modeling.py`, metrics, the training log and a full model card.
Code and write-up: [GitHub]({GITHUB}).

## Licence and intended use

- Weights: **DINOv3 License** (`LICENSE.md`). Acknowledge DINOv3 in any publication that uses them.
- Trained on RSNA competition data, licensed for research use only
  ([RSNA MIRA terms](https://www.rsna.org/mira-license)). No competition images, labels or reports are included.
- **Research use only. Not a medical device and not for clinical use.**
"""


def main():
    api = HfApi()
    print("logged in as", api.whoami()["name"])
    missing = [m["run"] for m in MODELS if not (OUT / m["run"] / "model.safetensors").exists()]
    if missing:
        raise SystemExit(f"not exported yet: {missing}")

    recipes, e4, e14 = ensembles()
    for repo in (REPO, REPO_DINO):
        d = FRONT / repo.split("/")[1]
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    main_d, dino_d = FRONT / REPO.split("/")[1], FRONT / REPO_DINO.split("/")[1]
    (main_d / "README.md").write_text(front_main(e4, e14), encoding="utf-8")
    (main_d / "ensembles").mkdir()
    for name, r in recipes.items():
        (main_d / "ensembles" / name).write_text(json.dumps(r, indent=2), encoding="utf-8")
    (dino_d / "README.md").write_text(front_dino(), encoding="utf-8")
    shutil.copy(OUT / "_download" / "knee-ten-mii-5folds" / "DINOV3_LICENSE.md", dino_d / "LICENSE.md")

    api.create_repo(REPO_DINO, repo_type="model", exist_ok=True)   # public, like the main repo
    for m in MODELS:
        print("uploading", m["run"], "->", repo_of(m))
        api.upload_folder(folder_path=str(OUT / m["run"]), path_in_repo=m["run"], repo_id=repo_of(m),
                          repo_type="model", commit_message=f"Add {m['run']}",
                          ignore_patterns=["__pycache__/*", "*.pyc"])
    api.upload_folder(folder_path=str(main_d), repo_id=REPO, repo_type="model", commit_message="Front page and ensembles")
    api.upload_folder(folder_path=str(dino_d), repo_id=REPO_DINO, repo_type="model", commit_message="Front page and licence")
    print(f"done: https://huggingface.co/{REPO}  and  https://huggingface.co/{REPO_DINO}")


if __name__ == "__main__":
    main()
