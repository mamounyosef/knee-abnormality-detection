# convnext_base_v5_96slices_per_finding_attn

The best single model from the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition solution ([code on GitHub](https://github.com/mamounyosef/knee-abnormality-detection)).
Given one knee MRI study, it outputs a score for each of 12 abnormalities.

| | Score (macro AUROC) |
| --- | --- |
| Kaggle public leaderboard | **0.939** |
| Kaggle private leaderboard | **0.922** |
| Validation (346 studies, report labels) | 0.9213 |
| Audit-58 (58 studies, expert panel labels) | 0.9272 |

AUROC is the area under the ROC curve: 1.0 is a perfect ranking of positive over negative studies, 0.5 is chance.
"Macro" means the average over the 12 findings.

## What it predicts

12 logits, in this order (apply a sigmoid for probabilities):
ACL tear, MCL tear, Medial Meniscus tear, Lateral Meniscus tear, Medial OA, Lateral OA, PF OA (patellofemoral
osteoarthritis), Effusion, Synovitis, Baker's cyst, Contusion (bone bruise), Fracture.

## Architecture

- **Backbone:** ConvNeXt-Base (`timm`: `convnext_base.fb_in22k_ft_in1k`), 88M parameters.
- **Input:** 96 slices from 5 MRI series: sagittal fluid-sensitive (26), sagittal plain (22), coronal fluid (18),
  coronal plain (12), axial fluid (18). Each series is resampled to 0.365 mm per pixel, 384 x 384 px (a 140 mm
  field), using the middle 6% to 94% of its slices. Left knees are mirrored to look like right knees.
- **Windows:** the 96 slices are joined into one stack and read as overlapping 3-slice windows (the 3 slices fill
  the 3 colour channels). All 94 windows are used at inference, 12 random ones in training.
- **Head:** per-finding attention. For each finding, a small network scores every window, a softmax turns the
  scores into weights, and the weighted mean of the window features is read out by a per-finding linear layer.
  Each finding can focus on the slices that show it.

## Training

- **Labels:** only 58 of 4,407 training studies have expert image labels. The rest use soft labels built from each
  study's radiology report (translated, then scored by an LLM), averaged with MedGemma image opinions on 5 findings,
  then blended 50/50 with out-of-fold predictions of earlier image models ("v4" labels).
- **Optimiser:** AdamW, weight decay 0.059. Learning rate 4e-5 for the backbone and 2e-3 for the head, cosine
  schedule, 5% warmup, gradient clipping at 3. 16 epochs plus a 4-epoch low-learning-rate extension.
- **Batch:** 8 studies (96 windows) per step. Mixed precision (PyTorch AMP).
- **Checkpoint:** step 9000 of 9750 (best validation). Trained on an RTX 4060, then an H100.

Full settings are in `config.json`; the full validation curve is in `val_history.csv`.

## How to use

```python
from huggingface_hub import snapshot_download
import sys

folder = snapshot_download("mamounyosef/knee-abnormality-detection",
                           allow_patterns="convnext_base_v5_96slices_per_finding_attn/*")
folder += "/convnext_base_v5_96slices_per_finding_attn"
sys.path.insert(0, folder)
from modeling import load_model

model = load_model(folder)          # eval mode, expects 94 windows per study
# x: (n_studies * 94, 3, 384, 384), ImageNet-normalised windows, in stack order
```

The DICOM-to-window preprocessing lives in the GitHub repository (`runs/shared_code.py`).

## Files

| File | What |
| --- | --- |
| `model.safetensors` | weights |
| `config.json` | model, input and training settings |
| `modeling.py` | standalone model code (needs `torch`, `timm`, `safetensors`) |
| `metrics.json` | step, validation, audit and leaderboard scores, per-finding AUROC |
| `val_history.csv` | every validation run during training |

## Limitations

- **Research use only. Not a medical device and not for clinical use.**
- Trained and evaluated on one competition dataset; performance on other scanners, protocols or populations is
  unknown.
- Most training labels come from radiology reports, not from images, so they carry report-level noise.

## Licence and credits

- **Weights:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): free for non-commercial use with
  credit. Trained on RSNA competition data, which is licensed for research use only
  ([RSNA MIRA terms](https://www.rsna.org/mira-license)). No competition images, labels or reports are included.
- **Backbone:** ConvNeXt-Base ImageNet weights via `timm` (Apache 2.0).
- **Recipe:** the input layout and attention head follow a public Kaggle notebook by dreaddevelopment.
- Thanks to RSNA and Kaggle for the dataset.
