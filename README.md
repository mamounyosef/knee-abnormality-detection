# RSNA Knee Abnormality Detection

My solution to the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition (July to October 2026). Given a knee MRI study, the task is to predict a confidence score for
each of **12 abnormalities**. The metric is the **macro-averaged area under the ROC curve (AUROC)** across all 12.

## Results at a glance

| Submission | Public LB | Private LB | What it is |
| --- | --- | --- | --- |
| **14-model ensemble** | **0.942** | **0.928** | ConvNeXt, DINOv3, SwinV2, MRI-CORE ViT and 5 MedImageInsight folds, flat mean |
| 4-model ensemble | 0.942 | 0.927 | 4 models (2 PerFindingAttn-96 + DINOv3 + SwinV2), weighted logit mean |
| PerFindingAttn-96 (best single model) | 0.939 | 0.922 | One ConvNeXt-Base over 96 slices with a per-finding attention head |

**Trained weights** for 28 models (both ensembles, the folds and the label teachers) are on Hugging Face:
26 in [mamounyosef/knee-abnormality-detection](https://huggingface.co/mamounyosef/knee-abnormality-detection),
and the two DINOv3 fine-tunes in
[mamounyosef/knee-abnormality-detection-dinov3](https://huggingface.co/mamounyosef/knee-abnormality-detection-dinov3)
(DINOv3 License). Each has a model card, a standalone `modeling.py` and its metrics.

Kaggle splits the hidden test set in two: the **public** leaderboard (LB) is shown during the competition, the
**private** one is revealed at the end and decides the final ranking. All scores above come from models trained in
this repository only.

---

## The problem in plain words

A knee MRI study is a set of 3D scans (series) taken from three directions (sagittal, coronal, axial), each with
several contrasts. Every study needs a score for:

| Group | Findings |
| --- | --- |
| Ligaments | ACL tear (anterior cruciate ligament), MCL tear (medial collateral ligament) |
| Menisci (the two cartilage "shock absorbers") | Medial Meniscus, Lateral Meniscus |
| Osteoarthritis (OA, cartilage wear) | Medial OA, Lateral OA, PF OA (patellofemoral: behind the kneecap) |
| Fluid and soft tissue | Effusion (excess joint fluid), Synovitis (inflamed joint lining), Baker's cyst |
| Bone injury | Contusion (bone bruise), Fracture |

It is a **code competition**: inference runs offline inside a Kaggle notebook on 2 x T4 GPUs, within **9 hours** for
about 1,300 hidden studies.

**What makes it hard:** 4,407 training studies, but only **58 have image-level labels** (from an expert panel).
Every other study comes with only its original **radiology report**, in 9 languages (English 39%). Reports are
not available at test time. So most of the work is (1) turning reports into training labels and (2) training image
models on those noisy labels.

## Results

### Milestones (public / private LB)

| Public | Private | What changed |
| --- | --- | --- |
| 0.914 -> 0.918 | 0.900 -> 0.907 | One ConvNeXt -> 5 folds of it |
| 0.916 -> 0.927 | 0.900 -> 0.914 | Same model, LLM report labels -> labels blended 50% with out-of-fold model predictions ("teacher" labels) |
| 0.928 -> 0.932 | 0.915 -> 0.917 | Mixed-architecture ensembles (ConvNeXt, DINOv3, SwinV2, MRI-CORE ViT with cross-slice attention) |
| 0.938 | 0.925 | 10 members, 6 of them retrained on full data (train + validation + audit) |
| **0.942** | **0.928** | 14 members, incl. 5 MedImageInsight folds |
| 0.929 -> 0.939 | 0.911 -> 0.922 | Best single model: MedImageInsight -> **PerFindingAttn-96** ConvNeXt |
| 0.942 | 0.927 | 4-model ensemble (configs/ens4.py): the same score as the 14-model ensemble from only 4 models |

### What mattered most

1. **Labels from reports.** A large language model (LLM) reading the translated reports reaches 0.911 AUROC against
   the panel labels (0.895 with a smaller model). Blending in out-of-fold image-model predictions at 50% raised the
   same model's public score from 0.916 to 0.927 (+0.011).
2. **Architecture diversity over data-split diversity.** 5 folds of one ConvNeXt gave 0.918 (+0.004) over a single
   model. Mixing ConvNeXt, DINOv3, SwinV2, MRI-CORE and MedImageInsight gave 0.932 (+0.018).
3. **More slices, smarter pooling (PerFindingAttn-96).** 96 slices from 5 series, a bag of 3-slice windows and a per-finding
   attention head: 0.929 to 0.939 for the best single model.
4. **Full-data retrains inside the ensemble.** Swapping members for retrains on train + val + audit lifted the
   ensemble from 0.935 to 0.937 to 0.938. (Alone, a retrained model did not always beat its parent.)

### What did not help (measured, then closed)

| Idea | Result |
| --- | --- |
| AUC-M loss + PESG optimiser | best 0.9023 vs 0.9049 baseline (validation) |
| Stacking (logistic regression on member logits) | loses to a constrained weight search out of fold |
| SAM optimiser | +0.0077 on audit, tie on the leaderboard (0.930 vs 0.930) |
| Few-shot prompts, labeller ensembles, self-consistency, re-translation | all inside the labeller noise floor (3.3% of decisions) |
| Greedy ensemble search over 100 candidates | in-sample +0.0069, held out -0.0002 (LB 0.941 vs 0.942) |
| PerFindingAttn-96 5 folds | 0.937, below the single full-split model (0.939); no gain inside the ensemble |
| SWA, mirror test-time augmentation, RandConv, multi-window inputs | flat or negative |

## How it works

```text
reports ──► translate ──► LLM labels (12 soft scores) ──► + out-of-fold teacher predictions ──► training labels
DICOM   ──► per-series resample (0.40 mm/px, 512 px, N real slices, uint8) ──► shards on cloud volumes
shards + labels ──► runs/<experiment> (one shared engine) ──► checkpoints + cached val/audit predictions
cached predictions ──► ensemble search on audit-58 / clean-121 ──► one Kaggle notebook (2 x T4, shared decode)
```

### 1. Labels (`data_code/02` to `28`)

Every report is first translated to English by Qwen3.8-Max (`02_translate_reports.py`), then turned into 12 soft
scores (probabilities) per study. Each version builds on the one before. "Audit-58" is the AUROC of the labels
themselves against the 58 panel-labelled studies:

| Version | Labelled by | How | Audit-58 |
| --- | --- | --- | --- |
| v1 | Qwen3.8-27B (LLM, reports only) | translated report -> 12 probabilities, reasoning effort low | 0.895 |
| v2 | Qwen3.8-Max, 2.4T parameters (LLM, reports only) | same prompt, reasoning effort medium (`03_label_reports.py`) | 0.911 |
| v3 | v2 + MedGemma 1.5 (vision-language model, images) | MedGemma's image opinion averaged 50/50 in logit space on Effusion, Synovitis, Baker's, Lateral Meniscus, Lateral OA (`17`) | 0.931 |
| v4 ("w50") | v3 + two image-model teachers | 50% v3 + 50% out-of-fold predictions of 5-fold ConvNeXt and 5-fold MedImageInsight models (`22`) | 0.945 |

"Out-of-fold" means each study's prediction comes from the one fold model that never trained on it. The 58
panel-labelled studies always keep their panel labels. On the leaderboard, the same model scored 0.914 on v2,
0.916 on v3 and 0.927 on v4 (0.922 with only a 30% teacher weight).

### 2. Preprocessing (`data_code/07`, `modal_scripts/modal_preprocess.py`)

Every series is resampled in-plane to a fixed spacing (no interpolation through the slices: N evenly spaced real
slices are picked), windowed per volume to uint8 and stored as one shard per study. Left knees are mirrored so
"medial" is always on the same side of the image.

### 3. Training (`runs/`)

- `runs/shared_code.py` is the single engine: dataset, augmentations, models, train loop, metrics, checkpointing,
  logging, cloud glue and the Kaggle inference path.
- `runs/shared_config.yaml` holds the shared defaults. Each `runs/<experiment>/config.yaml` holds only what
  differs, deep-merged on top. `train.py` stays thin.
- Training ran on [Modal](https://modal.com) (H100 / H200), with an RTX 4060 for small local runs. A config's
  `local:` block swaps in PC-sized settings automatically, so a run can move between the two without edits.

**PerFindingAttn-96, the strongest recipe:** 5 series slots (sagittal fluid + plain, coronal fluid + plain, axial fluid),
96 slices, 384 px at 0.365 mm/px, a ConvNeXt-Base (or MedImageInsight DaViT-Huge) trunk applied to 3-slice
windows (12 random windows in training, all 94 at evaluation), and a per-finding attention head over the windows.
The input recipe and head design follow a public Kaggle notebook by dreaddevelopment; the backbones, labels,
training and inference are ours. Run folders: `runs/*_96slices_per_finding_attn*`.

### 4. Validation

The leaderboard is graded by panel labels, so model choices were scored on two surfaces no member trained on:

- **audit-58**: the 58 panel-labelled studies;
- **clean-121**: audit-58 plus 63 held-out studies with report labels.

Paired bootstrap over studies gives the probability that a change is real. A held-out cross-fit checks searches
for overfitting.

### 5. Inference (`runs/others/ensembling/`)

`submit_ensemble.py` packages any set of checkpoints into one Kaggle dataset and notebook:

- members grouped by input geometry, with each study decoded once and shared across groups;
- members split across the two T4 cards, in fp16;
- a **stress gate**: a 128-study commit run must pass and project under the 9 h cap before a real submission is
  allowed.

## Repository layout

```text
data_code/       numbered pipeline: EDA, report labelling, splits, folds, teacher labels
modal_scripts/   download, preprocessing, shard sync, checkpoint and run sync for Modal
runs/
  shared_code.py       the engine
  shared_config.yaml   shared defaults
  <experiment>/        config.yaml (only what varies), train.py, results/ (logs, summaries)
  others/              cross-run analysis, ensembling, submission tooling
checks/          tests for engine features and run configs (most run on Modal CPU)
other_md/        research notes
```

Not in the repository: the competition data and every file derived from it (`data/`, per the competition rules),
checkpoints and image shards (hundreds of GB). Released weights are on Hugging Face (links at the top).

## Reproducing

1. Download the competition data and preprocess it into shards (`modal_scripts/modal_download.py`,
   `modal_scripts/modal_preprocess.py`).
2. Build labels and splits with the numbered `data_code/` scripts, in order.
3. Train a run: `python -m modal run runs/<experiment>/train.py` (or `python runs/<experiment>/train.py` locally).
4. Score combinations from the cached predictions (`runs/others/ensembling/`), then stress-test and submit with
   `python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/<name>_stress.py` and its
   `_submit.py` twin.

Kaggle and Modal credentials are read from the environment, never from files in this repository.

## Acknowledgements and licences

Code: Apache 2.0 (see `LICENSE`). Pretrained weights keep their own licences:

- **Built with DINOv3.** DINOv3 weights by Meta AI are used under the DINOv3 License; a copy ships with every
  submission that includes them.
- MedImageInsight (Microsoft), MRI-CORE, and ImageNet-pretrained ConvNeXt / SwinV2 from
  [timm](https://github.com/huggingface/pytorch-image-models).
- RadImageNet weights are non-commercial and were never used in a prize-eligible submission.
- Thanks to the Kaggle community notebooks this work learned from, and to RSNA and Kaggle for the dataset, the first
  RSNA AI Challenge to pair every study with its radiology report.
