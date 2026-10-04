# Pretrained 3D / video backbones for the knee MRI task: research report

Date: 2026-10-04. Method: web research against primary sources (GitHub LICENSE files, Hugging Face model cards, papers). Anything I could not verify from a primary source is marked "unverified". Parameter counts marked "(est.)" are my own estimates, not from a source. T4 feasibility is my engineering estimate (15 GB VRAM, fp16, 946 test studies, 9 h), not a measured number.

## 1. Executive summary

### Bottom line

3D is NOT worth trying as a standalone, "foundation-model-initialised" member. The evidence from every direction points the same way:

- A Siemens benchmark (arXiv 2512.12887, 12 tasks, 3 of them shoulder MRI, which is the closest to ours) fully fine-tuned the two best-known pretrained 3D medical models and compared them with 2D/2.5D methods. MedicalNet (3D ResNet-50) scored 0.783 average AUC over 10 tasks, VoCo (Swin-UNETR, 160k CT) 0.793, 3D ResNet from scratch 0.809, 3D DenseNet from scratch 0.833, the RSNA-Kaggle-style 2.5D EfficientNet + BiLSTM 0.810, and a DINOv3 / MedImageInsight 2D backbone with LoRA + attention-pool slice fusion 0.894. Their own sentence: pretrained 3D medical foundation models "underperform despite larger parameter counts".
- Every RSNA Kaggle 3D classification winner (2022 cervical spine, 2023 abdominal trauma, 2024 lumbar spine) used 2.5D (2D CNN per slice or slice-triplet + LSTM/GRU/attention). 3D CNNs were only used for localisation/segmentation. The 3rd place 2024 writeup is reported (via search snippet, page not fetchable) to say 3D CNNs did not work in the final submission.
- The only knee-specific 3D pretrained encoder (KneePreM, 3D U-Net MAE on OAI) reports macro AUC 0.777 on fastMRI+ (meniscus tear + cartilage defect) and 0.787 on an 8-target arthroscopy cohort. These are far below our 0.90 to 0.92 level, and I could not verify that weights are public.
- Our own TRIAD result (0.8145) matches this pattern.

### If you still want one 3D-ish diversity member, the ranked shortlist

| Rank | Candidate | Why | Prize-safe |
|---|---|---|---|
| 1 | Self-built inflated / (2+1)D ConvNeXt-B (spatial 1x7x7 kernels inflated from our own supervised ImageNet-22k ConvNeXt, plus a small identity-initialised temporal block) | Starts from the strongest supervised 2D weights we already use, respects anisotropy, no new license | YES (same licenses as the 2D weights already in the pipeline) |
| 2 | Video Swin-B (torchvision `swin3d_b`, Kinetics-400 + ImageNet-22k) | Off-the-shelf 3D transformer, 88.0M params, supervised pretraining, windowed attention, patch (2,4,4) | RISKY-LOW (torchvision code BSD; weights have no explicit license statement; Kinetics annotations CC BY 4.0, source videos are YouTube; treat as unverified) |
| 3 | MedicalNet 3D ResNet-50 (Tencent, MIT) | Only widely used, small, supervised (segmentation) 3D medical backbone with some MRI in pretraining; 46.2M params | MOSTLY SAFE (code and weights MIT; the licenses of the 8 source datasets are unverified) but weakest evidence (0.783 avg) |
| 4 | Merlin I3D ResNet-152 (Stanford, MIT) | Inflated-2D-ImageNet then CT-supervised, so conceptually close to rank 1 | SAFE on paper (MIT code and weights); CT abdomen only, no MRI evidence, heavy |
| 5 | V-JEPA 2 ViT-L (Meta, MIT) | Video SSL, handles variable frames | SAFE on paper (MIT); but self-supervised, and our own history says SSL backbones fine-tune badly; 300M params |

Recommendation: spend effort on rank 1 only if there is spare capacity, as an architecture-diversity experiment (it is closest to a "3D" view but costs almost nothing in license risk). Do not spend a run on MedicalNet, VoCo, SuPreM, Models Genesis, BrainIAC, or other 3D medical foundation models: either they are CT-only, or they are non-commercial/non-redistributable, or both, and the one benchmark that fine-tuned them put them 10 AUC points below 2D foundation backbones. A cheaper and better-evidenced route to diversity: more 2D-backbone variety with attention-pool fusion (the paper's AnyMC3D design, DINOv3 ViT-L with LoRA, 1.2M trainable params).

### Do-not-ship list (non-commercial or unclear)

BrainIAC (research-only), SuPreM (CC BY-NC-ND per LICENSE file), Models Genesis (ASU non-commercial), VideoMAE v1 and v2 (CC-BY-NC-4.0 on the HF cards), V-JEPA 1 (CC-BY-NC-4.0 LICENSE), SwinUNETR self-supervised weights (no license stated, unknown), M3D (MIT stated, but data used under "non-commercial ML" permission, treat as risky), TotalSegmentator restricted tasks (the `total_mr` task itself is Apache-2.0).

## 2. Comparison table

Legend: Safe = permissive license for code and weights verified. Risky = non-commercial, unknown, or data-terms doubt. "unv." = unverified.

| Name | Pretraining | Params | Code license | Weights license | Prize-safe | Weights URL | MRI evidence | T4 feasibility |
|---|---|---|---|---|---|---|---|---|
| Inflated / (2+1)D ConvNeXt-B (self-built) | Supervised ImageNet-22k (then our own knee fine-tune) | ~89M (est.) | Ours; timm Apache-2.0 | Same as the 2D weights we already ship | YES | n/a (our own checkpoints) | None directly; ACS/I3D inflation papers show gains over from-scratch 3D | Good: about same cost as the current 2.5D member, slightly more |
| Video Swin-B (torchvision swin3d_b) | Kinetics-400 supervised, init from ImageNet-22k Swin-B | 88.0M | BSD-3 (torchvision, unv. file read) | No explicit statement; Kinetics annotations CC BY 4.0 | RISKY-LOW (unv.) | https://download.pytorch.org/models/swin3d_b_22k-7c6ae6fa.pth | None found for knee; used as 3D backbone in medical papers (not verified here) | Fine: 140.7 GOps per 32x224x224 clip, fp16 OK |
| Video Swin-T/S | Kinetics-400 supervised | 28.2M / 49.8M | BSD-3 | as above | RISKY-LOW | https://download.pytorch.org/models/swin3d_t-7615ae03.pth , https://download.pytorch.org/models/swin3d_s-da41c237.pth | None | Easy |
| MedicalNet 3D ResNet-10/18/34/50/101/152/200 | Supervised segmentation on 3DSeg-8 (8 public CT and MRI datasets; paper says "23 datasets" in a later README update) | 14.4M / 33.0M / 63.3M / 46.2M / 85.3M / 117.5M / 126.7M | MIT | MIT (LICENSE file, HF card); dataset terms unv. | MOSTLY SAFE | https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view , https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50 | Siemens benchmark: 0.783 avg AUC (shoulder MRI tasks included), below from-scratch 3D DenseNet 0.833. MedNet-FS knee few-shot: ACL AUC 0.72 with 40 per class | Easy: ResNet-50 3D is small |
| Merlin I3D ResNet-152 | Inflated ImageNet ResNet-152, then abdominal CT: contrastive with reports + EHR phenotype labels; 15,331 CT scans | ~118M (est., same as 3D ResNet-152) | MIT | MIT (HF card and repo); Merlin dataset is separate and non-commercial but not shipped | SAFE on paper | https://huggingface.co/stanfordmimi/Merlin | None for MRI | Heavy: input 224x224x160; 118M params; OK at small batch |
| V-JEPA 2 ViT-L/16 (fpc64, 256px) | Self-supervised video, internet-scale | ~300M (0.3B) | MIT (some Apache-2.0 files) | MIT (HF card) | SAFE on paper | https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 | None found | Heavy: 300M ViT on 64x256x256 tokens; needs short clips or smaller V-JEPA 2.1 ViT-B (80M, 384px) |
| VideoMAE-B / v2-B | Self-supervised Kinetics-400 / UnlabeledHybrid-1M | 94.2M / 86.2M | repo unv. | CC-BY-NC-4.0 | NO | https://huggingface.co/MCG-NJU/videomae-base | none | n/a |
| V-JEPA 1 | Self-supervised video | various | CC-BY-NC-4.0 (LICENSE file) | CC-BY-NC-4.0 (unv. separately, assume same) | NO | https://github.com/facebookresearch/jepa | none | n/a |
| InternVideo2 | video-text, SSL, supervised stages | S/B/L/1B | Apache-2.0 (repo) | per-checkpoint, unv. | RISKY (unv.) | https://github.com/OpenGVLab/InternVideo | none | 1B too big; S/B only |
| X3D / SlowFast (PyTorchVideo) | Supervised Kinetics | small (unv.) | Apache-2.0 | unv. | RISKY (unv.) | https://github.com/facebookresearch/pytorchvideo | none | Easy |
| torchvision R3D-18 / R(2+1)D-18 / MC3-18 | Supervised Kinetics-400 | ~33M / ~31M / ~11M (est.) | BSD-3 | unv. | RISKY-LOW (unv.) | torchvision.models.video | none | Easy |
| VoCo (Large-Scale-Medical) | Self-supervised contrastive, 160k CT | 31M to 1.2B; B variants 53M / 72M | Apache-2.0 | no explicit weight license; datasets "publicly available for academic research", some need author approval | RISKY | https://huggingface.co/datasets/Luffy503/VoCo-10k (and links in https://github.com/Luffy03/Large-Scale-Medical) | CT only; Siemens benchmark 0.793 avg, 0.919 on trauma task | Medium |
| SwinUNETR SSL (Tang et al.) | Self-supervised, 5,050 CT | ~62M (est., SuPreM lists 62.19M for Swin UNETR) | Apache-2.0 for MONAI code (unv. for this folder) | none stated | RISKY (unknown) | `model_swinvit.pt` in MONAI-extra-test-data 0.8.1 releases | CT only | Medium |
| CT-FM (SegResNet) | Self-supervised contrastive, 148k CT (Imaging Data Commons) | 77.8M to 87.2M (two HF cards) | MIT (repo, search result) | Apache-2.0 (HF feature extractor card) | SAFE on paper | https://huggingface.co/project-lighter/ct_fm_feature_extractor | CT only, HU-windowed input | Medium; CT preprocessing mismatch with MRI |
| SuPreM (Swin UNETR, U-Net, SegResNet) | Supervised segmentation, AbdomenAtlas CT | 62.19M / 19.08M / 4.70M | CC BY-NC-ND 4.0 (LICENSE file; a search result said NC-SA, both NC) | same repo license | NO | https://github.com/MrGiovanni/SuPreM | CT only; KneePreM paper: SuPreM init was worse than random init on fastMRI+ (0.686 vs 0.714) | n/a |
| Models Genesis | Self-supervised on CT (LUNA16) | small U-Net | ASU custom non-commercial | same | NO | https://github.com/MrGiovanni/ModelsGenesis | CT | n/a |
| BrainIAC (3D ViT-B/16, 96^3, SimCLR) | Self-supervised, multi-cohort brain MRI | ~86M (est.) | Research-only (MGB) | Research-only; HF gated | NO | https://github.com/AIM-KannLab/BrainIAC | Brain only | n/a |
| KneePreM (3D U-Net MAE) | Self-supervised on 19,011 OAI knee series | unv. | unv. (uses FuseMedML, Apache-2.0 unv.) | unv.; weights release NOT confirmed; OAI data terms unv. | RISKY (unv.) | not found | fastMRI+ macro AUC 0.777, APM 0.787 (weak) | Medium |
| M3D (3D ViT encoder + LLM) | CT-text pairs (Radiopaedia) | 4B/7B LLM + ViT | MIT | MIT stated, data under non-commercial ML permission | RISKY | https://github.com/BAAI-DCAI/M3D | CT | Too heavy |
| SAM-Med3D | Supervised promptable segmentation, SA-Med3D-140K (multi-modality) | unv. | Apache-2.0 | Apache-2.0 on repo; source dataset terms unv. | RISKY (unv.) | https://github.com/uni-medical/SAM-Med3D | segmentation only | Medium |
| TotalSegmentator MRI (nnU-Net) | Supervised segmentation of 50 MRI structures | unv. | Apache-2.0 | Apache-2.0 for open tasks including `total_mr`; restricted tasks need a license | SAFE for `total_mr` (unv. exact weights terms per task) | via `pip install totalsegmentator` | MRI segmentation; knee bones not in `total_mr` list | Encoder reuse is non-trivial; low value |
| FOMO25 baselines (3D, brain MRI SSL) | Self-supervised FOMO-60K brain MRI | unv. | unv. | unv. | RISKY (unv.) | https://github.com/fomo25/baseline-codebase | Brain only | n/a |
| TRIAD (already tried) | MAE on 131k brain/breast/prostate MRI | | | | | | 0.8145 val in our runs | |

## 3. Detailed notes per candidate

### 3.1 Self-built inflated / (2+1)D ConvNeXt (top recommendation if any 3D-style member is built)
- Idea: keep our supervised ImageNet-22k ConvNeXt-B (known to work in our runs), turn each 7x7 depthwise conv into 1x7x7 (a free reshape of the pretrained weights) and add a temporal mixing block per stage (3x1x1 depthwise conv initialised to identity or a delta kernel) so slices exchange information early rather than only at the pooling head.
- This is the I3D inflation recipe (inflate 2D kernels along time) combined with (2+1)D factorisation. Merlin uses exactly this I3D inflation for its ResNet-152 (https://arxiv.org/html/2406.06512v1).
- Anisotropy: temporal kernels operate over real slices only; no resampling. Our N evenly spaced real slices are the "frames".
- License: inherits the ConvNeXt weight license already used. Safe.
- Expected gain: unproven. In the Siemens benchmark, the simplest slice fusion (attention pooling) beat LSTM (0.962 vs 0.903 AUC on PDAC, https://arxiv.org/pdf/2512.12887, appendix F), so early temporal mixing may add little over our existing cross-slice attention. This is why I rank it as a diversity experiment, not a score play.

### 3.2 Video Swin Transformer (torchvision `swin3d_*`)
- Pretraining: supervised Kinetics-400 (swin3d_b also has the variant initialised from ImageNet-22k); 88.0M params for B, 140.7 GOps, 81.6% K400 top-1 for the 22k variant. Source: https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/video/swin_transformer.py
- Original repo (https://github.com/SwinTransformer/Video-Swin-Transformer) is Apache-2.0; torchvision code is BSD-3 (my knowledge, license file not fetched here, mark unverified).
- Weights: no license statement in torchvision metadata I could see; Kinetics-400 annotations are CC BY 4.0 per a search result summarising the MCG-NJU/VideoMAE issue, but the underlying videos are YouTube. Treat as "risky-low, unverified".
- Window attention is 3D (default window 8x7x7, patch 2x4x4), which treats time like space, the anisotropy concern you recorded. Mitigation: set the temporal patch to 1 and window depth small, or feed slices as frames with temporal window 2 to 4.
- No knee-MRI result found.

### 3.3 MedicalNet / Med3D
- Chen, Ma, Zheng 2019, "Med3D" (https://arxiv.org/abs/1904.00625). Pretrained by supervised multi-dataset segmentation (3DSeg-8: MRI and CT, brain, heart, prostate, spleen, liver, pancreas, etc.). All volumes resampled to the median spacing of each domain, intensity z-scored after 0.5 to 99.5 percentile clipping.
- Weights: resnet_10/18/34/50 (23-dataset versions), plus 101, 152, 200. Sizes: 14.36M, 32.99M, 63.31M, 46.21M, 85.31M, 117.51M, 126.74M (README). Download links: Google Drive https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view, Tencent Weiyun, and Hugging Face https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50.
- License: LICENSE file is MIT (includes 3D-ResNets-PyTorch, MIT); HF card says MIT. The LICENSE file has no special data terms. Licenses of the underlying 8 datasets are unverified.
- Evidence: Siemens benchmark full fine-tune 0.783 avg AUC (10 tasks, incl. three shoulder MRI tasks: Bicep 0.679, Bursa 0.801, Labrum 0.676), worse than 3D ResNet from scratch (0.809) and 3D DenseNet from scratch (0.833). The same paper notes LoRA and linear probing are far worse (linear probe 0.654 on T4). MedNet-FS (https://pmc.ncbi.nlm.nih.gov/articles/PMC13458554/) on MRNet few-shot reports MedNet ACL AUC 0.72 at 40 samples per class, and that pretraining on non-medical volumetric data gave only 0.50.
- Verdict: weak evidence; keep only as a license-safe fallback.

### 3.4 Merlin (Stanford)
- 3D VLM for abdominal CT; image encoder is I3D ResNet-152 (ImageNet-inflated), trained on 15,331 CT scans with EHR phenotype supervision plus contrastive learning against radiology reports (Clinical Longformer, 4,096 tokens). Input 224x224x160. Sources: https://github.com/StanfordMIMI/Merlin, https://huggingface.co/stanfordmimi/Merlin, https://arxiv.org/html/2406.06512v1.
- Code and weights: MIT (HF card and repo). The Merlin abdominal CT dataset is under a separate non-commercial data-use agreement (not needed for the weights).
- No MRI result. CT HU windowing is baked into its preprocessing, domain gap to knee MRI is large. Not recommended.

### 3.5 V-JEPA 2 and other video SSL
- V-JEPA 2 (https://github.com/facebookresearch/vjepa2): ViT-L/16 300M at 256px; ViT-g up to 1B; V-JEPA 2.1 adds ViT-B/16 80M at 384px. Majority MIT, some Apache-2.0 files. HF card for vitl-fpc64-256 says MIT, 64 frames.
- All self-supervised. Your own lesson (SSL fine-tunes poorly: MedSAM, BEiTv2, TRIAD, Barlow Twins) applies. No MRI evidence found.
- VideoMAE and VideoMAE v2: CC-BY-NC-4.0 on the HF cards. V-JEPA 1: CC BY-NC 4.0 LICENSE. Do not use.
- InternVideo (https://github.com/OpenGVLab/InternVideo): repo Apache-2.0; per-checkpoint weight terms unverified; sizes S/B/L/1B; not attractive.
- X3D, SlowFast (https://github.com/facebookresearch/pytorchvideo): Apache-2.0 code; weights license not verified; supervised Kinetics. Small models, cheap, but natural-video features, 3D convs treat axes alike.
- TimeSformer: not checked (unverified; I believe the code is CC-BY-NC, do not rely on that).
- Used on MRI slice stacks with good results? Not found in my searches. The most relevant evidence is the Siemens paper, which found 2D backbone + slice fusion beats 3D models; it also tested MRI shoulder volumes (320x320x28, i.e. 28 slices, anisotropic like ours).

### 3.6 CT-only 3D foundation models (VoCo, SwinUNETR SSL, CT-FM, SuPreM, Models Genesis)
- VoCo (https://github.com/Luffy03/VoCo, https://github.com/Luffy03/Large-Scale-Medical): Apache-2.0 code, CT only, 160k CT volumes, SSL. Siemens benchmark: 0.793 avg, 50.49M params. The README says datasets are public for academic research and some need author approval, so weight-license status is murky.
- SwinUNETR SSL weights (https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain): 5,050 CT scans from HNSCC, LUNA16, TCIA colonography, COVID-19, LIDC; `model_swinvit.pt`; no explicit weight license found.
- CT-FM (https://huggingface.co/project-lighter/ct_fm_feature_extractor): Apache-2.0 on HF card, 77.8M params, SSL on 148k CT scans; input clipped to HU -1024 to 2048, SPL orientation. Prize-safe on paper but CT-only.
- SuPreM (https://github.com/MrGiovanni/SuPreM): LICENSE file is CC BY-NC-ND 4.0. Excluded.
- Models Genesis: ASU custom non-commercial. Excluded.
- Transfer CT to MRI is questionable: intensity semantics differ. KneePreM's own paper measured SuPreM init (CT, supervised) at 0.686 vs random init 0.714 on fastMRI+ knee, i.e. negative transfer.

### 3.7 MRI-specific 3D encoders
- BrainIAC (https://github.com/AIM-KannLab/BrainIAC): 3D ViT-B/16, 96^3 input, SimCLR; research-only license (Mass General Brigham). Excluded. Brain only anyway.
- FOMO25 baselines (https://github.com/fomo25/baseline-codebase): brain MRI SSL; license not visible in the page I could fetch (unverified); brain only.
- TRIAD: already tested (0.8145).
- KneePreM (https://arxiv.org/abs/2609.31461): the only knee-specific 3D encoder found. 3D U-Net MAE on 19,011 baseline OAI knee series (9,493 sagittal 3D DESS and 9,518 sagittal IW TSE), 40x224x224 input, 4 days on one A100. Downstream: fastMRI+ macro AUC 0.777 (meniscus tear, cartilage defect, 1,172 exams), APM eight-target 0.787. Code via FuseMedML (https://github.com/BiomedSciAI/fuse-med-ml, oai_example) but I found no statement that the KneePreM weights are public (unverified). OAI data use terms for derived weights: unverified. Verdict: weak numbers, uncertain availability.
- TotalSegmentator MRI (https://github.com/wasserth/TotalSegmentator): Apache-2.0, supervised nnU-Net on 50 MRI structures; `total_mr` is open; `appendicular_bones_mr` (patella, tibia, fibula) requires a license. Possible use is as a localisation tool, not a classifier backbone.

### 3.8 Other medical 3D models
- M3D (https://github.com/BAAI-DCAI/M3D): MIT, 3D ViT encoder at 32x256x256, but training data used under Radiopaedia non-commercial ML permission. CT, too heavy. Risky.
- SAM-Med3D (https://github.com/uni-medical/SAM-Med3D): Apache-2.0, segmentation only, data terms of its 16+ source datasets unverified.
- MONAI model zoo (https://github.com/Project-MONAI/model-zoo): repo Apache-2.0, but each bundle has its own license and "does not make any statement of suitability for diagnostic use". I did not find a knee or musculoskeletal MRI classification bundle.
- Not researched in depth (unverified leads from search snippets): TAP-CT (https://arxiv.org/html/2512.00872), Raptor (https://arxiv.org/html/2507.08254, train-free 3D embeddings from 2D foundation models), MedNeXt-v2, STU-Net.

## 4. What knee and spine competition winners used

| Source | Result | Architecture |
|---|---|---|
| RSNA 2022 cervical spine fracture, 1st | Siemens paper appendix E1 | 3D segmentation for localisation, then 2D EfficientNet-V2-S / ConvNeXt on 15 slices per vertebra with neighbour slices and segmentation-mask channels, LSTM over slices, ensemble |
| RSNA 2023 abdominal trauma, 1st (https://github.com/Nischaydnk/RSNA-2023-1st-place-solution) | 3D segmentation for crops, then 2.5D: 96 slices into 32 triplets, CoaT Lite / EfficientNet-V2-S, GRU head, auxiliary segmentation heads |
| RSNA 2024 lumbar spine, 1st | Localise then classify, 2.5D multi-view crops (sagittal + axial), ConvNeXt-S / EfficientNet-V2-S, BiLSTM + attention MIL across slices and views, auxiliary losses, ensembles |
| RSNA 2024, 5th (https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution) | per search snippet: 3D UNet was good, but 2D UNet + sequence model (transformer, LSTM) gave better z-axis accuracy, so they chose 2D + sequence |
| RSNA 2024, 3rd (https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453) | per search snippet: 3D CNN did not work well in the final submission |
| Stanford MRNet challenge (https://stanfordmlgroup.github.io/competitions/mrnet/) | Top leaderboard entry was the baseline MRNet (AlexNet per slice, max pool over slices, per plane), average AUC 0.917 (abnormal 0.939, ACL 0.967, meniscus 0.845). All top entries were slice-based 2D CNN variants |
| MRNet comparative papers | A 3D vs 2D comparison on knee MRI (via a Frontiers 2025 search result, paper not fetched) reports 3D abnormality AUC 0.937 vs 2D 0.923; another ACL study reports 3D 89% vs 2D 92% accuracy. Mixed; small datasets (about 1,000 exams) |

Siemens paper summary (appendix E): winning solutions share 2.5D representation, 2D CNN backbones with explicit sequential fusion, auxiliary heads, and ensembling. The paper's headline: "over the past five years, all the winning solutions for 3D classification challenges adopted the 2D/2.5D methods". I did not fetch the Kaggle writeups directly (JavaScript pages return only titles), so the 2024 details are second-hand (Siemens paper and search snippets).

## 5. Anisotropy handling options

1. Slice-wise 2D encoder + learned slice fusion (what we do): the dominant winning design. In the Siemens paper, attention-pool fusion (0.001M params) beat LSTM (5.5M) and transformer (7.0M) fusion on PDAC.
2. (2+1)D / pseudo-3D: spatial 1xkxk then temporal kx1x1. Spatial weights inflate directly from 2D ImageNet checkpoints. R(2+1)D-18 in torchvision follows this (Kinetics weights, license unverified).
3. I3D-style inflation (Carreira and Zisserman): replicate 2D kernel along time and divide by T. Merlin does this for ResNet-152. For anisotropic data, use temporal kernel size 1 early, 3 only late, which is equivalent to option 2.
4. ACS convolutions (Yang et al., "Reinventing 2D Convolutions for 3D Images", https://arxiv.org/abs/1911.10477, IEEE JBHI 2021): split 2D kernels by channel into axial, coronal, sagittal groups, so any pretrained 2D CNN becomes a 3D CNN with same params. Licence of the ACSConv repo: not verified (the URL I tried returned 404). Because our through-plane spacing is about 10x in-plane, the coronal and sagittal groups see very coarse z, so I would use the axial-only variant.
5. Anisotropic nnU-Net-style 3D: kernel (1,3,3) at the first stages until the in-plane resolution has dropped to the through-plane resolution, then (3,3,3). Applies to any from-scratch 3D net; no pretrained weights for knee.
6. Cross-slice attention (our MRI-CORE ViT-B with cross-slice attention already does this).
7. Resample to isotropic: rejected (interpolates about 50% of series). The Siemens benchmark resized MRI to fixed shape (320x320x28) without isotropic resampling and still found 2D + fusion best.

## 6. Risks and open questions

- Pretraining-data licenses. MIT-labelled weights (MedicalNet, Merlin, CT-FM) may be trained on data whose terms are restrictive. The competition rule exempts pretrained models with incompatible licenses from the CC-BY-NC grant, but a quick legal read of the rules text (hackathon_page.txt) is advisable before shipping anything not already in the pipeline.
- Video Swin and other Kinetics-trained weights: no explicit weight license found; treat as unverified until the specific repo/model card says otherwise.
- KneePreM: weights availability and OAI terms unverified; do not plan around it.
- Kaggle writeups for RSNA 2024 (1st, 2nd, 3rd, 5th) could not be fetched in full; details above are from the Siemens paper and search summaries. If the exact "3D CNN failed" claims matter, read the writeups in a browser.
- The Siemens benchmark uses private shoulder MRI (28 slices), not knee; the transfer of its ranking to our task is likely but not guaranteed.
- T4 feasibility figures are estimates. If any 3D candidate is tried, profile one batch on a T4-class card first (fp16, batch 1, 32 slices at 384 to 512 px). 3D transformers at 512 px are the dangerous case (token count).
- Our evidence base (TRIAD 0.8145, SSL and segmentation-pretrained failures) and the literature agree. The expected value of a 3D member is low; expected diversity gain in an ensemble that already holds five 2D/2.5D families is uncertain, so test cheaply (frozen-feature or short run, then ensemble test with cached predictions) before investing.

## Sources

- https://arxiv.org/pdf/2512.12887 (Siemens benchmark, AnyMC3D)
- https://arxiv.org/abs/2609.31461 and PDF (KneePreM)
- https://arxiv.org/abs/1904.00625 (Med3D)
- https://github.com/Tencent/MedicalNet , https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50
- https://github.com/StanfordMIMI/Merlin , https://huggingface.co/stanfordmimi/Merlin , https://arxiv.org/html/2406.06512v1
- https://github.com/Luffy03/VoCo , https://github.com/Luffy03/Large-Scale-Medical
- https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain
- https://huggingface.co/project-lighter/ct_fm_feature_extractor
- https://github.com/MrGiovanni/SuPreM (LICENSE: CC BY-NC-ND 4.0)
- https://github.com/MrGiovanni/ModelsGenesis (LICENSE: ASU non-commercial)
- https://github.com/AIM-KannLab/BrainIAC
- https://github.com/fomo25/baseline-codebase
- https://github.com/BAAI-DCAI/M3D , https://github.com/uni-medical/SAM-Med3D
- https://github.com/Project-MONAI/model-zoo
- https://github.com/wasserth/TotalSegmentator
- https://huggingface.co/MCG-NJU/videomae-base , https://huggingface.co/OpenGVLab/VideoMAEv2-Base
- https://github.com/facebookresearch/jepa (LICENSE: CC BY-NC 4.0) , https://github.com/facebookresearch/vjepa2 , https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
- https://github.com/SwinTransformer/Video-Swin-Transformer , https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/video/swin_transformer.py
- https://github.com/facebookresearch/pytorchvideo , https://github.com/OpenGVLab/InternVideo
- https://arxiv.org/abs/1911.10477 (ACS convolutions)
- https://stanfordmlgroup.github.io/competitions/mrnet/ (MRNet leaderboard)
- https://github.com/Nischaydnk/RSNA-2023-1st-place-solution
- https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups (2nd, 5th, 9th, 11th, 12th) and discussion/539453 (3rd)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13458554/ (MedNet-FS)
