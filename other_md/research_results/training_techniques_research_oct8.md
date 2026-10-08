# Training methods and techniques research (8 Oct 2026)

Scope: noisy and soft labels, report-derived uncertainty, distillation, auxiliary and report-supervised objectives, losses and optimisers (SAM), MRI-specific augmentation (TorchIO-style artefacts), medical self-supervised pretraining, weight averaging, per-label ensembling. Emphasis on medical imaging work (MICCAI, RSNA, Radiology: AI, MIDL, medical Kaggle winners) as requested.

Terms used below: Area Under the Receiver Operating Characteristic curve (AUROC), Magnetic Resonance Imaging (MRI), Self-Supervised Learning (SSL), Masked Autoencoder (MAE), Sharpness-Aware Minimization (SAM), Exponential Moving Average (EMA), Stochastic Weight Averaging (SWA), Symmetric Cross Entropy (SCE), Early-Learning Regularisation (ELR), Generalised Cross Entropy (GCE), Large Language Model (LLM), Vision-Language Model (VLM), Test-Time Training (TTT), Binary Cross Entropy (BCE), Graphics Processing Unit (GPU), Oct (October).

## 0. What our repo already settles (do not re-run)

Checked against other_md/outline.md, runs_timeline.md and runs/shared_code.py.

| Idea | Status in our docs |
|---|---|
| Soft LLM targets (trained on probability, not 0/1) | IN USE since v3/v4. Hard-vs-soft 0/1 test: no effect (outline line 104). |
| Label smoothing (hard mode) | Engine knob exists (`label_smoothing: 0.0`), only for hard mode. Ablation "planned", never run. |
| Mixup and weight EMA | Rejected before retrains (outline line 264, "small expected gain"). Not measured. |
| Focal loss | Tried (focal -0.0014 on 6 focal findings). Not a win. |
| pos_weight, pairwise hinge, AUC-M + PESG (7 runs) | Closed, zero or negative. |
| Stacking (per-class logistic, cross-finding stacker) | Closed. PF OA was the only finding helped, and audit-58 disagreed. |
| Head refit (linear, per-label, MLP) | Closed (MLP -0.0028). |
| Test-time training, distilling ensemble into a member | Closed (runs timeline ~line 4490): solo rises, ensemble adds nothing. |
| Duplicate-series and slot-variant TTA | Rejected (-0.0025, -0.0019). |
| Co-teaching, ELR, GCE | Listed as "still untried" (outline line 173). |
| Continued MAE on MRI-CORE ViT-B | Planned, and 40+ timeline mentions (outline ~line 731). Partly launched (`mricore_vitb_mae_cross_attention`). |
| Round-2 pseudo-labels from a diverse OOF teacher | In big_jump_ideas_oct7.md as candidate 1. Not yet run. |
| Transductive self-training on test set | In big_jump_ideas_oct7.md as candidate 2. Not run. |

Current engine augmentations (runs/shared_code.py `_augment`): rotation, scale, translate, brightness, contrast, horizontal flip, slice dropout, gamma, Gaussian noise, blur, square cutout. NOT present: MRI bias field, motion, ghosting, k-space spike, k-space truncation (Gibbs ringing), elastic deformation, sequence dropout.

Claim to verify before using: the "Chris Deotte: with pseudo-labels every single model matches the ensemble" statement (cited in big_jump_ideas_oct7.md). My search found no primary source tying Deotte to this result (only an NVIDIA interview about pseudo-labels in general). Treat it as forum hearsay.

## 1. Ranked top 10 (expected leaderboard gain per GPU dollar, honest)

Cost basis: one ConvNeXt-Base 2.5D run on our Modal setup is roughly $2 to $6 (outline, earlier runs). H100 hours are used as the unit below where a number is given. Gains are expected macro AUROC on the leaderboard, not on val.

**1. Report-hedge-aware soft targets (uncertainty words to target shrinkage).**
- What: map each finding's hedge language in the report ("possible", "may", "cannot exclude", "probable", "likely", "unlikely") to a target value, instead of the LLM's single probability. Rep-GLS (Zhang et al., arXiv 2508.02495, v3, Nov 2025) does this with an expert rubric: positive/is = -0.2 (sharpen), null = 0, unlikely/probable/likely = 0.2, may/could/possible = 0.4, "not exclude"/"not rule out" = 0.7, "cannot be evaluated" = 0.9, and loss = (1 - r) BCE + r KL(p || uniform). Source: https://arxiv.org/html/2508.02495v3. Reported mean AUC on MIMIC-CXR (chest X-ray, 14 labels): hard BCE 79.63, full Rep-GLS 84.14 (+4.5). No standard label-smoothing baseline in the paper, so the gain is vs hard labels only.
- CheXpert uncertainty policies (U-Ones, U-Zeros, U-Ignore, label-smoothing variants Pham et al.) are the older version of this. Source: https://arxiv.org/pdf/2102.13269 and https://arxiv.org/pdf/1911.06475. Evidence: a student comparison found U-Ignore beat U-Ones and U-Zeros. Weak.
- Fit: our LLM labels were produced from the same reports, so the hedge mapping can be recomputed from stored report text with no new LLM calls if the report text is kept (CPU minutes). Shrink only the uncertain cells, keep the 0/1 cells as they are.
- Cost: about one ConvNeXt run, roughly $2 to $6. Gain: plausibly +0.001 to +0.004 (weak-medium). Risk: our soft probabilities may already encode hedging (v4 mix), so the shrink could double-count. Test on audit-58 per finding and on the 704 val split, same sign on both.
- Overlap: partly in docs (soft labels, label smoothing ablation planned). The hedge-word mapping itself is new.

**2. Report vs radiologist label shift: per-finding label source weighting, fit on audit-58.**
- What: the repo already has the key measurement. Per-finding image-model AUROC vs report-target AUROC on the 58 panel studies (big_gain_ideas_research.md section 1): Effusion image 0.901 vs report 0.696, Contusion 0.915 vs 0.821, Lateral OA 0.782 vs 0.829, Lateral Meniscus 0.858 vs 0.894. Images win on some findings, reports on others. The idea is to train each head's target as a per-finding blend of report label and image-model OOF prediction, with the blend weight chosen on the 58 panel studies (and checked on val so it is not one split only).
- Source of method: Raykar et al. 2010 (jointly learning classifier and annotator quality), https://jmlr.org/beta/papers/v11/raykar10a.html; VisualCheXbert shows report labels disagree systematically with radiologist labels and a calibrated mapping beats raw report labels by +0.14 mean F1, https://arxiv.org/pdf/2102.11467.
- Cost: CPU only plus one run. Gain: could be the biggest single lever on panel labels, +0.002 to +0.008, but only 58 studies (per-finding standard error 0.04 to 0.09). Risk: overfitting the blend to 58 studies. Constrain to a grid of 3 to 5 weights per finding.
- Overlap: partly. Our v4 mix is a global 50/50 blend, not per-finding. Per-finding blend is new.

**3. Round-2 pseudo-labels with a diverse OOF teacher (already the top big-jump idea).**
- What: teacher = mean of OOF predictions from 3 to 4 architecturally different members (ConvNeXt 2.5D, MII, DINOv3 ConvNeXt, SwinV2), new targets = teacher probability blended with the round-1 target, then 5-fold students. Noisy Student (https://arxiv.org/abs/1911.04252) iterates the same way.
- Evidence: forum threads (735304: a single 5-fold ResNet-50 at 224 px scored 0.954), Raymond Yuen 42nd place (several teachers, different targets). Medical evidence: PANDA label cleaning (https://www.nature.com/articles/s41591-021-01620-2, Nature Medicine 2021), iterated by several top teams. Lumbar 3rd place: pseudo-labels "not significant" (negative-ish evidence).
- Cost: 5 folds x 4 teachers = about 20 runs, roughly $40 to $120 (H100 hours scale). Gain: +0.003 to +0.010 if it works. Risk: 58-study panel cannot rank small gains.
- Overlap: in big_jump_ideas_oct7.md (candidate 1). Not run.

**4. MRI artefact augmentation (TorchIO-style: bias field, motion, ghosting, k-space spike, Gibbs ringing from k-space truncation).**
- What: add to `_augment`. TorchIO provides BiasField, Motion, Ghosting, Spike and Anisotropy transforms (https://docs.torchio.org/2.0/tutorials/augmentation/). Gibbs ringing = keep only a centred part of k-space (the frequency data), which causes ringing at edges.
- Evidence: Cabrera and Fetit (NeurIPS 2022 workshop, https://neurips.cc/virtual/2022/63415) simulated Gibbs, spike and wraparound artefacts during training: lower texture bias and better robustness to unseen hospitals (segmentation, not classification). Bias field is common in MRI pipelines and is cheap.
- Fit: our test set is from other scanners and the local data is mixed-protocol. Robustness to site shift matters. Our engine lacks all of these.
- Cost: CPU-side, near zero GPU; one ConvNeXt run for testing ($2 to $6). Gain: +0.001 to +0.004 (weak). Risk: artefacts applied on the 2.5D stack before resample could distort the thin structures (meniscus, ACL) that the focal findings need. Apply at low probability (0.1 to 0.2) and check focal findings separately, as the 2026 runs did.
- Overlap: not in docs. Engine augmentation list is the place to add it.

**5. SAM (Sharpness-Aware Minimization) optimiser, one run.**
- What: `optimizer: sam` knob around AdamW. Doubles step cost.
- Evidence: Hassan et al. 2025 (IEEE Access) "Do Sharpness-based Optimizers Improve Generalization in Medical Image Analysis?" (https://arxiv.org/pdf/2408.04065): SAM was the only sharpness method that consistently improved generalisation across CNNs and ViTs on breast ultrasound. ISBI 2022 reports 5 to 10 percent generalisation gain on ultrasound and chest X-ray. No multi-label MRI study found. Imbalanced-SAM (ImbSAM) addresses tail-class overfitting, relevant for rare findings. Source: arXiv 2408.04065 above.
- Cost: 2x step time, so roughly $4 to $12 per ConvNeXt run. Gain: +0.001 to +0.004 (weak). Risk: fp16 instability (already a noted issue), and SAM with noisy targets is untested here.
- Overlap: listed as rank 4 in big_gain_ideas_research.md. Not run.

**6. Noise-robust losses: SCE (symmetric cross entropy), ELR (early-learning regularisation), GCE (generalised cross entropy).**
- What: add on top of soft BCE. Our targets are soft, so the usual sample-selection losses (co-teaching, DivideMix) would discard our noisy-but-useful labels; regularisation losses fit better.
- Evidence: LNMBench (Ma et al., arXiv 2512.09315, Dec 2025, 10 methods, 7 datasets, 6 modalities): on real medical noise, regularisation losses (SCE, CDR) were best and sample-selection methods were WORSE than plain cross entropy. Gains modest (a few accuracy points). Source: https://arxiv.org/html/2512.09315v1. I did not confirm ELR or SCE by name in the snippet; check that page. BoMD (ICCV 2023, https://mlanthology.org/iccv/2023/chen2023iccv-bomd/) is a multi-label noisy chest X-ray method that relabels noisy samples smoothly.
- Cost: one run each, $2 to $6. Gain: +0.000 to +0.003. Risk: low; the outline already lists these as untried.
- Overlap: listed as untried in outline.md line 173 and big_gain_ideas_research.md item D.

**7. Medical self-supervised pretraining: masked (MAE) and volume-contrastive (VoCo) continued pretraining on our unlabelled slices.**
- What: continue MAE on the 819k unlabelled slices from MRI-CORE ViT-B (already in the plan) or from the DINOv3 ConvNeXt (FCMAE, with the GRN caveat in outline). Models Genesis (https://arxiv.org/pdf/1908.06912) is the classic medical generic pretraining (CT and MRI; its MRI numbers were not in what I could retrieve). VoCo (CVPR 2024, https://arxiv.org/pdf/2410.09890) is CT-based. Revisiting MAE for 3D medical segmentation (https://arxiv.org/html/2410.23132v2) reports MAE pretraining beats earlier SSL and nnU-Net by about 3 Dice points, on brain MRI segmentation.
- Knee evidence: SSL on MRNet (ACL, meniscus) generally trailed supervised baselines in 2020 to 2021 (SKID, SB-SSL, jigsaw). OAI-DINO on knee MRI (ISMRM 2025 abstract) beat scratch and supervised ImageNet pretraining in its own report. Knee OA SSL with ConVIRT and GLoRIA on radiographs was weaker than ImageNet for grading (0.627 vs 0.646), https://arxiv.org/pdf/2603.24903.
- Cost: high. MAE on 819k slices is many GPU-hours (roughly 50 to 150 H100 hours, $100 to $400). Gain: uncertain, +0.000 to +0.010. Risk: budget is the hard limit; the timing before 22 Oct is tight.
- Overlap: planned (outline ~line 731), partly launched.

**8. Weight averaging over the last checkpoints (SWA), free at inference.**
- What: average weights of the last K epoch checkpoints (uniform), instead of EMA. Repo rejected EMA for "small expected gain" without measuring it. SWA costs no extra training if checkpoints or a running average are saved.
- Evidence: few-shot learning comparison (arXiv 2204.12466): SWA 67.18 vs EMA 66.05 (1-shot), so SWA slightly ahead there. Medical: SWA used for stabilisation in a GI bleeding model (arXiv 2412.16723) without ablation; EMA teacher gave +6.9 Macro-F1 on ISIC 2019 (arXiv citation in medical search, no direct SWA comparison). Weak evidence overall.
- Cost: near zero if the trainer already saves a running average; otherwise one run. Gain: +0.000 to +0.002. Risk: small.
- Overlap: EMA rejected in outline (line 264); SWA not discussed.

**9. Report-supervised contrastive pretraining (ConVIRT, GLoRIA, MedKLIP).**
- What: image-text contrastive pretraining on knee MRI plus report sentences, then fine-tune. ConVIRT (Zhang et al., MLHC 2022), GLoRIA (ICCV 2021), MedKLIP (https://arxiv.org/pdf/2301.02228, chest X-ray; ResNet-50 init).
- Evidence: on knee radiographs, ConVIRT and GLoRIA improved linear probe but lost on KL grading (0.627 vs ImageNet 0.646), https://arxiv.org/pdf/2603.24903. CMANet (medRxiv 2025, https://www.medrxiv.org/content/10.1101/2025.06.08.25329119.full.pdf) fuses 3D knee MRI with report text for OA grading on 642 patients: task-specific, not a pretraining comparison. A 2025 study (arXiv 2510.24385) found that distilling report text into an image model gives no gain when the label is already predictable from text. Our labels ARE report-derived.
- Cost: high (new pretraining, text encoder, and our report text for 4,407 studies). Gain: low, +0.000 to +0.005. Risk: high.
- Overlap: report-text auxiliary targets are in big_gain_ideas_research.md item F (rank 6). Contrastive pretraining is new, but I rate it low.

**10. Auxiliary attribute heads from report text (beyond the 12 labels).**
- What: extra heads for structured attributes read by the LLM (ACL partial vs complete, meniscus horn/tear, cartilage grade, bone marrow oedema location, osteophytes), tied to weak findings (Lateral OA 0.851, Synovitis 0.865, Lateral Meniscus 0.872, PF OA 0.880 on audit-58, runs/others/per_finding_gap.py).
- Evidence: RSNA 2023 1st place (auxiliary segmentation "+0.01 to +0.03", https://github.com/Nischaydnk/RSNA-2023-1st-place-solution); RSNA 2024 lumbar 3rd place (auxiliary losses helped, severity weights 1/2/4). Caution: arXiv 2510.24385 (text-predictable labels get no gain).
- Cost: one run plus LLM extraction of attributes (roughly $5 to $20 with a cheap model; check cost before a batch). Gain: +0.001 to +0.005. Risk: attribute labels are noisier than the 12 labels.
- Overlap: in big_gain_ideas_research.md item F. Not run.

Honourable mention (not in top 10): progressive resizing, layer-wise decay (not in our engine per grep), and TTT or per-label rank averaging. Per-label rank averaging and ridge stacking are already measured negative, so skip them.

Distillation (from big_gain_ideas item H): Noothout et al. (J Med Imaging 2022, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9142841/) compressed segmentation ensembles into one CNN without losing the ensemble's performance. Beyer et al. 2022 "A good teacher is patient and consistent" (https://arxiv.org/abs/2106.05237) is function matching with long schedules and the same augmented view for teacher and student. Our TTT simulation says distilling the ensemble into one member adds nothing. Value is efficiency, not AUROC. Not in top 10.

## 2. Ranking summary

| Rank | Idea | Expected gain | Cost (approx) | Risk | In docs? |
|---|---|---|---|---|---|
| 1 | Report-hedge-aware soft targets | +0.001 to +0.004 | $2 to $6 | Double-counting with v4 | Partly |
| 2 | Per-finding report/image label blend (fit on audit-58) | +0.002 to +0.008 | CPU + $2 to $6 | Overfit to 58 studies | Partly |
| 3 | Round-2 pseudo-labels, diverse teacher | +0.003 to +0.010 | $40 to $120 | Panel cannot rank small gains | Yes (candidate 1) |
| 4 | TorchIO-style MRI artefacts (bias field, motion, ghosting, k-space) | +0.001 to +0.004 | $2 to $6 | Thin-structure damage | No |
| 5 | SAM optimiser | +0.001 to +0.004 | $4 to $12 | fp16 instability | Yes (rank 4) |
| 6 | SCE / ELR / GCE on soft targets | +0.000 to +0.003 | $2 to $6 each | Low | Yes (untried) |
| 7 | Continued MAE / VoCo pretraining | +0.000 to +0.010 | $100 to $400 | Budget, timing | Yes (planned) |
| 8 | SWA over last checkpoints | +0.000 to +0.002 | ~$0 | Small | EMA rejected, SWA not |
| 9 | ConVIRT / GLoRIA / MedKLIP report-image pretraining | +0.000 to +0.005 | High | Knee evidence negative | Partly |
| 10 | Report-derived auxiliary attribute heads | +0.001 to +0.005 | $5 to $20 + run | Noisy attributes | Yes (item F) |

Recommended order for the remaining two weeks (before 22 Oct): 2, 1, 4 in one batch of three runs (about $10 to $20 total), then SAM (5) if the batch is positive on audit-58 and val. Round-2 pseudo-labels (3) stay the main bet and should start first if GPU time allows. Skip 7 unless the budget is clearly available.

## 3. Sources

- Rep-GLS: https://arxiv.org/html/2508.02495v3
- CheXpert uncertainty policies and label smoothing: https://arxiv.org/pdf/2102.13269, https://arxiv.org/pdf/1911.06475
- Raykar et al. 2010: https://jmlr.org/beta/papers/v11/raykar10a.html
- VisualCheXbert: https://arxiv.org/pdf/2102.11467
- PANDA label cleaning (Nature Medicine 2021): https://www.nature.com/articles/s41591-021-01620-2
- Noisy Student: https://arxiv.org/abs/1911.04252
- TorchIO augmentation: https://docs.torchio.org/2.0/tutorials/augmentation/
- Cabrera and Fetit (NeurIPS 2022 workshop): https://neurips.cc/virtual/2022/63415
- Sharpness-based optimisers in medical imaging (Hassan et al.): https://arxiv.org/pdf/2408.04065
- LNMBench (noisy medical labels): https://arxiv.org/html/2512.09315v1
- BoMD (multi-label noisy chest X-ray): https://mlanthology.org/iccv/2023/chen2023iccv-bomd/
- Models Genesis: https://arxiv.org/pdf/1908.06912
- VoCo: https://arxiv.org/pdf/2410.09890
- MAE for 3D medical segmentation: https://arxiv.org/html/2410.23132v2
- Knee SSL ConVIRT/GLoRIA (radiographs): https://arxiv.org/pdf/2603.24903
- CMANet knee MRI with reports: https://www.medrxiv.org/content/10.1101/2025.06.08.25329119.full.pdf
- MedKLIP: https://arxiv.org/pdf/2301.02228
- Beyer et al. 2022: https://arxiv.org/abs/2106.05237
- Noothout et al. 2022 (ensemble distillation, segmentation): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9142841/
- Report-text distillation gives no gain when text-predictable: https://arxiv.org/abs/2510.24385

## 4. Caveats

- Several claims could not be verified: the Deotte single-model-equals-ensemble statement, the Models Genesis MRI numbers, the ELR/SCE entries of LNMBench, and any SWA-vs-EMA medical comparison.
- No search found a multi-label MRI SAM study or a knee MRI report-contrastive pretraining comparison with numbers. Those gaps are real.
- Every gain estimate above is a prior, not a measurement. On the 58 panel studies a difference under about 0.01 per finding is inside noise (outline caveat). Decide on the same sign across audit-58 and val, as the repo already does.
