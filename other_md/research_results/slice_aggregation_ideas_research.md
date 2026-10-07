# Slice and view aggregation: new ideas (research, 6 Oct)

Scope: replacements or additions to the flat mean over 24 per-image vectors (4 views x 6 images, ConvNeXt-Base 2.5D trunk, linear head). Only ideas not already tried. Evidence ratings: A = replicated or directly measured on comparable task, B = one credible paper or competition result, C = plausible, indirect or weakly documented.

Caveat on sourcing: several Kaggle write-up pages (1st place cervical spine, 2nd place lumbar spine) would not render in the fetch tool. For those I rely on the GitHub READMEs and search snippets, and say so. The AnyMC3D paper (2512.12887) and the SmAP NeurIPS PDF could not be parsed, so I do not quote numbers from them.

## 1. Already tried (from runs_timeline.md and the public-notebook analysis)

| Idea | Run (timeline entry) | Result |
|---|---|---|
| Self-attention (transformer) over the 96 pooled slice vectors replacing the mean | convnext_small_transformer_pool (6) | -0.0148 val, worst of project (random head on pretrained trunk, no linear-probe warmup) |
| Token pooling: CLS vs mean/max/attn over ViT patch tokens | mricore_vitb_avgpool (27) | CLS lost to mean-style pooling in ensemble (0.9127 -> 0.9112) |
| Cross-slice attention blocks inside a ViT trunk (from scratch, finetuned, MAE-pretrained, full) | mricore_vitb_cross_attention, _ft, _mae_cross_attention, _full (28, 29, 37, 41) | Solo weak; MAE variant a useful ensemble member only; "cross-slice fusion closed as a direction to scale" (~1064) |
| Slices as channels (4 to 8 passes per study) | slice_input_ch1/ch2/proj_ramp (44-46) | -0.025 to -0.038, fatal on focal findings |
| Non-overlapping 2.5D (3 neighbouring slices as RGB) | convnext_base_25d_group3 (47) | -0.0031 solo at 36% of the cost; became the standard input |
| 5th view slot (cor_plain) | convnext_base_5slot (36) | negative, closed |
| Gated residual slot attention (12 label queries over 24 vectors, gate at 0) and token attention | convnext_base_25d_slotattn, _tokenattn (52) | Neutral; gate stayed ~0.006-0.012; logits correlate 0.994 with parent |
| Learned MLP weighting replacing the mean (no gate, no fallback) | _poolattn (53) | Worst solo (0.8977), +0.0006 as 7th ensemble member |
| 12 queries over 2400 spatial tokens, gate open (equivalent of public TokenResidualPool idea) | _spatial (53) | -0.0028 to -0.0070 solo; ensemble neutral; "flat mean close to optimal" |
| Per-finding pooling (max / top2 / mean), per-finding view restriction by anatomy | _perfinding_pool_features, _perfinding_slots_and_pooling (68, 69, 74) | Max 0/4 wins (-0.0106 mean delta); top2 gave MCL +0.0329 but total -0.0035; restricting views HURT |
| BiGRU / transformer sequence heads on cached features (frozen trunk, order-aware) | seq_head_sweep (65) | mean 0.9042, gru 0.8999, gru_joint 0.8980, tfm 0.8977, attn 0.9034 |
| Exhaustive ensemble-structure search, nesting | (73, 75) | +0.0023 in sample, -0.0003 held out |
| Sliding-window TTA, reverse-slice TTA | (79) | closed; reverse is vacuous under a mean |
| Plain Conv3D "triad" trunk | triad_plainconv3d_320 (81, later) | 0.8145 (unpretrained, weak) |
| Joint-zoom crop second stage | (recent) | small gain |
| Public pipeline equivalents (slot readers, window readers with per-finding attention, CoAtNet gated spatial head with protected GAP residual, 3-zone depth adapter) | public_notebooks_0946_analysis.md | Their gains come from ensemble breadth (5 families, 47 members), not from the heads |

Not found in the history (so treated as untried): learnable generalized-mean or log-sum-exp pooling initialised at the mean; learnable per-view weights or per-view concatenation; per-view auxiliary losses; random slice or view dropout with renormalisation in training (slot masking exists only for missing slots); zero-parameter slice-shift inside the trunk; fixed (non-learned) smoothing along the slice axis before top-k; manifold mixup or slice-order flip as head-training augmentation. Please grep before running any of them in case a negative note exists that I missed.

## 2. What the literature says overall

- The strongest direct evidence supports our own finding. A 7-dataset benchmark on 3D neuroimaging (3 CT, 4 MRI, 662 to 21,744 scans) found mean pooling matched or beat Max, ABMIL, TransMIL and SmAP on 4 of 6 moderate-size tasks, and that fancier aggregators gave under 0.025 AUROC (under 0.01 on CT) for 25x the training time. Evidence A for "mean is hard to beat". https://arxiv.org/html/2604.26807
- A companion synthetic study shows that when instances are correlated, no common aggregator (Max, Mean, ABMIL, TransMIL) approaches the Bayes optimum even at N=10,000 (about 0.84 vs 0.882), and only smoothness-aware attention (smAP) got within about 0.014. Evidence B, synthetic. https://arxiv.org/html/2510.25759
- Attention pooling tuned in vision: sandwich normalisation and a concat of average pool plus attention pool were the best variants, yet all still lost to plain average pooling on small datasets. Evidence B, blog level. https://benjaminwarner.dev/2022/07/14/tinkering-with-attention-pooling
- Every RSNA volumetric winner I could check used a trained sequence model, but each also had stage-one cropping to the organ and, in most cases, a segmentation or position auxiliary signal, so their gains cannot be attributed to the aggregator alone: 2023 abdominal trauma 1st place (2.5D CNN + GRU, max over sigmoid slice predictions, auxiliary segmentation loss weight 0.125 said to add 0.01-0.03) https://github.com/Nischaydnk/RSNA-2023-1st-place-solution ; 2019 intracranial hemorrhage 1st place (slice embeddings into LSTM, then a second LSTM on logits plus slice position metadata) https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117210 ; 2022 cervical spine 1st place (2.5D CNN + LSTM over 15 slices per vertebra, per search snippet) https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution ; 2024 lumbar spine 2nd place (encoder, LSTM, then attention pooling, with sequence-flip augmentation and manifold mixup on embeddings) https://github.com/brendanartley/RSNA-2024-Competition . Their tasks have a localised ROI (one vertebra, one organ, one disc level), which a whole-knee study does not, which is part of why our sequence heads lose.
- Knee-specific: MRNet fuses views by max over slices then concatenation across planes. CoPAS (Nature Communications 2024, 12 knee abnormalities, 1748 patients) uses 3D ResNet-18 per sequence, a pooling that keeps the slice axis (so each depth position becomes a token), cross-plane attention, plane-aware fusion, and BCE on every branch plus focal loss on the fused output. Average AUC 0.812 vs MRNet 0.781. No clean ablation of the individual fusion parts was visible in the text I could read, so the gain cannot be assigned to any one component. Evidence B. https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947

## 3. Candidate ideas, assessed

### A. Learnable power-mean (GeM) or log-sum-exp pooling over the 24 vectors, initialised at the mean, per finding
- What: pooled = (mean(x_i^p))^(1/p) on a non-negative feature, or (1/r) log mean(exp(r * z_i)) on per-image finding logits z_i, with r (or p) a learnable scalar per finding initialised so it equals the mean (r near 0, p = 1). A convex mix alpha*mean + (1-alpha)*top2 with learnable alpha per finding is the same family.
- Evidence: LSE is a classic MIL operator (smooth max) https://proceedings.mlr.press/v95/yan18a/yan18a.pdf ; GeM beat avg and max in image retrieval and was shown to help classification https://arxiv.org/pdf/1711.02512 . Evidence for MRI slice stacks: none found (C). Our own measurement is the real support: top2 lifted MCL by +0.0329 while max hurt 4 of 4 (entry 69), which is exactly the signature of a finding wanting something between mean and max.
- Why it might work where ours failed: it starts as the flat mean, adds 12 to 24 scalars (not a head), and lets only the findings that benefit (MCL, perhaps ACL) leave the mean. Our per-finding pooling forced a hard choice from step 0 and trained the trunk from scratch with it; poolattn replaced the mean with a 0.14M-parameter module.
- Cost: very cheap on cached features (seconds on the 4060, extract_features.py output exists). Then a short fine-tune only if a finding moves.
- Risk: the cached features were trained under mean pooling (entry 65 caveat), so a cached test will under-state any gain; positive cached signal is meaningful, a null is not decisive.

### B. View-aware linear readout: per-view means, a learnable weight per view and finding, and per-view auxiliary heads
- What: pool each of the 4 views to a vector (mean), apply a per-view linear map to 12 logits, and sum (equivalent to a concat of 4 vectors into a 4096-d linear head). Optionally add a BCE loss on each per-view logit set (CoPAS style) with weight about 0.3. The flat mean is a special case that forces every view to have the same weight vector per finding.
- Evidence: CoPAS branch-wise BCE plus fused loss (B, but not isolated); multi-loss "auxiliary model per view" feature fusion is a standard recipe in multi-view imaging https://www.nature.com/articles/s41467-024-51888-4 (CoPAS) and the multi-view fusion reviews (C).
- Why it might work: it addresses the user's exact complaint (view identity ignored) with 3x more head parameters rather than a new mechanism. Entry 74 restricted which views a finding sees by anatomy and lost, but that was a hard prior in a from-scratch run; this lets the head learn the weights and starts near the mean if initialised with equal weights. Per-view auxiliary losses force each view's features to be informative on its own, which the mean-trained trunk does not require.
- Cost: cached features for the readout (4096 x 12 weights), a short fine-tune for the auxiliary loss.
- Risk: the trunk already produces mostly view-agnostic features, so the head may learn near-equal weights (a null result that still closes the user's question cleanly). Overfitting of 4x parameters on 4400 noisy-label studies.

### C. Random slice and view dropout with renormalisation during training
- What: during fine-tuning drop K of the 24 images (and sometimes a whole view) before the mean, rescale by N/(N-K) (or just average the survivors), keep all at test.
- Evidence: MIL-Dropout (drop top-k important instances and similar ones) improved ABMIL by about 2% on Camelyon16 and TCGA-NSCLC and 6-17% accuracy on small MIL benchmarks (B, histology, not MRI) https://arxiv.org/html/2504.14783v1 ; token random dropout in ASMIL curbs late-epoch degradation of attention MIL (B) https://arxiv.org/pdf/2603.06658 ; modality dropout is a standard robustness regulariser (C) https://arxiv.org/html/2608.23161v1 .
- Why it might work: with a mean over 24 images the model can rely on a few highly informative images and ignore the rest; dropping the top-contributing images forces redundant evidence, which also helps focal findings seen on 2 to 3 slices. Our noise level (report-derived labels) is the regime in which this regulariser is reported to help.
- Cost: a few lines in the engine, one short fine-tune from an existing 2.5D checkpoint (about the cost of a seed run). Not testable on cached features.
- Risk: dropping the only informative image of a focal finding creates label noise; use K small (2 to 4 of 24) or drop low-contribution images only.

### D. Fixed (parameter-free) local smoothing along the slice axis, then top-k
- What: for each view, take the 6 ordered vectors, average neighbours with a window of 2 or 3 (or a [0.25, 0.5, 0.25] kernel), then apply top2 or LSE across the smoothed positions, then mean across views. No learned parameters.
- Evidence: smoothness-aware attention (SmAP) is the only aggregator that came close to the Bayes bound on correlated instances (synthetic) https://arxiv.org/html/2510.25759 ; on real CT and MRI the benchmark shows only modest or no gain https://arxiv.org/html/2604.26807 (so rating C for knee MRI).
- Why it might work: a lesion spans 2 to 3 adjacent slices; smoothing before taking a max removes single-slice noise, which is what killed max pooling in entry 69 (noise gets selected). Entry 47 found group3 helped focal findings because it concentrates a lesion in one image, the same effect at feature level.
- Cost: cached features only, minutes.
- Risk: only 6 images per view, so the smoothing window covers a large fraction of the view; the gain, if any, will be on MCL and ACL.

### E. Manifold mixup and slice-order flip as head and fine-tune augmentation
- What: mix the pooled study embedding (or the 24 vectors) of two studies and their labels, and randomly reverse each view's slice order (harmless under mean, but keeps any order-aware component honest).
- Evidence: used in the 2024 lumbar spine 2nd place solution together with LSTM and attention pooling (B; no public ablation) https://github.com/brendanartley/RSNA-2024-Competition .
- Why it might work: noisy labels plus small data; mixup on embeddings is a known remedy for overfit heads. Our sequence-head sweep (entry 65) lost partly through overfit (0.56 to 0.80M parameters on frozen features).
- Cost: cached features, minutes; mixing embeddings with multi-label targets is valid for AUROC.
- Risk: small expected gain, mainly useful as a regulariser to rescue a larger head from ideas A, B or D.

### F. Zero-parameter slice shift inside the trunk (temporal shift module), last stages only
- What: in the last two ConvNeXt stages, shift a fraction (for example 1/8 forward, 1/8 backward) of channels along the image axis within a view (images of a view are consecutive slices), so each image sees neighbour feature maps before the global pool. Fine-tune from our 2.5D checkpoint.
- Evidence: TSM gives temporal modelling at zero parameters in 2D CNNs https://pubmed.ncbi.nlm.nih.gov/33035158/ ; Slice-Shift UNet brought +0.021 DSC (shift alone) and +0.062 with multi-view fusion on AMOS segmentation https://arxiv.org/html/2307.12853 . No MRI classification result found (C).
- Why it might work where cross-slice attention failed: shift adds no new weights, so it cannot wreck pretrained features the way new attention blocks trained from random did (entries 28-29, 41), and works at the feature-map level, which entry 6 argued was the right place. Our 2.5D input already shares 3 neighbouring slices, so the added context is modest.
- Cost: moderate (engine change, one fine-tune from checkpoint).
- Risk: highest of the six. Cross-slice fusion has been closed for ViT trunks; shifting channels perturbs pretrained statistics; expect at best a decorrelated ensemble member.

### Examined and not recommended
- Token-level MIL over patch tokens (TokenResidualPool style): already tried as spatial / tokenattn; no gain.
- ABMIL, TransMIL, DSMIL, CLAM replacements: the benchmark shows no robust gain over mean on 3D MRI (https://arxiv.org/html/2604.26807); DSMIL and CLAM instance losses mostly help localisation, not classification (https://pmc.ncbi.nlm.nih.gov/articles/PMC12343447/ and the MIL transfer review). Our attention runs agree.
- Max over slice logits as the RSNA 2023 winner used: we measured max losing 4 of 4 (entry 69); their max sat on top of a GRU and an organ crop.
- Late fusion of separate per-view models: costs 4 trunks; the ensemble line already shows low returns (entry 75).
- Slice position as an explicit scalar (2019 hemorrhage 1st place): our views are 6 images each, so position resolution is poor; already approximated by ordered per-view layout.
- 2+1D slice-axis 1D convolution on cached vectors: effectively the GRU/tfm sweep (entry 65), which lost.

## 4. Ranked shortlist with first experiments

1. A. Learnable per-finding LSE/power-mean (or mean/top2 mix) pooling, mean-initialised. First experiment: extend seq_head_sweep.py with a head `lse`: per-image logits z = W x_i, pooled logit = (1/r_f) log mean exp(r_f z_i), r_f init 1e-3, 12 scalars, plus an `mix` variant alpha_f * mean + (1 - alpha_f) * top2 of z_i. Report per-finding deltas against the 0.9042 mean baseline over the same seeds. Success: MCL and at least one other finding above seed spread with the others within 0.0005.
2. B. View-weighted readout. First experiment: same cached features, `viewcat` head = per-view mean then 4 x (1024 to 12) linear, init all views to the flat-mean weights; plus `viewcat_aux` with per-view BCE at weight 0.3. Compare to mean and log the learned view weights per finding to see whether the user's "view matters" intuition is visible.
3. C. Slice and view dropout fine-tune. First experiment: add `train.slice_drop` (drop K=3 of 24 images, renormalise) and `view_drop` p=0.1 to the engine; short fine-tune from convnext_base_25d_18slices checkpoint, 3 epochs at the low final learning rate, compare val and audit-58 against a same-length no-dropout continuation (important control).
4. D. Smoothed top-k. First experiment: on cached features, per-view kernel [0.25, 0.5, 0.25] (edge-replicated) then top2 across 6 positions on logits, then mean across views; also mean(smoothed top2, flat mean). Seconds to run, compare per finding.
5. E. Manifold mixup and flip on cached embeddings. First experiment: re-run the gru and tfm heads of entry 65 with embedding mixup (alpha 0.4) and sequence-flip; the question is whether a regularised order-aware head can reach 0.9042. If it still loses, order-awareness is closed for good with a regularisation confound removed.
6. F. Slice-shift in the last two stages. First experiment: patch the ConvNeXt stages 3 and 4 with a shift op over the image axis within each view (1/8 channels each direction), load the existing checkpoint, fine-tune 3 epochs, and compare solo and as an ensemble member. Only do this if 1 to 4 show any positive cached signal.

Expected honesty check: the best outside evidence (Section 2) says the mean is hard to beat on 3D MRI at this data size, and our own entries 52, 53 and 65 agree. The most defensible expectation is a small per-finding gain (MCL, maybe ACL) from ideas 1 and 4, and a regularisation gain from 3. Ideas 2 and 5 are mostly diagnostic.

## 5. Source list

- https://arxiv.org/html/2604.26807 (multi-dataset MIL benchmark, 3D neuroimaging)
- https://arxiv.org/html/2510.25759 (correlated MIL, synthetic)
- https://arxiv.org/html/2504.14783v1 (MIL-Dropout)
- https://arxiv.org/pdf/2603.06658 (ASMIL, token random dropout)
- https://arxiv.org/html/2408.09449v1 (FocusMIL, max pooling and the standard MIL assumption)
- https://benjaminwarner.dev/2022/07/14/tinkering-with-attention-pooling
- https://proceedings.mlr.press/v95/yan18a/yan18a.pdf (dynamic pooling in deep MIL, includes LSE)
- https://arxiv.org/pdf/1711.02512 (GeM pooling)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947 (CoPAS, knee, 12 abnormalities)
- https://github.com/zqiuak/CoPAS
- https://arxiv.org/html/2005.02706v3 (ELNet) and https://arxiv.org/html/2108.08136 (spatial attention for MRNet) as knee context
- https://github.com/Nischaydnk/RSNA-2023-1st-place-solution
- https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117210
- https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution (content via search snippet only)
- https://github.com/brendanartley/RSNA-2024-Competition (RSNA 2024 2nd place)
- https://pubmed.ncbi.nlm.nih.gov/33035158/ (TSM) and https://arxiv.org/html/2307.12853 (Slice-Shift UNet)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12343447/ (learnable context in MIL; DSMIL and CLAM background)
- https://arxiv.org/pdf/2512.12887 (AnyMC3D, 2D foundation models for 3D classification; abstract only, not used for numbers)
