# Slices, slots (MRI sequences) and axes: aggregation research (8 Oct)

Scope: how to handle slices, slots and the three planes (sagittal, coronal, axial) and how to combine them. Research only, no runs, no edits outside this file.

Terms: Area Under the Receiver Operating Characteristic curve (AUROC), Multiple Instance Learning (MIL: one label for a bag of instances, here a study of many slices), Long Short-Term Memory network (LSTM), Gated Recurrent Unit (GRU), Test-Time Augmentation (TTA), Multi-Layer Perceptron (MLP), Magnetic Resonance Imaging (MRI), Convolutional Neural Network (CNN), Vision Transformer (ViT), Log-Sum-Exp (LSE, a smooth maximum), Generalized Mean (GeM, a power mean), Out-Of-Fold (OOF, predictions on a fold the model never trained on).

Plain-language anatomy for this topic: a knee MRI study is a stack of 2D images ("slices") per sequence ("slot"). Sagittal slices cut left-to-right (side view), coronal cut front-to-back, axial cut top-to-bottom. The anterior cruciate ligament (ACL) runs diagonally inside the knee and shows best on sagittal. The medial collateral ligament (MCL) is on the inner side and shows best on coronal. Menisci (C-shaped cartilage pads) and bone bruises (contusions) are spread over several slices, so no single slice is enough.

## 0. Already in our repo (do not re-run without a new angle)

Source: `other_md/research_results/slice_aggregation_ideas_research.md` section 1, plus `runs_summary.md`.

| Idea | Verdict in repo |
| --- | --- |
| Transformer over the 96 pooled slice vectors | Lost -0.0148 (convnext_small_transformer_pool) |
| Cross-slice attention blocks inside a ViT | Weak solo, MAE variant useful only as ensemble member |
| Slices as channels (4 to 8 per study) | Lost -0.025 to -0.038 |
| 3-slice 2.5D grouping | Adopted as standard (cheaper, -0.003 solo) |
| 5th view slot (cor_plain) | Negative |
| Gated residual slot attention, token attention | Neutral, logits correlate 0.994 with parent |
| Learned MLP weighting replacing mean | Worst solo (0.8977) |
| 12 label queries over 2400 spatial tokens | -0.003 to -0.007 |
| Per-finding pooling (max / top2 / mean) and per-finding view restriction | Max lost 0/4; top2 helped MCL +0.033 but total -0.0035; restricting views hurt |
| BiGRU / Transformer / attention heads on cached frozen features | Mean pooling best (0.9042 vs GRU 0.8999) |
| Exhaustive ensemble nesting | Held-out -0.0003 |
| Sliding-window and reverse-slice TTA | Closed (reverse is vacuous under a mean) |
| Plain Conv3D trunk | Weak (0.8145) |

Not yet tried per the repo notes: learnable GeM / log-sum-exp pooling, learnable per-view weights or per-view concatenation, per-view auxiliary losses, random slice or view dropout, fixed smoothing before top-k, manifold mixup or slice-order flip.

## 1. What the literature and winners actually did

### 1a. MRNet (Stanford knee MRI, Bien et al. 2018)
- Architecture: one ImageNet-pretrained 2D CNN per plane, run over every slice, features pooled over slices, then a small logistic regression that combines the three planes' outputs. Plane-level outputs are combined late (after pooling), not by concatenating features.
- Source: MRNet paper, PLoS Medicine 2018 (search: "MRNet Bien 2018 knee"). This is the baseline all follow-ups compare against.
- Relevance: our 2D ConvNeXt + mean pooling is an MRNet descendant. The main upgrade we have not tested is a learned plane combiner (1b) instead of one shared head over pooled features.

### 1b. Plane fusion evidence (mixed, important)
- Dergipark 2026 multi-plane study (MRNet-based, https://dergipark.org.tr/en/pub/jai/article/1926635): ACL best single plane = sagittal AUC 0.976; three-plane Hybrid Fusion = 0.949. So naive fusion LOST to the best single plane. Meniscus: fusion sagittal 0.791, axial 0.826. Mamba variant had lower mean AUC and higher variance.
- OrthoDiffusion (arXiv 2602.20752, musculoskeletal MRI): concatenation of planes gave the best macro-average AUROC; their adaptive per-patient, per-label plane gating (MPAE) was proposed on top. Fetch failed (file too large), so only the search snippet was read. Treat the numbers as unverified.
- GCCE 2025 (Waseda, sagittal T2 + coronal T1 + axial PD): attention fusion across sequences beats uniform averaging, motivated by sequences carrying different information. Numbers not retrieved.
- Conclusion from the sources: fixed fusion does not reliably beat the best plane. Gains appear only when the weights are LEARNED per finding and per study. Our per-finding restriction results agree (hard restriction hurt, soft weighting neutral).

### 1c. RSNA 2022 Cervical Spine Fracture (CSFx), 1st place write-up (secondary: Zenn, yuu, https://zenn.dev/yuto_mo/articles/f9fb1c8297f252)
- Stage 1: 3D segmentation of 7 vertebrae.
- Stage 2: each vertebra cropped separately; 15 slices taken evenly along the z axis; each slice is a 5-channel image (the slice plus 2 neighbours on each side), plus the vertebra mask as a 6th channel. A 2D CNN encodes each slice, an LSTM runs over the 15 slices to predict fracture for that one vertebra.
- Model 2: all 7 x 15 images fed together for a patient-level label.
- Lessons: (i) evenly spaced sampling (15 of many) is enough; (ii) per-region crops with the region mask as an extra channel; (iii) 5-channel 2.5D with a wider neighbourhood than our 3-slice version. Note the repo already found "slices as channels" harmful, so only the per-region crop idea transfers.
- No ablation numbers in the page.

### 1d. RSNA 2023 Abdominal Trauma, top solution (secondary: review in arXiv 2512.12887 and a summary, not the original write-up)
- Reported design: 96 CT slices grouped into 32 triplets; a 2D CNN (CoaT-Lite or EfficientNet-V2-S) per triplet; a GRU over triplets; auxiliary segmentation heads; max pooling over slice logits for the final prediction.
- Relevance: the max-over-logits step is the one aggregation we have not tested at logit level (our per-finding max was on features). Unverified details, treat as a lead.

### 1e. RSNA 2024 Lumbar Spine, 2025 Intracranial Aneurysm
- Could not retrieve write-ups. The 2024 winner (team Avengers, per RSNA results) and the 2025 winner (tomoon33, per RSNA results) have Kaggle write-up pages; the 2025 ones describe multi-axis (three-plane) ROI extraction with nnU-Net and a 3D multi-task head that predicts segmentation and classification jointly (secondary, Japanese slide deck). Not verified. Action for the user: read the Kaggle write-ups of Avengers (2024) and tomoon33 (2025) directly; I could not access them.

### 1f. Benchmarks (arXiv 2512.12887, "Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification")
- Lists these aggregators: mean, median, query-based attention pooling, bidirectional LSTM (the RSNA-Kaggle entry), Transformer (MST), and multi-plane fusion (M3T, fusing three orthogonal directions).
- Reported averages (different backbones, not isolated): AnyMC3D (MII backbone) 0.894 and AnyMC3D (DINOv3) 0.894 avg AUC, MST Transformer 0.869, RSNA-Kaggle BiLSTM 0.810. The aggregator ablation (Table 6) was cut off, so no clean mean-vs-attention number.
- Use: query-based attention over slices is a documented, competitive option; their own numbers cannot isolate it.

### 1g. Sequence models over slices (2025-2026)
- Glioblastoma follow-up MRI benchmark (arXiv 2511.18595): Vision-Mamba tiny per slice, mean-pooled over 50 slices. No head-to-head vs LSTM.
- Mamba scan order matters a lot (up to 27 Dice points, arXiv 2025 scan-order study). Not a classification result, but it warns that any sequence model is order-sensitive.
- No credible paper compares Mamba vs LSTM over the same slice features for knee classification. Low priority for us.

## 2. Candidate ideas, ranked

Scoring: expected leaderboard gain per dollar (GPU hours). Costs assume our cached per-slice features where noted; "cache" means the trunk is run once and only the head is trained, so GPU cost is minutes.

### Rank 1. Per-view (per-slot) heads with a learned, per-finding plane combiner (MRNet-style, learned weights)
- What: one linear head per slot (sag_plain, sag_fluid, cor_fluid, ax_fluid) on that slot's pooled features, plus per-slot auxiliary losses; combine the four slot logits with learned weights per finding (12 x 4 weights, softmax, initialised uniform), trained on OOF predictions or jointly.
- Repo status: "per-view auxiliary losses" and "learned per-view weights" are listed as untested. Restriction (hard routing) was tested and hurt; soft weighting was not.
- Evidence: MRNet plane combiner (1a); learned gating beats fixed fusion (1b); OrthoDiffusion MPAE; GCCE 2025.
- Why it fits us: our slots differ in content (fluid-sensitive vs plain), and the mean across all 96 images gives a sag_plain series the same weight as an ax_fluid series for every finding. Learned weights let ACL come from sagittal and MCL from coronal without hard routing.
- Cost: cache: near zero. Full retrain: about 1 h per fold on T4-class hardware with the current ConvNeXt-Base 2.5D. Fits 9 h inference (combiner is 48 weights).
- Risk: low to moderate. Fusion lost to the best single plane in 1b, so use learned weights with a fallback to the mean; judge on OOF, not val.

### Rank 2. Learnable GeM or log-sum-exp pooling across slices, initialised at the mean, per finding (repo idea A, untested)
- What: p-norm / LSE pooling with one learned exponent per finding and per slot. Initialised so it equals the mean.
- Evidence: standard in weakly-supervised MIL (MIL pooling literature); the repo's own note that the flat mean is "close to optimal" means the expected gain is small, so measure on cached features first.
- Cost: cache, minutes. Risk: low (can only move 12 + 4 parameters).

### Rank 3. Logit-level max or LSE over slices, per slot, blended with the mean (from RSNA 2023 abdominal trauma, 1d; repo tested feature-level max only)
- What: final logit = w * mean(logits) + (1 - w) * LSE(logits over slices), w learned per finding.
- Evidence: the 2023 top solution used max over slice logits (secondary source). Feature-level max lost, logit-level not tested.
- Cost: cache. Risk: low; the blend weight prevents collapse.

### Rank 4. Gated attention MIL over slices per slot (Ilse et al. 2018 gated-attention form), with the attention scores exposed as an interpretable output (repo idea variant of "learned MLP weighting", which lost)
- What: attention a_i = softmax(w^T tanh(V h_i) * sigmoid(U h_i)) per slot and per finding; study vector = sum a_i h_i.
- Evidence: gated-attention MIL is the standard fix for "which slices matter"; query-based attention pooling is competitive in 1f. But the repo's MLP-weighting and token attention were both neutral or negative, so the expected gain is low.
- Cost: cache. Risk: moderate (overfitting with 4,407 studies; the repo's own result warns of this).

### Rank 5. Per-region (per-compartment) crops with the region mask as a channel (from CSFx 1st place, 1c)
- What: instead of whole slices, crop each compartment (medial meniscus region, lateral meniscus region, ACL notch, patella region) via a cheap localisation model, and feed crops with a mask channel. Equivalent of the CSFx per-vertebra trick.
- Evidence: CSFx 1st place (secondary); the repo's joint-zoom crop gave a small gain already, so crops are not dead.
- Why it fits us: menisci and ACL are small; we currently feed whole 336-512 px slices where a meniscus tear may be a few pixels.
- Cost: high. A localiser (landmark or box model) plus training a second stage: several GPU hours and an extra labelling step. Fits 9 h only if the localiser runs once per study and is cached. Risk: localisation error (no ground-truth boxes in our data; would need a proxy).

### Rank 6. Cross-plane attention on the four slot tokens only (4 tokens per study, plane embeddings), applied on top of cached per-slot features
- What: a 2-layer transformer over 4 slot tokens (one per plane/slot), with plane embeddings; output replaces the mean head.
- Evidence: M3T multi-plane fusion (1f). Repo: transformer over 96 pooled slice vectors lost (-0.0148) but that was 96 tokens from a randomly initialised head on a pretrained trunk; 4 tokens and a linear-probe warmup is different. The repo's seq-head sweep on frozen features (transformer 0.8977) is the closest prior test, so expect a small gain at best.
- Cost: cache, minutes. Risk: low to moderate.

### Rank 7. Random slot dropout and slice dropout with renormalisation during training (repo idea C, untested)
- What: during training, drop each slot with p=0.1 and renormalise the mean; drop 20% of slices.
- Evidence: standard regularisation; the 5-slot test and the missing-slot handling suggest the model already copes with missing series.
- Why it fits us: slots are often missing in real studies; training on drops matches test-time missingness and may improve calibration across slots.
- Cost: one normal training run per candidate (about 1 h per fold). Risk: low.

### Rank 8. Five-channel 2.5D (slice plus two neighbours each side), CSFx-style (1c)
- What: widen the 3-slice RGB stack to 5 neighbours (5-channel input, pretrained stem adapted by channel averaging).
- Evidence: CSFx 1st place used 5-channel input (secondary). Repo found "slices as channels" harmful (4-8 passes), but that stacked slices as separate passes, not one wider stack.
- Cost: one run per fold (about 1 h per fold). Risk: moderate; the pretrained stem needs surgery and the 3-slice grouping was the cheaper winner.

### Rank 9. Evenly spaced key-slice sampling at larger spacing, plus fixed smoothing before top-k (repo idea D, untested)
- What: average each slice with its neighbours (fixed kernel), then take top-k slices by a cheap score; or sample 15 evenly spaced slices as in CSFx.
- Evidence: CSFx 15-slice sampling (1c); fixed smoothing is an untested repo idea.
- Cost: cache for the scoring stage; a retrain for the sampler. Risk: low on cache, moderate on the sampler (may drop focal lesions).

### Rank 10. Stacking meta-learner per finding over OOF per-view and per-model predictions (MRNet-style logistic combiner over our 4-slot heads, not over features)
- What: a logistic regression per finding over the OOF logits of each slot head and each ensemble member.
- Evidence: MRNet final step (1a). Repo: "exhaustive ensemble-structure search" gave +0.0023 in sample, -0.0003 held out, so stacking may repeat that result.
- Cost: minutes on existing OOF. Risk: low, but expect the held-out result to be near zero.

## 3. Novel combinations (not tried anywhere we could find)

- N1. Learned plane combiner with label-specific plane priors (MRNet-style logistic) initialised from anatomy: ACL weights high on sagittal, MCL high on coronal, meniscus on axial plus sagittal, all learnable with the anatomy as a prior, not a hard route. Combines the MRNet combiner (1a) with the repo's failed hard routing: soft prior instead of a route. Cost: same as Rank 1. Risk: low.
- N2. Slot tokens plus slice-position embeddings inside the cached pooled features: treat each slot's 24 slice vectors as a 1D sequence with learned position embeddings, then a 1-layer transformer per slot, then cross-slot attention. Splits the failed 96-token test into two smaller levels. Cost: cache. Risk: moderate.
- N3. Distillation from the per-slot heads to the mean head (self-distillation across views): train the mean head to match the best per-slot combiner's logits. Cheap, makes the single head carry the per-view information without changing inference cost. Cost: one run per fold. Risk: low.

## 4. Key caveats and forum evidence

- Forum (kaggle thread 735304, read 7 Oct in other_md/research_results/big_jump_ideas_oct7.md): 5-fold ResNet-50 at 224 px reaches 0.954; our 512 px single models peak 0.934. The forum view is that labels (pseudo-label quality) matter more than architecture. So aggregation work is worth doing only if cheap (cache-based ideas Ranks 1-3, 6, 10).
- Every fusion result in the literature used one dataset with strong single-plane signal (ACL). Our task has 12 labels with a mix of focal (meniscus, fracture) and diffuse (effusion, synovitis) findings; a learned combiner should be checked per finding.
- The sources above are mostly from MRNet-style ACL and meniscus studies, not the RSNA knee set. Measured gains there do not transfer directly.

## 5. Ranked top 10 (expected leaderboard gain per GPU dollar)

1. Per-view heads + learned per-finding plane combiner (Rank 1, cache-cheap, MRNet-based).
2. N1 anatomy-prior soft plane weights (novel, same cost as 1).
3. Logit-level LSE/max blended with mean (Rank 3, cache).
4. Learnable GeM / LSE pooling per finding (Rank 2, cache).
5. Cross-slot transformer over 4 slot tokens on cached features (Rank 6, cache).
6. Random slot and slice dropout during training (Rank 7, one run per fold).
7. N3 self-distillation from per-view combiner to mean head (one run per fold).
8. Per-region crops with mask channel (Rank 5, expensive, highest upside).
9. Five-channel 2.5D (Rank 8, one run per fold, moderate risk).
10. Gated attention MIL with interpretable scores (Rank 4, cache; likely near zero given repo history).

## 6. Sources

- MRNet, Bien et al. 2018, PLoS Medicine (search only; paper not fetched).
- Multi-plane hybrid fusion, Dergipark 2026: https://dergipark.org.tr/en/pub/jai/article/1926635
- OrthoDiffusion, arXiv 2602.20752: https://arxiv.org/pdf/2602.20752 (fetch failed on size; snippet only)
- Revisiting 2D foundation models for 3D medical classification, arXiv 2512.12887: https://arxiv.org/html/2512.12887v3 (partial read)
- CSFx 1st place summary, Zenn yuu: https://zenn.dev/yuto_mo/articles/f9fb1c8297f252
- Abdominal trauma 2023 top solution, secondary via arXiv 2512.12887 and an online summary (not the original write-up).
- Glioblastoma Mamba benchmark, arXiv 2511.18595: https://arxiv.org/pdf/2511.18595
- Kaggle RSNA 2024 lumbar and 2025 aneurysm winner lists: https://www.rsna.org/news/2024/november/2024-ai-challenge-winners and https://www.rsna.org/news/2025/november/2025-ai-challenge-winners (write-ups not retrieved).

Not verified (flag before relying on): RSNA 2023 abdominal trauma 1st-place architecture; RSNA 2024 lumbar and 2025 aneurysm write-up contents; OrthoDiffusion MPAE numbers; GCCE 2025 numbers.
