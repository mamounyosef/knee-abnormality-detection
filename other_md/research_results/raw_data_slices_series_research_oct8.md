# Raw data side: DICOM series, slices, slots, planes, intensity (8 Oct 2026)

Scope: research only. No training, no Modal, no repo edits except this file. Every number below was measured on local CSVs, the local shards' metadata, or cached out-of-fold (OOF) logits, with scripts in the session scratchpad (not in the repo). Ideas already tried are checked against outline.md, runs_summary.md, runs_timeline.md and the other four 8 Oct reports.

Terms: Area Under the Receiver Operating Characteristic curve (AUROC), Out-Of-Fold (OOF, predictions on a fold the model never trained on), Test-Time Augmentation (TTA), Field Of View (FOV), Image Position/Orientation Patient (IPP/IOP, DICOM tags giving each slice's 3D position and direction in the scanner frame), Fat Suppression (FS, a scan setting that darkens fat so water looks bright), Proton Density (PD), T1/T2 (two basic MRI contrasts), Class Activation Map (CAM).

## 0. Bottom line

The raw-data side is mostly saturated by what we already ran (slots, slices, crops, zoom, resolution, six views, duplicates, TTA, positions, flip, metadata). Three of the checks the coordinator listed (wrong-plane series, patient overlap, metadata/intensity-descriptor inputs) are dead, with numbers in section 2. Two genuinely new levers survive, and one of them is backed by a data fact nobody had measured:

- **Same-plane fluid and plain series share one scanner geometry** (median centre offset 0.0 mm; 75th percentile under 0.1 mm for sagittal and coronal). That makes pixel-level cross-sequence fusion (fluid + plain as channels of one image) possible without image registration. We only ever fused sequences late (after pooling). Radiologists call bone bruise and fluid-vs-fat by comparing sequences pixel by pixel. And the 2.5D neighbour-slice channels were measured null, so the three colour channels are "free".
- **Intensity normalisation was never varied** (every run uses the volume 1-99 percentile window). Cheap to test, and a good source of ensemble diversity.

Honest expectation: each is worth about +0.001 to +0.004 on the leaderboard (the OOF-to-leaderboard slope is 0.48), not a +0.02 jump. Subgroup analysis (section 2.6) shows no raw-data attribute where the model is badly broken, so no single data fix can open a big gap.

## 1. Facts measured on our data

| Fact | Number | Source |
| --- | --- | --- |
| Series types | exactly 6 (sag/cor/ax x fluid/plain); no localizers or scouts; 3 to 14 series per study (median 5) | 07_series_index.csv |
| Slot presence | sag_plain 96.8%, sag_fluid 94.2%, cor_fluid 96.4%, cor_plain 77.3%, ax_fluid 100%, ax_plain 19.4% | same |
| Studies with a duplicate series in some slot | 43.9% | same |
| Fluid + plain pair exists in the same plane | sagittal 91.0%, coronal 73.7%, axial 19.4%; all three planes 12.8% | measured |
| Plane label vs plane from IPP | 0 mismatches in 24,371 series; vs series-description text 62 of 18,126 (0.34%, the text is wrong, not the label) | IPP axis, a2.py |
| Slice spacing uniform inside a series | max gap / median gap = 1.0 at the 99th percentile; 0 series with a gap above 1.5x | slice_stats |
| 3D/thin series (spacing under 1.6 mm) | 565 studies (12.8%); 3.3% of the picked slot series (ax_fluid 6.9%, sag 3.0 to 3.3%, cor 0%); median 144 slices of 0.8 mm | picks by `coverage` rule |
| Wide frames (640x1280) | 259 series, all ax_fluid (6% of studies), one protocol, knee centred (centre at 0.50 of width) but filling 38% of the frame | series_stats |
| FOV | median 160 mm (5% to 95%: 140 to 189); 99% of series are under 205 mm, so every 512 px frame is padded with black; 336 px (134 mm) crops the central 84% of a typical FOV | series_stats |
| Tight-box centre offset | median 4.9 mm, 95th percentile 13 mm: knees are centred | series_stats |
| Distance between the picked slices | median 5.3 mm (5th to 95th: 4.2 to 7.1) against 3 mm slices; axial 6.3 mm. With 18 picks, 35 to 43% of stored slices are never seen at evaluation | picks, a7.py |
| Slice normal vs nominal axis (obliquity) | median angle sagittal 7.5 deg, coronal 8.4 deg, axial 4.2 deg; 90th percentile 17 to 19 deg for sag/cor | shard IOP |
| Cross-plane geometry | sag-cor centre distance median 9.9 mm, sag-ax 7.3 mm (different FOV prescriptions, same patient frame) | shard IPP/IOP |
| Same-plane fluid vs plain geometry | centre offset median 0.0 mm; share under 3 mm: sag 92%, cor 87%, ax 70%; share under 8 mm: 97%, 91%, 79%; normal angle under 3 deg: 97%, 90%, 92% | shard IPP/IOP, a9.py |
| Contrast mix inside "plain" (TE/TR rule: TR under 1000 = T1; else TE under 60 = PD; else T2) | sag_plain: T1 2038, PD 1747, T2 1174, unknown 238. cor_plain: T1 2896, PD 539, T2 549. ax_plain: T1 678, PD 269, T2 223. So "plain" = "not flagged fluid", it is not one contrast. | series_stats |
| Vendor / field | Siemens ~42%, GE ~20%, Philips ~31%, Toshiba 4%; 1.5 T 58%, 3 T 37% | series_stats |
| Window headroom | median 194 grey levels between the 1st and 99th percentile of the stored uint8 (5th percentile 132; under 100 levels in 0.4% of series) | slice_stats, a5.py |
| Percentile window top sits at the 97th percentile of the foreground on average (5% to 95%: 93.4 to 98.3) because 66% of pixels are background | measured |

## 2. Dead or closed by this session's numbers (do not spend on these)

1. **Wrong-plane or mislabelled slot picks.** IPP axis agrees with the plane flag for all 24,371 series. The only disagreements are 62 series where the description text disagrees (the text is the unreliable one). Fat-sat flag is the competition flag (fluid equals fat suppression everywhere); in descriptions, STIR/SPIR/SPAIR appear in plain slots 0 to 1% of the time. Nothing to clean.
2. **Patient overlap / near-duplicate patients.** `patient_id` is unique per study (4,407 of 4,407), `FrameOfReference` unique per series, no age, no dates, no institution. The earlier report-text similarity (2.6% of val within 0.95) is the only trace. No leak to exploit, in either direction.
3. **DICOM metadata as model input.** Study-level metadata alone (vendor, model, field strength, per-slot geometry and TE/TR, side) gets 0.63 macro AUROC (ACL 0.69) with gradient boosting (a4.py). This was already stacked on top of the ensembles and lost (`protocol_features_test.py`, timeline: audit -0.0008 to -0.0027), and sex adds nothing (`sex_signal.py`). Closed.
4. **Global intensity-histogram descriptors as input.** New check: five tail ratios (p95/p99, p99/p99.9, p50/p99.9, p5/p99, foreground fraction) for each of four slots predict Synovitis at 0.815, Baker's 0.790, Effusion 0.776 alone (macro 0.699), because bright fluid fattens the histogram tail. Added to the 2-family OOF logits (ConvNeXt folds + MedImageInsight folds, 4,342 studies): 0.9186 to 0.9158 (-0.0028), audit-58 0.9287 to 0.9226. The CNN already reads it. Closed (a14.py).
5. **Higher bit depth (uint16).** The stored uint8 leaves a median 194 grey levels inside the 1-99 window, under 100 in 0.4% of series. Not a bottleneck.
6. **A raw-data attribute where the model is broken.** Macro AUROC of the 2-family OOF logits (v3 labels), by subgroup:

   | Subgroup | AUROC | n |
   | --- | --- | --- |
   | all | 0.9189 | 4,342 |
   | GE / Philips / Siemens / Toshiba | 0.9221 / 0.9167 / 0.9162 / **0.9017** | 852 / 1281 / 1924 / 180 |
   | 1.5 T / 3 T | 0.9169 / 0.9208 | 2516 / 1569 |
   | has a 3D thin series / not | 0.9204 / 0.9188 | 655 / 3687 |
   | native pixel spacing fine / mid / coarse | 0.9170 / 0.9264 / **0.9116** | ~1450 each |
   | thickness at most 2.5 / 3 / at least 3.5 mm | 0.9253 / 0.9165 / 0.9175 | 734 / 2518 / 1090 |
   | has a duplicate series / not | **0.9129** / 0.9222 | 1900 / 2442 |
   | a missing slot / none | 0.9215 / 0.9183 | 552 / 3790 |

   Everything sits in 0.912 to 0.926. Thin 3D series are not a weak spot, so slab averaging of 3D series (a candidate I had) is dropped as a stand-alone idea. The two soft spots (Toshiba, coarse native resolution, studies with duplicate series; each about -0.005 to -0.01) are small and have no cheap fix (duplicates were already tested as TTA and schedule).
7. **Quantisation, bit-depth, bias-field (N4), Nyul, subtraction, localizers.** See section 4; evidence says small effect relative to what we already have.

## 3. Ranked top 10 (expected leaderboard gain per dollar)

Costs use $5 per H100 hour; a 12-epoch 2.5D ConvNeXt-Base run is about $5 to $9 on Modal (the round-2 folds cost $4.6). "Shard rebuild" is never needed here: the shards keep every slice at native resolution plus pos_first, pos_last, orientation and pixel spacing per series. T4 inference is judged against a current 2.5D member (about 24 images per study).

### Rank 1. Same-plane cross-sequence early fusion (fluid + plain as channels of one image). NOVEL

- What: for each plane, pick the fluid series A and the plain series B of the same study. For each picked slice of A at position t along the slice normal, take the nearest slice of B (positions are exactly linear: ipp0 + i*(ipp1-ipp0)/(n-1), verified uniform). The three colour channels become [A(k), B(nearest to k), A(k+1) or mean of A neighbours]. Three streams (sag, cor, ax) of 6 images each = 18 images per study instead of 24. Where B is missing (cor 26%, ax 81%) fill with A neighbours exactly as today. Align in-plane by translation from the IPP difference of the two slices projected on the row and column vectors (median 0.0 mm, 90th percentile 1.5 mm sag, 6.9 mm cor, 13.9 mm ax), so no image registration is needed.
- Evidence and rationale: early fusion of co-registered sequences is the BraTS standard (channel stacking) and is how a radiologist reads bone bruise (bright on fat-suppressed water image, dark on T1 or plain; marrow fat vs edema) and tells fluid from fat. Our current design fuses only after pooling, so the network cannot compare two sequences at the same pixel. Previous "channels" failures were about widening the stem (information budget); this keeps `in_chans=3`, and the 2.5D neighbour channels were measured null (convnext_small_25d -0.0005 val), so spending them on a second sequence costs nothing. Our own six-view result (flat) tells us more late-fused views are not enough, not that cross-sequence pixel contrast is useless.
- Our data: pair available in 91% (sag), 73.7% (cor), 19.4% (ax) of studies; geometry identical to the scanner (offset 0.0 mm median). Residual patient motion between series is not in the DICOM geometry and is the main risk. Findings expected to move: Contusion, Fracture (marrow), Synovitis, Effusion, Baker's (fluid vs other tissue), PF OA via axial where available.
- Cost: engineering about 1 day (Dataset pairing path, `plan_slots` already knows both series; the Kaggle path `dicom_series_volume` must also return slice positions, then a parity check like the existing ones). One run about $4 to $6 (25% fewer images than a 4-slot member). T4 inference about 25% cheaper than a standard member. Local check on a few studies before launch.
- Risk: moderate (motion misalignment; stem trained on different channel meaning, but ImageNet colour semantics already are arbitrary). A twin with the same recipe and only this change gives a clean answer. Also a strong diversity candidate for the ensemble (input representation differs from every member).
- Expected: solo -0.002 to +0.005; ensemble +0.001 to +0.004. Highest ceiling per dollar of anything left on the raw-data side.

### Rank 2. Intensity-normalisation variants, as ensemble members and as a zero-training TTA test. NOVEL (window never varied)

- What: replace the per-volume 1-99 percentile window with (a) foreground-referenced scaling (Otsu mask, divide by the foreground median or a 20-60 percentile mean, fixed gain, top clipped at a fixed multiple), (b) z-score on the foreground (nnU-Net style), (c) CLAHE on the 8-bit slice, (d) the existing `window: slice` option. Ordered steps: first a no-training test (score an existing checkpoint on the 346+58 val under 1 to 2 variants, and a mean over windows as TTA); then train the winning variant as a member.
- Evidence: a 2026 knee-MRI meniscus benchmark (arXiv 2607.20028) compared seven normalisations on a 3D U-Net: all similar internally, but z-score, Nyul and CLAHE significantly more robust on an external dataset; authors stress the difference is small next to domain shift. nnU-Net uses foreground z-score for MRI as default.
- Our data: the window top sits at the 97th foreground percentile on average but spans 93.4 to 98.3, and the ratio p99/p99.9 spans 0.52 to 0.93 (5th to 95th), 0.65 in fluid slots vs 0.87 in plain ones. Because the top is set by the brightest 1% of all pixels, it moves with how much fluid is present, so fluid brightness relative to muscle is not preserved. The tail shape alone predicts Synovitis 0.815 (section 2.4), so the intensity statistics carry real signal and are scaled by a pathology-dependent window. A reference-tissue scale keeps "fluid brightness relative to muscle" intact.
- Cost: loader-only change (the shards keep a 0.1-99.9 uint8 and the percentile table); the no-training test is local and free; each member run about $5 to $9. T4 cost unchanged.
- Risk: low. Expected gain: +0.000 to +0.003, mostly as ensemble diversity (same architecture, different input scaling decorrelates more than a seed change). Window TTA is bounded the same way the other TTA were (+0.0004 sliding windows), so treat it as a bonus.

### Rank 3. Bin-mean slab input (use every stored slice instead of picking one per bin). NOVEL

- What: today each of the 18 bins contributes one slice (centre at evaluation, random in training), so 35 to 43% of stored slices are never seen at evaluation and the gap between picks is 5.3 mm (axial 6.3 mm) against 3 mm slices. Instead feed the mean (or mean+max mix) of all stored slices inside the bin: same 18 images, same cost, every slice contributes, noise falls by about 1.3x, and a thin lesion cannot sit in a skipped slice. For the 12% of studies with a 3D thin series this also turns 0.8 mm single slices (13% of the depth seen) into about 5 mm slabs, matching the other series.
- Evidence: standard radiology practice (isotropic 3D reformatted to 3 to 5 mm thick-slab slices); RSNA 2022 cervical-spine winner sampled evenly and used neighbour context; "slices 24 vs 18" and sliding-window TTA gave no gain, which says more distinct slices per se is not the lever, so the honest prior is neutral to slightly positive.
- Cost: loader change (no shard rebuild, but 3D series decode more voxels; check the loader's time budget; the Kaggle path already holds the full volume), one run about $5 to $9. T4 unchanged apart from the extra averaging (negligible).
- Expected: +0.000 to +0.002. Cheap and cannot hurt the pass count.

### Rank 4. Common patient-frame feature volume (lift 2D features from sag, cor, ax into one 3D grid). NOVEL, expensive

- What: every slice has an exact position and orientation in one patient frame (IPP/IOP). Run the 2D trunk as today, then back-project each slice's feature map into a shared coarse voxel grid (about 16x16x16) by the slice geometry, average the votes, and put a small 3D convolution or attention on the grid before the head. This is the registration of sag/cor/ax into one 3D grid done at feature level (no image resampling, no interpolation of intensities), and it lets an ACL tear seen on a sagittal slice and a coronal slice at the same place reinforce each other, which late mean pooling cannot do.
- Evidence: multi-view 3D lifting is standard in 3D detection; the CAM analysis in the repo says evidence sits in 2 of 6 slice groups 66 to 85% of the time, i.e. it is local, which is the condition under which co-location helps. OrthoDiffusion-style plane concatenation led the macro AUROC in its paper (unverified here). Our own tests: the learned cross-slice and cross-slot attention heads were neutral to negative, so the gain must come from geometry, not from capacity.
- Our data: sag-cor centre distance median 9.9 mm, so a shared frame is meaningful; obliquity median 7 to 8 deg is handled exactly by the geometry; residual motion between series is the unknown.
- Cost: 3 to 5 days of engineering and checks, two or three runs ($15 to $30). T4 cost: a small increase (grid ops). Probably not finishable safely in 14 days next to the final submissions. Risk: high.

### Rank 5. Slot dropout and missing-sequence training (random drop of one view p=0.1, renormalised)

- Already listed in the 8 Oct slices report (idea "C"); I add the measured data: 12.6% of studies miss a slot and 26% of cor_plain / 81% of ax_plain are missing, so a stack that includes fused or six views would see missing inputs often. Brain-tumour work (npj Digit Med 2021; BraTS 2021) shows sequence dropout keeps full-input performance and helps when sequences are missing, but gains appear in segmentation, with AUROC flat in the one classification-like number reported. Run only with Rank 1 (where it matters more), not as a stand-alone.
- Cost: engine knob + run $5 to $9. Expected +0.000 to +0.001.

### Rank 6. Contrast-aware slot filling and a sequence-type embedding

- What: "plain" is T1 (39%), PD (34%), T2 non-fat-sat (23%) in sag_plain, and T1 for 73% of cor_plain. Choose among duplicates and fill empty slots by contrast type (for a missing fluid slot prefer a T2/PD plain series over a T1) and give the net a learned embedding (T1/PD/T2/FS) per slot. The current duplicate rule uses coverage and then resolution only.
- Evidence: none direct; stacking protocol features failed, so the embedding is probably worth little; the filling rule might matter for the 12.6% with a missing slot. Cheap check first: does the 5% of studies with the weak fallback differ in AUROC (the subgroup table says no: 0.9215 vs 0.9183).
- Expected: about 0.000 to +0.001. Cost: $5 to $9. Low priority.

### Rank 7. Background suppression or noise-texture harmonisation (zero the background by foreground mask)

- What: scanner and coil noise texture in the background identifies the vendor and the site, and sites carry report-style label bias (language groups differ: Synovitis prevalence from 0.39 (bg) to 0.51 (el); lateral OA 0.27 to 0.45). Zero-filling pixels outside the Otsu mask (or noise-matching all to one level) removes that shortcut.
- Evidence: protocol/scanner metadata alone reaches 0.63 macro AUROC and language alone 0.56 (a4.py), so a scanner shortcut exists; but the stack test shows the CNN already has it and extra metadata adds nothing, so a deliberate removal could help or hurt. The test labels come from a radiologist panel, not reports, so a shortcut trained on report style should not transfer.
- Cost: loader change plus one run, $5 to $9. Expected: -0.002 to +0.002, honest coin flip. Only worth it as a cheap diversity member.

### Rank 8. Quality-aware duplicate series selection

- What: choose among duplicate series by sharpness/motion score instead of depth coverage. Studies with a duplicate score 0.9129 against 0.9222 without (-0.009), which suggests a repeated scan (motion?) marks hard studies.
- Evidence against: duplicate TTA (-0.0025), the series schedule (flat), and the coverage rule measured right (alternate series worse: 0.8885 vs 0.8940). The weakness is probably the study, not the choice.
- Cost: score computed on the shards (CPU), one run. Expected: about 0.000 to +0.001. Low.

### Rank 9. Geometry-paired cross-sequence and cross-plane contrastive pretraining on our own 819k slices. NOVEL, high risk

- What: because same-plane series are exactly registered (and all planes share a frame), positive pairs for self-supervised contrastive learning are free: the fluid and plain slice at the same position, or a sagittal slice and the coronal slice that cuts it. In-domain pretraining on 24k series, then the normal fine-tune.
- Evidence against: five self-supervised or domain backbones were tried and lost (MRI-CORE MAE, Barlow Twins, TRIAD, MedSAM, BEiT v2: 0.77 to 0.86 solo). This adds only the geometry-paired positives.
- Cost: 6 to 10 H100 hours ($30 to $50) plus the fine-tune; not finishable safely. Expected: unknown, probably negative at our scale. Park.

### Rank 10. In-plane de-obliquing (rotate each series so the knee axis is vertical)

- Measured obliquity: slice normal 7 to 8 deg off the nominal axis (90th percentile 17 to 19 deg sag/cor), so some sagittal slices are not true knee-sagittal. De-rotation by IOP is exact, but the technologist prescribes the planes relative to the knee, not the scanner, and the training rotation augmentation (+/-10 deg, p=0.3) already covers the spread. Expected about 0, listed only because the numbers invite it.

## 4. Ideas from the brief and their verdict

| Idea | Verdict |
| --- | --- |
| Per-sequence z-score, Nyul, N4 vs per-volume percentile | Z-score/Nyul/CLAHE: small, external-only benefit in a knee segmentation benchmark (arXiv 2607.20028). N4: about 40 s/volume on CPU for 24k series, skip. Folded into Rank 2 (z-score, reference tissue, CLAHE), not Nyul (landmark fitting per sequence for a gain our window already mostly gives). |
| Multi-window channels (several windows of one slice as RGB) | RGB is already used for 3 slices in the best recipe (a wider stem lost). Window TTA instead (Rank 2). |
| DICOM metadata as input | Dead (section 2.3). |
| Physical-mm slice sampling | Series are uniform, the sampler is in index space; picks are 5.3 mm apart with a spread of 4.2 to 7.1 mm, so index and mm sampling differ by about 15% only. Captured better by Rank 3 (bin mean). |
| Anatomy-aligned crop via cheap localisation | Knee centred (median offset 4.9 mm), zoom crops already run (z030 shipped). Closed. |
| Use every series incl. duplicates and localizers | No localizers exist; duplicates tried (schedule, TTA) and flat. |
| Series-level vs study-level learning | Study-level pooling is what we do; per-series heads tried (per-view heads are untested at logit level, see the slices report). Not a raw-data change. |
| Aspect ratio / FOV preserving resize | The only odd frame is 259 ax_fluid series (640x1280, knee 38% of the frame, centred); centre crop covers it. No change. |
| Slice-thickness adaptive sampling | Thin series do not underperform (0.9204 vs 0.9188). Bin mean (Rank 3) addresses it for free. |
| Registration of sag/cor/ax into one 3D grid | Rank 4 (feature-level, geometry only). |
| Identify near-duplicate patients, left/right pairs, scanner effects | Dead / measured (sections 2.2, 2.3, 2.6, Rank 7). |
| Smart TTA over series choices | Closed twice (-0.0025, -0.0019). |
| Data cleaning (wrong-plane via orientation) | Nothing to clean (section 2.1). |
| Higher bit depth | Not needed (section 2.5). |

## 5. Recommended plan for the 14 days

1. Today: the free local tests of Rank 2 (window variants on an existing checkpoint, CPU data path only, the GPU is the user's RTX 4060).
2. Start Rank 1 engineering (pairing by IPP, parity with the Kaggle path); one local prototype and the standard check scripts before any Modal run. If the solo run lands within 0.003 of the parent, test it in the ensemble: it is the most different member we can still build.
3. Rank 3 can share the same Dataset change set as Rank 1 and runs as a second arm.
4. Ranks 4 and 9 are for after the deadline.

## 6. Method notes

- OOF source: 5 ConvNeXt folds (`convnext_base_25d_18slices_v4_flip_f0..4`) averaged with 5 MedImageInsight folds (`medimageinsight_davit_huge_336_v3_f0..4`), 4,342 studies, scored against v3 labels (the labels the folds' validation used), same loader as `runs/others/sex_signal.py`.
- Geometry source: per-series `orientation`, `pos_first`, `pos_last`, `pixel_spacing` read from the `meta` member of all 4,407 local shards (no pixel decode). FOV centre = midpoint of pos_first and pos_last plus half the image extent along the row and column vectors.
- Scratch scripts (a1.py to a14.py) are in the session scratchpad, not the repo.

## 7. Sources

- Knee MRI intensity-normalisation benchmark (7 methods, 3D U-Net meniscus, IWOAI to SKM-TEA): https://arxiv.org/abs/2607.20028
- Intensity standardisation and deep learning, multi-centre FLAIR: https://arxiv.org/pdf/2307.03827
- Nyul vs histogram matching critique: https://proceedings.mlr.press/v172/simko22a.html
- Sequence dropout for missing MRI inputs (brain metastases, npj Digital Medicine 2021): https://arxiv.org/pdf/1912.11966
- RSNA 2025 aneurysm winners and approach summary (secondary, Japanese slide deck): https://speakerdeck.com/k951286/kaggle-rsna-intracranial-aneurysm-detectionkonpe-fan-sheng-hui , results page https://www.rsna.org/news/2025/november/2025-ai-challenge-winners
- RSNA 2024 lumbar results (write-ups not retrieved): https://www.rsna.org/news/2024/november/2024-ai-challenge-winners
- Slice-thickness and scanner as site-effect drivers in MRI harmonisation: https://link.springer.com/article/10.1038/s41598-023-43715-5

Not verified: contents of the Kaggle write-ups for RSNA 2024 lumbar and 2025 aneurysm winners (not retrievable here); OrthoDiffusion plane-concatenation numbers; any claim that early fusion helps knee classification (no direct paper found), so Rank 1 rests on radiology practice, BraTS-style channel stacking and our own geometry measurement.
