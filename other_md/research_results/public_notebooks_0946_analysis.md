# Public notebooks at ~0.943-0.946: what they do and how to use them

Scope: four public Kaggle notebooks (d4-blend, public0946, public0946-no-correction, Speedy Raptors original), read in full from the downloaded .ipynb files. Cell numbers below follow the **pjmathematician d4-blend** notebook (33 cells, 0 to 32) unless stated. The code cells are byte-identical across the notebooks except where section 6 says so (verified by per-cell hashes). In yamadan96 and sushanth every cell number is pj's plus 1 (they add one markdown cell at the top). In Speedy Raptors the same code appears in cells 1 to 23 (no markdown cells between).

Notebook outputs are stripped (no timing prints, no scores in outputs). Everything about scores and runtimes below is from the authors' own markdown claims.

## 0. Headline findings (read this first)

1. The public pipeline alone scores **0.943**, not 0.946. yamadan96 and sushanth both say so explicitly (their cell 0): they copied the code of `ryokucha/rsna-knee-d4-blend-0946-ours10`, removed the 6 trailing cells that need private datasets, submitted on 2026-09-28 and got 0.943. The parent reaches 0.946 by blending a **private ConvNeXt ensemble at 10 percent**. The pj d4-blend banner also says 0.943 (cell 0).
2. pj d4-blend keeps three "OUR LEG" cells (30 to 32) that blend the author's own private model fleet (`rsna-knee-eff6-assets`, `rsna-knee-ens14-assets`, not attached) at rank weight 0.45 (default `RSNA_OURS_W`). Without those assets it soft-fails to "public only" (the public 0.943 output). So run as-is by anyone else it is the 0.943 pipeline.
3. The "correction" step: the only post-hoc correction in these four notebooks is the **MCL swap** (sushanth cell 31, section 2.5). It is a hard-coded hack for the 3-study visible test set and is **present in the sushanth notebook despite its "no-correction" name**; yamadan96 is the notebook that has none. On the hidden set it swaps the top two MCL values among about a thousand studies, i.e. it is noise. Not worth copying.
4. The pipeline is a very large rank-space ensemble of 5 families (DINOv2-small slot readers, A5 attention-pooling folds, RadImageNet ResNet50 heads, 4 Raptor CoAtNet views, 4 CoAtNet reader families) with per-finding weights. Final blend is dominated by the CoAtNet branch (weight 0.60 to 1.00 per finding).
5. **Licensing blocker for prize eligibility: Stage 3 (RadImageNet) is CC-BY-NC-SA-4.0 / built on RadImageNet.** It is entangled with a learned calibrator (section 2.3). See section 7.

## 1. Model inventory

Competition inputs used by all branches: `train.csv`, `test.csv`, `test_series.csv` (columns `Anatomical_Plane`, `Fat_Suppression`, `Fluid_Sensitive`, provided by the competition), `sample_submission.csv`, `test_series/`.

| # | Branch (stage) | Dataset source (Kaggle) | Architecture | Input | Members | Cells |
|---|---|---|---|---|---|---|
| 1 | DINO stage (Stage 1) | `pilkwang/rsna-knee-weights` (manifest.json + 20 member .pt + legacy bundle `rsna_20260807_v1.pt` unused), base `metaresearch/dinov2/PyTorch/small/1` | HuggingFace `AutoModel` DINOv2-small, **last 6 blocks fine-tuned** (`unfreeze_last=6`, first 6 frozen and shared), `SlotHead` (per-finding query attention over 6 slots, optional slot prior table, hidden 256), pooling `cls_mean` or `cls_mean_focal` (CLS + mean patch + top-k patch mean) | 6 slots (`SLOTS_RECOVERED`: SAG fluid FS, COR fluid FS, AX fluid FS, SAG fluid noFS, COR T1, SAG T1), 3 adjacent slices per window used as the 3 colour channels, 224 or 336 px, crop 130 mm, slice band 0.2-0.8; exact per-group values (img, group, slices, crop_mm, band) are read from the manifest (`adopt_config_globals`, cell 15) so not visible in the notebook text | **20 members** = 5 folds x 4 (comment in cell 15: "four members within each fold"), exactly 20 required, each checked by a stored fingerprint (tolerance 0.002) | 4 to 17 |
| 2 | A5 (Stage 2) | `mattiaangeli/knee-mri-fold-weights` (m_f0..m_f4.pt) | timm backbone named in the checkpoint cfg (not visible; created `pretrained=False`), `Net` with stem `native/compress/mixer`, `cond=token` ViT slot token option, readout `xres/xattn/attn/...` (label attention pooling over variable-length slice sets) | 6 slots = (plane x fat-sup flag), **16 slices per slot**, 336 px, crop 130 mm, band 0.12-0.88, per-slice 1/99 percentile windowing, `n_meta=0` | **5 folds**, rank mean | 19 to 23 |
| 3 | RadImageNet (Stage 3) | `marwanmath/resnet-50-radimagenet-marwan` (ResNet50.pt, sha 08629f7e...), heads: `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812` and `antoinegg1/rsna-knee-e9-radimagenet-heads-v15` (E10 "reference" heads, 5 folds), `antoinegg1/rsna-knee-e11-diverse-heads-v20` and kernel outputs `sofiaanjenje/rsna-knee-e11-train`, `...e13-train` (E13 heads, 5 folds) | torchvision ResNet50 (RadImageNet weights, frozen, global average pooled 2048-d) + `_RadHead` (LayerNorm, Linear 2048 to 512, plane+position embeddings, 8-head attention with 12 finding queries, fuse MLP) | grayscale slice replicated to 3 channels, 224 px, 8 slices per plane; layouts: E10 = 3 FS planes (SAG/COR/AX), full frame (crop 10000 mm, legacy rules); E13 = SAG_FS, COR_FS, AX_FS, SAG_NOFS at crop 130 mm; "second" = E13 heads fed the E11 layout (SAG_NOFS, COR_NOFS, AX_NOFS, SAG_FS) | 3 layouts, 5 heads each (E10 ref, E13, E13-on-E11) + linear calibrator | 24, 25 |
| 4 | Raptor (Stage 4) | `dreaddevelopment/raptor-knee-maxspan` (`raptor_ft_coatnet_v5_full_swa.pt`), `...native384dense` (`..._v10_full.pt`), `...native384` (`..._v8_full_swa.pt`) | timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` backbone (global avg pool, 3 input channels) + `RaptorClassifier` (LayerNorm, tanh attention MLP to 12 per-finding window attention, per-finding linear) | **3 adjacent slices as RGB per window, 384 px network resolution**, 5 source slots (SAG FS 18, SAG noFS 14, COR FS 12, COR noFS 8, AX 12 slices), 140 mm centre crop, span 0.02-0.98, up to **94 windows** per study (cell 27) | 4 views: maxspan-v5 (w 0.60, 336 px stored then resized to 384), native384dense-v10 (w 0.10), maxspan-v5-reverse (same checkpoint, windows flipped along slice axis, w 0.10), native384-v8 (w 0.20, 44-slice layout, span 0.06-0.94). 3 checkpoints, one used twice | 27 |
| 5 | CoAt residual gated (Stage 5a) | `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` | CoAtNet RMLP-2 384, per-finding gated spatial head with a protected global-average residual, second finding-specific attention across windows | 94 native-centre windows, six slots with quotas (20,12,12,8,16,8) (cell 3 `_dense_coat_specs`) | 3 checkpoints, epochs [4, 6, 8], prob mean | 27 |
| 6 | CoAt D4 depth-zone (5b) | `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2` | frozen CoAtNet Rank8 parent (SWA of epochs 15, 16) + 3-zone depth adapter with SWA over adapter epochs [11, 9, 5] (3255 state elements), 2 attention heads of width 128 for the "fsx" head | 96-slice global stack, slot widths (27,21,18,12,18), 384 px, canonical sagittal (cell 2 expects `canonical_sagittal: True`), patched to dense unique sampling | 1 model | 2, 27 |
| 7 | CoAt Global96 (5c) | `mattiaangeli/rsna-knee-coatnet-global96-top3` | same 96-slice stack as D4 but full-width gated head (no finding-to-slot attention) | 94 windows | 3 checkpoints, epochs [16, 23, 18] | 27 |
| 8 | CoAt Repair-v1 (5d) | inside the same attached sources (manifest `coatnet_repairv1_top3_manifest.json`, sha 99288a9a...); the dataset slug is not in kernel-metadata, it is located by manifest hash via `_asset_find_asset` | CoAtNet family "repair v1" | | 3 checkpoints, epochs [12, 7, 11] | 27 |
| - | Support | `mattiaangeli/opencv-python-headless-4120088-x86` (pinned cv2 4.12.0.88 wheel, installed with `pip install --no-deps --target`), timm 1.0.22 wheel shipped inside the D4/Global96 artifact | | | | 2, 27 |
| - | Attached but never used by code | `mattiaangeli/rsna-knee-cnx-m448-f0-public` (ConvNeXt M448 fold 0, only in the sushanth metadata; no code references it), `pilkwang/rsna-knee-llm-labels` (training provenance only, never read) | | | | |

Total checkpoints actually run per study: 20 (DINO) + 5 (A5) + 5 x 3 layouts of RadImageNet heads + 4 Raptor views (3 checkpoints) + 3+1+3+3 CoAt checkpoints. About 45 networks.

### Blend weights (summary)

- DINO members: rank-percentile of each member (`_combine`, cell 15), averaged with manifest weights (`weight`, optional `target_weight`), then re-ranked (`write_submission`).
- A5 weight: `A5_W = 0.52`, so stage-2 output = 0.48 * DINO rank + 0.52 * A5 rank (cell 22). Rows with no matching acquisition (metadata proves absence) keep the DINO-only rank.
- Rad: reference = rank(0.5 * E10 rank + 0.5 * E13 rank) (`_RAD_E13_MEMBER_WEIGHT = 0.50`); alpha 0.55 into the base for all findings except Baker's (`_RAD_EXCLUDE = ("Baker's",)`, in Speedy Raptors also Fracture); then second pass alpha 0.20 (`_RAD_V48_SECOND_ALPHA`); then calibrator gate (below).
- Raptor: probability blend with weights 0.60, 0.10, 0.10, 0.20 (after normalisation), then rank.
- CoAt family: mean of the 4 members' probabilities (25 percent each), then rank.
- Raptor + CoAt hybrid: 0.6 * Raptor rank + 0.4 * CoAt family rank (`private_alpha = 0.4`, cell 27).
- Outer blend (cell 27, last block): `ours` = DINO + A5 + Rad output ("transformer"), `theirs` = Raptor + CoAt hybrid ("coatnet"). Per-finding CoAt weight `w`: default **0.60** (Speedy Raptors original: 0.70), overrides ACL 0.75, Medial Meniscus 0.80, **Lateral Meniscus 1.00**, Lateral OA 0.75, Fracture 0.75. Output = rank((1-w) * rank_ours + w * rank_coat).

## 2. The blend in detail

### 2.1 Rank algebra
Everything is combined in **percentile-rank space** (`rank(method='average', pct=True)`), never logits or probabilities, except (a) the 4 Raptor views and (b) the CoAt family members, which are averaged as probabilities first and ranked afterwards ("rank of the mean probabilities, not mean of ranks", comment in cell 27). Final submission values are global percentile ranks in (0, 1], ties averaged. The test set must therefore be scored jointly (ranks depend on the cohort); blending with our predictions has to be rank based too.

### 2.2 Window pooling inside the DINO stage (a real trick)
`PUBLIC_FRONTIER_TARGET_POOL` (cell 15): over the overlapping 3-slice windows of a study, take the **max** of window probabilities for Fracture, Contusion, Medial Meniscus, Lateral Meniscus, Baker's, the **top-2 mean** for ACL and MCL, and the plain mean for the rest. A second "soft" focal pool (softmax with beta 6 to 10, blended with alpha 0.15 to 0.25) is computed (`LEGACY_FOLD_SOFTPOOL_*`) but the shipped path uses only the max/top2 variant (`public_frontier_members`). No flip or jitter TTA is used at the shipped inference (`plans = [(member, False)...]`, comment: "run_dinov2 retains only the unaugmented views").

### 2.3 The calibrator (inside Stage 3)
After the RadImageNet blend, an 88-feature **linear stacker** (decoded from the zlib/base64 payload `_RAD_CAL_PAYLOAD`, cell 25) is applied:
- features: base rank, E10+E13 reference rank, second rank (3 x 12), reference minus base (12), second minus base (12), mean of the three (12), 4 group means of the mean rank, and 12 **protocol counts from test_series.csv** (`n_series`, planes counts, fat-suppressed counts per plane, fluid-sensitive counts per plane): 36 + 24 + 12 + 4 + 12 = 88.
- groups: [ACL, MCL, Contusion, Fracture], [Medial Meniscus, Lateral Meniscus], [Medial OA, Lateral OA, PF OA], [Effusion, Synovitis, Baker's].
- 12 x 88 coefficients (|coef| mean 0.0026, max 0.157), standardised with stored mean/scale, sigmoid-free linear output, re-ranked.
- gate: applied only to ACL, Medial OA, Lateral OA, PF OA, Effusion, Baker's, Contusion with mix 0.6 * rank + 0.4 * calibrated rank.
This calibrator is "matched" to the Rad ranks (it needs `reference_rank` and `second_rank`), so Stage 3 cannot be dropped without also dropping the calibrator, and the 0.55/0.20/0.52/0.60 weights downstream were tuned with it in place. It was fitted offline by the upstream author (training labels or held-out target unknown).

### 2.4 Defensive wrapper code (not score related)
Pinned SHA256 hashes for every manifest, checkpoint, wheel; fingerprint test per member; strict state_dict loads; fail-closed on missing members except the CoAt readers, which are allowed to fail one by one ("continuing without it (flagged)", fallback to the remaining members, or Raptor only). The final cell 29 refuses to publish if DINO is not 20/20 or the calibrator was not applied.

### 2.5 The "correction": MCL swap (sushanth cell 31, only there)
```python
mcl_vals = sub['MCL'].values.copy()
mcl_max_idx = mcl_vals.argmax()        # currently S2 (row with highest MCL)
mcl_2nd_idx = mcl_vals.argsort()[-2]   # currently S3 (row with 2nd highest MCL)
sub_swapped.loc[mcl_max_idx, 'MCL'] = mcl_vals[mcl_2nd_idx]
sub_swapped.loc[mcl_2nd_idx, 'MCL'] = mcl_vals[mcl_max_idx]
```
Rationale in its comment: the CoAt arm and the maxspan-reverse view "both predict S3 as highest for MCL" on the **3-study visible test set** (S2 = 1.0, S3 = 0.667). It touches only MCL and exactly two rows; on the hidden set of about a thousand studies it exchanges the top two ranks (a difference of about 0.001), so its effect is negligible and it is a visible-test-set fit with no principled basis. The "0.946" in the pj ancestry (ryokucha, 10 percent private ConvNeXt) and pj's own "OUR LEG" (45 percent private fleet) are **blends with private models**, not corrections. There is no other post-hoc score correction in the code.

## 3. Preprocessing

### 3.1 Series selection
- DINO and Rad stages (cells 7, 8): header probe of the middle slice (`SeriesDescription`, `SequenceName`, `ScanOptions`, TR, TE, ...) and regex-derived `fatsat` (fs, fatsat, stir, spair, spir, we, water excit, tirm, ... incl. GE Dixon water "SMART FAT") and `weight` (T1 / T2 / PD / GRE; fluid = PD or T2). `pick_slots` picks per slot the series with plane == slot plane, fatsat == slot flag, fluid flag if specified, **most slices**. Plane label comes from `test_series.csv`.
- A5, Raptor and CoAt: use the metadata columns of `test_series.csv` directly (`Anatomical_Plane`, `Fat_Suppression`, `Fluid_Sensitive`). Raptor `_pick_series_for_slot` takes the first series of the plane with the preferred fluid flag and not used before.
- Note: the competition's `test_series.csv` does carry plane, fat suppression and fluid flags, and the notebooks rely on them heavily (also the calibrator's protocol counts).

### 3.2 Slice ordering and selection
- Order by the projection of ImagePositionPatient on the slice normal (`dot(ipp, cross(iop_row, iop_col))`), fall back to InstanceNumber / natural file name order.
- DINO: `np.linspace(lo, hi, n_slice)` indices inside the band (0.2-0.8 of the stack), then overlapping 3-slice windows (`window_starts`, stride 1).
- Raptor/CoAt: depth span 0.02-0.98 (Raptor v8: 0.06-0.94), **capacity-aware unique linspace picks** per slot (`_dense_allocate`, `_dense_unique_linspace`, no duplicated source slice) up to 96 slices, windows centred on every picked slice (94 windows max, triplet = centre-1, centre, centre+1).
- A5: 16 slices per slot in band 0.12-0.88.

### 3.3 Crop, field of view, pixel size, normalisation
- Physical centre crop using PixelSpacing: 130 mm (DINO, A5, Rad E11/E13), **140 mm** (Raptor and CoAt), full frame for Rad E10 (10000 mm). The crop is centred on the image centre (no localisation). Resize with bilinear (torch) or INTER_AREA (cv2) to 224 / 336 / 384.
- Intensity: DINO: apply Rescale slope/intercept, then per-slot-volume 1/99 percentile clip to [0, 1] then uint8 cache; A5: per-slice 1/99 percentile; Raptor: 2/98 percentile over the slot's original picks, clip to [0, 1], uint8, then ImageNet mean/std on the 3-slice RGB; MONOCHROME1 is inverted in the Raptor reader. Rad: [-1, 1] scaling, 1 channel replicated.
- Missing slot: masked (attention mask); missing study: all-zero row.

### 3.4 Laterality
- DINO / Rad (cell 10, `normalise_laterality`): right knees are mirrored to look like left: **Coronal and Axial flip the last (x) axis, Sagittal flips dimension 0 (the slice order)**. This is the same convention as our own default. Side is read from the DICOM Laterality tag, falling back to geometry (median x of the image centre from IPP/IOP, |x| < 20 mm = unresolved; Rad legacy rules use corner x with a 5 mm limit); disagreement tag vs geometry is logged.
- A5: flip x for non-sagittal series if ImagePositionPatient[0] of the first file is negative (no sagittal flip).
- Raptor / CoAt: no laterality flip is visible in the Raptor reader; the D4/Global96 runtime uses a "canonical sagittal" but their source files are not shown in the notebooks.

### 3.5 TTA
No flips or rotations at inference. Only: Raptor "reverse" view (same checkpoint on slice-reversed windows), dense overlapping windows with max/top-k pooling, and the fp16-then-fp32 retry on non-finite outputs. DINO jitter code exists but is disabled (jitter False).

## 4. Labels

- The notebooks are inference-only: no training label is read anywhere in the code. The attached dataset `pilkwang/rsna-knee-llm-labels` is **never opened** by any cell; it is attached as provenance.
- Dataset page (via the Kaggle API metadata, license **CC0-1.0**, 28 votes, 5858 downloads), title "RSNA Knee LLM-read report labels": "Twelve per-study scores with per-target confidences and the underlying verdicts, obtained by having a language model read each radiology report in its original language." Rule extractors miss reports in nine languages, so a language model reads the report instead. Each report is answered as 12 (verdict, severity) pairs under a constrained response schema; verdicts are YES / NO / UNK (NO = denied or normal, UNK = not mentioned); severity grades how emphatic the assertion is; "Scores are rank-usable, not calibrated probabilities". The description is truncated after "Columns: StudyInstanceUID, then per finding" and **does not name the LLM**. So the training labels of the models are most likely these report-derived scores (plus rule extractors), not the 58 panel labels.
- The 58 official panel studies are used as a held-out / selection set: dreaddevelopment's page says "on the 58 held-out labelled studies it scores 0.9174"; code strings `gold58_rank_ensemble_auc` (cell 2/ `_asset_write_coat_runtime`) refer to selection on those 58; the Global96 dataset is named "Gold Top3". Whether the gold studies also entered training (the CoAt runtime mentions `saved_training_gold_unique_center_slot_budgets`) cannot be settled from the text.
- Consequence for us: these models were trained on the official 4407 training studies (via report-derived labels), so **our own local val folds are contaminated for them**; only the panel-58 audit and the public leaderboard are clean tests.

## 5. Runtime

- Hard requirement: exactly 2 working Tesla T4 (`len(DEVS) != 2` raises; cells 2, 4), else the run aborts (no single-GPU fallback). Internal `TIME_BUDGET = 8 h` (cell 4) and per-stage deadlines (`rsna_deadline`).
- Authors' claims (cell 0 table): Raptor baseline about 9 h, DINOsaur blend about 4.5 h, Speedy Raptors v7 about 1.5 h, current version **under 30 minutes** on 2xT4. Mattia's original: "runs comfortably in less than 9 hrs". No printed timing evidence exists in the notebooks.
- Memory / speed tricks: shared frozen first 6 DINOv2 blocks computed once for all 20 members (cell 15); persistent model replicas on both GPUs; CPU decode threads prefetching one block ahead (`_RadLayoutPrefetch`, `_ke_prefetched_windows`); uint8 pixel caches in memmap on scratch disk (`rsna_array`); cached DICOM ordering and pixel LRU (192 MB); micro-batches of 8 windows for the backbones (`backbone_micro_images=8`); `CUDNN_CONV_WSCAP_DBG=1024` to bound cuDNN workspace; `cudnn.benchmark=False`; each CoAt reader runs in a **subprocess** (`_run_required_child`) with its own pinned cv2/timm wheels; for cohorts over 48 studies the 4 CoAt readers run one after another (cell 27), for tiny cohorts in 2 parallel pairs. Peak reserved memory per shard gated under 15.5 GB.
- Test cohort size: a code comment in pj cell 32 says "the hidden set (~1300)"; our task description says about 950. My own rough estimate (not from the notebooks): about 45 networks, CoAtNet-RMLP-2-384 forward on 94 windows per pass, 4 Raptor passes (the reverse view reuses the same decoded windows) plus 10 CoAt passes per study; about 3 to 5 s per study across both T4s, so roughly **0.5 to 1.5 hours** for 950 to 1300 studies. Treat as an estimate.

## 6. Differences between the four notebooks

| | pjmathematician d4-blend | yamadan96 public0946 | sushanth ...-no-correction | mattiaangeli Speedy Raptors original |
|---|---|---|---|---|
| Cells | 33 | 31 | 32 | 24 |
| Core pipeline (DINO, A5, Rad, Raptor, CoAt x4) | identical | identical | identical | same code with the differences below |
| Header claim | "0.943 SOTA", v9/v34 (Mattia v34 / Jiwei Liu v9) | cell 0: "0.943, not 0.946", code from ryokucha `...0946-ours10` minus 6 private cells | same text | TL;DR: added two CoAtNet MRI models (D4 depth-zone SWA3, Global96) and speed optimisations; pj's table gives Speedy Raptors v7 = 0.942 |
| Trailing cells | cells 30 to 32 "OUR LEG": stash public output, run private `infer_kernel.py` (eff6/ens14 assets: S_dist, S_coat, S_cnxl, 54 windows), rank blend `(1-0.45) public + 0.45 ours` (optionally EFF leg `RSNA_EFF_W` default 0), soft-fail to public-only | none | **MCL swap** (cell 31) | none |
| Outer CoAt weight default | 0.60 (+ overrides) | 0.60 | 0.60 | **0.70** (same overrides) |
| Rad blend excluded findings | Baker's | Baker's | Baker's | **Baker's and Fracture** |
| CoAt readers | 4, parallel pairs for <= 48 studies, serial otherwise; Global96 output in `g96_input/` | same as pj | same as pj | 4 readers always serial (no parallel pairs, no thread-safe JSON lock) |
| Stale audit dict | `family_mix` computed from members | same | same | hard-coded `family_mix=[.5,.5]`, `coats` lists only 2 families (stale, the code still runs 4) |
| Scores stated | 0.943 in the banner (own score not stated) | 0.943 (own submission, 2026-09-28) | 0.943 (same claim; the MCL swap cell makes it a modified version, its score is not stated) | not stated in the notebook |
| Dataset sources | 14 public + 3 blank (private) entries | 14 | 15 (adds the unused `mattiaangeli/rsna-knee-cnx-m448-f0-public`) | 14 |

Ancestry: Jiwei Liu (rsna-knee-fast-2xt4-inference), Mattia Angeli (Bend the Knee to the Dinosaurs, then Speedy Raptors), dreaddevelopment / Roman Tamrazov (Raptor CoAtNet weights), Pilkwang (DINO weights, LLM labels), Sofia Anjenje (E11/E13 training kernels), Antoine G. and prvsiyan (RadImageNet heads), Marwan Mahmoud (RadImageNet ResNet50), renta.k (hybrid raptor lateral meniscus), Anvith Pothula (rsna-base).

## 7. Dataset and model licences (from the Kaggle API metadata of each dataset)

Method: `kaggle datasets metadata <owner>/<slug>` was run for all datasets (the web pages are JavaScript rendered and WebFetch returned no content). Licence text below is what the dataset owners selected; it does not prove the licence of the upstream weights.

| Dataset | Kaggle licence | Built on | Flag |
|---|---|---|---|
| `dreaddevelopment/raptor-knee-maxspan`, `-native384`, `-native384dense` | CC0-1.0 | timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (Apache-2.0 on Hugging Face, pretrained on ImageNet-12k then fine-tuned on ImageNet-1k) fine-tuned on the knee data | OK (ImageNet pretraining, same as most backbones) |
| `mattiaangeli/rsna-knee-coat-resgated-ep10-top3`, `...coatnet-d4-depthzone-swa3-b2`, `...coatnet-global96-top3` | CC0-1.0 | CoAtNet from timm (same family), Repair-v1 same family | OK. Repair-v1 slug unknown (found by manifest hash), licence unchecked |
| `mattiaangeli/knee-mri-fold-weights` (A5) | CC0-1.0 | timm backbone named in checkpoint cfg (not visible) | backbone and its pretrained weights unverified |
| `pilkwang/rsna-knee-weights` | CC0-1.0 | DINOv2-small (Apache-2.0) fine-tuned | OK; Apache-2.0 notice to be carried if shipped |
| `pilkwang/rsna-knee-llm-labels` | CC0-1.0 | an LLM (unnamed) reading reports | not weights; LLM unknown, check its terms for output use |
| **`marwanmath/resnet-50-radimagenet-marwan`** | **CC-BY-NC-SA-4.0** | **RadImageNet ResNet50 weights** (`encoder_source_commit 0ce16f7...`) | **NON-COMMERCIAL, FLAG** |
| **`antoinegg1/rsna-knee-e9-radimagenet-heads-v15`** | **CC-BY-NC-SA-4.0** | heads on RadImageNet features | **FLAG** |
| **`antoinegg1/rsna-knee-e11-diverse-heads-v20`** | **CC-BY-NC-SA-4.0** | heads on RadImageNet features | **FLAG** |
| **`prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`** | "other" | heads on RadImageNet features (name and contract: `v52-radimagenet-resnet50`) | **FLAG** |
| kernel outputs `sofiaanjenje/rsna-knee-e11-train`, `...e13-train` (`v52_e11_heads.pt`) | n/a (notebook outputs) | E11/E13 heads on RadImageNet features | **FLAG** |
| `mattiaangeli/opencv-python-headless-4120088-x86` | "unknown" | OpenCV 4.12.0.88 wheel (Apache-2.0 upstream) | OK, not weights |
| `mattiaangeli/rsna-knee-cnx-m448-f0-public` | "other" | ConvNeXt M448 fold 0; not used by code | unused, unchecked |
| model `metaresearch/dinov2/PyTorch/small/1` | Kaggle model page did not render; the Hugging Face `facebook/dinov2-small` card states **Apache-2.0** | DINOv2 (Meta) | OK, Apache-2.0 (keep licence copy and notice) |

**RadImageNet:** Stage 3 (cells 24 and 25) loads RadImageNet ResNet50 weights plus heads trained on top of them, five datasets and two kernels in all. This is non-commercial material; do not ship it in a prize-eligible submission. Everything that feeds from Stage 3 (reference rank, second rank, the 88-feature calibrator, the 0.55 and 0.20 blend weights) has to go with it.

## 8. Assessment

### 8.1 Core of their advantage over a single model
1. **Dense, physically normalised 2.5D CoAtNet**: 3 adjacent slices as RGB, 140 mm centre crop by DICOM pixel spacing, 384 px network input, up to 94 windows per study spread uniformly over 2 to 98 percent of every selected series, window features pooled by finding-specific attention across planes. The dreaddevelopment page quantifies it: widening the span only helps if slice density grows with it (44 slices over the wide span: 0.917; 64 slices: 0.928 as a single model on the public leaderboard). The single Raptor model is already 0.928.
2. **Breadth of diverse families in rank space**: ImageNet-12k CoAtNet readers with different heads/geometry (residual gated, D4 depth zones, Global96, Repair-v1, Raptor views), DINOv2-small slot readers (20 members), A5 slice-attention folds, RadImageNet heads. Averaging checkpoints (SWA, top-3 epochs) inside each family and ranks between them. Public numbers from the authors: 0.924 (single) to 0.941, 0.942, 0.943 by adding families.
3. Per-finding choices: window pooling (max for menisci/fracture/contusion/Baker's, top-2 for ACL/MCL), per-finding family weights (Lateral Meniscus is 100 percent CoAt), calibrator with protocol counts.
4. Use of the competition-provided series metadata (plane, fat suppression, fluid sensitivity) to choose series and as stacker features.

Tricks we may lack (candidates): max/top-2 window pooling per finding; the dense capacity-aware unique-slice windows at 384 px with a 140 mm physical crop; `canonical sagittal`/slice-reverse view as TTA; rank-of-mean-probabilities per family; per-finding blend weights between a 2.5D-CoAt family and a slot-reader family; protocol-count features (series counts, fat-sup counts) in a linear stacker.

### 8.2 Cleanest way to add their predictions to our ensemble in ONE Kaggle notebook
Their pipeline is already organised as sequential stages that each write intermediate arrays, and our blend is rank based, so:

1. Copy the pj d4-blend notebook (or yamadan96 which is the same minus the private tail), drop cells 30 to 32 (private leg), and keep it as the first half of our notebook. Attach the 14 public datasets + both kernel outputs + `metaresearch/dinov2` + competition data (no internet needed; assets are found by pinned hashes, cells 2 and 4).
2. After cell 29 (it writes `/kaggle/working/submission.csv` and per-branch arrays), read the per-branch outputs rather than only the final CSV. Already saved by the notebook: `diagnostics/*.npz` (every DINO member `dino_<id>`, `a5_rank_mean`, `rad_*_raw`, `coat_*_rank`, `btkd_parent_after_rad`, `final`), `raptor_raw.npz` (4 views x studies x 12 probabilities), `speed_a5_raw.npz`, `_coat_*` CSVs and the 4 CoAt `*_predictions.npz`. Then our own cell blends: `rank(a * rank(ours) + (1 - a) * rank(theirs_final))` per finding, with `a` chosen from our panel-58 audit and public leaderboard checks, **not** from our local val (their models saw our val studies).
3. Reusable as-is: all of Stage 1 (DINO), Stage 2 (A5), Stage 4 and 5 (Raptor and the 4 CoAt readers) and the rank logic. Reusable as ideas for our own models: items in section 8.1.
4. **Blockers / caveats**
   - Prize eligibility: Stage 3 RadImageNet (CC-BY-NC-SA) must be removed, and with it the calibrator (it needs Rad ranks). The remaining public pipeline then has no validated blend weights (0.52 A5, 0.60/0.75/1.00 CoAt weights were tuned with Rad inside the "ours" side), so re-measure or fall back to a rank mean of DINO + A5 against CoAt. We cannot say from the text how much of 0.943 comes from Rad.
   - Hard-coded 2xT4 requirement (cuda device count must be 2) and subprocess readers; fine for the competition runtime, but a T4 x2 is also required to test.
   - A5 weights' backbone and the Repair-v1 slug are not identifiable from the text; confirm licences before shipping.
   - DINOv2 (Apache-2.0): ship the licence copy. LLM-label dataset (CC0) has no weights; the unnamed LLM's terms are not known.
   - Our own timing: their stack adds about 0.5 to 1.5 h on 2xT4 (estimate, section 5) out of 9 h, plus the time of our own models; the CoAt readers are the bulk. The notebook has an internal 8 h `TIME_BUDGET` which must be revised if our models run first.
   - Rank dependence: blending requires the full test cohort at once (no per-shard ranking), which their code respects.
   - Any claim of improvement from blending must be checked on the panel-58 audit and a public submission: the ensemble correlation (their own leg vs public measured 0.75 to 0.89 per finding, pj cell 32 comment) shows the gain comes from diversity, so blending a dissimilar family (ours) with theirs is plausible.
