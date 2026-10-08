# Anatomy- and finding-aware modelling, plus medical-imaging tricks (8 Oct 2026)

Scope: what could lift the weak findings (Synovitis, Fracture, MCL, Patellofemoral Osteoarthritis (PF OA), Lateral OA, Baker's cyst) and the strong ones, using knee MRI anatomy, sequence physics and how radiologists read each finding. Deadline is 22 Oct 2026, so **two weeks remain**. Ranking therefore weights what can be tested cheaply (cached predictions, one training run, or a CPU job) above anything needing new masks or a full shard rebuild.

Evidence grades: **[lit]** = found in a source this session (URL given); **[bg]** = background knowledge, not verified by search, check before relying on it.

## 0. Plain-language primer (the user has no medical background)

- **Sagittal**: slices cut left-to-right, like a side view of the knee. Best for the cruciate ligaments (ACL, PCL) and the menisci (the two C-shaped shock absorbers).
- **Coronal**: slices cut front-to-back. Best for the collateral ligaments (MCL on the inner side, LCL on the outer side).
- **Axial**: horizontal slices, top-down. Best for the kneecap (patella) and the groove it slides in.
- **Fluid-sensitive / fat-suppressed T2 (our "fluid" slots)**: fat is darkened so that fluid (effusion, Baker's cyst, bone bruises) shows bright.
- **Proton density (PD) / T1 ("plain" slots)**: show structure and anatomy well; fluid is less bright.
- **Bone marrow edema (bone bruise, our "Contusion")**: water-rich swelling inside bone. Looks grey-bright on fat-suppressed fluid images and is invisible on X-ray.
- **Effusion**: fluid in the joint, most visible in the suprapatellar pouch (the pocket above the kneecap).
- **Synovitis**: the joint lining thickens and takes up signal; hard to separate from effusion on images.
- **Field of view (FOV)**: how many millimetres of knee each image covers. We use 134 mm in the 420 px run, 205 mm in the main runs.
- **Voxel / pixel spacing (mm/px)**: physical size of one picture element. We resample to 0.40 mm/px.
- **Intensity standardisation**: making brightness mean the same thing across scanners and studies.
- **Bias field**: a smooth brightness gradient across the image caused by the scanner coil. Looks like a shadow.

## 1. Where we already stand (do not re-propose)

From `other_md/outline.md`, `runs_timeline.md`, `big_gain_ideas_research.md`:
- Per-volume percentile uint8 windowing is done. Cropping to the centre was decided "none" at 512 px, and a 336 px / 134 mm frame tied the baseline (run 43).
- Per-finding slot routing and per-finding pooling: closed (lost monotonically, selection bias).
- Restricting views by anatomy (slot masking) HURT (run 74, cut at 9/12).
- Compartment-aware (medial/lateral half) pooling: negative on the 346 clean val (`convnext_base_25d_v5_compartment`).
- Co-occurrence priors on silent report cells: rejected (-0.0108).
- Laterality flip: adopted.
- MedSAM, BEiT v2, TRIAD (3D masked autoencoder), Barlow Twins: all failed as backbones (0.77-0.85). Lesson: only label-supervised backbones work for us.
- Specialist takeover: 1 of 14 specialist/finding pairs beat the ensemble (Synovitis via `convnext_base_fluid`).
- Already in flight: `convnext_base_25d_v5_native_res_420px` (0.32 mm/px, 420 px, same 134 mm frame). This IS the "higher-resolution zoom on the joint" idea, so it is not new.
- Report attributes already extracted for 4,407 studies (`data_code/25_extract_report_attributes.py`, 45 graded attributes, 44% of cells reported). Only used as label noise checks so far, not as training targets.
- `cor_plain` is missing for 22.7% of studies and `ax_plain` for ~80%; the current fallback duplicates the same-plane series (not a true view).

## 2. Medical-imaging tricks (weighted heavily, per coordinator request)

### 2a. Intensity standardisation

- **N4 bias-field correction** [bg, widely used, N4ITK = Nick Tustison's algorithm in ANTs/SimpleITK]. Removes the coil shading. The classic is slow: a source reports ~40 s per volume for N4, against ~4 s for a learned replacement (HDDnet). Learned emulator DeepN4 (trained on 72 scanners, median PSNR 47.96 dB against N4ITK) [lit: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10680935/]. Expected gain for us: small. Our per-volume percentile windowing already handles global brightness; shading is mostly a concern for the surface-coil coronal T1. Cost: CPU for 24k series is many hours, far too much for two weeks. **Verdict: low, skip.**
- **Nyul histogram standardisation** (piecewise-linear histogram matching to learned landmarks per sequence) [lit: https://arxiv.org/pdf/2307.03827 reports N4 + Nyul/White-Stripe pipelines for multi-centre FLAIR; theoretical critique https://proceedings.mlr.press/v172/simko22a.html says Nyul approximates histogram matching no better than equalisation, on synthetic data only]. Needs per-sequence landmark fitting, gives a cross-scanner consistency gain that we partly have. **Verdict: low.**
- **Per-sequence z-score (White Stripe style)** [lit as above]. Our percentile windowing is a simpler, comparable choice. A test costs one shard-free change in the loader (apply z-score per series in the Dataset, guarded with `getattr` defaults per the engine rule). Worth a cheap A/B only if we have spare GPU. **Verdict: low, optional.**

### 2b. Multi-planar reformatting (MPR): build the missing planes from 3D sequences

- **Idea**: some series are true 3D acquisitions with thin slices (our 709 "long" series, 0.8 mm thick, 3D DESS or similar). A 3D volume can be re-cut into coronal or axial slices at no scan cost. Our missing coronal PD (`cor_plain`, 22.7% missing) and axial PD (`ax_plain`, ~80% missing) could be filled by MPR from a 3D sagittal volume where one exists, instead of duplicating a fluid series.
- **Why it matters**: MCL (coronal), PF OA (axial) and Medial OA are the findings the reports and our per-finding notes say depend on a plane we often lack. Our `cor_plain` test already lifted MCL (+0.0133 in run 74) which suggests plane coverage is the lever.
- **Caveats**: MPR from a 3D sequence is blurrier in the cut direction than a true 2D acquisition; the MRI physics (contrast) of the 3D sequence is not the PD contrast we want. Mark reformatted slots as a separate channel or flag, so the model knows they are reconstructed.
- **Cost**: needs new shards for the affected studies (CPU reformat, a few hours) plus one training run. Feasible only if started today.
- **Verdict: medium-high gain on MCL, PF OA, Medial OA; medium-high cost. Rank 3.**

### 2c. Sequence physics and channel design

- **Sequence choice by finding** [lit + bg]: fluid-sensitive sagittal and coronal read for effusion, Baker's, meniscus (signal reaching the surface), ACL (fibre discontinuity). Axial PD or T1 for patellofemoral cartilage and the kneecap. Our per-finding slot evidence matches this. A 2025 multi-plane study found sagittal best for ACL and axial best for meniscus [lit: https://qims.amegroups.org/article/view/137915/html], but another U-net study found coronal better than sagittal for meniscus [lit: search result, same Frontiers/medRxiv cluster]. Planes conflict across studies, so keep the choice learned, not hard-coded.
- **Channel stacking of co-registered sequences** (3 sequences of the same plane at the same location stacked as RGB-like channels, so one ImageNet-pretrained first layer sees all three in one forward pass). Our current pipeline pools slots after the backbone, so a stacked input is a genuinely different design. Registration: we can align series by `ImagePositionPatient` (already in the shard metadata) without image registration, since the same knee is scanned in one session. Limitation: series have different slice positions and counts, so alignment is approximate (sub-slice error is probably fine). Cost: one new Dataset mode and a 3-channel first-layer adapter. Gain unknown. **Verdict: medium, testable in one run. Rank 5.**
- **Subtraction** (fluid minus plain to highlight fluid): classic for contrast-enhanced MRI, but our fluid and plain series are different contrasts, not pre- and post-contrast, so subtraction has no physical basis here. **Verdict: skip.**
- **Fat-saturation check**: our `sag_fluid`, `cor_fluid`, `ax_fluid` are fat-suppressed by label. Confirm the `series_description` text ("FS", "fat sat", "STIR", "SPIR") rather than assume it; a mislabelled fat-sat series would hurt Contusion specifically. Cheap, one grep on the metadata. **Verdict: quick sanity check, do it.**

### 2d. Radiologist reading patterns (what the eye uses, per finding)

- **Meniscus tear** [bg, Stoller grading]: a normal meniscus is a dark triangle on sagittal images. A tear shows bright signal inside it. Grade 3 (the "tear" grade) = bright signal that reaches the joint surface. Grade 2 = linear signal not reaching the surface. Radiologists read the posterior horn (back third) most carefully and scroll through 3-5 central sagittal slices, where the "double PCL sign" (a false appearance of two posterior cruciate ligaments, a known pitfall at the meniscus root) and meniscal root tears live [bg]. Source for the sagittal normal appearance: https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.747217/pdf [lit, partial]. Exact Stoller definitions not confirmed by search, verify in the 1987 paper before use.
- **ACL tear** [bg]: read on sagittal fluid or PD fat-sat; a torn ACL shows fibre discontinuity (gap) or a bright, wavy, thickened ligament. Radiologists check the notch (the groove where the ACL sits). Also associated with lateral femoral condyle bone bruise (the pivot-shift pattern). Our ACL is already at 0.97 from the labels, so the anatomy gain here is small.
- **Bone bruise (Contusion)** [bg]: grey-bright patchy signal in bone on fat-sat fluid images, usually in the lateral femoral condyle or the tibial plateau after a twisting injury. Needs fat suppression, so the fluid slots are the right ones.
- **Effusion and Baker's cyst** [bg]: effusion is the bright fluid in the suprapatellar pouch (sagittal). Baker's cyst is bright fluid in the back of the knee (popliteal fossa), usually on sagittal fluid slices at the back of the joint. These are local, so a crop around the pouch or the popliteal fossa would be the anatomy-aware variant (see 3b).
- **Synovitis** [bg]: thickened, enhancing or bright lining tissue, often hard to separate from effusion. Radiologists judge it by the thickness of the lining against the fluid. Weakest finding for us (0.80 LLM-label AUROC), and report-silent in 51% of cells. Image-only reading is the only route, which is why MedGemma's image opinion helped (+0.02 on the audit).
- **PF OA** [bg]: cartilage thinning or bone changes on axial images where the kneecap meets the groove. Axial PD is the best view, and our `ax_plain` is 80% missing: this links directly to 2b.

### 2e. Radiomics and texture

- Radiomics = computing many hand-made texture and shape numbers per region (first-order intensity, grey-level co-occurrence, shape). Evidence [lit]: cartilage radiomics on MRI and CT (138 knees) found texture and shape features as key predictors, with feature robustness tested across segmentations (https://link.springer.com/10.1007/s00330-021-07951-5 family; Information Systems Frontiers 2024 abstract); tibial subchondral bone radiomics reached AUC 0.80 for tibiofemoral OA (Rotterdam, https://link.springer.com/article/10.1007/s00330-021-07951-5); meniscus radiomics predicted knee replacement with 87.2% accuracy (ISMRM 2024 abstract). Radiomics needs masks of the region, which we do not have, so it would need an automatic segmentation first. **Verdict: low-medium. Only viable if a free knee segmenter is run (see 3a). Rank 9.**

## 3. Anatomy-aware modelling and ROI localisation

### 3a. Automatic knee segmentation as a crop or channel

- **TotalSegmentator MRI** (nnU-Net, 50 to 80 MRI structures incl. bones and some soft tissue) [lit: https://github.com/wasserth/TotalSegmentator (Apache-2.0 code); search result: https://arxiv.org/abs/2405.19492v2]. Licence caveat: the code is Apache-2.0 and the research doc says `total_mr` is open, but a Slicer forum thread says some tasks are academic-only or licensed, and the search did not confirm the weight-file licence. Verify on the repo before any use. Gains: a bone mask gives a tight crop for PF OA and Contusion (bone), a femur/tibia split for the medial/lateral findings. Cost: 3D nnU-Net on ~1,300 studies x 2 sequences on one T4 is roughly 20-60 s per volume (estimate, not measured), i.e. 15-40 GPU-hours. **Does not fit the 9 h budget. Verdict: high gain, impossible in two weeks. Rank 7 (research only).**
- **Cheap 2D knee box**: a 2D detector or the existing intensity threshold on the mid-sagittal slice gives a joint bounding box in a few ms. Since the field of view is already centred (knees centred, ~15% air per `frame_usage_stats.py`), gain is limited to the air margin, which is the `native_res_420px` run. **Verdict: already covered.**
- **SAM-family masks as channels** (MedSAM, SAM-Med3D): MedSAM failed as a backbone (0.7664). A mask channel is different, but without a validated knee prompt it is guesswork. SAM-Med3D data terms unverified. **Verdict: skip.**

### 3b. Per-structure crops (suprapatellar pouch, popliteal fossa, notch)

- Localise the anatomy from the existing 96-slice stack (e.g. the slice where the kneecap is widest) and read a high-resolution crop around the suprapatellar pouch (effusion, synovitis), the popliteal fossa (Baker's), or the notch (ACL). The RSNA 2022 cervical spine and 2024 lumbar winners used localise-then-classify crops [lit: https://github.com/Nischaydnk/RSNA-2023-1st-place-solution and the 2024 lumbar 2nd-place write-up in `big_gain_ideas_research.md`]. For the lumbar 2024 winner, I could not confirm the exact sagittal and axial crop strategy by search [search result, not verified].
- **Cost**: needs a localiser (new training or a crude rule) and a second model or a second input. Two weeks is not enough to build and validate a two-stage pipeline safely. **Verdict: high upside, high cost. Rank 6 (only if the team has spare GPU next week).**

### 3c. Slice-position encoding

- Each slice has a physical position (`ImagePositionPatient`, already in the shard metadata). Give the 2.5D model a normalised position channel (e.g. 0 at the top of the joint, 1 at the bottom), so the head knows whether a bright patch is at the posterior horn or the body of the meniscus. Mean pooling currently throws away where in the joint a slice sits.
- **Cost**: small Dataset change (an extra scalar per slice, guarded with `getattr`), one run. Engine rule: new Dataset attributes need defaults while a local run trains.
- **Verdict: low-medium gain, very cheap. Rank 2.**

### 3d. Report attributes as auxiliary targets (weak localisation)

- The 45 graded attributes (per compartment cartilage grade, meniscus state, ACL/PCL/MCL/LCL grades, effusion size, synovitis stated, Hoffa, Baker's size, bruise location x5, fracture, insufficiency fracture, osteochondral lesion) already exist for 4,407 studies. Rank correlation with the matching v3 label: ACL 0.966, medial meniscus 0.942, PF OA cartilage 0.938, medial OA 0.912, MCL 0.911, effusion 0.892, Baker's 0.873, lateral OA 0.750, lateral meniscus 0.740, fracture 0.627, synovitis 0.605 (from the timeline).
- Idea: a masked multi-task head that predicts these attributes where the report states them (loss masked where unreported), alongside the 12 labels. This is "weak localisation" in the sense that the attribute "posterior horn" or "grade 3 cartilage" tells the model where and how severe. It is different from the rejected co-occurrence prior (that was a logistic prior on the labels, not a shared representation).
- Expected gain: most on Synovitis and Fracture (weakest), where extra signal from the same images is the only lever left. Risk: the attribute labels inherit report noise (same 51% silent-cell problem), so the head may learn the labeller rather than the image.
- **Cost**: one loss term and a masked target tensor; no new data. One training run. **Verdict: highest expected gain per dollar. Rank 1.**

### 3e. Label-correlation head (learned, not a prior)

- A learned head on the 12 logits (e.g. a small graph or a 12x12 learned matrix) lets Effusion and Synovitis and Baker's inform each other. Prior version rejected (-0.0108) because it used silent cells. A learned version on the logits is untested. Expected gain small (the logits are already correlated 0.95 within an ensemble). **Verdict: low. Rank 10 or skip.**

### 3f. Laterality and side consistency

- Already adopted: laterality flip. Side-consistency across the two knees is not available at inference (only one knee per study). **Verdict: done.**

## 4. External tools and data (licence and cost)

| Item | Use | Licence (verified or unknown) | GPU cost on T4 | Verdict |
| --- | --- | --- | --- | --- |
| TotalSegmentator MRI (`total_mr`) | bone/soft-tissue masks | code Apache-2.0; weight licence to verify | ~15-40 GPU-h for test set (estimate) | Too slow for 9 h |
| SKM-TEA | 16 pathologies as boxes | Stanford non-commercial agreement | n/a | Banned (non-commercial) |
| OAI / MOAKS grades | effusion-synovitis, cartilage, osteophytes | NDA institution-cosigned | n/a | Risky, disqualified without host confirmation |
| fastMRI+ knee | pathology boxes | CC BY 4.0 for boxes, but research/education data use | n/a | Banned for us (agreement) |
| MedSAM, SAM-Med3D | masks | Apache-2.0 (weights caveat for MedSAM CC-BY-4.0 per memory note) | n/a | Failed as backbone, skip |

## 5. Ranked top 10 (expected macro-AUROC gain per dollar, two-week horizon)

1. **Masked multi-task report-attribute head (3d).** Gain on Synovitis, Fracture, Baker's most. Data already extracted; one run. Risk: labeller noise.
2. **Slice-position encoding from `ImagePositionPatient` (3c).** Cheap; helps meniscus-horn and effusion localisation. One run.
3. **MPR fill of missing coronal and axial planes from 3D series (2b).** Targets MCL, PF OA, Medial OA via real `cor_plain` and `ax_plain` coverage (currently 22.7% and ~80% missing). Needs partial shard rebuild, start today or drop.
4. **Channel stacking of co-registered same-plane sequences (2c).** Different design from slot pooling, untested; one run.
5. **Sequence sanity check: fat-sat flags from series description (2c).** Cheap; protects Contusion. Do regardless.
6. **Per-structure crops from a localiser (3b).** High upside, too big for two weeks; park for the next competition or a post-deadline experiment.
7. **TotalSegmentator MRI masks for crops and radiomics (3a, 2e).** Best anatomy tool, does not fit 9 h GPU; only if licence and runtime change.
8. **Native-resolution zoom (420 px, 0.32 mm/px).** Already in flight. Wait for its result before adding more resolution.
9. **Radiomics texture features into a stacker (2e).** Needs masks; CPU-heavy; low.
10. **Learned label-correlation head on logits (3e).** Small gain, low risk, low priority.

Skipped with reasons: N4 bias correction (CPU hours, small gain), Nyul standardisation (similar to our percentile windowing), subtraction imaging (no physical basis for our contrasts), SAM-family masks (MedSAM failed as backbone), OAI/fastMRI/SKM-TEA (licence).

## 6. Recommended next step for the coordinator

- Start items 1 and 2 now as one run (they share the Dataset and loss change and can be tested locally with `getattr` defaults). Check item 5 with a grep on `series_description` first.
- Decide by Monday whether item 3 (MPR fill) can finish a shard rebuild before the deadline. If not, drop it.
- Do not chase item 6 or 7 this competition.

## 7. Sources (search-verified)

- https://qims.amegroups.org/article/view/137915/html (2025 multi-plane ACL/meniscus comparison)
- https://www.medrxiv.org/content/10.1101/2025.09.09.25335391.full.pdf (dual-branch EfficientNet, ACL, MRNet)
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10680935/ (DeepN4)
- https://arxiv.org/pdf/2307.03827 (intensity standardisation and deep learning, FLAIR)
- https://proceedings.mlr.press/v172/simko22a.html (Nyul vs histogram matching critique)
- https://github.com/wasserth/TotalSegmentator and https://arxiv.org/abs/2405.19492v2 (TotalSegmentator)
- https://cds.ismrm.org/protected/23MProceedings/PDFfiles/3105_MzGLVsPrF.html (synthetic fat-sat from multi-contrast input, ISMRM abstract)
- https://escholarship.org/uc/item/7294c75t (UCSF effusion-synovitis volume, Dice 0.79)
- https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.747217/pdf (normal meniscus on sagittal MRI)
- https://arxiv.org/pdf/2108.08136 (spatial attention and localisation for knee injury, MRNet)
- https://www.ams.giti.waseda.ac.jp/data/pdf-files/2025GCCE_kunitomi.pdf (slice attention plus sequence fusion on MRNet, 2025)
- https://arxiv.org/pdf/2602.20752 (OrthoDiffusion, 2026, not usable per-slice)
- https://arxiv.org/abs/2601.18250 (OrthoFoundation, 2026, self-supervised, DINOv3-Large; licence not checked)
- https://www.rsna.org/news/2024/november/2024-ai-challenge-winners (RSNA 2024 lumbar results; 1st place write-up not reached)

Not verified by search (background knowledge): Stoller and Crues grading details, double PCL sign, bone bruise locations, Baker's cyst sign, ACL notch reading, the claim that radiologists scroll 3-5 central sagittal slices for the meniscus root.
