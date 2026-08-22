# Pretrained Medical-Image Backbones for RSNA Knee Abnormality Detection — Research Report

## 0. Executive summary

The honest headline: there is no strong, drop-in, knee-MRI-pretrained 2D backbone that is both publicly downloadable and clearly better than your ImageNet-22k ConvNeXt-Small. The medical-pretraining space is dominated by (a) chest X-ray models, (b) 3D CT/segmentation models, (c) gated or unreleased MRI foundation models. The two genuinely actionable options are RadImageNet (ResNet-50, the only MRI-containing 2D classification pretraining with real knee-MRI transfer evidence) and self-supervised pretraining on the competition's own unlabeled data (DINOv2/MAE), which sidesteps the availability problem entirely.

Critically, top Kaggle medical-imaging solutions — RSNA 2019, 2022, 2023, 2024 — have essentially all used ImageNet-pretrained backbones, not medical-pretrained ones. That is the single strongest practical data point in this report.

## 1. Candidate table

| # | Model | Arch / params | Pretrain data | 2D/3D | Weights available? | License | Size | timm-drop-in? | MRI/MSK evidence |
|---|---|---|---|---|---|---|---|---|---|
| 1 | RadImageNet ResNet-50 (HF convergedmachine/RadImagenet, GitHub) | ResNet-50, 25M | 1.35M CT/MRI/US images, 11 anatomies incl. knee, 165 labels, supervised | 2D | ✅ Yes (also official PyTorch via Google Drive) | MIT (repo) / Apache-2.0 (refined fork) | 191 MB | ✅ Excellent — timm.create_model('resnet50', num_classes=165) then load | Strongest: +4.8% AUC ACL, +4.5% meniscus on MRNet vs ImageNet |
| 2 | RadImageNet ResNet-18 / ResNet-10t | ResNet-18 (11M) / ResNet-10t | same | 2D | ✅ | Apache-2.0 | 90 MB / 40 MB | ✅ | same paper family |
| 3 | RadImageNet DenseNet121 / InceptionV3 / InceptionResNetV2 | CNN | same | 2D | ⚠️ Official PyTorch via Google Drive only (I could not verify the link resolves) | MIT | ~30–220 MB | ⚠️ Non-timm state dicts, needs remapping | same |
| 4 | MRI-CORE (GitHub, paper) | ViT-B (SAM-style), 86M | 6M MRI slices / 110k volumes, 18 body regions (Duke-110K), SSL | 2D | ✅ Claimed released | CC BY 4.0 | ~350 MB | ⚠️ SAM-style ViT, not a timm classifier; needs adapter | Segmentation-only evals; no classification vs ImageNet comparison |
| 5 | RAD-DINO (HF microsoft/rad-dino) | ViT-B/14, 87M | 883k chest X-rays, DINOv2 | 2D | ✅ Ungated | MIT | ~350 MB | ⚠️ HF AutoModel, not timm; easy enough | Wrong modality (CXR); no MRI evidence |
| 6 | BiomedCLIP (HF) | ViT-B/16, 86M | PMC-15M figure-caption pairs (paper figures, incl. MRI panels), CLIP | 2D | ✅ Ungated | MIT | ~800 MB (full), ~350 MB vision | ✅ It is a timm ViT under open_clip | Weak — figures ≠ clinical slices; no MSK MRI numbers |
| 7 | MedSigLIP-448 (HF) | SigLIP ViT, 400M vision | CXR, derm, ophtho, path, CT/MRI slices | 2D | ⚠️ Gated (must accept HAI-DEF terms) | health-ai-developer-foundations | ~1.6 GB (vision) | ⚠️ HF transformers | No MSK MRI numbers; 448px native |
| 8 | MedGemma-4B vision tower (HF) | = MedSigLIP 400M | as above | 2D | ⚠️ Gated | HAI-DEF | large | ⚠️ Awkward extraction | — |
| 9 | MedImageInsight (paper; community mirror HF lion-ai/MedImageInsights) | DaViT-B dual-encoder, ~360M | Multi-domain incl. MRI, image-text | 2D | ⚠️ Official route = Azure Foundry; community HF mirror exists (legality/permanence uncertain) | MIT claimed on mirror; Microsoft terms ambiguous | ~1.2 GB | ❌ Custom UniCL codebase | SOTA on CXR/derm/OCT; no knee MRI |
| 10 | Curia / Curia-2 (HF raidium/curia, raidium/curia-2, GitHub) | ViT ~86M | 150k CT/MRI exams (130 TB), one hospital, SSL | 3D-ish/slice | ✅ | RESEARCH-ONLY RAIL-M — ⚠️ likely incompatible with a Kaggle prize competition | ~350 MB | ⚠️ AutoModel | 19-task bench, mostly CT/neuro |
| 11 | Swin UNETR SSL (weights, MONAI) | 3D Swin encoder, ~62M | 5,050 CT (v1) / ~50k CT+MR volumes (v2) | 3D | ✅ | Apache-2.0 | ~120 MB | ❌ 3D only — incompatible with your 2D-per-slice pipeline | Segmentation gains only |
| 12 | MedSAM / SAM-Med2D (HF flaviagiammarino/medsam-vit-base) | ViT-B SAM, 86M | 1.5M medical image-mask pairs | 2D | ✅ | Apache-2.0 | ~375 MB | ❌ Segmentation encoder; poor classification features in practice | Negative-ish for classification |
| 13 | 3DINO (GitHub) / BrainSegFounder (HF) | 3D ViT / SwinUNETR | Brain MRI (UK Biobank etc.) | 3D | ✅ | varies | — | ❌ | Brain-only, wrong anatomy |
| 14 | PRISM (arXiv 2508.07165) | 3D, arch unstated | 336,476 MRI volumes, 34 datasets, whole-body multi-sequence | 3D | ❌ Not confirmed released | unknown | — | ❌ | Best-in-class claims (39/44 benchmarks) but I could not verify any weight download |
| 15 | Decipher-MR (arXiv 2509.21249, npj Dig Med) | 3D ViT + text | 200k MRI series / 22k studies | 3D | ❌ Explicitly withheld ("safety, IP, commercial viability") | n/a | — | ❌ | Unusable |
| 16 | SKELEX (arXiv 2602.03076) | ViT-L, ImageNet-1k init | 1.3M MSK radiographs (SNUH-1M), SSL | 2D | ❓ Unverified | unknown | ~1.2 GB | ⚠️ | Right body region, wrong modality (X-ray). Includes OA grading + fracture — conceptually adjacent to your labels |
| 17 | OAI-DINO (ISMRM 2025 abs. 4014) | DINO on 3D knee MRI, OAI | Knee MRI — perfect match | 3D | ❌ No release found | n/a | — | ❌ | Beat both scratch and supervised pretraining — but no weights |
| 18 | MRNet-derived models (Stanford) | AlexNet-era | 1,370 knee MRI | 2D | ❌ Community repos ship code, not weights; dataset requires a signed research agreement | restrictive | — | ❌ | Too small/weak to matter as a backbone anyway |
| 19 | DINOv2 / DINOv3 (natural images) (timm/HF) | ViT-S→L, ConvNeXt-B variants | LVD-142M natural images, SSL | 2D | ✅ | DINOv2 Apache-2.0; DINOv3 has a Meta license — check terms before Kaggle use | 90 MB–1.2 GB | ✅ timm | See §4 — genuinely strong evidence at 512px |
| 20 | MedMAE / PaCX-MAE / assorted MAE CXR checkpoints | ViT-B | CXR | 2D | ⚠️ Scattered, often dead links | varies | — | ⚠️ | Weak, CXR-only |

## 2. Detail on the ones that matter

### 2.1 RadImageNet — the only candidate with direct knee-MRI transfer numbers

What it is. 1.35M annotated CT/MRI/ultrasound images (the 5M figure quoted in some abstracts refers to the raw collection before curation), 3 modalities, 11 anatomies, 165 pathology labels, from Mount Sinai. Supervised multi-class classification pretraining at 256×256, 3-channel, standard [0,1]/Keras-style scaling. MIT license. Anatomies explicitly include musculoskeletal — knee MRI is in the pretraining distribution.

The headline result (Radiology: AI 2022, PMC9530758), on the MRNet knee dataset with 5-fold CV:

| Task | RadImageNet AUC | ImageNet AUC | Δ |
|---|---|---|---|
| ACL tear | 0.97 ± 0.03 | 0.91 ± 0.08 | +4.8% (P<.001) |
| Meniscal tear | 0.96 ± 0.02 | 0.92 ± 0.06 | +4.5% (P<.001) |

That is exactly two of your twelve labels, on knee MRI, at small sample size. It is the single most on-point piece of evidence in the entire literature for your task.

Caveats you must weigh:

- The tasks are near-in-distribution to RadImageNet's own 165 labels (which include MSK MRI findings). Some of the gain is "the pretraining task nearly is the downstream task," which won't generalize equally to Baker's cyst, synovitis, or contusion.
- The comparison is ResNet-50/DenseNet121/InceptionV3 class models at 256px — an architecture generation below your ConvNeXt-Small. Your baseline is 0.894; a RadImageNet ResNet-50 may simply start from a weaker place and lose despite better pretraining. There is no RadImageNet ConvNeXt, and no RadImageNet ViT (the repo says "Swin will be released later" — still unreleased as far as I can verify).
- A dental-radiograph reproducibility study found ImageNet beat RadImageNet on larger datasets (pneumonia, COVID) and did not cleanly reproduce the advantage.
- A follow-up ACL-tear architecture study broadly supports RadImageNet's edge for ACL tear, but I could not extract precise per-architecture AUCs from the PDF — treat as directional support only.

Practical access. The cleanest path is convergedmachine/RadImagenet — resnet50.pth (191 MB), resnet18.pth (90 MB), resnet10t.pth (40 MB), all native timm state dicts (timm.create_model('resnet50', num_classes=165)), from the Refined-RadImagenet project (Apache-2.0). Note: the HF README is empty, so input normalization is undocumented there — assume timm ImageNet stats unless you verify otherwise. These are retrained refinements, not exact ports of the original Keras weights; their reported 165-class macro-F1 is 0.51 for ResNet-50. The original authors' PyTorch weights are behind a Google Drive link on the official repo which I did not verify resolves.

### 2.2 MRI-CORE — the most interesting underexplored option

MRI-CORE is a 2D ViT-B trained self-supervised on 6M MRI slices from 110k volumes across 18 body regions (Duke-110K, 2016–2020, 9+ sequence types) at 1024×1024 with per-slice min-max normalization. Weights are stated as released under CC BY 4.0 at github.com/mazurowski-lab/mri_foundation.

This is, on paper, the best modality/dimensionality match available: 2D, MRI-only, huge, permissive license, high native resolution (fits your 512px pipeline better than 224px models). But: every published evaluation is segmentation (Dice vs SAM/MedSAM), with zero classification benchmarks and zero ImageNet comparison. It is a SAM-style encoder, so create_model(..., num_classes=0) won't work directly — you'd wrap the ViT and pool patch tokens. I could not verify the checkpoint download from the search results, so confirm the weights actually exist before planning around this.

### 2.3 Things that look attractive but are traps

- PRISM and Decipher-MR are the two best MRI foundation models by scale/claims. Decipher-MR's authors state outright they are not releasing weights. PRISM's release status is unverified. Don't plan around either.
- Curia is downloadable but RESEARCH-ONLY RAIL-M — for a prize-money Kaggle competition this is a real licensing hazard. Flagging clearly.
- MedSigLIP / MedGemma are gated behind Google's HAI-DEF terms. Gating is survivable (download once, ship in a Kaggle Dataset) but you should read whether HAI-DEF permits redistribution inside a Kaggle Dataset — I did not verify that it does.
- MedImageInsight's only convenient access is a community HF mirror; the official route is an Azure endpoint. Provenance and permanence are both shaky.
- Swin UNETR / 3DINO / BrainSegFounder / MedSAM are architecturally incompatible with a 2D-slice-encoder pipeline, or are segmentation encoders whose features transfer poorly to classification.
- MRNet weights: no credible public backbone weights; dataset itself requires a research agreement. Dead end.

## 3. Does medical pretraining actually beat ImageNet? Both sides.

**For "no, ImageNet is fine":**

- Transfusion (Raghu et al., NeurIPS 2019) — the canonical skeptic paper. ImageNet transfer gives "little benefit"; gains come largely from over-parameterization, not feature reuse; feature reuse is concentrated in early layers; ImageNet accuracy does not predict medical accuracy.
- Dental RadImageNet study (2024) failed to reproduce RadImageNet's advantage and found ImageNet superior on larger datasets.
- Comparative analysis of ImageNet models vs DINOv2 in medical classification and DINOv2 on radiology benchmarks both find general-purpose SSL competitive with or better than domain models.
- Resolution scaling governs DINOv3 transfer (2025) found that frozen features from a 7B medical-scale teacher underperformed a fully finetuned 86M ImageNet-init backbone. Fine-tuning capacity > pretraining pedigree.
- Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification — benchmarks DINOv2, ConvNeXt, ResNet, DenseNet, Med3D, MedImageInsight, MedGemma across RSNA 2022/2023/2024 Kaggle datasets. Reported outcome: DINOv2 beat ImageNet ConvNeXt/ResNet, and medical-pretrained backbones were "mixed — not universally superior to DINOv2." (I could not extract the numeric tables from the PDF; treat the direction as reliable, the magnitudes as unverified.)

**For "yes, domain pretraining helps":**

- RadImageNet original — the +4.8% / +4.5% knee MRI numbers above, with the strongest gains explicitly on small datasets, which is your regime.
- Moving from ImageNet to RadImageNet — independent-ish confirmation for classification and segmentation, again with larger gains on smaller data.
- OAI-DINO (ISMRM 2025) — knee-MRI SSL beat supervised pretraining on knee downstream tasks.
- SKELEX — MSK-domain SSL on top of an ImageNet-1k ViT-L init improved fracture detection and OA grading across 12 tasks.

My read of the evidence quality: the pro-domain evidence that is strongest and most on-point (RadImageNet knee) is also the evidence most vulnerable to task-leakage and to being architecturally obsolete. The anti-domain evidence is broader, more recent, and better controlled. The one pattern both sides agree on: domain-adaptive continued pretraining on data close to your own distribution beats swapping in a foreign-domain checkpoint. That points straight at option #1 below.

## 4. What Kaggle winners actually used

| Competition | Winning backbone(s) | Medical-pretrained? |
|---|---|---|
| RSNA 2019 Intracranial Hemorrhage (1st place, writeup) | SE-ResNeXt-101, DenseNet-121/169 + sequence model over slices | ❌ ImageNet |
| RSNA 2022 Cervical Spine (1st, 3rd, 5th) | 2.5D EfficientNet-V2-S / ConvNeXt + LSTM over slices | ❌ ImageNet |
| RSNA 2023 Abdominal Trauma (1st) | 2.5D CNN (ConvNeXt/EfficientNet family) + GRU, multi-scale feature maps | ❌ ImageNet |
| RSNA 2024 Lumbar Spine (1st) | localize-then-classify; ResNet-50-FPN detector + Swin/ConvNeXt classifiers on 2.5D crops | ❌ ImageNet |

Zero of the four used medical-domain pretrained weights. Every one instead invested in: slice-sequence aggregation, localization/cropping, resolution, and ensembling. This is a strong prior that your remaining headroom is not in the backbone checkpoint.

One relevant signal from your own competition: at least one public approach for RSNA Knee uses a DINOv2-small (ViT-S/14) backbone with target-specific attention pooling — i.e. others are also probing SSL backbones rather than medical ones.

## 5. Ranked shortlist — what I'd actually do

### #1 — Self-supervised continued pretraining on the competition's own images (DINOv2 or MAE), initialized from your ConvNeXt/ViT

Expected gain: +0.005 to +0.020. Risk: medium. Cost: ~$15–40.

You have 3,639 studies × 96 slices ≈ 350k unlabeled in-domain images, plus whatever unlabeled/test-adjacent data the competition provides. This is exactly the regime where SSL pays, and it is the one intervention where the literature is unanimous: OAI-DINO beat supervised pretraining on knee MRI; SKELEX's whole recipe is "ImageNet init + domain SSL"; the DINOv3 resolution paper shows fully-adapted mid-size backbones win. It has zero licensing, gating, or download risk, and produces a checkpoint you already own. This dominates every off-the-shelf option on expected-value-per-risk.

Reasoning for #1: it's the only option that both (a) matches your exact data distribution and (b) has no availability caveats. And it directly attacks the "capacity exhausted" symptom — ConvNeXt-Base not helping suggests a data/representation bottleneck, which is what SSL fixes.

### #2 — DINOv2 (and DINOv3 if licensing permits) ViT-B/ConvNeXt-B at 512px

Expected gain: +0.003 to +0.012. Risk: low. Cost: ~$15.

Cheapest credible experiment. Resolution scaling governs DINOv3 transfer found DINOv3+ConvNeXt-B at exactly 512×512 to be the strongest initialization, with gains concentrated on small, focal, boundary-dependent findings — which describes meniscal tears, Baker's cyst, and fractures precisely. Your pipeline is already 512px ConvNeXt. This is nearly a one-line change. Two warnings: (a) fully fine-tune, don't freeze — frozen giant-teacher features lost to ImageNet init in that same paper; (b) verify the DINOv3 license permits competition use (DINOv2 is Apache-2.0 and safe).

### #3 — RadImageNet ResNet-50 as a diversity member, not a replacement

Expected gain: +0.002 to +0.008 via ensembling. Risk: low. Cost: ~$15.

It is the only checkpoint with direct evidence on two of your twelve labels. But ResNet-50 at 256px is a weaker architecture than ConvNeXt-Small at 512px, so I would not expect it to beat your baseline standalone. Its value is as a decorrelated ensemble member — a backbone that saw knee MRI during pretraining will make different errors than an ImageNet one, and macro-AUROC ensembles reward exactly that. 191 MB, MIT/Apache, native timm load: trivially cheap to test. If it lands within ~0.01 of baseline solo, it's almost certainly ensemble-positive.

### #4 — MRI-CORE ViT-B, contingent on verifying the download

Expected gain: unknown, −0.02 to +0.015. Risk: high. Cost: ~$20 + integration time.

Best theoretical fit in the whole report (2D, MRI-only, 6M slices, 18 body regions, 1024px native, CC BY 4.0). But it is a segmentation-oriented SAM-style encoder with no published classification evidence whatsoever, and I could not confirm the weights actually download. Spend 20 minutes verifying the repo has a real checkpoint before spending anything else. High variance — this is the "could be the unlock nobody else found" slot.

### #5 — Non-backbone alternatives you should weigh against all of the above

Expected gain: plausibly larger than #1–#4.

Given that four consecutive RSNA winners got their edge from pipeline rather than pretraining, and that ConvNeXt-Base gave you +0.0008: before spending more on backbones, consider that your 24-slices-per-slot sampling, your pooling head (attention pooling per-target rather than global mean), slot-aware feature fusion, and resolution/cropping are all plausibly worth more than any checkpoint swap. The Base-vs-Small null result is evidence that capacity is exhausted — it is not evidence that pretraining is the binding constraint. It may equally point at the pooling head or the slice sampler.

## 6. Explicit uncertainty log

- Not verified downloadable: PRISM weights; MRI-CORE checkpoint file; SKELEX weights; original RadImageNet Google Drive PyTorch links; MedImageInsight official weights.
- Confirmed NOT released: Decipher-MR (authors declined), OAI-DINO (conference abstract only), MRNet backbone weights.
- Gated: MedSigLIP, MedGemma (HAI-DEF terms — redistribution rights inside a Kaggle Dataset unverified).
- License hazard: Curia (RESEARCH-ONLY RAIL-M), DINOv3 (Meta license, not Apache).
- Numbers I could not extract: per-architecture AUCs in arXiv 2508.17567; the comparison tables in arXiv 2512.12887. Both PDFs were fetched but tables were not machine-readable — directions reported, magnitudes not.
- Could not access: the Kaggle competition discussion forum (returns title only to WebFetch — it needs a logged-in session). The single DINOv2-small datapoint for your competition came via search snippet, not a primary read. Worth you checking the forum manually — it's the highest-density source for what's actually working on this exact data.
- The RadImageNet dataset size is quoted inconsistently across sources (1.35M curated images per the GitHub repo, "5 million" in several abstracts). I've used 1.35M as the primary-source figure.

## Sources

RadImageNet GitHub · RadImageNet paper (Radiology: AI) · PMC9530758 · Moving from ImageNet to RadImageNet · Refined-RadImagenet · HF convergedmachine/RadImagenet

Transfusion · Dental RadImageNet vs ImageNet · DINOv2 on radiology benchmarks · ImageNet vs DINOv2 in medical classification · Resolution scaling governs DINOv3 transfer · Revisiting 2D Foundation Models for 3D Medical Classification

MRI-CORE / repo · RAD-DINO · BiomedCLIP · MedSigLIP-448 · MedGemma-4B-pt · MedImageInsight / community mirror · Curia / Curia-2 / GitHub · PRISM · Decipher-MR · SKELEX · OAI-DINO ISMRM 2025 · MONAI Swin UNETR SSL · MedSAM · 3DINO · BrainSegFounder · MSK MRI foundation segmentation study

Kaggle winners: RSNA 2019 1st / writeup · RSNA 2022 winners analysis · RSNA 2023 1st · RSNA 2024 1st
