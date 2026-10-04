# Knee MRI image labellers: commercial, open-source and API options

Research date: 2026-10-04. Method: web search plus direct fetch of primary pages (vendor pages, arXiv/PMC papers, GitHub/HF repos, API terms, competition rules in `hackathon_page.txt`, RSNA MIRA agreement PDF). Anything not confirmed from a primary source is marked "unverified".

## 1) Executive summary

### Bottom line

1. **No candidate is likely to beat our own ~0.90 to 0.93 per-model / 0.93 ensemble as a zero-shot labeller.** Everything strong is either sales-only (and cannot legally receive our images), or open but only a pretrained backbone (not a finding classifier for our 12 labels), or a general VLM that publications show is weak at knee MRI.
2. **A hard blocker applies to every cloud or vendor service.** Competition rule 4.a: "You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition." The RSNA MIRA agreement adds "You may not otherwise copy, reproduce, retransmit, distribute, publish, commercially exploit or otherwise transfer any material." Uploading our DICOMs (or slice images) to Incepto KEROS, OpenAI, Google Gemini, Anthropic or any hosted API is a transmission to a non-participant. This is independent of price or ToS. Verdict: NOT USABLE for all hosted services. (Running a model fully locally or on our own Modal containers is fine.)
3. **The only genuinely new, usable asset found** is a knee-MRI-pretrained 2D ViT-L: **OrthoFoundation-L** (MIT repo, weights via Git LFS, ~1.13 GB). It is continued self-supervised pretraining of DINOv3-L on ~894k knee MRI slices plus ~358k knee radiographs. It is a backbone to fine-tune on our 4,400 studies, not a labeller. Its reported strength is ligament/meniscus/cartilage (private Peking University Third Hospital data), not our weak findings (lateral OA, synovitis, PF OA). Expect a different, possibly useful ensemble member, not a labeller that beats 0.93. DINOv3 license obligations carry over (see risks).
4. Second asset: **OrthoDiffusion** weights are on Hugging Face (MIT, 3 x 553 MB, one 3D diffusion U-Net per plane). Earlier internal reports said "no weight release confirmed"; that is now outdated. It is a 3D diffusion feature extractor with an 8-label head (ACL, PCL, MM, LM, MCL, LCL, patellar dislocation, joint effusion). Heavy to adapt, architecture unlike our pipeline.

### Ranked shortlist

| Rank | Candidate | Role | Verdict |
|---|---|---|---|
| 1 | OrthoFoundation-L (ytrsk/OrthoFoundation) | Fine-tune as backbone on our data; new ensemble member | SAFE for offline label/training use; RISKY if weights ship (inherits DINOv3 license, README/paper inconsistent on init) |
| 2 | OrthoDiffusion (lt-0123, HF lanstat0123/orthodiffusion) | 3D multi-plane features, effusion/MM/LM signal | SAFE on license (MIT), low feasibility (3D diffusion, 16x256x256 input) |
| 3 | MRI-CORE ViT (already used) | Already in our stack | SAFE (already judged) |
| 4 | Medical Slice Transformer recipe (DINOv2 + slice transformer) | Method only, no weights | SAFE (method) |
| 5 | Fine-tuned open VLM (Qwen2.5-VL / MedGemma, LoRA on our data) | Supervised labeller | SAFE if weights licence permits; evidence says SFT gets ~0.60 accuracy on MOAKS grading, not competitive with CNN/ViT |
| X | KEROS (Incepto), OpenAI, Gemini, Anthropic APIs | Hosted labellers | NOT USABLE (data transmission ban) |

### Does anything add a strong, different signal on lateral OA / synovitis / PF OA / lateral meniscus?

- **Lateral meniscus**: commercial KEROS reports lateral meniscus AUC 0.91 (one study) or 0.84 internal / 0.77 external (another, attribution to KEROS unverified), OrthoDiffusion reports LM tear AUC 85.0. Our report labels score 0.916 vs panel; none of these is clearly better, and KEROS is unusable.
- **Lateral OA, PF OA, synovitis**: no candidate was found that reports these as labels. KEROS has "cartilage lesions" and "joint effusions", not synovitis or tibiofemoral/patellofemoral OA grade. OrthoFoundation reports "cartilage damage". The only OA-grade-like open supervision is OAI MOAKS (cartilage, osteophyte, BML) which could pre-train a grader but needs the OAI data use agreement and is a different sequence (DESS, double-echo steady-state) from our PD/T2 fat-sat.
- Honest expectation: the best realistic gain is OrthoFoundation-L as one more diverse backbone, judged by ensemble test and the audit-58 split. With only 58 audit studies, AUROC differences under about 0.03 are within noise.

## 2) Comparison table

| Name | Type | Findings covered (of our 12) | Published accuracy | Access | Cost | License / ToS | Prize-safety | URL |
|---|---|---|---|---|---|---|---|---|
| KEROS (Incepto) | Commercial, CE IIa | ACL, MCL, MM, LM, cartilage lesions, bone edema (contusion proxy), effusion, popliteal cyst (Baker's) | One study: AUC 0.96 MM, 0.91 LM, 0.99 ACL; another Incepto-affiliated study: ACL 0.89, MM 0.88 int / 0.83 ext, LM 0.84 / 0.77 (attribution unverified) | Sales/demo only, platform subscription, no public API or research tier found | Not published | Device IFU: "shall not be used alone to recommend medical care for diagnostic purposes"; terms for outputs unverified | NOT USABLE (data transmission ban, sales-only) | https://incepto-medical.com/product/keros |
| Other vendors (Gleamer, Radiobotics RBknee, IB Lab KOALA, Qure.ai, Lunit, Annalise, VUNO, Siemens, Philips) | Commercial | No knee MRI pathology product found | n/a | n/a | n/a | n/a | NOT USABLE (nothing to use) | see section 3 |
| OrthoFoundation-L | Open weights, 2D ViT-L (DINOv3-L continued SSL) | ACL, PCL, MCL, MM/LM, cartilage, bone bruise, patellar dislocation (private data) | ACL AUC 98.5, PCL 84.3, MCL 93.3, cartilage 93.4 (repo README); ACL 97.7 with 50% labels | GitHub LFS pointer, 1.13 GB | Free | Repo MIT; paper CC BY-NC-ND 4.0; base DINOv3 license applies | SAFE offline / RISKY if shipped | https://github.com/ytrsk/OrthoFoundation |
| OrthoFoundation-MSK-5M | Same, 5M images | Same plus hip/shoulder/ankle | Not extracted | GitHub | Free | No license stated | RISKY (no license) | https://github.com/ytrsk/OrthoFoundation-MSK-5M |
| OrthoDiffusion | Open weights, 3 x 3D diffusion U-Net | ACL, PCL, MM, LM, MCL, LCL, patellar dislocation, effusion | Macro AUROC 0.908; MM 87.2, LM 85.0, effusion 86.4 (their data) | HF 3 x 553 MB | Free | MIT | SAFE (license) | https://huggingface.co/lanstat0123/orthodiffusion |
| MRI-CORE | Open weights, 2D SAM-style ViT | Backbone only | Segmentation only for knee-adjacent | GitHub | Free | CC BY 4.0 (per paper) | SAFE (already in use) | https://github.com/mazurowski-lab/mri_foundation |
| Res3DKneeModel (RWTH Aachen) | Code only | 23 conditions (cartilage, menisci, bone marrow, ligaments) | AUC >= 0.85 for 8 and >= 0.75 for 18 conditions | Code only, no weights seen | Free | No license shown | Not usable as weights; recipe only | https://github.com/spacecheck/Res3DKneeModel |
| Medical Slice Transformer | Method | n/a | Knee MRNet meniscus AUC 0.85 vs 0.69 for 3D ResNet | Paper (CC BY 4.0), no official code found | Free | n/a | SAFE (method) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/ |
| KneePreM | 3D SSL, OAI 19,011 series | Classification + segmentation | fastMRI+ ROC AUC 0.722 (50% data) | No code/weights found | n/a | unverified | Not available | https://arxiv.org/abs/2609.31461 |
| Decipher-MR (GE) | 3D ViT-B VLM | General MRI | n/a | Weights not shared | n/a | Paper CC BY-NC-SA 4.0 | NOT USABLE | https://www.nature.com/articles/s41746-026-02596-4 |
| PRISM / MARS multi-sequence FM | Foundation model | 44 tasks, knee tasks unverified | Ranked first in 41/44 benchmarks (their claim) | Weights unverified | n/a | unverified | Unverified | https://arxiv.org/abs/2508.07165 |
| Co-plane attention (Nat Commun 2024), 9-abnormality multitask (EClinMed 2025) | Papers | 12 and 9 knee abnormalities | Overall AUC 0.812; primary 0.898 int / 0.815 ext | No code found | n/a | n/a | Not available | PMC11368947, PMC12508577 |
| GPT-5 / GPT-4o | Hosted API | Any via prompt | GPT-5 meniscus (100 MRNet cases): acc 76%, sens 84%, spec 68% | API | per token | See section 3 | NOT USABLE (transmission) | n/a |
| Gemini 2.5 Pro | Hosted API | Any via prompt | ACL, 150 PD-FS volumes: best F1 0.31 | API | per token | See section 3 | NOT USABLE | n/a |
| Claude vision | Hosted API | Any | No knee MRI study found | API | per token | See section 3 | NOT USABLE | n/a |
| Qwen2.5-VL / Med3DVLM (SFT) | Open VLM | MOAKS grades | SFT overall acc 0.60 to 0.61 (OAI DESS) | HF | Free | model-specific | SAFE if licence ok; weak | https://psb.stanford.edu/psb-online/proceedings/psb26/sambara.pdf |

## 3) Detailed notes

### 3.1 Commercial products

**KEROS (Incepto Medical, Paris).**
- What it detects (vendor page): "ACL tears, MCL tears, meniscal tears, ligament, menisci, and cartilage lesions along with identifying bone edema, joint effusions, and popliteal cysts", plus patellar height, trochlear dysplasia, TT-TG. CE-marked Class IIa, notified body IMQ 0051. FDA status not stated on the page.
- Mapping to our 12: ACL, MCL, medial meniscus, lateral meniscus, joint effusion, Baker's cyst, bone contusion (via bone edema). Cartilage lesion output is a partial proxy for OA but is not tibiofemoral/patellofemoral OA grading. No synovitis, no fracture.
- Accuracy: a study of the Keros algorithm reports AUC 0.96 (0.93 to 0.99) medial meniscus, 0.91 (0.85 to 0.96) lateral meniscus, 0.99 (0.98 to 0.997) ACL. A separate multi-continental study quotes ACL AUC 0.89 (partial and complete), medial meniscus 0.88 internal / 0.83 external, lateral meniscus 0.84 / 0.77 (this one is an Incepto-affiliated model but I could not confirm it is the shipping KEROS version). A knee AI assistant study with Incepto-affiliated authors trained on 23,074 studies: radiologist sensitivity 81% to 86%, accuracy 86% to 91%. These are real-world clinical-grade PD-FS/T2 sets, not directly comparable to our panel labels.
- Access/price: "Request a demo"; business model is a hospital subscription platform. No public API, research tier or price found (unverified beyond the vendor page).
- ToS / outputs for training or competition: not published. The IFU line: "KEROS is a diagnostic tool and shall not be used alone to recommend medical care for diagnostic purposes."
- Competition fit: it would require sending competition DICOMs to a vendor (or installing a vendor container); rule 4.a forbids transmitting data to non-participants, and a sales-only product fails "publicly available and equally accessible ... at no cost" (rule 6.a). Verdict: NOT USABLE.

**Other vendors checked (no knee MRI pathology product found):**
- Gleamer: X-ray (fracture, BoneMetrics full-leg measurements). Not MRI.
- Radiobotics RBknee: knee X-ray osteoarthritis, FDA 2019. Not MRI.
- ImageBiopsy Lab (IB Lab): KOALA is standing knee radiograph OA (87% sensitivity, 83% specificity for KL mild vs moderate/severe, per search summary), LAMA leg alignment, HIPPO, PANDA, FLAMINGO (CT). The vendor site lists no knee MRI product (a snippet describing an MRI knee classification product could not be tied to a live product).
- Qure.ai: qMSK is fracture detection; vertebral compression on CT. No knee MRI.
- Lunit: distributes Radiobotics RBfracture; chest/breast core. No knee MRI.
- Annalise, VUNO: no knee MRI product found (VUNO lists chest X-ray, chest CT, bone age, brain MRI, ECG).
- Siemens Healthineers: GOKnee3D is an acquisition acceleration application, not interpretation; AI-Rad Companion covers brain, chest, prostate, spine in search results. Philips SmartSpeed Precise is reconstruction. GE: no knee interpretation product found.
- Mediaire: a snippet says it built software for automatic detection and classification of cartilage damage on knee MRI; status and availability unverified.
- Avicenna, Arterys, Nanox, Subtle, Rad AI, Neurophet, DeepTek, CureMetrix, Infervision, United Imaging, Ever Fortune, Medality: no knee MRI pathology product confirmed in my searches (unverified; absence of search hits is not proof).
- A 2023 review of commercial MSK/neuro algorithms (PMC10122717) lists no FDA-cleared knee MRI interpretation product; only RBknee (X-ray) and BoneMRI (3D bone visualization).

### 3.2 Open-source knee MRI models with weights

**OrthoFoundation (Yu et al., arXiv 2601.18250, Jan 2026; repo ytrsk/OrthoFoundation).**
- Architecture: 2D ViT-L, DINO student-teacher continued self-supervised pretraining on ~1.25M knee images (357,670 radiographs, 893,985 MRI slices) from OAI, fastMRI and private Peking University Third Hospital data. Trained on four A100 GPUs.
- MRI benchmark: ten binary tasks (ACL, PCL, medial/lateral menisci, collateral ligaments, patellar dislocation, bone bruise, medial patellar retinaculum, cartilage damage), "OrthoFoundation consistently ranked first"; PCL AUC 84.3, README lists ACL 98.49, PCL 84.31, MCL 93.34, cartilage 93.38 AUROC, meniscus 84.74 accuracy. Test data are private, so none of this is comparable to our panel labels. No OA grade, synovitis or effusion task is reported for MRI.
- Weights: `OrthoFoudation-L.pth` is a Git LFS pointer (sha256 385a7757..., 1,213,056,638 bytes). Whether LFS bandwidth allows download is unverified.
- License: repo LICENSE is MIT (README: "The project operates under the MIT License"). The paper itself is CC BY-NC-ND 4.0. The backbone is DINOv3-L ("OrthoFoundation uses a Vision Transformer backbone initialized with DINOv3-L weights", MSK-5M README). The paper methods text says "initialized with DINOv2 weights" in one passage and DINOv3-L in others, so the base is inconsistent in the paper (flag for verification).
- DINOv3 obligations (from our memory notes, not re-verified here): if DINOv3-derived weights ship we need the license copy, "Built with DINOv3", and publication acknowledgement.
- Feasibility: ViT-L (about 304M parameters) fine-tuning on 8 GB needs bf16, gradient checkpointing, small batch or LoRA; on Modal it is straightforward. Inference over 4,400 studies is cheap.
- Verdict: SAFE for offline use as a training/labelling signal; RISKY if the weights go into the submission without handling the DINOv3 license.

**OrthoFoundation-MSK-5M (ytrsk/OrthoFoundation-MSK-5M).** ~5M images (about 4M knee, 1M ankle/hip/shoulder). README names ACL, PCL, MCL, meniscus, cartilage plus hip labral, supraspinatus, ATFL/CFL tasks. No license stated, so treat as RISKY until clarified. Weights described as released.

**OrthoDiffusion (Lan et al., arXiv 2602.20752, Feb 2026; Renmin University and Peking University Third Hospital).**
- 3D U-Net diffusion denoisers on 16 x 256 x 256 volumes, one per plane (sagittal, coronal, axial), pretrained SSL on 15,948 unlabeled knee MRI scans. Supports segmentation of 11 knee structures and 8-label diagnosis, with transfer to ankle and shoulder (11 diseases).
- Per-label AUROC (macro 0.908): ACL 97.04, PCL 93.73, MM 87.24, LM 85.02, MCL 91.91, LCL 85.77, patellar dislocation 99.26, joint effusion 86.35. No OA, synovitis, Baker's cyst, contusion or fracture label.
- Weights: Hugging Face `lanstat0123/orthodiffusion` (MIT): axial_model.pt, coronal_model.pt, sagittal_model.pt, 553 MB each. Code MIT on GitHub (lt-0123/OrthoDiffusion). README quote: "Due to privacy restrictions, MRI data is not publicly released."
- Compute: authors pretrained on one H20 143 GB GPU; diffusion pretraining about 2.15e4 steps with batch 10. Fine-tuning or linear probing a 3D diffusion encoder on our variable-sequence studies is a research project, not a quick win.
- Verdict: SAFE on license, low feasibility. Useful mainly if you want a 3D multi-plane feature extractor for an effusion/meniscus head.

**Res3DKneeModel (Vuskov et al., Eur Radiol 2026, RWTH Aachen).** 3D ResNet-18 plus transformer over sagittal, coronal, axial and an optional T1; 23 conditions; 3,121 internal and 448 external studies; AUC >= 0.85 for 8 and >= 0.75 for 18 conditions. Repo github.com/spacecheck/Res3DKneeModel contains README, requirements, train.sh, src/, images/; no weight files seen and no license shown. Paper is CC BY. Useful as a recipe only.

**Medical Slice Transformer (Sci Rep 2025).** DINOv2 per-slice encoder plus transformer across slices. Knee MRNet meniscus AUC 0.85 +/- 0.04 vs 0.69 +/- 0.05 for 3D ResNet. No official code found (a GitHub repo arthurcc0/MST is breast-only and unlicensed). It is the same family of approach we already use (2.5D / slice aggregation).

**MRI-CORE (Mazurowski lab).** Weights at github.com/mazurowski-lab/mri_foundation, paper states CC BY 4.0, already in our stack (MRI-CORE ViT). Knee-adjacent results are segmentation only.

**KneePreM (arXiv 2609.31461).** 3D U-Net masked autoencoder pretrained on 19,011 OAI series; fastMRI+ two-label ROC AUC 0.722 with 50% data. No weights/code link found. Unavailable.

**Decipher-MR (GE HealthCare).** 3D ViT-B vision-language on 200,000 series, knee included. Per secondary summaries "pretrained weights might not be directly shared at this time"; paper CC BY-NC-SA 4.0. NOT USABLE.

**PRISM / MARS (Qiu et al., Nature BME July 2026, arXiv 2508.07165).** 336,476 volumes from 34 datasets; ranked first in 41 of 44 benchmarks. Weights availability, license and knee task list could not be confirmed (Nature page paywalled). Unverified.

**Published models without released weights found:** co-plane attention for 12 knee abnormalities (Nat Commun 2024, overall AUC 0.812, 1,748 subjects), 9-abnormality multitask system (EClinicalMedicine 2025, AUC 0.898 internal, 0.815 and 0.812 external), UCSF 3D CNN grading of cartilage/bone marrow/menisci/ACL (Radiology AI 2021, code availability unverified), MRNet-derived community repos (research quality, trained on MRNet which is under a Stanford research use agreement).

**Segmentation-only resources (not labellers):** KneeXNet-2.5D (cartilage and meniscus segmentation, weights and app released per its abstract, license unverified), OAI-ZIB / nnU-Net cartilage models, MedSAM / SAM2 fine-tunes (AutoMedLabel). They could generate cartilage thickness or meniscus-volume features for lateral OA but would need a downstream classifier trained on our labels.

**Datasets that can supervise a labeller (all licensing-sensitive):**
- fastMRI+ knee annotations (Microsoft, MIT repo; Nature Sci Data paper says CC BY 4.0): 16,154 bounding boxes, 13 study-level labels, 22 pathology categories over the whole fastMRI knee set. Each image labelled by a single radiologist. But the underlying fastMRI images are licensed "for internal research or educational purposes only", "Not SELL OR OTHERWISE MONETIZE", no redistribution, and "destroy all copies" on completion. RISKY (and trained weights are derived from them).
- OAI (MOAKS cartilage, BML, osteophyte, effusion-synovitis): free after a data use agreement. DESS sequence mostly, domain shift from clinical PD/T2 fat-sat. RISKY on the equal-accessibility and redistribution clauses; unverified details.
- MRNet (Stanford, 1,370 exams: abnormal, ACL, meniscus): Research Use Agreement. RISKY.
- SKM-TEA (155 patients, qDESS, 16 pathology boxes): Stanford AIMI; license unverified; tiny.
- 3DReasonKnee (rajpurkarlab/3DReasonKnee): 494k quintuples from 7,970 OAI DESS volumes with MOAKS grades; paper CC BY-NC 4.0; underlying OAI DUA applies.

### 3.3 Large general vision-language APIs

**Published evaluations (all weak):**
- GPT-5, zero-shot meniscal tear on 100 MRNet validation exams (coronal and sagittal, prompt "Analyze these knee MRI images ... Respond only with 'yes' or 'no'"): accuracy 76% (CI 0.668 to 0.833), sensitivity 84%, specificity 68%, F1 0.778; authors: "falls short of these specialized models", high false-positive rate. Specialised MRNet-class models reach meniscus AUC ~0.85 (MRNet paper: abnormality 0.937, ACL 0.965, meniscus 0.847).
- Gemini 2.5 Pro, ACL on 150 PD-FS volumes (50 healthy, 50 partial, 50 complete): best F1 0.31, precision 0.32 with region-of-interest prompting.
- MeniOmni (arXiv 2605.28161) meniscus grading: GPT-4o 62.5% acc, Claude 3.5 Sonnet 60.2%, Gemini 1.5 Pro 58.4%, Qwen2-VL-72B 55.4% vs fine-tuned Video Swin-B 63.8% (these numbers are from the paper's table as extracted; grading, not binary detection).
- 3DReasonKnee MOAKS grading: o1 zero-shot overall accuracy 0.477, with instruction schema 0.568; Qwen2.5-VL-7B 0.470 / 0.504; SFT of Qwen2.5-VL-3B 0.613 and Med3DVLM 0.596. Supervised fine-tuning beats frontier APIs, consistent with our MedGemma / Hulu-Med / Gemma 4 results.
- Generalist biomedical MLLMs (LLaVA-Med, RadFM) "struggle in specialized orthopedics" per the MeniOmni/related summaries. Merlin and CT-CHAT are CT models, CheXagent is chest only. Med-Gemini is closed. M3D, Med3DVLM, RadFM are CT-centred; no knee MRI numbers except Med3DVLM SFT above.

**Terms of service (outputs, medical use, competing models):**
- Google Gemini API terms (ai.google.dev/gemini-api/terms, fetched): "You may not use the Services to develop models that compete with the Services (e.g., Gemini API or Google AI Studio)." and "You may not use the Services in clinical practice, to provide medical advice, or in any manner that is overseen by or requires clearance or approval from a medical device regulatory agency." Free tier: Google "uses the content you submit ... to provide, improve, and develop Google products and services and machine learning technologies"; paid tier: Google "doesn't use your prompts ... or responses to improve our products".
- Anthropic Commercial Terms (fetched): customers may not "access the Services to build a competing product or service, including to train competing AI models or resell the Services except as expressly approved by Anthropic." Anthropic AUP for healthcare use: "a qualified professional in that field must review the content or decision prior to dissemination or finalization" and AI-use disclosure required. Using Claude vision outputs as a soft label for our own knee classifier is not an obviously competing AI model, but a Kaggle-winning model that others may reuse is a gray area (unverified legal view).
- OpenAI Terms of Use (page returned 403, so no verbatim quote; secondary sources and search summary): users may not use Output "to develop models that compete with OpenAI", with an exception for models "primarily intended to categorize, classify, or organize data (such as embeddings or classifiers)" if not distributed or made commercially available to third parties. Treat as paraphrase, unverified verbatim.
- None of these overrides the real blocker: sending Competition Data to these services breaches rule 4.a and the MIRA agreement. Verdict for all three: NOT USABLE.

## 4) Fine-tuning options (our ~4,400 studies, license permitting)

| Option | Weights/licence | Compute | Expected value |
|---|---|---|---|
| OrthoFoundation-L (ViT-L, knee MRI SSL) | MIT repo, DINOv3 base license applies | Modal: easy (ViT-L, 2D slices, our existing 2.5D / per-finding head). Local 4060 8GB: bf16 + checkpointing + LoRA, small batch | Best new candidate; a knee-MRI-specific, different pretraining from MedImageInsight, MRI-CORE, DINOv3. Untested on OA/synovitis. |
| OrthoDiffusion planes | MIT | 3D diffusion U-Net with 16x256x256 inputs; need custom data pipeline, several GPU-hours at minimum (unverified), authors used 143 GB GPU for pretraining | Low; architecture mismatch with current pipeline |
| MRI-CORE | CC BY 4.0 (paper) | Already done | In stack |
| Open VLM LoRA (Qwen2.5-VL-3B/7B, MedGemma 4B) | model licences vary (verify HF card; MedGemma has Health AI Developer Foundations terms, unverified here) | 3B LoRA fits 24 GB comfortably; on 8 GB only with 4-bit and tiny context | Evidence says SFT VLMs reach ~0.60 accuracy on MOAKS grading; unlikely to beat CNN/ViT; limited value |
| Pretrain a grader on OAI MOAKS or fastMRI+ then transfer | Data agreements, domain shift (DESS) | Moderate | RISKY licensing and shaky transfer; skip unless equal-accessibility is confirmed |
| Slice-transformer recipe (MST) on top of any 2D backbone | Apache/CC BY method | Already analogous to our 2.5D | No new signal |

Suggested low-cost experiment: load OrthoFoundation-L into the existing engine as a backbone, train per-finding heads with the usual flip/ensemble protocol, run the ensemble test and audit-58 comparison. Compare each finding against MedImageInsight, with attention to lateral OA, synovitis, PF OA, lateral meniscus.

## 5) Risks and open questions

1. **Transmission ban is absolute for hosted tools.** Even a "private" vendor demo or a paid API receives Competition Data. Only local or own-Modal inference is allowed. If the host later clarifies that de-identified single slices to a paid API are acceptable, the verdicts for APIs change from NOT USABLE to RISKY, but their accuracy still does not justify it.
2. **DINOv3 provenance of OrthoFoundation.** Paper text says DINOv2 initialisation in one place, DINOv3-L elsewhere. MIT on the repo does not remove upstream DINOv3 license duties. Needs a written check against the DINOv3 license before shipping weights; pure offline label use is lower risk.
3. **OrthoFoundation weights via Git LFS**: the pointer is real (1.13 GB) but LFS bandwidth/availability and integrity (sha256 given) are untested here. The 5M variant has no license text.
4. **Benchmarks are on private data** (Peking University Third Hospital). None reflects OA, synovitis or panel-read labels, so published AUCs say little about our weak findings.
5. **Audit-58 is small.** Any claim of improvement on lateral OA / synovitis / PF OA must be judged with the ensemble test and wide confidence intervals; differences of a few hundredths of AUROC are not detectable on 58 studies.
6. **Dataset-derived weights.** Anything trained on fastMRI (non-commercial, internal research only, destroy-on-completion), MRNet (Research Use Agreement) or OAI (DUA) inherits those terms. fastMRI+ annotations alone are CC BY 4.0 / MIT, but they only mean something with the images. OrthoFoundation itself is pretrained on OAI + fastMRI + private data; exposure is the licensor's, but winners must disclose and the host may ask.
7. **Unverified items:** KEROS pricing, API, and whether its figures in the second study refer to the shipping product; FDA status of KEROS; PRISM/MARS weights and license; KneeXNet-2.5D license; MedGemma and Qwen2.5-VL license text; OpenAI terms verbatim; Mediaire product; whether any other vendor (Arterys, Nanox, etc.) has a knee MRI interpretation product.
8. **Competition rule context.** Rule 6.a/6.b (external data and models acceptable if publicly available at no cost or reasonable), rule 8.2 "a small subscription charge ... acceptable", but "Purchasing a license to use a proprietary dataset that exceeds the cost of a prize ... would not be considered reasonable." A vendor product like KEROS (unpublished price, sales-only) would almost certainly fail "equally accessible".

## Sources

- KEROS: https://incepto-medical.com/product/keros ; Eur Radiol 2025 AI assistance study https://link.springer.com/article/10.1007/s00330-025-11820-w ; resident study https://link.springer.com/article/10.1007/s00330-025-12052-8 ; Keros multi-continental ACL https://link.springer.com/article/10.1007/s00330-022-08923-z ; Keros meniscus/ACL arthroscopy crossover https://link.springer.com/article/10.1007/s00264-025-06531-2
- Vendors: https://www.imagebiopsy.com/ ; https://pmc.ncbi.nlm.nih.gov/articles/PMC10122717 ; https://www.itnonline.com/content/fda-clears-siemens-healthineers-goknee3d-mri-application ; https://www.qure.ai/us/blog/the-importance-of-ai-in-msk-radiology
- OrthoFoundation: https://arxiv.org/abs/2601.18250 ; https://github.com/ytrsk/OrthoFoundation ; https://github.com/ytrsk/OrthoFoundation-MSK-5M
- OrthoDiffusion: https://arxiv.org/abs/2602.20752 ; https://github.com/lt-0123/OrthoDiffusion ; https://huggingface.co/lanstat0123/orthodiffusion
- Res3DKneeModel: https://github.com/spacecheck/Res3DKneeModel ; MST: https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/ ; MRI-CORE: https://arxiv.org/html/2506.12186v1 ; KneePreM: https://arxiv.org/abs/2609.31461 ; PRISM/MARS: https://arxiv.org/abs/2508.07165 ; Decipher-MR: https://www.nature.com/articles/s41746-026-02596-4
- Nat Commun 12-abnormality: https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947 ; EClinicalMedicine 9-abnormality: https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577
- LLM/VLM evaluations: https://pmc.ncbi.nlm.nih.gov/articles/PMC12811926/ ; https://pmc.ncbi.nlm.nih.gov/articles/PMC12961630/ ; https://arxiv.org/pdf/2605.28161 ; https://psb.stanford.edu/psb-online/proceedings/psb26/sambara.pdf
- Terms: https://ai.google.dev/gemini-api/terms ; https://www.anthropic.com/legal/commercial-terms ; https://www.anthropic.com/legal/aup ; OpenAI terms (403 on fetch): https://openai.com/policies/terms-of-use/
- Datasets/licences: https://fastmri.med.nyu.edu/ ; https://github.com/microsoft/fastmri-plus ; RSNA MIRA agreement http://rsna.org/mira-license ; competition rules `D:\My Projects\knee_abnormality_detection\hackathon_page.txt` (sections 4, 6)
