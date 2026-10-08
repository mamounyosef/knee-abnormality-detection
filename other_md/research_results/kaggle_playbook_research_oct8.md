# Kaggle playbook and competitive intelligence, 8 Oct 2026

Scope: competitive intelligence on the live RSNA Knee Abnormality Detection competition, plus write-ups from analogous past RSNA challenges, plus inference-budget tricks for 2x Tesla T4 (T4 = a data-centre GPU with 15 GB memory, fp16 capable) in a 9 hour (h) code-competition cap.

## 0. Retrieval limits (read first)

- Kaggle discussion, writeup and notebook pages are rendered by JavaScript. The fetch tool returned only page titles for every Kaggle URL tried (competition discussion index, 1st place writeup pages). So I could NOT read any forum thread, writeup text or public notebook from Kaggle directly.
- Everything about the knee forum (thread 735304, 745861, 743416, 741819, 734109, 740718) and the public notebooks in this report comes from our existing docs (`big_jump_ideas_oct7.md`, `public_notebooks_0946_analysis.md`, `public_notebook_scored_0_941_summary.md`), NOT from a fresh read. Treat as "as recorded on 7 Oct", not re-verified.
- Web search found NO post newer than 7 Oct 2026 about the knee competition, no score above 0.946 beyond what our docs already record, and no new host clarification on external data or LLM use.
- Reliable text was found only for the GitHub repos and news pages listed in section 5. Past-winner technique details below are therefore thin, and I say so per item.
- Fix for next time: a browser session (Claude in Chrome or Playwright) that renders Kaggle pages, or a logged-in Kaggle CLI call for discussion text. Not done here (research-only brief, no browser or Kaggle write access used).

## 1. Current competition facts (rules, from local hackathon_page.txt)

- Deadlines: team merger 15 Oct 2026; winners' requirement deadline 5 Nov 2026 (training code, video, method description); final submission 22 Oct 2026.
- Prizes: Leaderboard 1st to 10th = $9,000 down to $5,000 (total $77,000 with efficiency prizes). **Efficiency Prize: 1st $7,000, 2nd $6,000, 3rd $5,000**, judged on runtime AND predictive performance on private data. Only submissions the team selected for the Leaderboard Prize (or auto-selected) are eligible, and a submission can win both.
- Rules section 2.6 (external data): allowed if publicly available and equally accessible to all at no cost, OR meets the "Reasonableness" standard (reasonably accessible to all, minimal cost). Pre-trained models are explicitly allowed ("Freely & publicly available external data is allowed, including pre-trained models").
- Team limits: max 5 members; hackathon-style "one submission per team" rule applies to Hackathons only (not this competition's text as far as grepped; re-check before merger).
- Winners must open-license the solution (section 2.8) and may be asked for a reproducible method description and a recorded or panel call. Incompatible-license data or models do not need the open-source grant, but must still be disclosed.

**Efficiency Prize implication (new to our docs?).** Our docs track the efficiency prize as a separate tuning track (outline line 9, 348, 438). The rules say the efficiency score is computed on the same private test data, so a slightly weaker but much faster ensemble may win $5k to $7k that the 0.943 ensemble cannot. Cost to check: the efficiency leaderboard is public during the training period (rank only), so we can see where a candidate lands.

## 2. Past analogous competitions (what the sources actually say)

| Competition | Winner (rank 1) | Source | What we could verify | Transferable idea |
|---|---|---|---|---|
| RSNA 2024 Lumbar Spine Degenerative Classification | Avengers (2nd IanPan-Kevin-Yuji-Bartley, 3rd SonySpine s & tkmn & Moyashii); 1,874 teams | [RSNA results](https://www.rsna.org/news/2024/november/2024-ai-challenge-winners), [AuntMinnie](https://www.auntminnie.com/imaging-informatics/artificial-intelligence/article/15708869/rsna-names-winners-of-lumbar-spine-ai-challenge) | Winner names only. Writeup NOT read. Educational Merit Award (code clarity and efficiency) went to Avengers and SonySpine team. | Code clarity is scored separately: keep the submission tidy for the 5 Nov package. |
| RSNA 2023 Abdominal Trauma Detection | Team Oxygen | [Albumentations listing](https://albumentations.ai/adoption/competitions/kaggle/rsna-2023-abdominal-trauma-detection/), [ITN Online](https://itnonline.com/node/89790) | Rank only. Peer-reviewed review (Radiology: AI, 2024) of winners: mean AUC 0.85 for bowel and mesenteric injury. A third-party GitHub (siwooyong) describes one-stage resnest50d + GRU attention (13 outputs) plus two-stage with segmentation; its rank is NOT stated. | Multi-label CT classifier with a GRU over slices, plus an optional segmentation stage. |
| RSNA 2022 Cervical Spine Fracture | Not confirmed from RSNA page; Zenn (Japanese) walkthrough describes a 1st place two-stage pipeline | [Zenn walkthrough](https://zenn.dev/yuto_mo/articles/f9fb1c8297f252) (secondhand) | Stage 1: 3D segmentation of each vertebra (7-channel mask, 128 px, resnet18 or efficientnet_v2s + U-Net), trained on only 87 masked volumes, then predicts masks for all about 2k studies. Stage 2: 2.5D (three-adjacent-slice) CNN + LSTM classifier, two model types. Writeup is Kaggle topic 362787 (not read). | Two-stage: segmentation or localisation first, then a 2.5D classifier. Pseudo-masks from a small model can train a stage-1 model on few labels. |
| RSNA 2023 Screening Mammography Breast Cancer | Conflicting: a GitHub repo (dangnh0611) claims 1st place and links Kaggle discussion 392449; Albumentations lists "mr.robot" gold at rank 1 | [dangnh0611 repo](https://github.com/dangnh0611/kaggle_rsna_breast_cancer), [Albumentations](https://albumentations.ai/adoption/competitions/kaggle/rsna-breast-cancer-detection/) | Repo README only: YOLOX breast ROI detector, then four ConvNeXt-small classifiers ensembled. Rank claim unverified. | ROI detector then ensemble of 4 small backbones. Matches the "crop then ensemble" pattern. Cropping only pays when it removes irrelevant area; our crop-safety data (outline line 308: cropping saves only 16%) argues against a big crop stage for knees. |
| RSNA 2025 Intracranial Aneurysm Detection | tomoon33 (2nd BraveCoWCoW, 3rd BTYND); 1,147 teams | [RSNA winners](https://www.rsna.org/news/2025/november/2025-ai-challenge-winners); Japanese summary via [AuntMinnie/RSNA search](https://www.auntminnie.com/imaging-informatics/artificial-intelligence/article/15751740/rsna-launches-2025-intracranial-aneurysm-detection-ai-challenge) | Secondhand only (a competitor's Japanese slide deck): vessel segmentation, then 3D classifier on the vessel mask, coarse-to-fine, nnU-Net heavy, location as auxiliary segmentation. Official writeup URL given but not readable. | Use the segmentation as an attention guide for the classifier; coarse-to-fine search. Heavy for our budget. |
| 2025 Educational Merit Award | Ian, Theo and Bartley; MIC-DKFZ | RSNA winners page | Given for code clarity, completeness, organisation and efficiency. | Same as 2024: a clean package can earn an award. |

Not found: a writeup or method text for any 2026 knee forum post with 0.954 (the 5-fold ResNet-50 @224 result in our docs). It could not be confirmed here.

## 3. Ranked playbook for our pipeline (top 10, by expected leaderboard gain per dollar)

Evidence grades: A = reported by multiple sources or a measured result of ours, B = one credible secondhand report, C = speculation.

1. **Round-2 pseudo-labels from a diverse out-of-fold (OOF) teacher.** Already in `big_jump_ideas_oct7.md` (item 1). Evidence B-A: forum consensus (Chris Deotte "with pseudo-labels every single model matches the ensemble", tennogh "OOF pseudo-labels well correlated with LB"), Noisy Student (arXiv 1911.04252). Cost: 5-fold training of 3 to 4 members on v4 labels (GPU hours on Modal, real money). Risk: low for rules (no human labels). Status: NOT run yet. Top priority.
2. **Small models at low resolution as diverse members.** Forum 0.954 claim (ResNet-50 @224, 5-fold, from our docs) beats our 512 px singles (best 0.934). Evidence B (unverified here). Cost: cheap (224 px is far less compute than 512). Risk: low. Action: add a ResNet-50 @224 5-fold member and check it in the existing ensemble (no inference cost increase beyond one small model).
3. **Per-target teacher weighting (different teachers for different findings).** Raymond Yuen (42nd, from docs): "different teachers help different targets". Evidence B. Cost: zero GPU, just OOF AUROC per finding and a weighting table. Risk: overfitting the OOF with 12 targets; keep weights coarse (e.g. 2 to 3 levels). Fits our existing per-finding weights habit.
4. **Fold grouping by scanner or site.** GitHub repo (homeshwarnelakurthi, from search snippet): random K-fold inflates AUC by about 0.053 through scanner memorisation; the repo recommends grouped folds. Evidence B (self-reported by one author). Cost: the grouping key comes from DICOM tags (manufacturer, model, field strength), so it's CPU only. Risk: may make CV look worse and change model choice; that is the point. Relevance: our OOF teachers and early stopping depend on this.
5. **Inference speed for the efficiency prize: fp16 autocast, channels_last, batch by fixed shape.** Our docs already: decode cache (2.7x), two-GPU member split, thread caps (outline lines 924 onwards). Not yet in docs (verify in shared_code.py before claiming): `torch.autocast` fp16 for inference, `channels_last` memory format, and `torch.backends.cudnn.benchmark` off with fixed-size buckets (commit e2c04be says buckets are already used). Evidence A for speed in general, C for our gain. Cost: CPU check, no training. Risk: fp16 can shift logits slightly; run the ensemble on val and compare AUROC before submit.
6. **TensorRT or ONNX export of the backbone.** Common path (NVIDIA docs: PyTorch to ONNX to TensorRT engine, reduced precision and memory savings). Evidence A that it speeds up, C for our models (ConvNeXt, DINOv3, Qwen VLM are not all exportable). Cost: engineering time, an engine is GPU-specific (must build on the T4 inside the Kaggle notebook, not on Modal A10G/L4). Risk: high (build time counts toward the 9 h, silent accuracy drift). Recommend only for the 1 or 2 slowest members, after item 5.
7. **Cheaper ensemble of diverse weaker models.** BirdCLEF 2023 write-up (analogous CPU budget): an ensemble of weaker models beat one strong model under a time cap (secondhand). Evidence B. Action: use the efficiency leaderboard to test whether dropping the 2 weakest members costs less than 0.001 AUROC. Could win the efficiency prize without losing the leaderboard prize.
8. **Hosted frontier VLM verdicts on the most disputed label cells only.** Our docs: image-aware LLM labels 80-90% correct; use as one vote, not hard labels. Evidence B. Cost: small (API calls on maybe a few hundred studies). Risk: rules (section 2.6 "Reasonableness", small subscription acceptable per rule example) and NOT allowed at inference (no internet in the notebook). Only for training labels. Ask the user before any paid API call.
9. **Transductive self-training on the hidden test set inside the notebook.** Our docs (big_jump item 2). Evidence C (only weak-medium papers). Cost: one extra fine-tune inside the 9 h budget, risky. Risk: rules look allowed (no human labels), but any bug silently hurts the leaderboard. Simulate on val first.
10. **Two-stage localisation (knee ROI or meniscus crop, then 2.5D classifier).** Cervical 2022 1st place and breast 2023 (secondhand) both used a detector then classifier. Evidence B for those tasks. For knees our crop test (outline line 308 and 311) found cropping saves only 16% of compute and clips anatomy on 0.28% of studies, so the expected gain is small. Cost: a new stage to train and fit in 9 h. Risk: medium. Deprioritise.

Not recommended (from the research, for the record):
- fastMRI knee DICOM (10k clinical knee MRIs, same sequences) for Noisy Student: access agreement limits use to "internal research or education only", same disqualification risk as KneeCoT (banned per host, docs). High risk, do not pursue without host written answer.
- RadImageNet weights: non-commercial licence, already blocked (see `public_notebooks_0946_analysis.md`).
- Public-notebook "correction" MCL swap: noise, not worth copying (docs).

## 4. Inference-budget notes (9 h, 2x T4)

- Our current bottleneck from docs: decode 4.048 s per study, GPU about 16 s per study for 9 members; 1,300 test studies. Rough arithmetic: 1,300 x 20 s = 26,000 s = 7.2 h, close to the cap. Verify against the real timing log before any new member is added. Any new member must be charged against this budget.
- Two cards: put each model on one card (members split), not the batch (docs: batch split runs out of memory on 15 GB).
- Keep 20 to 30% headroom for Kaggle variance on the grader machine (outline gives a measured T4 figure; use the slowest observed run, not the mean).

## 5. Sources

- Kaggle competition page (title only retrieved): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion
- RSNA 2025 aneurysm winners: https://www.rsna.org/news/2025/november/2025-ai-challenge-winners
- RSNA 2024 lumbar winners: https://www.rsna.org/news/2024/november/2024-ai-challenge-winners
- RSNA 2024 lumbar winners (AuntMinnie): https://www.auntminnie.com/imaging-informatics/artificial-intelligence/article/15708869/rsna-names-winners-of-lumbar-spine-ai-challenge
- RSNA 2023 abdominal trauma (Albumentations): https://albumentations.ai/adoption/competitions/kaggle/rsna-2023-abdominal-trauma-detection/
- RSNA 2023 abdominal trauma review (ITN Online): https://itnonline.com/node/89790
- RSNA 2023 abdominal trauma repo (siwooyong): https://github.com/siwooyong/RSNA-2023-Abdominal-Trauma-Detection
- RSNA 2022 cervical (Zenn walkthrough, Japanese): https://zenn.dev/yuto_mo/articles/f9fb1c8297f252
- RSNA 2023 breast repo (dangnh0611): https://github.com/dangnh0611/kaggle_rsna_breast_cancer
- RSNA 2023 breast (Albumentations): https://albumentations.ai/adoption/competitions/kaggle/rsna-breast-cancer-detection/
- RSNA 2025 aneurysm (AuntMinnie): https://www.auntminnie.com/imaging-informatics/artificial-intelligence/article/15751740/rsna-launches-2025-intracranial-aneurysm-detection-ai-challenge
- Knee GitHub, scanner-grouping and 0.664/0.809 notes (homeshwarnelakurthi): https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection
- Knee GitHub, EfficientNet and DINOv2 notes (JunhaoLiXD): https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection
- RSNA 2026 knee launch (press release, listing only): https://www.rsna.org/media/press/2026/2669
- Local: `hackathon_page.txt` (rules, prizes), `other_md/outline.md` (efficiency and crop data), `other_md/research_results/big_jump_ideas_oct7.md`.

## 6. Open questions for the user

- Do you want a browser-rendered pass over the Kaggle forum (Claude in Chrome, logged in) to read thread 735304 and any post after 7 Oct? It is the single biggest gap in this report.
- Should the efficiency-prize track (section 1 and item 7) be scored as its own candidate set?
