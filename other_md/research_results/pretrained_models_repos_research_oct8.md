# Pretrained models, foundation models and repos: research pass (8 Oct 2026)

Scope: models not yet tried in `runs_summary.md`, plus 2025-2026 releases. Checked against the existing notes in `pretrained_backbones_research_v2.md`, `_v3.md`, `3d_pretrained_backbones_research.md`, `3_agents_model_to_use_research.md` and `best_pretrained_oss_knee_model_*.md`.

Abbreviations: Area Under the Receiver Operating Characteristic curve (AUROC), Vision Transformer (ViT), Self-Supervised Learning (SSL), Non-Commercial (NC), Research Use Agreement (RUA), Health AI Developer Foundations (HAI-DEF), Hugging Face (HF), Magnetic Resonance Imaging (MRI), Vision-Language Model (VLM), Masked Autoencoder (MAE), Knee Osteoarthritis (OA).

Time left: about 14 days to the 22 Oct deadline. Anything that costs more than roughly 2 h of T4 time for one member is unlikely to be worth the risk.

## 1. Headline

- No public knee MRI foundation model exists with usable weights. The ISMRM 2025 OAI-based 3D DINO model and the ISMRM 2026 3D knee model have no released weights we could find.
- The one knee MRI classifier with weights on HF (ACL-LKNet) was trained on MRNet data, which is governed by a non-commercial Research Use Agreement. It is not prize-eligible.
- Two cheap, prize-safe 2D candidates are new to our pool: C-RADIOv3-B (NVIDIA Open Model License) and PE-Core-B16 (Meta Perception Encoder, Apache-2.0 on the card). Both need one checkpoint-level licence check before use.
- The existing notes say Perception Encoder is FAIR non-commercial. That is true for Perception-LM, but the PE-Core-B16-224 card says Apache-2.0. Verify per checkpoint before relying on either claim.

## 2. Ranked top 10 (expected gain per cost, prize-safe first)

| Rank | Candidate | Params | Weights licence | Prize-safe | Tried? | T4 cost vs ConvNeXt-B | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C-RADIOv3-B (NVIDIA Agglomerative RADIO v3, Base) | 90M (98M safetensors) | NVIDIA Open Model License ("ready for commercial/non-commercial use") | Yes, after reading the licence file in the repo | No | About 0.9x (ViT-B/16 at 512, similar to SigLIP-B) | Run first. Cheap, different teachers, any-resolution |
| 2 | PE-Core-B16-224 (Meta Perception Encoder, core vision tower) | ~0.09B vision tower | Apache-2.0 per HF card (conflicts with our notes) | Likely, pending checkpoint check | No | About 0.9x at 512 (needs position interpolation from 224) | Run only if the checkpoint licence is confirmed Apache |
| 3 | MedSigLIP-448 vision tower (Google, medical SigLIP) | 400M vision, 400M text | HAI-DEF terms; no NC clause found in the model card, but terms are only linked | Probably yes, needs legal read | No (listed as "Stage 2" in 3_agents research, never run) | About 6-8x (OVER budget for full fine-tune) | Only as frozen feature extractor or a small-scale probe. Its pretraining includes MRI slices |
| 4 | C-RADIOv4-SO400M / -H | 412M / 631M | NVIDIA Open Model License on HF cards; a third-party page says NC source licence (unverified) | Check | No | OVER budget | Skip for the run. Its teachers (SigLIP2, DINOv3, SAM3) are the best knowledge source, but too big |
| 5 | Knee cartilage and meniscus U-Net masks (DOSMA, HF `arjundd/dosma-models`, qDESS and OAI models, 2D, uploaded 2022) | 2D U-Net, small | Not stated on the repo we checked | Unknown | No | Cheap (inference only, about 1,300 test studies) | Possible ROI prior (crop around joint). Sequence differs from RSNA, so gain is uncertain |
| 6 | LM-CartSeg (arXiv Dec 2025, nnU-Net on SKM-TEA 138 knees and OAIZIB-CM 404 knees, cartilage and subchondral bone) | 3D nnU-Net | Unknown, weights not confirmed public | Unknown | No | 3D, heavy | Only if weights appear. Low priority |
| 7 | V-JEPA 2 / video models on slice stacks | 300M+ | MIT for V-JEPA 2 | Yes on paper | Covered in 3D research doc | OVER | Already assessed. Not recommended |
| 8 | Web-SSL (Meta, DINO and MAE ViT-1B, 2B MetaCLIP data) | 1B-2B | Not found in search results | Likely FAIR NC (unverified) | No | OVER | Reject unless licence is clean |
| 9 | SAM 3 image encoder (Meta) | ~848M | Custom "SAM License", commercial terms unclear, gated | Unclear | No | OVER | Reject |
| 10 | ACL-LKNet (HF `shareefch1413/ACL-LKNet`, ConvNeXt-Tiny 2.5D, masked slice pretraining on MRNet) | ~28M | Code MIT; weights derived from MRNet (RUA prohibits commercial use) | NO | No | Cheap | Knee-MRI-specific SSL prior, but weights are NC-tainted. Use only for private analysis, never in a shipped submission |

Excluded for licence or architecture (not in top 10): `leminhhung0101/knee-model` (ConvNeXtV2-Tiny FCMAE init, which carries CC-BY-NC weights from the ConvNeXt-V2 family, and no licence on the card). Also GE HealthCare 3D MRI foundation model (no weights).

## 3. Candidate notes

### 3.1 C-RADIOv3-B (rank 1)
- Link: https://huggingface.co/nvidia/C-RADIOv3-B
- Pretraining: 700M web images (NV-CC-Img-Text-Dataset) with teacher distillation from RADIO teachers. Patch size 16. Supports 256-2048 px in steps of 16, so 512 needs no interpolation.
- Licence: NVIDIA Open Model License on the HF card. A third-party plugin lists NVIDIA Source Code License-NC. The GitHub README calls C-RADIOv3 "commercially viable". Read the licence file in the repo.
- Load: `AutoModel.from_pretrained("nvidia/C-RADIOv3-B", trust_remote_code=True)` (Transformers, not timm). Offline use needs the remote code copied into the Kaggle dataset.
- Why it fits: a different self-supervised and distillation recipe from our pool (ConvNeXt, SwinV2, DINOv3-ConvNeXt, SigLIP2, EVA-02, CoAtNet, MedImageInsight, MRI-CORE). Expected to be decorrelated.
- Evidence on MRI: none found. Gain is a diversity bet.

### 3.2 PE-Core-B16-224 (rank 2)
- Link: https://huggingface.co/facebook/PE-Core-B16-224
- Card: licence apache-2.0 (quoted from the HF page), vision tower 0.09B at 224 px, text tower 0.31B, synthetic video data engine.
- Conflict: our existing research (3_agents_model_to_use_research.md) says Meta Perception Encoder ships non-commercial. That may be about PE-Lang or Perception-LM, which do carry the FAIR NC licence. Check each checkpoint (PE-Core, PE-Spatial, PE-Lang) individually.
- Caveat: trained at 224 px. Using it at 512 needs interpolated position embeddings, which hurt the 224-trained convnext_base in our earlier tests.

### 3.3 MedSigLIP-448 (rank 3)
- Link: https://huggingface.co/google/medsiglip-448
- Licence: HAI-DEF terms of use (gated, accept on HF). The card says the model "must be independently validated" before medical use. The terms text was not retrieved; read it before shipping.
- Pretraining: de-identified medical image-text pairs including CT and MRI slices, plus natural images. Sources include MIMIC-CXR, TCGA and others.
- Size: 448 px input, 400M vision encoder. A full fine-tune on T4 at our slice counts is over budget (about 6-8x ConvNeXt-B by the timing method in pretrained_backbones_research_v3.md). Possible uses: one frozen feature pass (cheap, but frozen 7B features already lost to fine-tuned 86M backbones in our history) or a small-resolution fine-tune.
- Evidence on MRI: included in pretraining, no knee result known.

### 3.4 C-RADIOv4 (rank 4)
- Links: https://arxiv.org/abs/2601.17237 (tech report, 24 Jan 2026), https://github.com/NVlabs/RADIO
- Variants: SO400M (412M) and H (631M). Teachers: SigLIP2-g-384, DINOv3-7B, SAM3.
- Verdict: skip for runs (too heavy). Useful only if C-RADIOv3-B shows promise and we need a stronger distillation target.

### 3.5 DOSMA knee segmentation models (rank 5)
- Link: https://huggingface.co/arjundd/dosma-models (2D U-Nets for cartilage and meniscus; qDESS and OAI variants, weights from 2022).
- Idea: a segmenter gives a joint crop so the classifier sees the knee at higher effective resolution. Also gives per-slice masks as auxiliary targets.
- Risks: licence not stated on the repo we checked; qDESS (quantitative double-echo steady-state) is a sequence we may not have in RSNA; cropping adds a dependency and test-time cost.

### 3.6 ACL-LKNet (rank 10, not prize-safe)
- Link: https://huggingface.co/shareefch1413/ACL-LKNet
- Architecture: ConvNeXt-Tiny 2D backbone, parametric slice attention, 2-head self-attention over the three planes. Input 24 slices at 224 px per plane.
- Pretraining: masked slice modelling (self-supervised) on MRNet. Card reports AUROC 0.9639 on the 120-exam MRNet test set (self-reported, 95% CI 0.928-0.992).
- Licence problem: code MIT. MRNet is accessed under a Research Use Agreement that prohibits commercial use, redistribution and re-identification (secondary source: arXiv 2508.14151). Weights derived from that data should be treated as NC.
- Use: could be a private benchmark for our own 2.5D design, not a shipped member.

## 4. Things this pass could not settle

- Licence text for MedSigLIP HAI-DEF terms, DOSMA weights, LM-CartSeg weights, Web-SSL weights, and the PE checkpoints other than PE-Core-B16.
- Whether C-RADIOv3 loads offline inside a Kaggle notebook (remote code dependency).
- T4 timings for C-RADIOv3-B and PE-Core-B16 at 512 px: estimated from the ViT-B ratio in pretrained_backbones_research_v3.md, not measured.

## 5. Suggested next actions (for the user to decide)

1. Verify the C-RADIOv3-B licence file in the repo (read-only check) and the PE-Core-B16 and PE-Spatial checkpoint licences.
2. If both are clean, C-RADIOv3-B is the first run (about 1 h on T4), PE-Core-B16 second.
3. MedSigLIP only as a frozen-feature probe, and only after the HAI-DEF terms are read.
4. Do not spend a run on ACL-LKNet or any MRNet-derived weights.

## 6. Sources

- https://huggingface.co/nvidia/C-RADIOv3-B
- https://huggingface.co/nvidia/C-RADIOv3-L (licence tag: nvidia-open-model-license)
- https://huggingface.co/nvidia/C-RADIOv4-SO400M (licence: other, NVIDIA Open Model License per card)
- https://arxiv.org/abs/2601.17237
- https://github.com/NVlabs/RADIO
- https://huggingface.co/facebook/PE-Core-B16-224
- https://huggingface.co/facebook/PE-Spatial-G14-448 (Apache-2.0 per commit page)
- https://huggingface.co/facebook/Perception-LM-3B (FAIR NC licence)
- https://huggingface.co/google/medsiglip-448
- https://huggingface.co/shareefch1413/ACL-LKNet
- https://huggingface.co/leminhhung0101/knee-model
- https://huggingface.co/arjundd/dosma-models
- https://huggingface.co/facebook/sam3 (SAM License)
- https://aimi.stanford.edu/datasets/mrnet-knee-mris (MRNet access terms)
- arXiv 2508.14151 (MRNet RUA, secondary)
- arXiv 2504.01017 (Web-SSL, Fan et al. 2025)
