# Swin Transformer backbones for knee Magnetic Resonance Imaging (MRI) - research findings

Date: 2026-09-01

## 1. Headline

No Swin, SwinV2, Video Swin, or Swin-UNETR checkpoint pretrained on knee MRI, musculoskeletal (MSK) imaging, or the Osteoarthritis Initiative (OAI) / MRNet datasets could be found publicly available for download. The only medical Swin checkpoints that are genuinely public and verifiable are Swin-UNETR encoders pretrained on **CT** (5,050 scans, no MRI) and one head-brain-MRI variant (SwinBrain) - neither is musculoskeletal. For 2D, the practical and verifiable answer is: use a plain ImageNet-22k SwinV2 from `timm` (not medical-pretrained) - it is the only category with confirmed, license-clear, easy-to-load weights that also solves the 512px problem via SwinV2's log-spaced continuous position bias (Log-CPB). For the cross-slice idea, Video Swin Transformer gives native temporal shifted-window attention, but its only public checkpoints are pretrained on Kinetics natural video, not medical volumes, and adapting 96 slices x 512px to it is very likely infeasible inside a 9-hour Kaggle inference budget. Bottom line: **a medical-pretrained Swin does not exist for our use case** - the honest choice is an ImageNet-pretrained SwinV2 backbone (2D, per-slice) or, if cross-slice fusion is pursued, a 3D Swin-UNETR encoder pretrained on CT as a starting point despite the CT-to-MRI domain gap being unverified for our task.

## 2. Verified usable candidates, ranked

### 1. `timm` SwinV2, ImageNet-22k, high resolution (RECOMMENDED, non-medical but verified and load-ready)
- Model id: `swinv2_large_window12to24_192to384.ms_in22k_ft_in1k` (confirmed on Hugging Face model card).
- Confirmed: 196.7M parameters, MIT license, pretrained ImageNet-22k then fine-tuned ImageNet-1k at 384x384, window scales 12→24 as resolution scales 192→384.
- Loading: `timm.create_model("hf_hub:timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k", pretrained=True)`.
- **Experiment to run**: load at `img_size=512` (timm supports resizing SwinV2 positional/window parameters because SwinV2 uses Log-CPB, a small meta-network that generates the relative position bias for any window size instead of a fixed lookup table - so it extrapolates smoothly to windows it never saw, unlike Swin v1's plain bicubic interpolation). Concretely: `timm.create_model(model_id, pretrained=True, img_size=512)`, verify no shape errors, and fine-tune per-slice. This is a straightforward drop-in swap in `shared_code.py`'s backbone factory. There is a smaller `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` (87.9M params) if 8GB VRAM (local RTX 4060) forces a smaller model for local debugging before moving to Modal H200.

### 2. Swin-UNETR self-supervised encoder (MONAI / NVIDIA) - only for the 3D/cross-slice experiment, CT-pretrained, MRI domain gap unverified
- Confirmed checkpoint: `https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt` (and a second file `ssl_pretrained_weights.pth` at the same GitHub releases page, referenced in a MONAI GitHub discussion but with undocumented provenance - treat as unverified duplicate, not a second independent source).
- Confirmed pretraining data: **5,050 CT scans** (not MRI) from five public sources: HNSCC, LUNA16, TCIA CT Colonography, TCIA COVID-19, TCIA LIDC. Trained with reconstruction, 3D rotation prediction, and contrastive pretext tasks (paper: "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis", CVPR 2022, ranked 1st on Medical Segmentation Decathlon and BTCV leaderboards at time of publication - segmentation, not classification).
- License: Apache 2.0 (confirmed from repo license terms) - commercial/competition use is fine.
- Architecture (from architecture documentation, not this specific checkpoint's config file, so treat as typical-configuration, not this-exact-file-confirmed): embed dim 48, 4 encoder stages of 2 transformer blocks each, patch size 2x2x2, window size 7x7x7, encoder alone roughly 60-90M parameters depending on `feature_size`.
- Loading: MONAI `SwinUNETR` class exposes the encoder (`SwinViT`) separately; state dict keys in the release are `encoder.*` (per the GitHub discussion), so you would instantiate `monai.networks.nets.swin_unetr.SwinTransformer` (the ViT part only, not the U-Net decoder) and `load_state_dict` with `strict=False`, dropping the decoder/`out.conv.*` keys not needed for classification.
- **Honest caveat**: this is a segmentation encoder pretrained on CT. I could not find any published evidence of this specific checkpoint transferring to MRI or to any classification task, let alone knee/MSK. Whether the CT-learned low-level filters help versus hurt on MRI is untested in anything I could verify.
- **Experiment to run** (only if pursuing the 3D/cross-slice angle seriously): reshape a small neighborhood of slices (e.g. 8-16 adjacent slices per plane, since native window is 7x7x7 and Swin-UNETR was trained on volumes, not 96-slice studies) into a 3D volume, load this encoder, and compare AUROC against the current per-slice mean-pool baseline on a held-out fold before any larger investment - given the CT-to-MRI gap is unverified, this should be treated as a quick falsifiable test, not a default plan.

### 3. SwinBrain (self-supervised Swin-UNETR on 75,861 head MRI scans) - not musculoskeletal, but worth knowing it exists
- Repo: `https://github.com/MAI-Lab-West-China-Hospital/SwinBrain` (confirmed to exist and describe itself as claimed).
- Confirmed: pretrained on 75,861 plain head MRI scans (T1-weighted, T2-weighted, and Fluid-Attenuated Inversion Recovery/FLAIR - three different MRI contrast settings that make different tissues look bright or dark), architecture is Swin-UNETR (3D).
- Weights distributed via Google Drive link in the repo (not fetched/verified further - could not confirm the file is still live or its exact size).
- License: **not specified in the repo** - this is a hazard; do not use without the authors clarifying terms, since an unspecified license cannot be assumed to permit a Kaggle competition submission.
- Relevance to us: it is at least MRI-pretrained (unlike the CT Swin-UNETR above), but it is brain tissue, not knee/MSK, and it is unclear whether brain MRI low-level features transfer to knee MRI at all. No knee evaluation exists for it. Rank it below option 2 mainly because of the missing license, despite the better modality match.

## 3. Full table of everything found

| Candidate | 2D/3D | Params | Native res / window | Pretraining data | License | Loading | Knee/MSK evidence |
|---|---|---|---|---|---|---|---|
| `timm` SwinV2 (`swinv2_large_window12to24_192to384.ms_in22k_ft_in1k`) | 2D | 196.7M | 384px, window 12→24 (Log-CPB extrapolates) | ImageNet-22k → ImageNet-1k (natural images, not medical) | MIT | `timm.create_model`, one line | None (not medical) |
| `timm` SwinV2 base variant (`swinv2_base_window12to24_192to384...`) | 2D | 87.9M | 384px, window 12→24 | Same as above | MIT (timm default) | `timm.create_model` | None |
| Swin v1 (`swin_base_patch4_window12_384.ms_in22k`) | 2D | ~88M | 384px, window 12 | ImageNet-22k | MIT (timm default) | `timm.create_model` | None; window-size interpolation to 512 is bicubic, cruder than SwinV2's Log-CPB |
| Swin-UNETR SSL encoder (`model_swinvit.pt`) | 3D | ~60-90M (encoder only, config-dependent) | trained on volumetric patches, window 7x7x7, patch 2x2x2 | 5,050 CT scans (HNSCC, LUNA16, TCIA CT Colonography, TCIA COVID-19, TCIA LIDC) - CT only, no MRI | Apache 2.0 | MONAI `SwinTransformer`/`SwinUNETR`, `load_state_dict(strict=False)` | None found; segmentation-only evaluation in the source paper |
| `ssl_pretrained_weights.pth` (same MONAI releases page) | 3D | unknown | unknown | **undocumented** - referenced only in a GitHub discussion, provenance unconfirmed | unclear | unclear | None |
| SwinBrain | 3D | unknown (not confirmed) | Swin-UNETR defaults (unconfirmed for this checkpoint) | 75,861 head MRI scans (T1, T2, FLAIR) | **not specified - hazard** | Google Drive download, custom repo | None (brain, not knee) |
| RadImageNet Swin | 2D | n/a | n/a | 1.35M CT/MRI/ultrasound images, 11 anatomies incl. musculoskeletal | MIT (repo license) | n/a | **Not released yet** - repo explicitly states "we will release pretrained Swin Transformer weights later" |
| Video Swin Transformer (Swin-T/S/B) | "3D" (spatio-temporal, natural video) | 28M/50M/88M | 224px spatial, temporal window typically 8 frames, spatial window 7 | ImageNet-1k/22k image init, then Kinetics-400/600 or Something-Something-v2 video | Apache-2.0 (per repo terms fetched) | mmaction2/bespoke repo, config-driven, not a one-liner | None (natural video, no medical transfer evidence found) |
| Swin Transformer for Bankart lesion detection (arXiv 2412.06717) | 2D | unspecified variant | unspecified | Pretrained on a public knee MRI dataset (not named/confirmed which - could not verify if MRNet or other), then fine-tuned | unknown, no code/weight release found | n/a - no public checkpoint found | AUROC 0.87 (standard MRI), 0.90 (MR arthrogram) for Bankart lesion (a shoulder, not knee, cartilage/labrum tear) - note this is a **shoulder** pathology despite "knee MRI dataset" pretraining language; no weights released |
| SwinFUSE (arXiv 2405.12781) | 3D | unspecified | unspecified | CT + MRI, multi-modal, "modality-agnostic" pretraining pretext | CC BY 4.0 | Code repo at `github.com/devalab/SwinFUSE` (existence not independently verified beyond the paper's self-reported link) | No knee-specific evaluation found |
| 3DINO-ViT (arXiv 2501.11755) | 3D | unspecified | unspecified | ~100,000 3D medical scans, 10+ organs, multimodal | Not specified in abstract | Authors state weights "will be made available" - **no confirmed download exists today** | Not a Swin architecture (plain ViT-based) - listed only because it surfaced in the medical-Swin search; does not answer the question |
| PRISM foundation model (arXiv 2508.07165) | unconfirmed (could not fetch full text; PDF too large for this tool) | unspecified | unspecified | 336,476 volumetric MRI scans, 34 datasets, whole-body | unspecified | unspecified | Could not confirm backbone is Swin-based; could not confirm knee coverage or checkpoint availability. **Flag for manual follow-up** - abstract alone was not conclusive |
| MONAI HF `swin_unetr_btcv_segmentation` / `darragh/swinunetr-btcv-tiny` | 3D | unspecified | trained on CT | BTCV challenge CT (multi-organ abdominal CT) | Apache 2.0 (MONAI default) | HF `AutoModel`-style download, MONAI bundle | Fine-tuned end-to-end for CT organ segmentation, not a general encoder release; less suited to reuse than the raw SSL encoder above |

## 4. The cross-slice / 3D angle

Your architectural instinct is sound (shifted windows sliding along a third axis would give native adjacent-slice attention instead of the bolted-on pooling that already lost 0.0148 Area Under the Receiver Operating Characteristic curve (AUROC) points), but the available pretrained starting points do not match our setup well:

- **Video Swin Transformer** is the only architecture with genuinely native temporal shifted-window attention and public pretrained weights, but those weights come from Kinetics-400/600 natural video (people, objects, motion) - there is no evidence in anything I found that this transfers usefully to MRI slice stacks, and no medical-pretrained Video Swin exists that I could verify. Its native design also assumes short temporal windows (~8-32 frames) at ~224px; our pipeline is 96 slices at 512px per one of 4 slots - both frame count and resolution are well beyond anything Video Swin was designed or evaluated at. Running this at full scale is unlikely to fit a sane training budget on Modal H200s, let alone the 9-hour **inference** budget on the far weaker Kaggle GPU, since attention cost still scales with total tokens even with windowing.
- **Swin-UNETR (3D)** is CT-pretrained, segmentation-oriented, and untested for classification transfer or MRI transfer in anything I found. Its native window (7x7x7) is built for volumetric patches carved from a full 3D scan, not a stack of pre-selected sparse 2D slices - reshaping our 24-slice-per-slot data to fit its assumptions is a nontrivial adaptation, not a drop-in.
- Realistically, if you want to test the cross-slice hypothesis cheaply, a **lighter-weight custom step** (e.g., a 3D-windowed attention block or a small temporal conv/attention stage that consumes a 2D SwinV2's per-slice feature maps, not a full pretrained 3D Swin) is more tractable within the 9-hour budget than adopting any of the found 3D Swin checkpoints. That is an architecture-design question outside the scope of "does a pretrained checkpoint exist" - and the honest research answer to the checkpoint question here is **no adequate pretrained cross-slice medical Swin exists today**.

## 5. 512px feasibility

- **SwinV2** is the best-supported path to 512px: its Log-CPB relative-position mechanism was specifically designed (per the SwinV2 paper, arXiv 2111.09883) so a model trained at one window size/resolution can be extended to a larger one by re-evaluating the small meta-network, rather than needing a fixed table sized for one resolution. The largest official `timm` SwinV2 checkpoint tops out at 384px/window24 (confirmed on Hugging Face), so 512px means extrapolating one step further than the officially released checkpoint went (window24→~32) - this is within the mechanism's design intent but is not a configuration NVIDIA/Microsoft themselves shipped and benchmarked, so expect to validate it empirically rather than assume it works out of the box.
- **Swin v1** can also be run at 512px via bicubic interpolation of the relative position bias table (a common trick used in object detection/segmentation Swin configs), but this is the cruder of the two methods per the SwinV2 paper's own comparison, and accuracy degradation grows with the resolution gap.
- **Practical recommendation**: start from `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` or the large variant, instantiate with `img_size=512` in `timm.create_model`, and treat any window/resolution mismatch errors or the resulting accuracy as the actual answer for our case, since I could not find anyone else's published number for exactly this configuration.

## 6. Ruled out, and why

- **RadImageNet Swin** - explicitly not released yet ("we will release... later"); nothing to download today.
- **3DINO-ViT** - not a Swin architecture at all (plain ViT), and its own paper says weights are not yet available.
- **SwinFUSE, PRISM** - architecture and/or checkpoint availability could not be confirmed with confidence; do not treat as usable until someone reads the full paper/repo directly (PRISM's PDF exceeded this session's fetch size limit).
- **SwinBrain** - MRI-pretrained but brain-only, no knee evidence, and no stated license - a genuine license hazard, not just an anatomy mismatch.
- **RSNA prior Kaggle winners (2022 cervical spine, 2023 abdominal trauma, 2024 lumbar spine)** - none of the winning solutions found used a Swin backbone. 2022 cervical spine 1st place used a 2.5D CNN-RNN (EfficientNet-V2-S/ConvNeXt + LSTM); 2023 abdominal trauma 1st place used CoaT Lite/EfficientNet-V2-S with a Gated Recurrent Unit (GRU) over slice triplets. This is evidence (not proof) that Swin has not been the backbone of choice for winning solutions on adjacent RSNA slice-stack problems - CNN-plus-recurrent-fusion has a track record here that Swin does not.
- **Video Swin at full scale (96 slices x 512px)** - ruled out as impractical for the 9-hour inference budget; see section 4.

## 7. Verification log

Fetched and read directly (confirmed content):
- `https://github.com/MAI-Lab-West-China-Hospital/SwinBrain` - repo description, pretraining data, no license found.
- `https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain` and its `README.md` - checkpoint URL, CT dataset list, training hyperparameters.
- `https://github.com/SwinTransformer/Video-Swin-Transformer` - checkpoint table, license terms, loading commands.
- `https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/LICENSE` - confirmed Apache 2.0 text.
- `https://github.com/BMEII-AI/RadImageNet` - confirmed MIT license, confirmed Swin weights not yet released, confirmed dataset scale/anatomy coverage.
- `https://github.com/Project-MONAI/MONAI/discussions/7208` - confirmed second checkpoint filename `ssl_pretrained_weights.pth`, confirmed provenance was left unanswered in the thread.
- `https://huggingface.co/timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k` - confirmed param count, license, resolution, loading snippet.
- `https://arxiv.org/abs/2405.12781` (SwinFUSE) - confirmed abstract content, GitHub link as self-reported by authors (not independently browsed).
- `https://arxiv.org/abs/2501.11755` (3DINO-ViT) - confirmed abstract content; confirmed it is not Swin-based.
- `https://arxiv.org/abs/2412.06717` (Bankart lesion) - confirmed AUROC numbers, confirmed no checkpoint release found in the fetched content.
- `https://arxiv.org/pdf/2512.12887` (Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification) - confirmed this paper is about DINOv2/DINOv3, not Swin; noted only as adjacent context, not a Swin answer.
- Search results (not independently fetched beyond snippets) for RSNA 2022/2023/2024 winner architectures - taken from search-engine summaries of GitHub solution repos, not directly opened; treat backbone claims there as second-hand.

Could not access / could not confirm:
- `https://arxiv.org/abs/2508.07165` (PRISM) full text - PDF exceeded the 10MB fetch limit; abstract alone did not confirm architecture, license, or knee coverage. **Needs manual follow-up** if this looks promising.
- `https://openaccess.thecvf.com/content/ACCV2024/papers/Kim_Domain_Aware_Multi-Task_Pre-Training_of_3D_Swin_Transformer_for_Brain_ACCV2024_paper.pdf` - server returned HTTP 403 Forbidden, could not read.
- SwinBrain's actual Google Drive weight file - link presence confirmed in the repo text, but the file itself was not downloaded or checked for validity/size.
- `ssl_pretrained_weights.pth` provenance (modality, scan count) - never resolved in the MONAI discussion thread itself; do not treat as a second confirmed 50,000-volume CT/MRI checkpoint, since that number appeared only as an unresolved side-comment, not a released artifact.
- Whether MRNet or OAI have ever been used to pretrain a Swin checkpoint that was subsequently released publicly - no such release surfaced in any search performed.
