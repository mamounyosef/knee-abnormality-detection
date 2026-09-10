# Pretrained Backbones Research v3 - RSNA Knee Abnormality Detection

Follow-up to `pretrained_backbones_research_v2.md`. Where v1 and v2 hunted for MEDICAL
pretraining, v3 asks a different question: which downloadable 2D backbone adds the most
ENSEMBLE DECORRELATION, given that our best single model is 0.9050 and our ensemble is
0.9156. Members earn a slot by being different, not by scoring well alone.

Abbreviations: Area Under the Receiver Operating Characteristic curve (AUROC), Vision
Transformer (ViT) - an image model that splits a picture into square patches and processes
them the way a language model processes words, Masked Image Modeling (MIM) - pretraining by
hiding parts of an image and predicting them, Rotary Position Embedding (RoPE) - a way of
telling a transformer where a patch sits that works at any image size, Self-Supervised
Learning (SSL) - training without human labels.

## 1. Headline

`vit_base_patch16_siglip_512.v2_webli` is the clear pick, and it is the rare case of the
most-decorrelated candidate also being the CHEAPEST. Timed locally at 512px it runs at
0.90x the cost of the ConvNeXt-Base we already ship, and it is pretrained NATIVELY at
512x512, so it needs no position-embedding interpolation - unlike every backbone currently
in our pool, including `convnext_base`, which is pretrained at 224.

The second finding is a constraint the research itself did not apply: RUNTIME. Four of the
report's eight recommendations are ViT-Large class and do not fit under the 9 h Kaggle cap
alongside our existing members. Decorrelation ranking alone is not a usable ranking for us
any more.

Third: several backbones previously floated as candidates, including by this project's own
notes, turn out to be NON-COMMERCIAL and therefore ineligible for a prize submission -
ConvNeXt-V2 (all FCMAE variants), Meta Hiera, and Apple AIMv2. In each case the CODE is
permissive and the WEIGHTS are not, which is exactly the trap RadImageNet set earlier.

## 2. Verified against the installed timm (1.0.27)

Model names and native resolutions are the two things a research pass can most easily get
wrong, and both are checkable offline. Result: 7 of 8 exist exactly as named.

| model string | exists | native res |
| --- | --- | --- |
| `vit_base_patch16_siglip_512.v2_webli` | yes | 512x512 |
| `maxvit_base_tf_512.in21k_ft_in1k` | yes | 512x512 |
| `beit_large_patch16_512.in22k_ft_in22k_in1k` | yes | 512x512 |
| `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` | yes | 448x448 |
| `vit_large_patch14_reg4_dinov2.lvd142m` | yes | 518x518 |
| `vit_so400m_patch14_siglip_384.webli` | yes | 384x384 |
| `deit3_large_patch16_384.fb_in22k_ft_in1k` | yes | 384x384 |
| `internimage_large_384` | **NO** | - |

InternImage is not in timm at all: it needs OpenGVLab's DCNv3 CUDA extension compiled from
source, which is a real integration cost and a Kaggle-offline hazard.

`deit3_large_patch16_384` additionally REFUSES 512 input outright (`Input height (512)
doesn't match model (384)`), so the report's claim that it tolerates interpolation to 512
is wrong as far as timm's implementation is concerned.

## 3. Priced against the 9 h cap

Timed at 512x512, fp16, random weights, on the local RTX 4060; ratios taken against
`convnext_base` and scaled by its MEASURED 3.71 s/study on Kaggle's T4. Relative cost
transfers across GPUs much better than absolute cost. Headroom after the 6-member
submission is about 3.37 h.

| model | params | x ConvNeXt-B | s/study | hours | verdict |
| --- | --- | --- | --- | --- | --- |
| `vit_base_patch16_siglip_512` | 94M | **0.90** | 3.34 | 0.93 | **FITS, cheaper than ours** |
| `beit_base_patch16_384` | 86M | 1.27 | 4.71 | 1.31 | FITS |
| `maxvit_small_tf_512` | 68M | 1.56 | 5.79 | 1.61 | FITS |
| `maxvit_base_tf_512` | 119M | 2.65 | 9.82 | 2.73 | fits, but eats the budget |
| `vit_large_patch14_reg4_dinov2` | 304M | 3.49 | 12.94 | 3.60 | OVER |
| `eva02_large_patch14_448` | 304M | 4.43 | 16.44 | 4.57 | OVER |
| `beit_large_patch16_512` | 305M | 4.80 | 17.80 | 4.94 | OVER |
| `vit_so400m_patch14_siglip_384` | 429M | 6.02 | 22.34 | 6.21 | OVER |

## 4. Recommendation

Run `vit_base_patch16_siglip_512.v2_webli` first. It differs from our pool on three axes
at once - sigmoid language-image objective (nothing we have is language-supervised),
web-scale WebLI data, and native 512 geometry - while costing less than a member we
already ship. Apache-2.0, ungated.

If it pays, `beit_base_patch16_384` is the natural second: discrete-codebook masked
prediction is an objective unlike anything in the pool, at 1.31 h. `maxvit_small_tf_512`
is the third, adding a hybrid convolution/attention topology at 1.61 h.

Do not pursue the Large-class models unless a cheaper member is dropped first.

## 5. Ineligible: non-commercial WEIGHTS behind permissive CODE

| model | code | weights | source |
| --- | --- | --- | --- |
| ConvNeXt-V2, all FCMAE variants | MIT | **CC-BY-NC-4.0** | facebookresearch/ConvNeXt-V2 README |
| Meta Hiera | Apache-2.0 | **CC-BY-NC-4.0** | facebookresearch/hiera README |
| Apple AIMv2 | - | **apple-amlr, research only** | Apple ML Research license |
| SKELEX | - | CC-BY-NC-ND-4.0, and repo is EMPTY | HF `skhoha/SKELEX` |
| PRISM | - | CC-BY-NC-ND-4.0, gated | HF Paige |

This kills the ConvNeXt-V2 FCMAE idea that v2's notes and this project's own planning had
treated as a live option. The pattern to remember: for research releases, ALWAYS check the
weight licence separately from the code licence. It has now caught us three times
(RadImageNet, ConvNeXt-V2, Hiera).

## 6. Ineligible on architecture

- **Decipher-MR**: native 3D volumetric, cannot take independent 2D slices.
- **OrthoDiffusion**: diffusion generative model, not a feed-forward slice encoder. Still
  the best-matched pretraining domain in any of these reports (real knee MRI SSL), so
  still worth monitoring for an encoder-only release.
- **InternImage**: needs a compiled CUDA extension, not in timm.

## 7. What to trust in the source report, and what not to

The report's model identifiers, resolutions and licence findings held up on every point I
could check independently, which is a good sign. Two things did not:

- `deit3_large_patch16_384` does not accept 512 input in timm, contradicting its
  "high tolerance" claim.
- `internimage_large_384` is not a timm model.

It also asserts our cohort is "1,200 to 2,500 MRI studies"; we have 3,639 train / 704 val /
58 audit. And it recommends `torch.compile`, which we turned off after it broke checkpoint
loading with `_orig_mod.` prefixes.

No candidate in the report has ANY published knee, musculoskeletal or MRI evaluation
number. Every recommendation rests on architectural and objective-level reasoning, not
transfer evidence. That is honest of it to state, and it matches v1 and v2's finding that
prior RSNA Kaggle competitions were won 4-for-4 with ImageNet-only backbones.
