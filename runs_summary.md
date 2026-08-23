# Runs summary

Macro AUROC on all 12 findings. **Bold** marks the best in each column.

- **val** — 704 held-out studies, soft LLM labels. What we optimise.
- **audit-58** — 58 studies with the radiologist panel's 0/1 labels, taken at the same
  checkpoint as the best val. Report-only; it never picks a checkpoint. 58 rows is ±0.05,
  so read it as a sanity check, not a ranking.
- **public LB** — Kaggle's public leaderboard, ~1300 studies.

## Single runs

| run | params | val | audit-58 | public LB | notes |
| --- | --- | --- | --- | --- | --- |
| `convnext_base_4slot` | 88M | **0.8948** | 0.8900 | — | ConvNeXt-Base. +0.0008 over Small - capacity exhausted |
| `convnext_small_4slot` | 50M | 0.8940 | 0.8901 | 0.908 | **The baseline.** ConvNeXt-Small, 4 slots x 24 slices @512px |
| `convnext_small_25d` | 50M | 0.8935 | 0.8865 | 0.905 | 2.5D: neighbouring slices in the RGB channels |
| `convnext_small_posweight` | 50M | 0.8923 | 0.8838 | **0.909** | pos_weight auto_clamped. Val says -0.0017, LB says +0.001 - neutral |
| `convnext_small_672px_28slices` | 50M | 0.8879 | **0.8921** | — | 672px @0.31mm/px, 28 slices. Stopped at 3400/4540, -0.005 at every matched step |
| `mricore_vitb` | 86M | 0.8850 | 0.8775 | — | MRI-CORE ViT-B/16, DINO on 6M MRI slices. Extended to 16 epochs |
| `convnext_small_transformer_pool` | 52M | 0.8792 | 0.8709 | — | Self-attention over the 96 slices instead of the mean |
| `convnext_tiny_4slot` | 28M | 0.8596 | 0.8597 | 0.859 | First real run |
| `convnext_tiny_5slot` | 28M | 0.8585 | 0.8516 | 0.856 | A 5th slot (sag_t2). Lost |
| `radimagenet_resnet50` | 24M | — | — | — | _queued_ - RadImageNet ResNet-50, supervised on 1.35M CT/MRI/US |

## Ensembles

Equal-weight average in logit space. Only the ones worth keeping — see
`runs/others/ensembling/ensemble_results/` for every run of the search.

| members | val | audit-58 | public LB | notes |
| --- | --- | --- | --- | --- |
| cx_small_4slot + mricore_vitb + cx_base_4slot + posweight | **0.8983** | **0.8934** | — | **Best ensemble.** Equal-weight logit average |
| cx_small_4slot + mricore_vitb + cx_base_4slot | **0.8983** | 0.8923 | — | posweight adds nothing on val, +0.0011 on audit |
| cx_small_4slot + mricore_vitb | 0.8962 | 0.8902 | — | Two members, +0.0022 - mricore is the weakest model and the biggest jump |
| cx_small_4slot alone | 0.8940 | 0.8901 | 0.908 | Reference: the best single model |
| mricore_vitb top-10 checkpoints | 0.8849 | 0.8751 | — | Within-run checkpoint ensembling. Never beats its own best single (0.8850) |

## What this says

- **Ensembling is the biggest lever left.** +0.0043 over the baseline, for no GPU.
- **Solo score does not decide ensemble value.** `mricore_vitb` is the weakest
  full-size model and contributes the largest single jump, because it is the only one
  with a different architecture AND different pretraining.
- **Within-run checkpoint ensembling is dead** — flat for K=2..15, never beats the
  run's own best checkpoint. Snapshots of one trajectory share their errors.
- Capacity, pooling, input geometry and loss shaping have all been tried and none
  beat 0.8940 alone.
