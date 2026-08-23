# Runs summary

## Single runs

| run | params | val | audit-58 | public LB | notes |
| --- | --- | --- | --- | --- | --- |
| `convnext_base_4slot` | 88M | **0.8948** | 0.8900 | - | ConvNeXt-Base. +0.0008 over Small - capacity exhausted |
| `convnext_small_4slot` | 50M | 0.8940 | 0.8901 | 0.908 | The baseline. ConvNeXt-Small, 4 slots x 24 slices @512px |
| `convnext_small_25d` | 50M | 0.8935 | 0.8865 | 0.905 | 2.5D: neighbouring slices in the RGB channels |
| `convnext_small_posweight` | 50M | 0.8923 | 0.8838 | **0.909** | pos_weight auto_clamped. -0.0017 val, +0.001 LB - neutral |
| `convnext_small_672px_28slices` | 50M | 0.8879 | **0.8921** | - | 672px @0.31mm/px, 28 slices. Stopped 3400/4540, -0.005 at every matched step |
| `mricore_vitb` | 86M | 0.8850 | 0.8775 | 0.896 | MRI-CORE ViT-B/16, DINO on 6M MRI slices. Extended to 16 epochs |
| `radimagenet_resnet50` | 24M | 0.8799 | 0.8718 | - | RadImageNet ResNet-50, supervised on 1.35M CT/MRI/US. Extension gave +0.0005 then flattened, stopped at epoch 12 |
| `convnext_small_transformer_pool` | 52M | 0.8792 | 0.8709 | - | Self-attention over the 96 slices instead of the mean |
| `convnext_tiny_4slot` | 28M | 0.8596 | 0.8597 | 0.859 | First real run |
| `convnext_tiny_5slot` | 28M | 0.8585 | 0.8516 | 0.856 | A 5th slot (sag_t2). Lost |

## Ensembles

| members | val | audit-58 | public LB | notes |
| --- | --- | --- | --- | --- |
| cx_small_4slot + cx_base_4slot + mricore_vitb + radimagenet_resnet50 + 25d | **0.9005** | 0.8910 | - | Best val. 25d adds only +0.0005 here, its diversity overlaps |
| cx_small_4slot + cx_base_4slot + mricore_vitb + radimagenet_resnet50 | 0.9000 | 0.8928 | - | radimagenet adds +0.0017 while scoring lowest solo |
| cx_small_4slot + cx_base_4slot + posweight + mricore_vitb | 0.8983 | **0.8934** | - | posweight adds 0.0000 - another ConvNeXt-Small |
| cx_small_4slot + cx_base_4slot + mricore_vitb | 0.8983 | 0.8923 | pending | Submitted |
| cx_small_4slot + mricore_vitb | 0.8962 | 0.8902 | - | Pair: +0.0022, different architecture |
| cx_small_4slot + 25d | 0.8961 | 0.8910 | - | Pair: +0.0021, different INPUT beats different loss |
| cx_small_4slot + posweight | 0.8951 | 0.8894 | - | Two ConvNeXt-Smalls - correlated errors |
| mricore_vitb top-10 checkpoints | 0.8849 | 0.8751 | - | Within-run: never beats its own best single (0.8850) |
