# Public 0.949 CoAtNet notebook vs our pipeline (10 Oct)

Source: kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror (inference source, read in full), forum post 746792,
built on dreaddevelopment's public Raptor CoAtNet/MIL recipe (0.924 with public LLM labels, 64 slices / 42 windows).
Their score history: 80 slices 0.939 -> 96 slices 0.940 -> + blended labels (JEV, stevenleehans, pilkwang, own) 0.942
-> + OAI external knees 0.945-0.948 -> mirror TTA 0.949. Without OAI their single model is 0.942.

| | theirs (0.949) | ours (best single LB 0.930) |
| --- | --- | --- |
| slots | 5: sag fluid 26, sag plain 22, cor fluid 18, cor plain 12, axial (either) 18 = 96 slices | 4: 18 each = 72 slices |
| images per study, eval | 94 overlapping 3-slice windows (stride 1, centre slice decides the plane) | 24 non-overlapping triplets |
| images per study, train | 12 random windows per study (cheap), batch 1 x accum 16 | all 24, batch 8 |
| crop | 140 mm box from DICOM spacing -> 384 px (0.365 mm/px), inference centre crop 320 px (117 mm) | 0.40 mm/px, 336 px (134 mm) |
| slice range | evenly over 6-94% of the stack | margin 5% |
| pooling | LayerNorm -> Linear(768, 256) -> tanh -> dropout 0.2 -> Linear(256, 12): per-finding softmax attention over windows, per-finding linear head W (12, 768) | mean over 24 images, one Linear(1024, 12) |
| lr | backbone 4.0e-5, HEAD 1.95e-3 (~50x), wd 0.059, 16 epochs | backbone 5e-5, head 1e-4 (2x), wd 0.05, 12 epochs |
| checkpoint | average of 3 epoch checkpoints (SWA) | best step |
| TTA | anatomical mirror average (+0.001) | none |
| backbone | CoAtNet (timm), 384 | ConvNeXt-Base, 336 |

Our attention runs (poolattn, perfinding) used head lr 1e-4 with 24 images; theirs uses ~2e-3 with 94 windows.
