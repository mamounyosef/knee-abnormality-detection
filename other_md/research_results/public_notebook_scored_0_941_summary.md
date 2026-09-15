# Report: Bend the Knee to the Dinosaurs (0.941 RSNA Knee Abnormality Detection)

This report provides a comprehensive breakdown of the public Kaggle notebook "Bend the Knee to the Dinosaurs", which achieves a score of 0.941 (macro AUROC) on the RSNA Knee Abnormality Detection competition.

## 1. Data and Preprocessing

The notebook contains multiple independent inference pipelines, each with its own preprocessing logic. DICOM files are read using `pydicom.dcmread`. Caching is heavily utilized to store entire studies in memory (up to `CACHE_BUDGET_MAX_GB = 24.0` GB) before inference to optimize disk I/O.

### Series and View Selection

- **DINOv2 Pipeline:** Selects series based on the `SLOTS_RECOVERED` dictionary which defines 6 slots: `SAG_FLUID_FS`, `COR_FLUID_FS`, `AX_FLUID_FS`, `SAG_FLUID_NOFS`, `COR_T1`, `SAG_T1`. Series are mapped to these slots by plane (Sagittal, Coronal, Axial), fluid sensitivity, and fat suppression. When multiple series match a slot, the one with the maximum number of slices is chosen.
- **A5 Pipeline:** Uses 6 slots defined solely by plane and fat suppression (e.g., `('Sagittal', 1)`).
- **V48 Pipeline (RadImageNet):** Uses two slot configurations. The E13 configuration requires 4 slots: `SAG_FS`, `COR_FS`, `AX_FS`, and `SAG_NOFS`. The E11 configuration requires 4 slots: `SAG_NOFS`, `COR_NOFS`, `AX_NOFS`, and `SAG_FS`.
- **Raptor Pipeline:** Uses 5 slots: `("Sagittal", 1)`, `("Sagittal", 0)`, `("Coronal", 1)`, `("Coronal", 0)`, `("Axial", -1)`.

### Slice Selection and Interpolation

Slices are **not** interpolated along the Z-axis in any pipeline. They are selected as discrete images:

- **DINOv2:** Uses a group of 3 contiguous slices per slot (`GROUP = 3`), sampled uniformly from the middle 60% of the volume (`SLICE_BAND = (0.2, 0.8)`).
- **A5 Pipeline:** Selects exactly 16 slices linearly spaced from the 12% to 88% marks of the series (`SLICE_BAND = (0.12, 0.88)`).
- **V48 Pipeline:** Selects 8 slices per plane from the middle 60% (`0.2, 0.8`).
- **Raptor Pipeline:** Evaluates up to 64 or 44 slices depending on the arm, sampled between `0.02` and `0.98` (or `0.06` and `0.94`).

### Resolution, Cropping, and Normalization

- **DINOv2:** Center crops based on a physical field of view of `CROP_MM = 130.0`. Slices are bilinearly interpolated to `IMG = 336` or `224`. Intensity is clipped to the 1st and 99th percentiles of the full volume, then scaled to `[0, 255]`.
- **A5 Pipeline:** Resizes to `SIZE = 336` using `cv2.INTER_AREA`. Intensity is clipped per-slice or per-series (1st-99th percentile).
- **V48 Pipeline:** Resizes to `IMG = 224`. Normalizes image tensors to `[-1.0, 1.0]` by dividing by 127.5 and subtracting 1.0.
- **Raptor Pipeline:** Center crops to `CROP_MM = 140.0` and resizes to `IMG = 336` or `384`. Uses the 2nd and 98th percentile for intensity clipping across the concatenated valid slices. Output is normalized using ImageNet mean and std.
- **Laterality:** Right-sided knees are horizontally flipped for Coronal/Axial views and vertically flipped for Sagittal views to standardize orientation.

## 2. Labels

The notebook targets the exact 12 competition labels: `['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']`.

- **No external or auxiliary labels** are used or predicted.
- The models predict soft probabilities (via sigmoid), which are essential for the AUROC metric.
- A post-processing label smoothing technique (`legacy_fold_soft_window_pool`) is applied to specific findings using a softmax-weighted sum over TTA windows (e.g., `beta = 10.0` for Fracture, `alpha = 0.15`).

## 3. Model Architecture

The notebook is an ensemble of several fundamentally different architectures:

- **DINOv2 Pipeline:** Uses a pretrained `metaresearch/dinov2/PyTorch/small/1` (DINOv2-S) backbone. It processes 3-slice (2.5D) windows. A custom `SlotHead` aggregates features using slot embeddings, queries for the 12 labels, and cross-attention.
- **V48 Pipeline:** Uses a ResNet-50 backbone pretrained on RadImageNet (`ResNet50.pt`). It operates as a 2D CNN followed by global average pooling to a 2048-dim token. A custom head projects this to 512, adds plane and position embeddings, applies an 8-head `MultiheadAttention` layer, and fuses the result to output 12 logits.
- **A5 Pipeline:** Uses `timm` models (Vision Transformers and hybrids) equipped with custom multi-slice aggregation blocks like `SlotDepthMixer` or `DepthCompress`, and attention-based readout heads (`TokenResidualPool`, `LabelAttentionPool`).
- **Raptor Pipeline:** Uses the `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` backbone from `timm`. It evaluates 3-slice 2.5D windows and aggregates them using a custom `RaptorClassifier` that computes attention weights over the windows via an MLP.
- **CoAtNet Pipeline:** Explicitly noted in the markdown, it uses a CoAtNet RMLP-2 backbone with a finding-specific gated spatial head and global-average residual.

## 4. Training

Because this is an inference-only notebook, the training loop, loss functions, optimizer, schedules, and gradient clipping logic are **not present** in the code. However, the notebook explicitly defines the following hyperparameter constants intended for the DINOv2 model's training configuration:

- `EPOCHS = 10`
- `BATCH_STUDIES = 8`
- `LR_HEAD = 0.001`
- `LR_BACKBONE = 8e-06`
- `UNFREEZE_LAST = 6` (Only the last 6 transformer blocks of DINOv2 and LayerNorms are trainable)
- `WEIGHT_DECAY = 0.02`
- `AUG_ROT_DEG = 8.0`, `AUG_SCALE = 0.08`, `AUG_SHIFT = 0.05`, `AUG_INTENSITY = 0.1`

Inference operates in mixed precision (bfloat16 or float16) depending on CUDA device capability, using `torch.autocast`.

## 5. Inference and Ensembling

### Models in Submission

The final ensemble consists of over 47 distinct model checkpoints:

- **DINOv2:** 20 members (checked strictly via fingerprinting).
- **V48 Families:** 5 E10 public heads, 5 E10 reference heads, and 5 E13 heads (run twice on different layouts).
- **A5 Pipeline:** All checkpoint folds loaded from the `knee-mri-fold-weights` directory (presumably 5).
- **Raptor Pipeline:** 4 different arm configurations.
- **CoAtNet Pipeline:** 3 checkpoints (epochs 4, 6, 8).

### Ensembling Strategy

Predictions are exclusively combined using **percentile rank averaging** (`pd.DataFrame.rank(pct=True)`). Raw probabilities are rarely averaged directly between different families to prevent models with uncalibrated scales from dominating the AUROC.

- **Raptor Arms:** 4 arms are blended with weights `[0.60, 0.10, 0.10, 0.20]`.
- **CoAtNet:** Blended into the Raptor arm using `private_alpha = 0.40`.
- **V48 Branch:** The E13 model is blended into E10 at `0.50`. A second E13 pass (on the E11 layout) is blended at `0.20`.
- **Final Blend:** The A5 (Transformer) arm acts as the base. The Raptor+CoAtNet arm is given a weight of `0.60` globally, but class-specific weights override this: ACL (`0.75`), Medial Meniscus (`0.80`), Lateral Meniscus (`1.00`), Lateral OA (`0.75`), and Fracture (`0.75`).

### Test-Time Augmentation (TTA)

TTA is heavily utilized via sliding window evaluation. The pipeline applies class-specific pooling over these windows. The `PUBLIC_FRONTIER_TARGET_POOL` dictates that targets like Fracture, Contusion, and Meniscus use `max` pooling; ACL and MCL use `top2` pooling; and Synovitis uses `original_mean` pooling.

### Runtime Constraints

To ensure the notebook does not exceed the competition's 9-hour limit, the script employs a strict dynamic time budget (`TIME_BUDGET = 8.0 * 3600` seconds). During the DINOv2 inference, the code measures the time cost per window (`est['win']`) and fixed overhead (`est['fixed']`). If the remaining time is insufficient, it dynamically surrenders models or truncates windows to guarantee a successful submission.

## 6. Anything Unusual

This notebook is a masterclass in Kaggle engineering and contains several highly unusual, non-standard approaches:

- **Code Injection and Subprocessing:** The codebase strings together multiple foreign codebases. The Raptor inference code is stored entirely as a massive string (`_KE_SRC`) and executed on the fly via `exec(compile(...))`.
- **Offline Dependency Installation:** It installs an OpenCV headless wheel directly via a `subprocess.run` call (`pip install --no-deps`) using a local `.whl` file from a dataset to bypass internet restrictions.
- **External Inference Scripts:** It executes the CoAtNet pipeline by spawning a separate Python process (`subprocess.run`) that imports a script attached as a dataset.
- **CUDA Execution Probe:** Before initializing, the script runs a dummy `nn.Conv2d` forward pass (`_cuda_execution_probe`) to verify the CUDA compute capability and ensure the GPU is healthy, falling back to CPU gracefully if it fails.
- **Metadata Rank Calibration:** The script contains a heavily obfuscated base64+zlib compressed payload (`_RAD_CAL_PAYLOAD`). When decompressed, it applies a post-processing linear regression to the final ranks. The regression features include metadata extracted from the DICOMs (e.g., the number of series, and slice counts grouped by plane and fluid sensitivity).
- **Fingerprinting Gate:** It enforces strict cryptographic and "fingerprint" checks on the weights. It passes a fixed noise tensor through the loaded models and compares the output to a pre-recorded tensor (`FINGERPRINT_TOL = 0.002`). If the architecture or preprocessing changes, the fingerprint fails and execution aborts.

## 7. Stated Results

The notebook claims a score of **0.941 macro AUROC** on the RSNA Knee Abnormality Detection competition leaderboard. No local validation scores, per-fold scores, or per-finding metrics are reported in the notebook's markdown or code.

Ran command: `grep -n "LEGACY_MEMBER_WEIGHT" C:\Users\mamou\.gemini\antigravity\scratch\notebook.py`
Searched for "DINOv2"
Searched for "DINOv2"
Searched for "DINOv2"
Ran command: `cat C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "EXPECTED_DINOV2_MEMBERS"`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "combine_public_members_by_fold" -Context 3,10`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "groups.setdefault" -Context 5,10`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "combine_public_members_by_fold"`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "_combine"`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "def _combine" -Context 0,15`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "models = \[\]" -Context 3,15`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "coat_resgated_ep10_top3_inference as rt" -Context 5,10`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "coatnet_resgated_ep10_top3_inference" -Context 5,10`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "DINOv2" -Context 3,3`

Based on a detailed review of the notebook code, here is the breakdown of exactly where the ~47 model checkpoints come from and the nature of their diversity.

### 1. DINOv2 (20 members)

- **Path:** Loaded via `/kaggle/input/datasets/pilkwang/rsna-knee-weights/manifest.json`.
- **Count:** 20 members (`EXPECTED_DINOV2_MEMBERS = 20`).
- **Diversity:** The code parses 5 cross-validation folds from the manifest (a comment states `"give all five folds equal weight"`). The manifest groups them by `pixel_group` configurations containing distinct input settings (image resolution, crop mm, slice band, and slots). Thus, the 20 members consist of **5 CV folds evaluated across 4 different input configurations**.
- **Test-time variations:** These appear to be genuinely different trained checkpoints per fold and config (as opposed to one checkpoint evaluated 4 times), since each has its own fingerprint and state dict in the manifest.

### 2. A5 (5 members)

- **Path:** Loaded via glob `*_f*.pt` from `/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights`.
- **Count:** 5 members.
- **Diversity:** These are **different cross-validation folds**. The filename pattern `_f*.pt` and a code comment (`"Our independently trained five-fold family"`) confirm this.

### 3. V48 / E10 / E11 / E13 (15 distinct heads + 1 shared encoder)

This branch loads a single shared ResNet50 backbone (`ResNet50.pt` from `marwanmath/resnet-50-radimagenet-marwan`) and attaches different classification heads.

- **E10 Public Heads (5 members):** Loaded from `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`. These represent **5 CV folds**.
- **E10 Reference Heads (5 members):** Loaded from `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`. These represent **5 CV folds** of a "twin" branch, meaning they are a separate training run (e.g., a different random seed) of the same E10 setup.
- **E13 Heads (5 members):** Loaded from `sofiaanjenje/rsna-knee-e13-train/rsna_rad_e11`. These represent **5 CV folds** of a different diverse training setup.
- **Test-time variations:** The E13 heads are evaluated **twice** under different inference settings. They are evaluated once using the E13 slot layout (pass 1), and then the exact same weights are evaluated again using the E11 slot layout (pass 2).

### 4. Raptor (3 checkpoints, 4 inference arms)

- **Paths:** Loaded from `dreaddevelopment/raptor-knee-maxspan` (`v5_full_swa.pt`), `raptor-knee-native384dense` (`v10_full.pt`), and `raptor-knee-native384` (`v8_full_swa.pt`).
- **Count:** 4 inference arms.
- **Diversity:** The 3 files represent **different training runs/configurations** with varying resolutions and slice counts.
- **Test-time variations:** The `v5` checkpoint is evaluated **twice**: once normally, and once with the test-time parameter `reverse=True` (which flips the order of the slices).

### 5. CoAtNet (3 members)

- **Path:** Evaluated via an external script using an artifact from `mattiaangeli/rsna-knee-coat-resgated-ep10-top3`.
- **Count:** 3 members.
- **Diversity:** The notebook includes the strict assertion `assert [i['epoch'] for i in r['checkpoints']] == [4, 6, 8]`. This confirms these are **3 different epochs of one single training run**.

***

### Summary Table

| Group | Count | Backbone | Variation Within Group | Blending Weight |
| :--- | :--- | :--- | :--- | :--- |
| **DINOv2** | 20 | DINOv2-S | 5 CV folds × 4 input configurations (`pixel_group`). | Percentile rank average using specific weights defined in the `manifest.json`. |
| **A5** | 5 | timm ViTs / Hybrids | 5 CV folds. | Percentile rank averaged. Contributes 40% to the final global blend (adjusted per-class). |
| **V48 E10 (Public)** | 5 | ResNet50 (RadImageNet) | 5 CV folds. | Percentile rank averaged. Weighted equally (~50%) against the Reference E10 twin. |
| **V48 E10 (Reference)** | 5 | ResNet50 (RadImageNet) | 5 CV folds (twin run / different seed). | Percentile rank averaged. Weighted equally (~50%) against the Public E10 twin. |
| **V48 E13 (Pass 1)** | 5 | ResNet50 (RadImageNet) | 5 CV folds. | Percentile rank averaged. Blends into the E10 mix with a weight of 50%. |
| **V48 E13 (Pass 2)** | (Same 5 as Pass 1) | ResNet50 (RadImageNet) | **Same underlying weights** as Pass 1, but evaluated on a different test-time view/slot layout. | Percentile rank averaged. Blends into the E10/E13 mix with a weight of 20%. |
| **Raptor** | 4 | CoAtNet (RMLP-2) | 3 different training runs (v5, v8, v10). <br>**Same underlying v5 weights** evaluated twice (normal vs reversed slice order). | Probability weighted average: v5 (60%), v10 (10%), v5-reverse (10%), v8 (20%). |
| **CoAtNet** | 3 | CoAtNet (RMLP-2) | 3 different epochs (4, 6, 8) of a single training run. | Ranks are averaged, then blended into the Raptor arm with a `private_alpha` of 40%. |

Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "predict_member" -Context 5,20`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "class Model" -Context 3,12`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "def read_slot" -Context 3,30`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "class Net" -Context 3,25`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "def build_study" -Context 3,35`

Error: The stream was interrupted. Please continue the task you were working on. (Error ID: 38b081e2-6f9b-4be0-bdd4-a121db83e984-60)

Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py -TotalCount 1145 | Select-Object -Skip 1100`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py -TotalCount 1586 | Select-Object -Skip 1550`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py -TotalCount 1500 | Select-Object -Skip 1465`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "class DepthCompress" -Context 0,15`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py -TotalCount 1953 | Select-Object -Skip 1925`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py -TotalCount 2450 | Select-Object -Skip 2400`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\notebook.py | Select-String -Pattern "def eval_windows" -Context 3,15`
Ran command: `python -c "import ast; src = open('C:/Users/mamou/.gemini/antigravity/scratch/notebook.py').read(); ke_src = [n.value.s for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and len(n.targets)==1 and getattr(n.targets[0], 'id', '') == '_KE_SRC']; open('C:/Users/mamou/.gemini/antigravity/scratch/ke_src.py', 'w').write(ke_src[0]) if ke_src else None"`
Ran command: `Get-Content C:\Users\mamou\.gemini\antigravity\scratch\ke_src.py | Select-String -Pattern "def eval_windows" -Context 3,15`

Based on the code in `notebook.py`, here is precisely how slices are packed into the model's input channels for each arm:

### DINOv2

* **Channel contents:** 3 different slices per group.
- **Overlap:** Non-overlapping. The caching mechanism explicitly restricts the loaded slices per slot to exactly the group size (`CACHE_SLICES = GROUP * 1`). Because the available slices perfectly equal the window size, it evaluates exactly 1 static window starting at index 0.
- **Stride:** N/A (only 1 group per slot). No slice appears in more than one group.
- **Forward passes per study:** 6 passes (6 slots × 1 group).
- **Code:**

```python
starts = window_starts(cache.shape[2], group) if starts is None else list(starts) # Evaluates to [0]
# ...
rows = torch.from_numpy(np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
```

### A5

* **Channel contents:** The A5 arm **does not use 3-channel packing** to feed slices into the model. Instead, it passes all 16 slices of a slot simultaneously (`n_slice=16`). The encoder natively takes a 16-channel input, unless the `compress` stem is active, in which case it uses a 1x1 Convolution (`DepthCompress`) to project the 16 channels down to 3 channels.
- **Overlap:** N/A. All 16 slices are processed at once per slot.
- **Stride:** N/A.
- **Forward passes per study:** 6 passes (6 slots × 1 full block of 16 slices).
- **Code:**

```python
self.compress = DepthCompress(n_slice, 3) if stem == 'compress' else None
# ...
if self.compress is not None:
    im = self.compress(im)
```

### V48 (E10/E11/E13)

* **Channel contents:** V48 plainly **duplicates a single slice three times** across the input channels. 3-channel packing here is strictly to satisfy the input requirements of the pretrained RadImageNet ResNet50 backbone, not to add 2.5D context.
- **Overlap:** N/A.
- **Stride:** N/A.
- **Forward passes per study:** 32 passes (4 slots × 8 slices per slot, encoded independently).
- **Code:**

```python
image = _rad_torch.from_numpy(flat[indices]).to(device).float().div_(127.5).sub_(1.0)
image = image.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
```

### Raptor

* **Channel contents:** 3 different slices stacked as a 2.5D volume (a center slice `c` alongside its immediate neighbors `c-1` and `c+1`).
- **Overlap:** Heavily overlapping / sliding windows.
- **Stride:** The stride is not a fixed integer. The code uses `np.linspace` to sample `k_eval` (62 or 42) window centers across the entire concatenated study volume (max 64 or 44 slices). Because it requests roughly the same number of windows as there are slices, the stride effectively averages 1 slice. A single slice will appear in up to 3 consecutive groups.
- **Forward passes per study:** 62 or 42 passes (1 slot/volume × `k_eval` windows).
- **Code:**

```python
idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
# ...
c = max(1, min(c, D - 2))
tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
```

### CoAtNet

* The CoAtNet inference code is not present in the notebook; it is executed by loading an external module (`import coatnet_resgated_ep10_top3_inference as rt`). Therefore, its slice packing strategy is not determinable from the provided code.
