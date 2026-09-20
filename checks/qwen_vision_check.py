"""qwen_vision_check.py - prove the Qwen3.5 vision tower run is correct before it costs a GPU.

Counting tensors is not enough on its own. The loader makes three claims that would each
produce a valid-looking tensor of the right shape if they were wrong:

  1. patch_embed.proj is a Conv3d of shape (768, 3, 2, 16, 16), and summing the temporal
     axis is EXACTLY equivalent to the Conv3d over two identical frames. A wrong reduction
     (mean, or taking one frame) halves or skews every feature silently.
  2. pos_embed is an nn.Embedding over a 48x48 grid, not timm's (1, N, C) parameter.
     Resampling it wrongly leaves the model spatially scrambled while everything still runs.
  3. Qwen's tower has no cls token and no trunk-final norm, so the timm skeleton must be
     built with class_token=False and global_pool='avg'. Get that wrong and the LayerNorm
     that ends the tower stays at random init.

Run:  python checks/qwen_vision_check.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RUN = "qwen3_5_vision_tower_336"
REF = "beitv2_base_336"                      # the ViT-B/16 this must match geometrically
CKPT = ROOT / "data" / "pretrained" / "qwen3_5_vision_vitb16.safetensors"

sys.path.insert(0, str(RUNS))
sys.path.insert(0, str(RUNS / RUN))
import shared_code as sc  # noqa: E402
import external_backbones as eb  # noqa: E402

PASS, FAIL = [], []


def ck(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  %s %-62s %s" % ("PASS" if ok else "FAIL", name, detail))


print("\n1. config: geometry copied from %s, weights the only variable" % REF)
cfg = sc.load_config(RUNS / RUN, verbose=False)
ref = sc.load_config(RUNS / REF, verbose=False)
ck("the run loads its OWN config", cfg["experiment"]["name"] == RUN, cfg["experiment"]["name"])
for k in ("size", "n_slices", "slices_as_channels", "mm_per_px"):
    ck("input.%s matches %s" % (k, REF), cfg["input"][k] == ref["input"][k],
       "%r" % (cfg["input"][k],))
for k in ("train_csv", "val_csv"):
    ck("paths.%s matches %s" % (k, REF), cfg["paths"][k] == ref["paths"][k], cfg["paths"][k])
ck("epochs match %s" % REF, cfg["training"]["epochs"] == ref["training"]["epochs"],
   str(cfg["training"]["epochs"]))
ck("lp_ft_epochs is 1, so epoch 1 IS the frozen linear probe",
   cfg["training"].get("lp_ft_epochs") == 1, str(cfg["training"].get("lp_ft_epochs")))
ck("grad_clip is at least 3", cfg["training"]["grad_clip"] >= 3.0,
   str(cfg["training"]["grad_clip"]))
ck("early stopping patience is 7", cfg["training"]["early_stopping"]["patience"] == 7)

print("\n2. normalisation: Qwen publishes it, so this is checked not guessed")
ck("norm_mean is Qwen's [0.5, 0.5, 0.5]", list(cfg["input"]["norm_mean"]) == [0.5] * 3,
   str(cfg["input"]["norm_mean"]))
ck("norm_std is Qwen's [0.5, 0.5, 0.5]", list(cfg["input"]["norm_std"]) == [0.5] * 3,
   str(cfg["input"]["norm_std"]))
ck("no norm_mismatch_ok override is needed", not cfg["input"].get("norm_mismatch_ok"))

print("\n3. the weights file")
ck("checkpoint present", CKPT.exists(), str(CKPT) if CKPT.exists() else "MISSING")
if not CKPT.exists():
    print("\n  cannot continue without the weights")
    raise SystemExit(1)
from safetensors.torch import load_file, save_file  # noqa: E402
sd = load_file(str(CKPT))
ck("holds only vision tensors", all(".visual." in k for k in sd), "%d tensors" % len(sd))
ck("no language-model tensors leaked in",
   not any(".layers." in k and ".visual." not in k for k in sd))
nparam = sum(v.numel() for v in sd.values())
ck("about 100M parameters, i.e. ViT-B sized not 0.8B", 80e6 < nparam < 120e6,
   "%.1fM" % (nparam / 1e6))
w = sd["model.visual.patch_embed.proj.weight"]
ck("patch_embed.proj is Conv3d (out, in, temporal, kh, kw)",
   tuple(w.shape) == (768, 3, 2, 16, 16), str(tuple(w.shape)))
pe = sd["model.visual.pos_embed.weight"]
ck("pos_embed is a 48x48 grid", tuple(pe.shape) == (2304, 768), str(tuple(pe.shape)))

print("\n4. the Conv3d -> Conv2d collapse is EXACT, not an approximation")
torch.manual_seed(0)
x = torch.randn(2, 3, 336, 336)
b = sd["model.visual.patch_embed.proj.bias"]
refy = F.conv3d(torch.stack([x, x], dim=2), w, b, stride=(2, 16, 16)).squeeze(2)
got = F.conv2d(x, w.sum(dim=2), b, stride=16)
d = (refy - got).abs().max().item()
ck("Conv3d(two identical frames) == Conv2d(summed weights)", d < 1e-4, "max diff %.2e" % d)
wrong = F.conv2d(x, w[:, :, 0], b, stride=16)
ck("and taking ONE frame instead would be visibly wrong",
   (refy - wrong).abs().max().item() > 1e-2,
   "max diff %.3f" % (refy - wrong).abs().max().item())

print("\n5. position-embedding resampling")
same = eb._qwen_pos_embed(pe, 48 * 48)
ck("resampling to its own grid is the identity",
   torch.allclose(same.reshape(-1, 768), pe, atol=1e-6))
r = eb._qwen_pos_embed(pe, (336 // 16) ** 2)
ck("336px gives 21x21 = 441 tokens", tuple(r.shape) == (1, 441, 768), str(tuple(r.shape)))
ck("resampling preserves the statistics", abs(float(r.std()) - float(pe.std())) < 0.01,
   "std %.4f vs %.4f" % (r.std(), pe.std()))
try:
    eb._qwen_pos_embed(pe, 440)
    ck("a non-square token count is rejected", False, "it was accepted")
except RuntimeError:
    ck("a non-square token count is rejected", True)

print("\n6. the model builds and loads every tensor")
import importlib  # noqa: E402
tm = importlib.import_module("train")
model = tm.build_model(cfg)
bb = model.backbone
msd = bb.state_dict()
ck("no cls token in the skeleton", "cls_token" not in msd)
ck("the trunk-final norm is Identity, the LayerNorm moved to fc_norm",
   type(bb.norm).__name__ == "Identity" and type(bb.fc_norm).__name__ == "LayerNorm",
   "norm=%s fc_norm=%s" % (type(bb.norm).__name__, type(bb.fc_norm).__name__))
ck("num_features is 768", bb.num_features == 768, str(bb.num_features))
fresh = type(bb)(**{}) if False else None
n = eb.load_qwen3_5_vision(bb, CKPT)
ck("every tensor is filled from the checkpoint", n == len(msd), "%d/%d" % (n, len(msd)))

print("\n7. the weights really are Qwen's, not timm's random init")
ref_w = sd["model.visual.blocks.0.attn.qkv.weight"]
ck("block 0 qkv matches the checkpoint bit for bit",
   torch.allclose(msd_now := bb.state_dict()["blocks.0.attn.qkv.weight"], ref_w, atol=0),
   "max diff %.2e" % (msd_now - ref_w).abs().max().item())
ck("fc_norm came from merger.norm",
   torch.allclose(bb.state_dict()["fc_norm.weight"], sd["model.visual.merger.norm.weight"]))

print("\n8. the strict counter fires rather than half-loading")
import timm  # noqa: E402
m2 = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0,
                       img_size=336, class_token=False, global_pool="avg",
                       no_embed_class=True)
p = Path(tempfile.mkdtemp()) / "truncated.safetensors"
save_file({k: v for k, v in sd.items() if ".blocks.5." not in k}, str(p))
try:
    eb.load_qwen3_5_vision(m2, p)
    ck("a checkpoint missing one block raises", False, "it loaded silently")
except RuntimeError as e:
    ck("a checkpoint missing one block raises", True, str(e)[:52])

print("\n9. a real forward pass at the run's own geometry")
per = sc.images_per_study(cfg)
model.eval()
with torch.no_grad():
    y = model(torch.randn(2 * per, 3, cfg["input"]["size"], cfg["input"]["size"]))
ck("windows per study", per == len(cfg["input"]["slots"]) * 6, str(per))
ck("study logits are (2, 12)", tuple(y.shape) == (2, len(cfg["tasks"])), str(tuple(y.shape)))
ck("logits are finite", bool(torch.isfinite(y).all()))
ck("logits are not constant", float(y.std()) > 1e-4, "std %.4f" % y.std())

print("\n10. frozen features separate real knee windows")
# NOT on Gaussian noise, and NOT on the raw vectors. Mean-pooled ViT features always share
# one large direction, so raw pairwise correlation is ~0.99 for any input and says nothing.
# A linear head cannot see that direction anyway - its bias absorbs the batch mean, and
# feat_norm removes the shared scale - so measure what the head actually gets.
_df = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg))
_shards = Path(cfg["paths"]["shards_dir"])
_have = [u for u in _df["StudyInstanceUID"] if (_shards / ("%s.npz" % u)).exists()]
if len(_have) >= 2:
    _idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    _ds = sc.KneeShardDataset(_df[_df["StudyInstanceUID"].isin(_have)].reset_index(drop=True),
                              cfg, sc.plan_slots(_idx, cfg), train=False)
    _x = sc.prepare_batch(torch.stack([_ds[i][0] for i in range(len(_ds))]), cfg, "cpu")
    with torch.no_grad():
        a = bb(_x[:24])
    ck("features are not collapsed to a constant", float(a.std()) > 1e-3, "std %.4f" % a.std())
    ac = a - a.mean(dim=0, keepdim=True)
    c = np.corrcoef(ac.numpy())
    off = c[~np.eye(len(ac), dtype=bool)]
    ck("batch-centred windows are decorrelated, not duplicates", float(off.mean()) < 0.5,
       "mean off-diagonal r %+.4f over %d windows" % (off.mean(), len(ac)))
    _s = np.linalg.svd(ac.numpy(), compute_uv=False)
    _ev = (_s ** 2) / (_s ** 2).sum()
    _er = float(np.exp(-(_ev * np.log(_ev + 1e-12)).sum()))
    ck("the feature space has real dimensionality, not one direction", _er > 3.0,
       "effective rank %.1f of %d windows" % (_er, len(ac)))
else:
    print("      skipped: %d local shards, needs 2" % len(_have))

print("\n" + "=" * 76)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    for f in FAIL:
        print("    FAILED: %s" % f)
    raise SystemExit(1)
