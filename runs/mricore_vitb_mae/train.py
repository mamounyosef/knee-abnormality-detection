"""
train.py — MAE continued pretraining of MRI-CORE ViT-B/16 on our knee slices.

Self-supervised, no labels: mask 75% of each slice's patches, reconstruct them, keep the
encoder. Output is an encoder-only checkpoint in MRI-CORE's own format, so the next run
loads it with `weights: "MRI_CORE_vitb_mae.pth"` and no code change.

Settings follow "Masked Autoencoders Are Scalable Vision Learners" (arXiv 2111.06377),
Table 8 and the Table 1 ablations. Two deliberate deviations, both marked DEVIATION below.

Local:  python runs/mricore_vitb_mae/train.py
Modal:  modal run runs/mricore_vitb_mae/train.py
"""

import math
from collections import deque
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EXP_NAME = "mricore_vitb_mae"


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402


# =============================================================================
# data: one item = one series, K random slices from it
# =============================================================================
class MAESliceDataset(sc.KneeShardDataset):
    """Slices, not studies. Reuses the parent's shard read, resample, window and warp.

    A shard stores a whole series as one array, so pulling a single random slice would
    decompress ~33 and throw the rest away. Each item takes K slices from one series
    instead, and the series list is repeated so one epoch is ~one pass over the corpus.
    """

    def __init__(self, df: pd.DataFrame, cfg: dict, series_df: pd.DataFrame, train: bool):
        super().__init__(df, cfg, plan={}, train=train)
        m = cfg["mae"]
        self.k = int(m["slices_per_series"])
        self.n_slices = self.k                    # what _load's pick_slices reads
        studies = set(df["StudyInstanceUID"])
        s = series_df[series_df.study.isin(studies)]
        s = s[s.n_slices >= 1]
        pairs = list(zip(s.study.tolist(), s.series.tolist()))
        # repeat so len(dataset) * k is roughly the real slice count
        reps = max(1, int(round(float(s.n_slices.mean()) / self.k)))
        self.index = pairs * reps
        self.n_series, self.reps = len(pairs), reps

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        study, series = self.index[i]
        rng = np.random.default_rng([self.seed, int(sc._EPOCH.value), i] if self.train
                                    else [self.seed, i])
        try:
            v = self._load(study, series, rng)     # (k, size, size) float, windowed
        except Exception:
            v = np.zeros((self.k, self.size, self.size), np.float32)
        if self.aug is not None:
            v = self._augment(v, rng)
        # (1, k, H, W): one "slot" of k slices, so prepare_batch's 5D path applies unchanged
        return torch.from_numpy(v[None].astype(np.uint8))


# =============================================================================
# model
# =============================================================================
class MAE(nn.Module):
    """MRI-CORE ViT-B/16 encoder + a small disposable decoder."""

    def __init__(self, cfg: dict):
        super().__init__()
        import timm
        m, mm = cfg["model"], cfg["mae"]
        size, patch = cfg["input"]["size"], 16
        self.encoder = timm.create_model(m["name"], pretrained=False, num_classes=0,
                                         img_size=size)
        self.grid = self.encoder.patch_embed.grid_size[0]      # 512/16 = 32
        self.num_patches = self.grid * self.grid               # 1024
        self.patch = patch
        self.mask_ratio = float(mm["mask_ratio"])

        d_enc = self.encoder.embed_dim                         # 768
        d_dec = int(mm["decoder_dim"])                         # 512
        depth = int(mm["decoder_depth"])                       # 4
        heads = int(mm["decoder_heads"])

        self.decoder_embed = nn.Linear(d_enc, d_dec, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_dec))
        # +1 for the CLS slot, matching the encoder's pos_embed layout
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_dec), requires_grad=False)
        from timm.models.vision_transformer import Block
        self.decoder_blocks = nn.ModuleList(
            [Block(d_dec, heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(depth)])
        self.decoder_norm = nn.LayerNorm(d_dec)
        # 1 channel, not 3: our slices are grey copied 3x, predicting 3 would learn to triplicate
        self.decoder_pred = nn.Linear(d_dec, patch * patch, bias=True) # maps from d_dec which is 512 to 256 (which are the pixel values for each patch)""
        self.norm_pix = bool(mm["norm_pix_loss"])

        self._init_decoder()

    def _init_decoder(self):
        """Fixed sin-cos decoder positions, xavier_uniform elsewhere (paper A.1)."""
        pe = _sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        for mod in list(self.decoder_blocks) + [self.decoder_embed, self.decoder_pred]:
            for p in mod.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)

    def patchify(self, imgs):
        """(B, 1, H, W) -> (B, L, patch*patch). which is (B, 1, 512, 512) -> (B, 1, 1024, 256)"""
        p, h = self.patch, self.grid   # p = 16 pixels per patch, h = 32 patches per side
        # (B, 1, 512, 512) -> (B, 1, 32, 16, 32, 16): split each 512 axis into
        # "which patch" (32) and "which pixel inside it" (16)
        x = imgs.reshape(imgs.shape[0], 1, h, p, h, p)
        # einsum just reorders axes: n=batch c=channel h,w=patch row/col p,q=pixel row/col.
        # nchpwq -> nhwpqc puts both patch axes first and both pixel axes last, so each
        # patch's 256 pixels sit together and the final reshape can flatten them.
        return torch.einsum("nchpwq->nhwpqc", x).reshape(imgs.shape[0], h * h, p * p)

    def unpatchify(self, x):
        """(B, 1024, 256) -> (B, 1, 512, 512). Exactly patchify run backwards.

        patchify flattened each patch's 16x16 pixels into a row of 256; this puts them
        back on the board. Needed only for viewing - training never leaves patch space.
        """
        p, h = self.patch, self.grid
        # (B, 1024, 256) -> (B, 32, 32, 16, 16, 1): split the patch axis into row/col and
        # the pixel axis into row/col inside the patch
        x = x.reshape(x.shape[0], h, h, p, p, 1)
        # nhwpqc -> nchpwq is the inverse of patchify's reorder
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], 1, h * p, h * p)

    @torch.no_grad()
    def reconstruct(self, imgs):
        """One masked pass, returned as pictures instead of a loss.

        imgs: (B, 3, 512, 512) normalized, the same tensor training uses.
        Returns four (B, 1, 512, 512) tensors:
          original  - the ground truth, channel 0
          masked    - what the encoder actually saw (hidden patches blanked to grey)
          recon     - the decoder's prediction for EVERY patch
          paste     - visible patches kept as-is, hidden ones filled in from recon.
                      This is the one to look at: it is the model's completed image.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)    # (B, 1024, 256)
        target = self.patchify(imgs[:, :1])                 # (B, 1024, 256)
        if self.norm_pix:
            # training asked for the per-patch NORMALIZED pattern, so the prediction has
            # no brightness or contrast of its own. Put the target's back, or every patch
            # renders as grey noise.
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            pred = pred * (var + 1e-6) ** 0.5 + mu
        recon = self.unpatchify(pred)
        # mask is 1 where HIDDEN, so 1-mask is 1 where the encoder could see. Broadcast one
        # flag per patch out to all 256 of its pixels, then unpatchify to a pixel mask.
        vis = self.unpatchify((1.0 - mask).unsqueeze(-1).expand(-1, -1, self.patch ** 2))
        orig = imgs[:, :1]
        return orig, orig * vis, recon, orig * vis + recon * (1.0 - vis)

    def random_masking(self, x):
        """Pick a random 256 of the 1024 patches to keep, hide the other 768.

        Returns the kept patches, a 0/1 flag saying which ones were hidden, and
        the instructions for putting everything back in order later.
        """
        B, L, D = x.shape                       # x: (B, 1024, 768) - B images, 1024 patches
        keep = int(L * (1 - self.mask_ratio))   # each described by 768 numbers. keep = 256

        # How to choose 256 patches at random: give every patch a random number, sort the
        # patches by that number, and take the first 256. Sorting is how you do a random
        # choice on a GPU without writing a loop.
        rand_val = torch.rand(B, L, device=x.device)   # (B, 1024) one random float per patch
        shuffled = torch.argsort(rand_val, dim=1)      # (B, 1024) patch NUMBERS in random order
        ids_restore = torch.argsort(shuffled, dim=1)   # (B, 1024) argsort twice = the inverse:
                                                       # ids_restore[b, i] = where patch i went
        ids_keep = shuffled[:, :keep]                  # (B, 256) the patch numbers we keep

        # torch.gather(input, dim, index) picks values by index ALONG ONE AXIS. Here it
        # reads out[b, i, d] = x[b, index[b, i, d], d], i.e. row `index` of the patch axis.
        # It returns something the same shape as `index`, so `index` has to be 3-D like x.
        # ids_keep is only (B, 256), so:
        #   ids_keep[:, :, None]      -> (B, 256, 1)    add a trailing axis
        #   .expand(-1, -1, D)        -> (B, 256, 768)  repeat that patch number across all
        #                                768 feature slots. -1 means "leave this axis alone",
        #                                and expand is a free view - it copies no memory.
        x = torch.gather(x, 1, ids_keep[:, :, None].expand(-1, -1, D))   # -> (B, 256, 768)
        # The other 768 patches stop existing right here. That is what makes the encoder
        # 4x cheaper: it never sees them at all.

        # Now record WHICH patches were hidden, so the loss can score only those. Easiest
        # to build in shuffled order: the first 256 are the kept ones, the rest are hidden.
        mask = torch.ones(B, L, device=x.device)   # (B, 1024) start with 1 = hidden
        mask[:, :keep] = 0                         # first 256 of the SHUFFLED order = kept
        # same gather, but mask is 2-D so index needs no expand: out[b,i] = mask[b, ids_restore[b,i]]
        mask = torch.gather(mask, 1, ids_restore)  # (B, 1024) now mask[b, i] is about patch i
        return x, mask, ids_restore

    def forward_encoder(self, imgs):
        e = self.encoder
        x = e.patch_embed(imgs)        # (B, 3, 512, 512) -> (B, 1024, 768), one row per patch
        # Add "which position in the image am I" to every patch. This MUST happen before
        # we throw patches away, or the survivors would have no idea where they came from.
        # pos_embed is (1, 1025, 768); slot 0 is CLS, so patches are slots 1..1024.
        x = x + e.pos_embed[:, 1:, :]                     # (B, 1024, 768) + (1, 1024, 768)
        x, mask, ids_restore = self.random_masking(x)     # -> x (B, 256, 768)
        # CLS is a summary token, never hidden. cls_token is (1, 1, 768) for the whole
        # batch, so expand repeats it to (B, 1, 768) - a view, no memory copied.
        cls = (e.cls_token + e.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)   # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)   # (B, 257, 768) = 1 CLS + 256 surviving patches
        for blk in e.blocks:             # MRI-CORE's 12 layers, on 257 tokens not 1025
            x = blk(x)
        return e.norm(x), mask, ids_restore    # (B, 257, 768), (B, 1024), (B, 1024)

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)   # (B, 257, 768) -> (B, 257, 512), the decoder is narrower
        B, L = ids_restore.shape    # L = 1024, the full patch count we have to rebuild

        # The decoder needs all 1024 slots back, but only 256 hold real content. Fill the
        # rest with copies of ONE learned "something was here" vector, self.mask_token,
        # which is (1, 1, 512). expand stretches it to (B, 768, 512) without copying:
        # every one of those 768 slots is the same vector, viewed many times.
        pad = self.mask_token.expand(B, L + 1 - x.shape[1], -1)   # (B, 768, 512)
        y = torch.cat([x[:, 1:, :], pad], dim=1)   # (B, 1024, 512) = 256 real + 768 blanks
                                                   # x[:, 1:, :] drops CLS from the front

        # Those 256 real ones are bunched at the front in SHUFFLED order. Same gather as
        # in random_masking: out[b, i, d] = y[b, ids_restore[b, i, d], d], and ids_restore
        # is (B, 1024) so it needs [:, :, None].expand(-1, -1, 512) to match y's 3 axes.
        # Result: slot i holds patch i of the original image again.
        y = torch.gather(y, 1, ids_restore[:, :, None].expand(-1, -1, y.shape[2]))

        # Only NOW add positions. This is what tells each blank which part of the image it
        # is meant to reconstruct - before this line every blank is the identical vector.
        y = torch.cat([x[:, :1, :], y], dim=1) + self.decoder_pos_embed   # (B, 1025, 512)
        for blk in self.decoder_blocks:          # 4 fresh layers, deleted after pretraining
            y = blk(y)
        # (B, 1025, 512) -> (B, 1025, 256): each slot predicts its patch's 16x16 pixels.
        # [:, 1:, :] drops CLS again, leaving one prediction per real patch.
        return self.decoder_pred(self.decoder_norm(y))[:, 1:, :]   # (B, 1024, 256)

    def forward(self, imgs):
        """imgs: (B, 3, H, W) normalized. Loss on masked patches only."""
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)   # predicted pixels, (B, 1024, 256)
        # What the answer SHOULD be: the real image, cut into the same 1024 patches.
        # Channel 0 only, because our 3 channels are one grey image copied 3 times.
        target = self.patchify(imgs[:, :1])
        if self.norm_pix:                        # paper Table 1d: 85.4 vs 84.9 fine-tune
            # Rescale each patch by its own brightness and contrast, so the model is asked
            # to predict the PATTERN inside a patch, not how bright that patch happens
            # to be. Brightness is a scanner setting; the pattern is the anatomy.
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5
        loss = ((pred - target) ** 2).mean(dim=-1)   # squared error per patch -> (B, 1024)
        # Multiplying by mask zeroes every patch the encoder was allowed to see. We score
        # only the ones it had to guess, so it cannot win by copying what was in front of it.
        # mask.sum() is B * 768
        return (loss * mask).sum() / mask.sum().clamp(min=1)


def _sincos_pos_embed(dim, grid, cls_token=True):
    """A fixed table saying "this is patch (row, col)", one vector per patch.

    WHY THIS EXISTS. Attention has no idea where anything is - shuffle the patches and it
    gives the same answer. So every patch gets a vector added to it encoding its position.
    The encoder LEARNS its own (that is MRI-CORE's pos_embed). The decoder does not: this
    table is computed once from a formula and frozen, which is what the MAE paper does.
    Fixed is fine here because the decoder is thrown away, so there is nothing to gain
    from letting it learn the layout.

    WHY SIN AND COS. You could number the patches 0, 1, 2, ... but a plain number is a bad
    input to a network: it grows without bound and 31 vs 32 looks nearly the same as 3 vs
    32. Instead each position is written as a set of sine and cosine waves at many
    different frequencies - fast waves that distinguish neighbours, slow waves that
    distinguish far-apart patches. Every value stays in [-1, 1], and every position gets
    a unique combination.

    Standard ViT: pos_embed is learnable, trained via backprop.
    Here, the decoder specifically uses fixed sin-cos instead (as MAE paper does), 
    because it's discarded after pretraining. 
    The encoder (MRI-CORE) still uses learnable pos_embed, like normal ViT.
    
    Returns (grid*grid [+1], dim), e.g. (1025, 512) for our 32x32 grid plus the CLS slot.
    """
    def _1d(d, pos):
        """Encode ONE coordinate (all the row numbers, or all the column numbers).

        pos: any shape of position numbers. Returns (n_positions, d).
        """
        # d//2 frequencies, spread geometrically from 1 down to 1/10000. The first waves
        # cycle every ~6 patches, the last ones barely change across the whole image.
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        # every position x every frequency -> (n_positions, d//2) of angles
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        # take sin AND cos of each angle, giving d numbers per position. Both are needed:
        # sin alone is ambiguous (sin(x) = sin(pi - x)), the pair pins the angle down.
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    g = np.arange(grid, dtype=np.float32)          # 0..31, the coordinates along one side
    # meshgrid builds the full 32x32 board: gy holds every patch's ROW number, gx its
    # COLUMN number. indexing="ij" means row-major, matching how patchify flattened them.
    gy, gx = np.meshgrid(g, g, indexing="ij")
    # encode the two coordinates separately with half the budget each, then glue them
    # side by side -> (1024, dim). First half of every vector says "which column",
    # second half says "which row".
    pe = np.concatenate([_1d(dim // 2, gx), _1d(dim // 2, gy)], axis=1)
    if cls_token:
        # CLS is a summary of the whole image and belongs to no position, so its slot is
        # all zeros. It sits at index 0 to match the encoder's pos_embed layout.
        pe = np.concatenate([np.zeros([1, dim]), pe], axis=0)
    return pe


def save_mae_debug(model, x_prep, cfg: dict, out_dir, epoch: int, step: int, rows: int = 4):
    """original | masked | reconstruction | paste, one row per slice.

    This is the check that actually matters. If the paste column looks like a knee, the
    encoder is learning anatomy; if it looks like uniform blur, it is predicting local
    averages and the pretraining is not buying anything.
    """
    cv2 = sc.cv2
    if cv2 is None:
        return
    was_training = model.training
    model.eval()
    n = min(rows, x_prep.shape[0])
    orig, masked, recon, paste = model.reconstruct(x_prep[:n])

    # undo the ImageNet normalization so these are viewable 0..255 greys again
    mean = float(cfg["input"]["norm_mean"][0])
    std = float(cfg["input"]["norm_std"][0])

    def _u8(t):
        v = t.detach().float().cpu()[:, 0] * std + mean
        return (v.clamp(0, 1) * 255).to(torch.uint8).numpy()

    cols = [_u8(orig), _u8(masked), _u8(recon), _u8(paste)]
    h, w = cols[0].shape[1:]
    sheet = np.zeros((n * h, len(cols) * w), np.uint8)
    for r in range(n):
        for c, col in enumerate(cols):
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = col[r]

    sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    for c, name in enumerate(("original", f"masked {cfg['mae']['mask_ratio']:.0%}",
                              "reconstruction", "paste (vis + predicted)")):
        cv2.putText(sheet, name, (c * w + 6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1, cv2.LINE_AA)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"mae_e{epoch}_s{step}.png"
    cv2.imwrite(str(path), sheet)
    print(f"  🖼️  MAE reconstruction -> {path}", flush=True)
    if was_training:
        model.train()


def build_model(cfg: dict) -> nn.Module:
    model = MAE(cfg)
    path = _weights_path(cfg)
    _load_mri_core(model.encoder, path, model.grid)
    return model


def _weights_path(cfg: dict) -> Path:
    name = cfg["model"]["weights"]
    for c in (Path(cfg["modal"]["runs_mount"]) / "_pretrained" / name,
              Path(cfg["paths"]["data_dir"]) / "pretrained" / name):
        if c.exists():
            return c
    raise FileNotFoundError(f"{name} not found in /_pretrained or data/pretrained/")


def _load_mri_core(backbone: nn.Module, path: Path, grid: int):
    """MRI-CORE's DINO checkpoint -> a timm VisionTransformer. Same as the relabel run."""
    from timm.layers import resample_abs_pos_embed
    sd = torch.load(path, map_location="cpu", weights_only=False)["teacher"]
    out = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        k = k[len("backbone."):]
        m = re.match(r"blocks\.\d+\.(\d+)\.(.*)", k)
        if m:
            k = f"blocks.{m.group(1)}.{m.group(2)}"
        out[k] = v
    out.pop("mask_token", None)
    out["pos_embed"] = resample_abs_pos_embed(out["pos_embed"], new_size=[grid, grid],
                                              num_prefix_tokens=1)
    r = backbone.load_state_dict(out, strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError(f"MRI-CORE load mismatch - missing={r.missing_keys} "
                           f"unexpected={r.unexpected_keys}")
    print(f"  ✅ MRI-CORE loaded from {path} ({len(out)} tensors, pos_embed -> {grid}x{grid})")


def save_encoder(model: MAE, path: Path, cfg: dict):
    """Encoder only, wrapped in MRI-CORE's {"teacher": {"backbone.*"}} layout.

    That way the existing _load_mri_core reads it unchanged and the finetune run needs
    only a one-line config edit. Its block regex leaves plain names alone.
    """
    sd = {"backbone." + k: v.cpu() for k, v in model.encoder.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": sd, "mae": {"epochs": cfg["training"]["epochs"],
                                       "mask_ratio": cfg["mae"]["mask_ratio"]}}, path)
    print(f"  \U0001F4BE encoder -> {path} ({len(sd)} tensors)")


# =============================================================================
# training loop: no labels, no AUROC, so the engine's supervised loop does not apply
# =============================================================================
def run_training(cfg: dict, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = Path(out_dir) / cfg["output"]["run_dir"]
    ckpt_dir = res / cfg["output"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    tr_df = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg))
    # the 704 VAL studies are the holdout. We never train on them, so reading a
    # reconstruction loss there is free and leaks nothing - and it keeps the whole
    # train split available for pretraining.
    ho_df = pd.read_csv(sc.resolve(cfg["paths"]["val_csv"], cfg))

    ds = MAESliceDataset(tr_df, cfg, series, train=True)
    ho = MAESliceDataset(ho_df, cfg, series, train=False)
    dl = cfg["dataloader"]
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=dl["batch_size"], shuffle=True,
                        num_workers=dl["num_workers"], pin_memory=True,
                        persistent_workers=dl.get("persistent_workers", True),
                        prefetch_factor=dl.get("prefetch_factor", 4), drop_last=True)
    ho_loader = DataLoader(ho, batch_size=dl["batch_size"], shuffle=False,
                           num_workers=dl.get("val_num_workers") or dl["num_workers"])

    model = build_model(cfg).to(device)
    if cfg["training"].get("channels_last"):
        model = model.to(memory_format=torch.channels_last)

    t = cfg["training"]
    accum = dl.get("accum_steps", 1)
    k = cfg["mae"]["slices_per_series"]
    imgs_per_step = dl["batch_size"] * accum * k
    # paper A.1: lr = base_lr * batchsize / 256
    scale = imgs_per_step / 256.0
    lr_dec = float(t["optimizer"]["lr"]) * scale
    lr_enc = lr_dec * float(t["optimizer"]["encoder_lr_mult"])
    opt = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": lr_enc},
         {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
          "lr": lr_dec}],
        betas=tuple(t["optimizer"]["betas"]), weight_decay=t["optimizer"]["weight_decay"])

    epochs = int(t["epochs"])
    steps_per_epoch = max(1, len(loader) // accum)
    total = epochs * steps_per_epoch
    warm = max(1, int(total * float(t["scheduler"]["warmup_frac"])))
    lo = float(t["scheduler"]["min_lr_mult"])

    def lr_at(step):
        if step < warm:
            return step / warm
        prog = (step - warm) / max(1, total - warm)
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"])
    log = sc.CSVLogger(res / "mae_log.csv")
    dbg_dir = res / "debug_images"
    gstep, start_epoch = 0, 1

    last = ckpt_dir / "last.pt"
    best_path = ckpt_dir / cfg["output"].get("best_name", "best.pt")
    best_loss = [float("inf")]          # lower reconstruction loss is better

    def _save(path, epoch, complete, note=None):
        """One writer for every checkpoint here, so they all carry the same payload.

        sc._write_checkpoint is the engine's atomic writer: torch.save to a .tmp then
        rename, so a kill mid-write leaves the PREVIOUS checkpoint intact instead of a
        truncated file. Not sc.save_checkpoint, whose payload has no slot for a
        reconstruction loss - writing 1 GB twice to add one float is not worth it.
        """
        sc._write_checkpoint(
            {"model": model.state_dict(), "optimizer": opt.state_dict(),
             "scaler": scaler.state_dict(), "cfg": cfg, "epoch": epoch, "step": gstep,
             "epoch_complete": complete, "score_loss": best_loss[0],
             "elapsed_sec": round(time.time() - t0, 1)}, path)
        if note:
            print(note, flush=True)

    # newest usable checkpoint, whatever it is called: last.pt if the epoch finished,
    # otherwise the highest stepNNNN.pt. _resolve_resume picks by step, so a run killed
    # mid-epoch comes back from the periodic checkpoint rather than from the epoch start.
    _r = sc._resolve_resume("last", ckpt_dir)
    if _r is not None:
        ck = torch.load(_r, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        gstep = ck["step"]
        best_loss[0] = ck.get("score_loss", float("inf"))
        # epoch_complete=False marks a mid-epoch checkpoint: replay that epoch rather
        # than skipping the rest of its data
        start_epoch = ck["epoch"] + 1 if ck.get("epoch_complete", True) else ck["epoch"]
        print(f"  ↻ resuming from {_r.name} (epoch {ck['epoch']}, "
              f"step {gstep}, best holdout {best_loss[0]:.4f})")

    sc._banner("MAE PRETRAINING", "\U0001F3AD")
    print(f"  {ds.n_series} train series x{ds.reps} reps x{k} slices "
          f"= {len(ds) * k:,} slice-views/epoch")
    print(f"  {len(ho.index) * k:,} held-out | mask {cfg['mae']['mask_ratio']:.0%} | "
          f"decoder {cfg['mae']['decoder_depth']}x{cfg['mae']['decoder_dim']}")
    print(f"  lr: encoder {lr_enc:.2e}  decoder {lr_dec:.2e}  "
          f"(base {t['optimizer']['lr']:.1e} x {scale:.1f}), warmup {warm}/{total}")

    # ---- instrumentation state, same meanings as the engine's supervised loop ----
    retr_prev = (0, 0, 0, 0.0)      # last (imgs, bytes, ns, elapsed) shard-read snapshot
    flops_per_img = [None]          # counted once on a real batch, then reused
    last_print = [None, None]       # (elapsed, gstep) of the previous console line
    _peak = sc._peak_tflops(torch.cuda.get_device_name(0)) if device.type == "cuda" else 0.0
    prev_loss = None
    _params = [q for g in opt.param_groups for q in g["params"]]

    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        sc._EPOCH.value = epoch
        model.train()
        run_loss, seen = 0.0, 0
        # two throughput numbers so a slow shard volume cannot hide behind a fast GPU:
        #   compute = images / GPU time only, real = images / wall time incl. the data wait.
        # next() is timed by hand rather than using `for x in loader`, so the time spent
        # in the holdout pass and the checkpoint save is never miscounted as waiting on data.
        roll = deque(maxlen=20)          # (imgs, wall, data_wait) for the last 20 steps
        it, _lit = 0, iter(loader)
        while True:
            _s_t0 = time.time()
            _w0 = time.time()
            try:
                x = next(_lit)
            except StopIteration:
                break
            _wait = time.time() - _w0    # how long this step sat idle waiting for data
            it += 1

            with torch.autocast("cuda", enabled=t["amp"]):
                loss = model(sc.prepare_batch(x, cfg, device))
            scaler.scale(loss / accum).backward()

            _do_print = (it % (cfg["output"]["console_log_every"] * accum) == 0) or gstep == 0
            gnorm = float("nan")
            if it % accum == 0:
                # paper A.1: no gradient clipping. We still MEASURE the norm, but only on
                # print steps - unscale_ plus a pass over every gradient is not free.
                # scaler.step below detects the already-unscaled state and does not redo it.
                if _do_print:
                    scaler.unscale_(opt)
                    # max_norm=inf means "report the norm, clip nothing"
                    gnorm = float(torch.nn.utils.clip_grad_norm_(_params, float("inf")))
                for g, base in zip(opt.param_groups, (lr_enc, lr_dec)):
                    g["lr"] = base * lr_at(gstep)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                gstep += 1
            # .detach() first: float() on a tensor still attached to the graph makes
            # torch warn, and holding the graph alive for a running total is pointless
            run_loss += loss.detach().item() * x.shape[0]; seen += x.shape[0]
            roll.append((x.shape[0] * k, time.time() - _s_t0, _wait))

            if _do_print:
                el = time.time() - t0
                l = run_loss / max(seen, 1)
                eta = el * (total / max(gstep, 1) - 1)
                mem = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
                # compute = wall minus the data wait; real = raw wall. Both over the last
                # 20 steps, so a stall shows up immediately instead of being averaged away.
                _ri_ = sum(r[0] for r in roll)
                _rw_ = sum(r[1] for r in roll)
                _rd_ = sum(r[2] for r in roll)
                real_ips = _ri_ / max(_rw_, 1e-9)
                ips = _ri_ / max(_rw_ - _rd_, 1e-9)
                wait_pct = 100.0 * _rd_ / max(_rw_, 1e-9)
                overall_ips = gstep * imgs_per_step / max(el, 1e-9)

                d = "" if prev_loss is None else f" ({l - prev_loss:+.4f})"
                # wall time since the previous console line, so the cost of a window is
                # readable without differencing two elapsed= values. Nothing to difference
                # against on this process's FIRST line, and elapsed survives a resume while
                # last_print does not, so subtracting a zero there would report the whole
                # prior run as one window.
                _since = ""
                if last_print[0] is not None and gstep > last_print[1]:
                    _since = f"  +{sc.fmt(el - last_print[0])}/{gstep - last_print[1]}st"
                last_print[0], last_print[1] = el, gstep

                print("\u2500" * 70)
                print(f"  Epoch {epoch}/{epochs}  epoch_step: {it // accum}/{steps_per_epoch}"
                      f"  global_step: {gstep}/{total}")
                _gs = "" if gnorm != gnorm else f"  gnorm={gnorm:.2f} (unclipped)"
                print(f"     loss={l:.4f}{d}  lr[decoder={opt.param_groups[1]['lr']:.3e} "
                      f"encoder={opt.param_groups[0]['lr']:.3e}]{_gs}  "
                      f"elapsed={sc.fmt(el)}  ETA={sc.fmt(eta)}  "
                      f"gpu_mem(peak)={mem:.2f}GB{_since}")
                print()                      # split what the model is doing from how fast
                print(f"     throughput: {ips:.0f} img/s compute | {real_ips:.0f} img/s real "
                      f"({len(roll)}-step) | data-wait {wait_pct:.0f}%")
                _eff = 100.0 * overall_ips / max(real_ips, 1e-9)
                print(f"     \u27a4 END-TO-END: {overall_ips:.0f} img/s over the whole "
                      f"run ({_eff:.0f}% of in-loop) - {gstep * imgs_per_step / 1e6:.2f}M "
                      f"images in {sc.fmt(el)}")
                # slices/s and MB/s off the shard volume, summed over every worker, for the
                # window since the last print. ms/img is per-slice read latency.
                _ri, _rb, _rns = sc._retr_snapshot()
                _pi, _pb, _pns, _pel = retr_prev
                d_i, d_b, d_ns = _ri - _pi, _rb - _pb, _rns - _pns
                d_el = max(el - _pel, 1e-6)
                retr_prev = (_ri, _rb, _rns, el)
                if d_i > 0:
                    print(f"     retrieval:  {d_i / d_el:.0f} img/s | "
                          f"{d_b / 1e6 / d_el:.0f} MB/s | "
                          f"{d_ns / 1e6 / d_i:.1f} ms/img avg read (shards)")
                # Is the GPU actually WORKING, or just occupied? img/s says how fast this
                # model runs; MFU says how much of the card that speed uses, so a number
                # that looks fine can still be leaving 80% on the table.
                if device.type == "cuda":
                    if flops_per_img[0] is None:
                        flops_per_img[0] = sc._measure_flops_per_image(model, x, cfg, device)
                    _sm, _mc = sc._gpu_busy()
                    _bits = []
                    # x3: backward costs roughly twice forward, so fwd+bwd ~ 3x fwd
                    _ach = 3.0 * (flops_per_img[0] or 0.0) * ips / 1e12
                    if flops_per_img[0] and _peak:
                        _bits.append(f"MFU {100.0 * _ach / _peak:.0f}% "
                                     f"({_ach:.0f} of {_peak:.0f} TFLOP/s fp16/bf16)")
                    elif flops_per_img[0]:
                        _bits.append(f"{_ach:.0f} TFLOP/s (no peak figure for this card)")
                    if _sm is not None:
                        _bits.append(f"SM busy {_sm:.0f}%")
                    if _mc is not None:
                        _bits.append(f"mem ctrl busy {_mc:.0f}%")
                    if _bits:
                        print("     compute:    " + " | ".join(_bits))
                print("", end="", flush=True)
                prev_loss = l

            # what the encoder actually receives, plus what it reconstructs from it.
            # MAE has no AUROC, so a broken input pipeline would otherwise be invisible:
            # the loss would still fall and nothing would look wrong.
            _dbg = int(cfg["output"].get("save_debug_images") or 0)
            if _dbg and (gstep % _dbg == 0 or gstep == 1) and it % accum == 0:
                _xp = sc.prepare_batch(x, cfg, device)
                sc.save_debug_batch(_xp, cfg, dbg_dir, epoch, gstep)
                save_mae_debug(model, _xp, cfg, dbg_dir, epoch, gstep)
                del _xp

            # rolling mid-epoch checkpoint, so a kill between epoch boundaries costs
            # minutes instead of a whole epoch
            _ce = int(cfg["output"].get("checkpoint_every_steps") or 0)
            if _ce and it % accum == 0 and gstep % _ce == 0:
                _keep = int(cfg["output"].get("keep_last_n", 3))
                _save(ckpt_dir / f"step{gstep}.pt", epoch, False,
                      note=f"  💾 periodic checkpoint @ step {gstep} "
                           f"(rolling, keep_last_n={_keep})")
                sc.prune_checkpoints(ckpt_dir, _keep)

        model.eval(); hl, hn = 0.0, 0
        with torch.no_grad():
            for x in ho_loader:
                with torch.autocast("cuda", enabled=t["amp"]):
                    hl += float(model(sc.prepare_batch(x, cfg, device))) * x.shape[0]
                hn += x.shape[0]
        print(f"  ✓ epoch {epoch}: train {run_loss / max(seen, 1):.4f}  "
              f"holdout {hl / max(hn, 1):.4f}")
        log.log({"epoch": epoch, "step": gstep, "train_loss": run_loss / max(seen, 1),
                   "holdout_loss": hl / max(hn, 1), "lr": opt.param_groups[1]["lr"],
                   "elapsed_s": round(time.time() - t0, 1)})
        _hl = hl / max(hn, 1)
        # decide BEFORE writing last.pt, so last.pt records the current best and not the
        # previous one - that field is what a resume reads back
        _improved = _hl < best_loss[0]
        _prev = best_loss[0]
        if _improved:
            best_loss[0] = _hl
        _save(last, epoch, True)
        # the LAST epoch is not automatically the best one, so keep the best separately.
        # The encoder export tracks best.pt, since that is the trunk we would fine-tune.
        if _improved:
            _save(best_path, epoch, True)
            save_encoder(model, res / cfg["mae"]["encoder_out"], cfg)
            print(f"  🏆 new best holdout {_hl:.4f}"
                  + ("" if _prev == float("inf") else f" (was {_prev:.4f})"))

    print(f"\n  done in {sc.fmt(time.time() - t0)}")


def main():
    cfg = sc.load_config(EXP_DIR)
    run_training(cfg, EXP_DIR)


try:
    import modal

    cfg = sc.load_config(EXP_DIR, verbose=False)
    app = modal.App(f"{EXP_NAME}-train")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={cfg["modal"]["shards_mount"]: _shards,
                           cfg["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(cfg))
    def train_remote():
        rcfg = sc.remote_cfg(cfg)
        out = Path(cfg["modal"]["runs_mount"]) / EXP_NAME
        try:
            run_training(rcfg, out)
        finally:
            _runs.commit()

    @app.local_entrypoint()
    def modal_main():
        try:
            train_remote.remote()
        finally:
            sc.sync_back(cfg, "mae-pretrained")

except ImportError:
    pass


if __name__ == "__main__":
    main()
