"""
train.py — qwen35_2b_vlm_sys

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_25d_18slices_v5_aug_base/train.py
Modal:  modal run runs/convnext_base_25d_18slices_v5_aug_base/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "qwen35_2b_vlm_sys"


def _pkg_root() -> Path:
    """Find the runs/ folder holding shared_code.py — /root/runs on Modal, ../ locally."""
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))          # so `import shared_code` works from either location
import shared_code as sc  # noqa: E402

cfg = sc.load_config(EXP_DIR, verbose=False)   # shared_config.yaml + this folder's config.yaml


VLM_ID = "Qwen/Qwen3.5-2B"        # Apache-2.0, ungated (HF card checked 6 Oct)


def _patchify(x, t=2, p=16, m=2):
    """Engine images (N, 3, H, W), already normalised to Qwen's (x - 0.5) / 0.5 -> Qwen2-VL vision patches.

    Byte-for-byte the flatten of transformers' Qwen2VLImageProcessor for one still image: the frame is repeated
    `t` times along time, then cut into p x p patches ordered in m x m merge blocks.
    -> pixel_values (N * gh * gw, 3 * t * p * p), grid_thw (N, 3)
    """
    n, c, h, w = x.shape
    gh, gw = h // p, w // p
    v = x[:, None].expand(n, t, c, h, w)                                      # (N, T, C, H, W)
    v = v.reshape(n, 1, t, c, gh // m, m, p, gw // m, m, p)
    v = v.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)                               # N, gt, gh/m, gw/m, m, m, C, T, p, p
    pv = v.reshape(n * gh * gw, c * t * p * p)
    grid = torch.tensor([[1, gh, gw]], device=x.device).expand(n, 3)
    return pv, grid


class VLMModel(nn.Module):
    """Qwen3.5-2B (vision encoder + language model) reading one knee study -> 12 logits.

    A study is 4 views x 6 images (3 neighbouring slices as RGB, the 2.5D parent's input) at 384 px = 24 images,
    144 visual tokens each after the 2x2 merge, wrapped in a FIXED chat prompt (input_ids built once with the
    processor and stored as a buffer, so inference needs no tokenizer). The language model reads all visual tokens
    with LoRA adapters (q/k/v/o, linear-attention in/out, MLP); the vision encoder trains in full at its own lr
    (backbone group). Head: [last-token state, mean of the visual-token states] -> LayerNorm -> Linear(12).
    No text is generated. self.backbone holds the vision encoder so the engine's lr groups treat it as the trunk.
    """

    def __init__(self, cfg: dict, pretrained: bool = True):
        super().__init__()
        import transformers as tf
        v = cfg["model"]["vlm"]
        i = cfg["input"]
        self.n_img = len(i["slots"]) * int(i["slices_as_channels"])
        src = v.get("local_dir") or VLM_ID
        if pretrained:
            full = tf.Qwen3_5ForConditionalGeneration.from_pretrained(src, dtype=torch.float32)
        else:                                               # inference: config only, the checkpoint fills the weights
            full = tf.Qwen3_5ForConditionalGeneration(tf.AutoConfig.from_pretrained(src))
        core = full.model                                   # visual + language_model; lm_head (tied) dropped
        self.backbone = core.visual                         # the trunk for the engine (backbone lr group)
        lm = core.language_model
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=int(v["lora_r"]), lora_alpha=int(v["lora_alpha"]), lora_dropout=float(v["lora_dropout"]),
                        target_modules=list(v["lora_targets"]), bias="none")
        self.lm = get_peft_model(lm, lc)
        for n_, p_ in self.lm.named_parameters():
            p_.requires_grad = "lora_" in n_
        self._core = [core]                                 # rope-index helper, not a registered child
        self.image_token_id = int(full.config.image_token_id)
        if v.get("grad_checkpointing"):
            self.backbone.gradient_checkpointing_enable() if hasattr(self.backbone, "gradient_checkpointing_enable") else None
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            lm.config.use_cache = False
        hid = int(full.config.text_config.hidden_size)
        self.register_buffer("input_ids", torch.zeros(1, int(v["n_tokens"]), dtype=torch.long))
        self.register_buffer("mm_token_type_ids", torch.zeros(1, int(v["n_tokens"]), dtype=torch.long))
        self.feat_norm = nn.LayerNorm(2 * hid)
        self.head = nn.Linear(2 * hid, sc.num_output_logits(cfg))
        del full

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Restore ONLY the trained tensors, verbatim (training resume and the Kaggle inference path alike)."""
        # FROZEN weights (the language model outside LoRA) are never changed by training, and 18 of them do not
        # survive a state_dict round trip exactly (knee-vlm-smoke v7: linear_attn.norm.weight 0.9160 saved -> 0.9157
        # reloaded). So they always stay as built from Qwen's released weights (build with pretrained=True), and only
        # what training changed is restored: vision encoder, LoRA, feat_norm, head, prompt buffers.
        saved = set(super().state_dict().keys())            # persistent tensors only (rotary inv_freq is rebuilt, never saved)
        trainable = ({n for n, p in self.named_parameters() if p.requires_grad}
                     | {n for n, _ in self.named_buffers() if n in saved})
        sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}
        keep = {k: v for k, v in sd.items() if k in trainable}
        missing = sorted(trainable - set(keep))
        if strict and missing:
            raise RuntimeError(f"checkpoint lacks {len(missing)} trained tensors, e.g. {missing[:3]}")
        own = dict(self.named_parameters())
        own.update(dict(self.named_buffers()))
        with torch.no_grad():
            for k, v in keep.items():
                own[k].copy_(v.to(own[k].dtype))
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def build_prompt(self, cfg):
        """Once, at training start: the chat prompt with n_img image slots, tokenised by the real processor."""
        from PIL import Image
        import transformers as tf
        v = cfg["model"]["vlm"]
        proc = tf.AutoProcessor.from_pretrained(v.get("local_dir") or VLM_ID)
        s_ = int(cfg["input"]["size"])
        imgs = [Image.new("RGB", (s_, s_)) for _ in range(self.n_img)]
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(self.n_img)] + [{"type": "text", "text": v["prompt"]}]}]
        if v.get("system_prompt"):
            msgs = [{"role": "system", "content": [{"type": "text", "text": v["system_prompt"]}]}] + msgs
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = proc(text=[text], images=imgs, return_tensors="pt")
        ids = enc["input_ids"]
        assert ids.shape[1] == self.input_ids.shape[1], f"prompt has {ids.shape[1]} tokens, config n_tokens says {self.input_ids.shape[1]}"
        self.input_ids.copy_(ids)
        self.mm_token_type_ids.copy_(enc["mm_token_type_ids"])
        return enc

    def forward(self, x):                                    # x: (B * n_img, 3, H, W), normalised (x - 0.5) / 0.5
        b = x.shape[0] // self.n_img
        pv, grid = _patchify(x)
        ids = self.input_ids.expand(b, -1)
        # transformers' own Qwen3.5 forward: vision encoder, image-token scatter and multimodal rope positions exactly as
        # the released model; its language model carries our LoRA layers (peft injects them in place)
        out = self._core[0](input_ids=ids, pixel_values=pv, image_grid_thw=grid,
                            mm_token_type_ids=self.mm_token_type_ids.expand(b, -1), use_cache=False)
        hs = out.last_hidden_state                          # (B, L, hid)
        last = hs[:, -1]
        m = (ids == self.image_token_id)[..., None].to(hs.dtype)
        vmean = (hs * m).sum(1) / m.sum(1)
        return self.head(self.feat_norm(torch.cat([last, vmean], -1).float()))


def build_model(cfg: dict) -> nn.Module:
    """ALWAYS Qwen's released weights (Modal / this PC: the hub; Kaggle: model.vlm.local_dir, a folder the bundle
    attaches). No config-only fallback: the frozen language-model weights are never in our checkpoints (see
    VLMModel.load_state_dict), so a model without them would silently score with random layers. Fails loudly instead."""
    model = VLMModel(cfg, pretrained=True)
    model.build_prompt(cfg)
    print("  🧠 Qwen3.5-2B loaded: prompt %d tokens, %d images/study" % (model.input_ids.shape[1], model.n_img), flush=True)
    return model


def run_local():
    """Train on this PC's GPU — only useful for smoke tests, real runs go to Modal."""
    sc.set_seed(cfg["reproducibility"]["seed"], cfg["reproducibility"]["deterministic"])
    sc.print_config(cfg)
    sc.run_experiment(cfg, build_model(cfg), EXP_DIR)


try:
    # everything below is skipped when modal isn't installed, so local runs still work
    import modal

    app = modal.App(cfg["experiment"]["name"])
    # flash-linear-attention: the fast (Triton) kernels for the 18 linear-attention layers; without it transformers silently
    # falls back to a slow pure-torch chunk loop (the first launch ran ~1 study/s)
    _image = sc.modal_image(extra_pip=("transformers>=5.5", "peft>=0.18", "flash-linear-attention"))
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)  # input data
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)      # outputs
    _secrets = ([modal.Secret.from_name(cfg["modal"]["wandb_secret"])]
                if cfg.get("wandb", {}).get("enabled") else [])

    @app.function(image=_image,
                  volumes={cfg["modal"]["shards_mount"]: _shards,
                           cfg["modal"]["runs_mount"]: _runs},
                  secrets=_secrets, **sc.modal_resources(cfg))
    def train_remote(run_id: str = ""):
        """Runs inside the Modal container on the GPU named in config.yaml."""
        rcfg = sc.remote_cfg(cfg)                  # same config, container paths
        if run_id:
            rcfg["experiment"]["run_id"] = run_id  # allocated on the launching PC
        out_dir = Path(cfg["modal"]["runs_mount"]) / cfg["experiment"]["name"]
        sc.set_seed(rcfg["reproducibility"]["seed"], rcfg["reproducibility"]["deterministic"])
        sc.print_config(rcfg)
        model = build_model(rcfg)

        wb = cfg.get("wandb", {})
        run, log_fn = None, None
        if wb.get("enabled"):
            import wandb
            run = wandb.init(project=wb.get("project"), name=cfg["experiment"]["name"],
                             config=cfg, settings=wandb.Settings(x_disable_stats=True))
            log_fn = lambda d, s: run.log(d, step=s)
        try:
            sc.run_experiment(rcfg, model, out_dir, persist_fn=_runs.commit, log_fn=log_fn)
        finally:
            if run:
                run.finish()
            _runs.commit()                         # always save results, even if training crashed

    @app.local_entrypoint()                        # what `modal run train.py` calls
    def modal_main():
        sc.sync_on_interrupt(cfg)                  # ctrl+c still pulls the artifacts
        run_id = sc.launch_run_id(cfg, EXP_DIR)    # sequential, reserved here
        status = "crashed"
        try:
            train_remote.remote(run_id)
            status = "finished"
        finally:
            # the container can't reach this PC, so the pull happens here — and it
            # has to happen whether training finished or blew up
            sc.sync_back(cfg, status)

except ImportError:
    pass


if __name__ == "__main__":
    run_local()
