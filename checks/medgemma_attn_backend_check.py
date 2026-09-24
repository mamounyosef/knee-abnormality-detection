"""medgemma_attn_backend_check.py - does flex attention give the SAME P(yes) as sdpa?

Flex attention compiles a fused kernel for Gemma 3's own mask (causal, sliding window, AND
bidirectional attention inside each image). It is only a speed change if the answers are
identical, so this builds one tiny random Gemma 3 with images, loads the same weights under
both backends, and compares yes_prob_shared (prefix cache + batched questions) prompt by prompt.

    python checks/medgemma_attn_backend_check.py
"""
import sys
from pathlib import Path

import torch
from transformers import Gemma3Config, Gemma3ForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_code" / "label_v2"))
import medgemma_audit58 as mg  # noqa: E402

assert torch.cuda.is_available(), "flex attention needs the GPU"
torch.manual_seed(0)
IMG, BOI, EOI, BOS = 250, 251, 252, 2
cfg = Gemma3Config(
    text_config=dict(vocab_size=260, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, sliding_window=64, max_position_embeddings=4096,
                     layer_types=None),
    vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=2, image_size=112, patch_size=14),
    mm_tokens_per_image=16, image_token_index=IMG, boi_token_index=BOI, eoi_token_index=EOI)
base = Gemma3ForConditionalGeneration(cfg).eval()
sd = base.state_dict()


def build(attn):
    c = Gemma3Config.from_dict(cfg.to_dict())
    c._attn_implementation = attn
    mdl = Gemma3ForConditionalGeneration(c).eval()
    mdl.load_state_dict(sd)
    mdl.config._attn_implementation = attn
    mdl.set_attn_implementation(attn) if hasattr(mdl, "set_attn_implementation") else None
    return mdl.to("cuda", torch.float32)


n_img = 6
pix = torch.randn(n_img, 3, 112, 112)
ids = [BOS]
for _ in range(n_img):
    ids += [10, BOI] + [IMG] * 16 + [EOI, 11]
shared = torch.randint(30, 200, (40,)).tolist()
ids_list = [ids + shared + torch.randint(30, 200, (k,)).tolist() + [20, 21] for k in (5, 9, 3, 12)]
out = {}
for attn in ("sdpa", "flex_attention"):
    try:
        m = build(attn)
        p, _, L, chk = mg.yes_prob_shared(m, ids_list, pix, IMG, [100, 101], [102, 103], check=True)
        out[attn] = p
        print(f"  {attn:15s} P(yes) {[round(x, 6) for x in p]}  built-in full-pass check {chk[0]:.1e}")
    except Exception as e:                                    # noqa: BLE001
        print(f"  {attn:15s} FAILED: {type(e).__name__}: {str(e)[:300]}")
if len(out) == 2:
    d = max(abs(a - b) for a, b in zip(out["sdpa"], out["flex_attention"]))
    print(f"max |P(yes) sdpa - flex| = {d:.2e}  {'SAME' if d < 1e-4 else 'DIFFERENT'} "
          f"(prompt {len(ids_list[0])} tokens, {n_img} images, sliding window 64)")
    sys.exit(0 if d < 1e-4 else 1)
sys.exit(1)
