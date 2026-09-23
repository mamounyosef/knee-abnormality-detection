"""medgemma_prefix_cache_check.py - does "encode the images once, then run the 12 questions
as one batch on the shared cache" give the SAME answer as a full forward pass per question?

MedGemma 1.5 is a Gemma 3 model. This builds a tiny random-weight Gemma 3 (same code path:
image tokens with bidirectional attention, sliding-window layers mixed with global ones),
makes several prompts that share images + text and differ only at the end (with DIFFERENT
lengths, so the right-padding of the batch is exercised), and compares yes_prob_shared (the
one the audit script uses) against a full forward for every prompt. The sliding window is
SHORTER than the prompts, so the windowed cache path is exercised too.

    python checks/medgemma_prefix_cache_check.py
"""
import sys
from pathlib import Path

import torch
from transformers import Gemma3Config, Gemma3ForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_code" / "label_v2"))
import medgemma_audit58 as mg  # noqa: E402

torch.manual_seed(0)
IMG_TOK, BOI, EOI, BOS, PAD = 250, 251, 252, 2, 0
N_IMG_TOK = 4                                   # mm_tokens_per_image (2x2 after pooling)
cfg = Gemma3Config(
    text_config=dict(vocab_size=260, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, sliding_window=8, max_position_embeddings=512,
                     layer_types=None),
    vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=2, image_size=56, patch_size=14),
    mm_tokens_per_image=N_IMG_TOK, image_token_index=IMG_TOK, boi_token_index=BOI,
    eoi_token_index=EOI)
model = Gemma3ForConditionalGeneration(cfg).eval()
print("layer types:", model.config.text_config.layer_types)


def prompt(n_img, shared_text, question):
    ids = [BOS]
    for _ in range(n_img):
        ids += [10, BOI] + [IMG_TOK] * N_IMG_TOK + [EOI, 11]
    return ids + shared_text + question + [20, 21]      # "Final Answer:"


def full_p(ids, pix, yes_ids, no_ids):
    x = torch.tensor([ids])
    kw = {"input_ids": x, "attention_mask": torch.ones_like(x),
          "token_type_ids": (x == IMG_TOK).long()}
    if pix is not None:
        kw["pixel_values"] = pix
    with torch.inference_mode():
        lg = model(**kw).logits[0, -1].float()
    return float(torch.sigmoid(torch.logsumexp(lg[yes_ids], 0) - torch.logsumexp(lg[no_ids], 0)))


fails = 0
yes_ids, no_ids = [100, 101], [102, 103]
for n_img in (0, 1, 3):
    pix = torch.randn(n_img, 3, 56, 56) if n_img else None
    shared = torch.randint(30, 200, (15,)).tolist()   # e.g. the report
    questions = [torch.randint(30, 200, (int(k),)).tolist() for k in (5, 9, 3, 12)]
    ids_list = [prompt(n_img, shared, q) for q in questions]
    probs, mass, L, chk = mg.yes_prob_shared(model, ids_list, pix, IMG_TOK, yes_ids, no_ids,
                                             pad_id=PAD, check=True)
    worst = max(abs(full_p(ids, pix, yes_ids, no_ids) - p) for ids, p in zip(ids_list, probs))
    ok = worst < 1e-5 and chk[0] < 1e-4 and all(0 < v <= 1 for v in mass)
    fails += not ok
    print(f"  {n_img} images: prompts {[len(x) for x in ids_list]} tokens, shared prefix {L}, "
          f"sliding window 8, batch {len(ids_list)} right-padded, max |P(yes) full - batched| = "
          f"{worst:.2e}, built-in check {chk[0]:.1e}  {'OK' if ok else 'FAIL'}")
assert model.config.image_token_id == IMG_TOK, "config.image_token_id is what the audit uses"
print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
