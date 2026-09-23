"""medgemma_prefix_cache_check.py - does "encode the images once, reuse the cache for 12
questions" give the SAME answer as a full forward pass per question?

MedGemma 1.5 is a Gemma 3 model. This builds a tiny random-weight Gemma 3 (same code path:
image tokens with bidirectional attention, sliding-window layers mixed with global ones),
makes several prompts that share images + text and differ only at the end, and compares
yes_prob_shared (the one the audit script uses) against a full forward for every prompt.
The sliding window is set SHORTER than the prompts, so the windowed cache path is exercised.

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
IMG_TOK, BOI, EOI, BOS = 250, 251, 252, 2
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
    ids += shared_text + question + [20, 21]            # "Final Answer:"
    ids = torch.tensor([ids])
    tt = (ids == IMG_TOK).long()
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "token_type_ids": tt}


fails = 0
for n_img in (0, 1, 3):
    pix = torch.randn(n_img, 3, 56, 56) if n_img else None
    shared = torch.randint(30, 200, (15,)).tolist()   # e.g. the report
    questions = [torch.randint(30, 200, (int(k),)).tolist() for k in (5, 9, 3, 12)]
    inps = []
    for q in questions:
        x = prompt(n_img, shared, q)
        if pix is not None:
            x["pixel_values"] = pix
        inps.append(x)
    yes_ids, no_ids = [100, 101], [102, 103]
    probs, _, L, _ = mg.yes_prob_shared(model, inps, yes_ids, no_ids)
    worst = 0.0
    for x, p in zip(inps, probs):
        with torch.inference_mode():
            lg = model(**x).logits[0, -1].float()
        pf = float(torch.sigmoid(torch.logsumexp(lg[yes_ids], 0) - torch.logsumexp(lg[no_ids], 0)))
        worst = max(worst, abs(pf - p))
    total = inps[0]["input_ids"].shape[1]
    ok = worst < 1e-5
    fails += not ok
    print(f"  {n_img} images: prompt {total} tokens, shared prefix {L}, sliding window 8, "
          f"max |P(yes) full - shared| = {worst:.2e}  {'OK' if ok else 'FAIL'}")

# the built-in check on the first prompt must agree too
_, mass, _, diff = mg.yes_prob_shared(model, inps, yes_ids, no_ids, check=True)
assert all(0 < v <= 1 for v in mass), mass
assert model.config.image_token_id == IMG_TOK, "config.image_token_id is what the audit asserts on"
print(f"  built-in check: max |logit diff| = {diff:.2e}  {'OK' if diff < 1e-4 else 'FAIL'}")
fails += diff >= 1e-4
print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
