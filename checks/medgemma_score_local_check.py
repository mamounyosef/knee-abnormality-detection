"""medgemma_score_local_check.py - run the REAL medgemma_audit58.score() end to end on this PC.

The Modal container is replaced by the local GPU, MedGemma by a tiny random-weight Gemma 3
built with MedGemma's real token ids (262k vocab, 256 tokens per 896 px image), the gated
processor by an ungated copy of the Gemma 3 processor, and the knee-shards volume by the
local sample shards. Everything else is the production code path: the CPU-prep thread one
study ahead, images processed once, the processor-equality assertions, the batched 12
questions on the shared cache, the full-pass check, the logging and the per-study saves.

The probabilities are meaningless (random weights); what is checked is that it runs, that
every mode of every study returns 12 probabilities, that the cache check agrees, and that
the log and rows files are written.

    python checks/medgemma_score_local_check.py
"""
import json
import sys
import tempfile
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, Gemma3Config, Gemma3ForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_code" / "label_v2"))
import medgemma_audit58 as mg  # noqa: E402

assert torch.cuda.is_available(), "needs the local GPU"
proc = AutoProcessor.from_pretrained("unsloth/gemma-3-4b-it", use_fast=True)
tok = proc.tokenizer
cfg = Gemma3Config(
    text_config=dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                     num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, sliding_window=64, max_position_embeddings=32768,
                     layer_types=None),
    vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                       num_attention_heads=2, image_size=896, patch_size=14),
    mm_tokens_per_image=256, image_token_index=tok.image_token_id,
    boi_token_index=tok.convert_tokens_to_ids(tok.boi_token),
    eoi_token_index=tok.convert_tokens_to_ids(tok.eoi_token))
tiny = Gemma3ForConditionalGeneration(cfg).to(torch.bfloat16)


class _Vol:
    def commit(self):
        pass


# score() does `from transformers import AutoModelForImageTextToText, AutoProcessor` inside
# the function, so the loaders on those classes are what must be swapped
transformers.AutoModelForImageTextToText.from_pretrained = \
    staticmethod(lambda *a, **k: tiny.to(k.get("device_map", "cuda")))
transformers.AutoProcessor.from_pretrained = staticmethod(lambda *a, **k: proc)
tmp = Path(tempfile.mkdtemp(prefix="mg_local_"))
mg.REMOTE_DIR = str(tmp)
mg.SHARDS_DIR = str(ROOT / "data/shards")
mg.runs_vol = _Vol()
mg.hf_cache = _Vol()
mg.MAX_SLICES = 8                  # keep the CPU-side tiny model quick; the path is identical
mg.REASON_MAX_NEW = 12             # tiny model never writes "Final Answer:"; keep the forced path short
#                 # keep the CPU-side tiny model quick; the path is identical
local = sorted(p.stem for p in (ROOT / "data/shards").glob("*.npz"))
jobs = mg.build_jobs(local, reports=False)
for j in jobs:
    j["report"] = "Moderate joint effusion. Intact cruciate ligaments."
import os  # noqa: E402
os.environ.setdefault("HF_TOKEN", "local-test")
MODES = ["all", "all_num", "all_reason"]     # batch E, the next real run
raw = mg.score.get_raw_f()
rows, log_text = raw(jobs, MODES, "localtest", part=0)

fails = 0
exp = len(jobs) * len(MODES)
bad = [r for r in rows if "error" in r]
print(f"\nrows {len(rows)} (expected {exp}); errors {len(bad)}")
for r in bad:
    print("  ERROR", r["uid"][-8:], r["mode"], r["error"])
fails += len(rows) != exp or bool(bad)
ok = [r for r in rows if "error" not in r]
fails += not all(all((0 <= r[t] <= 1) or r[t] != r[t] for t in mg.TASKS) for r in ok)
gen = [r.get("gen_check") for r in ok if "gen_check" in r]
print("generation checks (greedy on shared cache == model.generate):", gen)
fails += gen != [True]
checks = [r["cache_diff"] for r in ok if r.get("cache_diff") is not None]
print("cache checks (max |logit diff|, bf16 tiny model):", [round(c, 4) for c in checks])
fails += len(checks) != len([m for m in MODES if mg.MODE_OPT.get(m, {}).get("answer") not in ("number", "reason")])
rc = [r["reason_check"] for r in ok if "reason_check" in r]
print("reasoning checks (|P(yes)| shared decode vs one full pass):", rc)
fails += len(rc) != 1 or not rc[0] < 1e-3
part = tmp / "localtest/part0"
saved = [json.loads(x) for x in (part / "rows.jsonl").read_text().splitlines()]
print("saved rows.jsonl", len(saved), "log.txt lines", len((part / "log.txt").read_text().splitlines()))
fails += len(saved) != exp
print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
