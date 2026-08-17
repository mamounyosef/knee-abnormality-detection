"""Does the shared system prompt actually hit Modal's prompt cache?

Cached prompt tokens bill at $0.25/MTok instead of $2.00, but KV caches are not
shared between replicas, so a hit needs the request to reach the container that
already holds the prefix. llm_client sends a `Modal-Session-ID` to make routing
sticky; this checks that it works.

    python data_code/test_prompt_cache.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import chat, stats  # noqa: E402

SYSTEM = ("You are a musculoskeletal radiologist labeling knee MRI reports. " * 60)

prev_in = prev_cached = 0
print(f"{'call':>4} {'prompt':>8} {'cached':>8} {'hit%':>6}")
for i in range(4):
    chat([{"role": "system", "content": SYSTEM},
          {"role": "user", "content": f"Reply with the single word OK. (call {i})"}],
         effort="low", max_tokens=20)
    d_in = stats["in_tok"] - prev_in
    d_cached = stats["cached_tok"] - prev_cached
    prev_in, prev_cached = stats["in_tok"], stats["cached_tok"]
    print(f"{i+1:>4} {d_in:>8} {d_cached:>8} {d_cached / max(d_in, 1):>6.0%}")

print(f"\ntotals: prompt={stats['in_tok']:,}  cached={stats['cached_tok']:,}  "
      f"completion={stats['out_tok']:,}  reasoning={stats['reason_tok']:,}")
print("call 1 is expected to miss (nothing cached yet); calls 2+ should hit.")
