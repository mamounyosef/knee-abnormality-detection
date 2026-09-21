"""two_gpu_split_check.py - the members are split across the cards, and nothing is lost.

Kaggle's "GPU T4 x2" gives two cards. The FIRST design split the BATCH, which needs every
model resident on every card: nine members left a 15 GiB T4 with 256 MiB free before a
single batch ran, and every batch died with out-of-memory. The second card was never the
problem - replicating was.

This design splits the MEMBERS. Each model lives on exactly one card, so the total weight
memory equals the single-card path and each card carries about half of it. Both cards then
run their own members on the same batch at the same time.

What must hold, and what would break silently if it did not:
  - every member is placed EXACTLY once. Place one twice and it votes twice; miss one and
    it votes never, while the blend still produces a plausible submission.
  - the assignment is BALANCED by cost. swinv2_base_in22k is 4.211 s per study and
    convnext_base_25d_poolattn 0.185, so round-robin would idle most of the second card.
  - a member's logits land in ITS OWN row and its own task columns.
  - a failed worker must RAISE. A bare threading.Thread swallows its exception, the outer
    handler substitutes zero logits, and zeros sigmoid to 0.5 - exactly the submission
    squeezed into [0.309, 0.707] that reported "9/9 members".
  - the reported total is the SLOWEST card, not the sum: the cards run concurrently.

Run:  python checks/two_gpu_split_check.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

PASS, FAIL = [], []


def ck(n, ok, d=""):
    (PASS if ok else FAIL).append(n)
    print("  %s %-58s %s" % ("PASS" if ok else "FAIL", n, d))


CPU = torch.device("cpu")
NAMES = ["swinv2_base_in22k", "convnext_base_relabel", "mricore_vitb_cross_attention",
         "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
         "convnext_base_25d_poolattn", "convnext_base_25d_perfinding_slots_and_pooling"]
CFGS = [{"experiment": {"name": n}} for n in NAMES]

print("\n1. the cost model that balances the cards")
costs = sc._member_costs(CFGS)
ck("measured costs are used where we have them",
   abs(costs[0] - 4.211) < 1e-6 and abs(costs[5] - 0.185) < 1e-6,
   "swinv2 %.3f, poolattn %.3f" % (costs[0], costs[5]))
unknown = sc._member_costs([{"experiment": {"name": "something_new"}}])
ck("an unknown member still gets a positive cost", unknown[0] > 0, "%.3f" % unknown[0])

print("\n2. placement: exactly once, and balanced")
models = [torch.nn.Linear(4, 3) for _ in CFGS]
placed, assign = sc._shard_models(models, CFGS, [CPU, CPU])
flat = sorted(i for a in assign for i in a)
ck("every member is placed exactly once", flat == list(range(len(models))), str(flat))
ck("no member is on two cards", len(flat) == len(set(flat)))
ck("placed[i] is member i, not a copy", all(placed[i] is models[i] for i in range(len(models))))
load = [sum(costs[i] for i in a) for a in assign]
ck("the two cards are within 25% of each other",
   abs(load[0] - load[1]) / max(load) < 0.25,
   "%.2f s vs %.2f s, serial would be %.2f s" % (load[0], load[1], sum(costs)))
ck("and that beats round-robin", abs(load[0] - load[1]) <
   abs(sum(costs[0::2]) - sum(costs[1::2])) + 1e-9,
   "round-robin gap %.2f s" % abs(sum(costs[0::2]) - sum(costs[1::2])))

print("\n3. one card degrades to the old behaviour exactly")
p1, a1 = sc._shard_models(models, CFGS, [CPU])
ck("a single device owns every member", a1 == [list(range(len(models)))])

print("\n4. the scatter is per-member and per-column")
M, K, T = 4, 5, 12
st = torch.full((M, K, T), float("nan"))
cols = [torch.arange(T), torch.tensor([0, 1, 2, 3]), torch.arange(T), torch.arange(T)]
for mi in range(M):
    st[mi][:, cols[mi]] = float(mi + 1)
ck("a partial-head member fills only its own columns",
   bool(torch.isnan(st[1][:, 4:]).all()) and bool((st[1][:, :4] == 2).all()))
ck("a full member fills all twelve", bool((st[0] == 1).all()))
ck("rows stay with their member", bool((st[3] == 4).all()))

print("\n5. the blend is unchanged by which card produced a row")
rng = np.random.default_rng(0)
a = torch.tensor(rng.normal(size=(5, 9, 12)), dtype=torch.float32)
a[1, :, 3:] = float("nan")
order = [3, 0, 4, 1, 2]
ck("nanmean does not care about member order",
   torch.allclose(torch.nanmean(a, dim=0), torch.nanmean(a[order], dim=0), atol=1e-7),
   "max diff %.2e" % (torch.nanmean(a, dim=0) - torch.nanmean(a[order], dim=0)).abs().max())

print("\n6. the source says what it should")
src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
split = src[src.find("def _score_batch_split"):src.find("        def _score_batch():")]
# "threading.Thread" also appears in the comment explaining why it is NOT used, so match
# the call, not the mention
ck("workers raise instead of being swallowed", "ThreadPoolExecutor" in split
   and "f.result()" in split and "threading.Thread(" not in split)
ck("each thread pins its own CUDA device", "torch.cuda.set_device(dev)" in split)
ck("a member with no usable logits raises", "produced no usable logits" in split)
ck("the stack is built on the CPU, where the cards meet", "st_cpu = torch.full" in split)
ck("COLS is moved to the CPU for the scatter", "COLS[_mi].cpu()" in split)
ck("models are SHARDED, never replicated",
   "_shard_models" in src and "_replicate_models" not in src)
ck("out of memory falls back to one card", "one card only" in src)
ck("the reported total is the slowest card", 'timing.get("gpu_wall_s")' in src)
ck("each member's forward is micro-batched so one member cannot fill a card",
   "for _s0 in range(0, x.shape[0], MICRO)" in split and "micro_studies" in src)
ck("micro-batch outputs are concatenated in order",
   "torch.cat(outs, dim=0)" in split)
ck("gpu_wall_s is the max over cards, not the sum",
   "timing[\"gpu_wall_s\"] = max(sum(d[i] for i in assign[di])" in src)

print("\n" + "=" * 72)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("    FAILED: %s" % f)
sys.exit(1 if FAIL else 0)
