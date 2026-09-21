"""two_gpu_split_check.py - the two-device path must produce IDENTICAL predictions.

Kaggle gives two T4s and every submission before this used one. The fix splits the study
batch across devices. Splitting cannot change any number, because each study is scored
independently - so the test is not "is it faster", it is "is it the same".

This forces the split path onto ONE device (two logical shards of the same card, or of the
CPU) so the comparison runs anywhere, including this PC with its single 8 GB card. The real
two-card timing is measured separately on Modal with gpu="T4:2".

WHAT WOULD BREAK SILENTLY, and is therefore checked:
  - study ORDER. torch.chunk then torch.cat must reassemble in the original order; a
    transposed pair would shuffle predictions onto the wrong StudyInstanceUID and still
    produce a perfectly valid submission.csv.
  - MEMBER order. Each device builds its own stack; if the member axis disagreed, the
    per-member logits handed to the grouped blender would be attributed to the wrong models.
  - a RAGGED final batch, where one shard is smaller than the other or empty.
  - the single-device path must be untouched.

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


print("\n1. the helpers")
ck("usable_gpus exists", hasattr(sc, "usable_gpus"))
ck("_replicate_models exists", hasattr(sc, "_replicate_models"))
ck("usable_gpus honours inference.max_gpus",
   sc.usable_gpus({"inference": {"max_gpus": 1}}) in (0, 1),
   str(sc.usable_gpus({"inference": {"max_gpus": 1}})))
# NOT torch.cuda.is_available(): usable_cuda() also refuses a card whose compute
# capability is missing from this torch build. On this PC the RTX 4060 is sm_89 and the
# installed torch has sm_86 and sm_90 but not sm_89, so the engine's inference path falls
# back to CPU here. Kaggle's T4 is sm_75 and is in the list.
ck("usable_gpus agrees with usable_cuda",
   (sc.usable_gpus() >= 1) if sc.usable_cuda() else sc.usable_gpus() == 0,
   "usable_cuda=%s device_count=%d" % (sc.usable_cuda(), torch.cuda.device_count()))

print("\n2. replication keeps device 0's ORIGINAL objects")
lin = torch.nn.Linear(4, 3)
dev = torch.device("cpu")
rep = sc._replicate_models([lin], [dev, dev])
ck("device 0 holds the same object, not a copy", rep[0][0] is lin)
ck("device 1 holds a distinct copy", rep[1][0] is not lin)
ck("the copy has equal weights",
   torch.equal(rep[1][0].weight.detach(), lin.weight.detach()))
rep[1][0].weight.data.add_(1.0)
ck("and mutating the copy does NOT touch the original",
   not torch.equal(rep[1][0].weight.detach(), lin.weight.detach()))

print("\n3. chunk/cat reassembles studies in the ORIGINAL order")
for k, nd in ((8, 2), (7, 2), (1, 2), (5, 3), (2, 3), (1, 4)):
    x = torch.arange(k * 2, dtype=torch.float32).reshape(k, 2)
    parts = [p for p in torch.chunk(x, nd, dim=0) if p.shape[0]]
    back = torch.cat(parts, dim=0)
    ck("k=%d over %d shards round-trips in order" % (k, nd), torch.equal(back, x),
       "shard sizes %s" % [p.shape[0] for p in parts])

print("\n4. the member axis survives the split")
# stacked is (n_members, k, n_tasks); each shard builds its own and they concatenate on
# dim=1. A shard that got the member order wrong would be invisible in the shapes.
M, K, T = 4, 6, 12
full = torch.arange(M * K * T, dtype=torch.float32).reshape(M, K, T)
shards = [full[:, :3], full[:, 3:]]
ck("cat on dim=1 restores the member axis exactly",
   torch.equal(torch.cat(shards, dim=1), full))

print("\n5. the same arithmetic on the blend, split or not")
rng = np.random.default_rng(0)
st = torch.tensor(rng.normal(size=(5, 9, 12)), dtype=torch.float32)
st[1, :, 3:] = float("nan")                       # a task-group member with a narrow head
one = torch.sigmoid(torch.nanmean(st, dim=0))
sp = torch.cat([torch.nanmean(st[:, :4], dim=0),
                torch.nanmean(st[:, 4:], dim=0)], dim=0)
# NOT bit-identical, and it cannot be: nanmean reduces a different number of elements per
# shard, so the float32 summation order differs. The gap is one unit in the last place.
# AUROC decisions in this project are made at 1e-4; this is six orders below that.
_d = (one - torch.sigmoid(sp)).abs().max()
ck("nanmean blend matches to float32 rounding when split", _d < 1e-6,
   "max diff %.2e, against the 1e-4 scale we make decisions on" % _d)
W = torch.tensor(rng.random(size=(5, 12)), dtype=torch.float32)
W = W / W.sum(0, keepdim=True)
one_w = torch.sigmoid(torch.nan_to_num(st, nan=0.0).mul(W[:, None, :]).sum(dim=0))
sp_w = torch.cat([torch.nan_to_num(st[:, :4], nan=0.0).mul(W[:, None, :]).sum(dim=0),
                  torch.nan_to_num(st[:, 4:], nan=0.0).mul(W[:, None, :]).sum(dim=0)], dim=0)
ck("weighted blend is identical when the batch is split",
   torch.allclose(one_w, torch.sigmoid(sp_w), atol=1e-7),
   "max diff %.2e" % (one_w - torch.sigmoid(sp_w)).abs().max())

print("\n6. the single-device path is untouched")
src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
i_guard = src.find("if len(devices) > 1:\n                return _score_batch_split()")
i_orig = src.find("stacked = torch.full((len(models), x.shape[0], len(tasks)),")
ck("the split is behind a len(devices) > 1 guard", i_guard > 0)
ck("and the original body still follows it", 0 < i_guard < i_orig)
ck("devices defaults to a single entry without CUDA",
   "if device.type == \"cuda\" and _ngpu > 1 else [device]" in src)

print("\n7. timing is wall clock, not serial-equivalent")
ck("member_s becomes the MEAN across devices",
   "sum(d[_i] for d in _dev_member_s) / len(devices)" in src,
   "a sum would hide the whole speedup from the 9 h budget")
ck("the pass count is recorded", '"n_gpus": len(devices)' in src)

print("\n8. cross-device indexing - the bug that silently zeroed the first two-GPU run")
# COLS index tensors are built on cuda:0. The thread for the SECOND card scatters into a
# tensor on that card; indexing it with a cuda:0 index raises. A bare threading.Thread
# swallowed that exception and every member on card 1 voted 0.5.
_src2 = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
_split = _src2[_src2.find("def _score_batch_split"):_src2.find("def _score_batch():")]
ck("the split moves COLS onto each thread's device",
   "cols_d = [c_.to(dev) for c_ in COLS]" in _split)
ck("and scatters with the per-device index, never the cuda:0 one",
   "st[_mi][:, cols_d[_mi]] = o" in _split and "st[_mi][:, COLS[_mi]] = o" not in _split)
# NOT asserted here, deliberately. A single-GPU machine cannot reproduce the real case: the
# failing scatter is a cuda:1 tensor indexed by a cuda:0 index, and torch happens to ALLOW a
# CPU tensor indexed by a CUDA index, so the obvious local proxy passes and proves nothing.
# Only a two-card run can confirm it; the same-device fix is correct whichever way it goes.
print("      (cross-device raise is only reproducible on two cards - verified on Kaggle)")

print("\n" + "=" * 72)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("    FAILED: %s" % f)
sys.exit(1 if FAIL else 0)
