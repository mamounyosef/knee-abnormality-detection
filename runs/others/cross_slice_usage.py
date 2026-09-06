"""
cross_slice_usage.py - is the model actually USING the spliced-in cross-slice blocks,
or has it learned to route around them?

WHY THIS SCRIPT EXISTS
----------------------
When the block was gated (`out = x + gamma * h`) the question had a one-number answer:
watch gamma. mricore_vitb_cross_attention_ft did exactly that and watched it decay
9.1e-4 -> 4.0e-4 while val sat at 0.8936. The gate is now deleted, so there is no single
scalar to read and the question has to be asked of the weights themselves.

The trap to avoid: comparing raw weight norms ACROSS different modules is meaningless.
A Linear's init scale is a function of its fan-in, so a fresh 768x768 out_proj and a
pretrained 768x3072 fc2 start at different magnitudes for reasons that have nothing to do
with whether either is used. Every number below is therefore a RATIO - against the same
tensor at an earlier step, or against the block's own residual stream.

WHAT ACTUALLY SHUTS AN UNGATED BRANCH DOWN
------------------------------------------
With `out = x + attn(norm(x))` there is no gate, but the model still has three ways to
make the branch a no-op, and they are distinguishable:

  1. SHRINK THE OUTPUT PROJECTION.  h is produced by attn.out_proj, so driving its weights
     toward zero is the exact analogue of closing a gate. This is the one to watch: it is
     the gate the model would build for itself if it wanted one.

  2. GO UNIFORM OVER SLICES.  If the attention distribution over the S=96 slices flattens,
     every slice gets the same h - the block degenerates into a constant offset carrying no
     slice-specific information. Invisible in any weight norm. Needs an activation probe
     (see `--data` mode) or an entropy read.

  3. LET THE NEXT LayerNorm SWALLOW IT.  Even a large h is irrelevant if it is tiny next to
     the residual stream x. So the meaningful magnitude is always ||h|| / ||x||, never ||h||.

METRICS
-------
  drift        ||W_late - W_early|| / ||W_early||  - did this tensor move at all?
               Near 0 means frozen. Compare the NEW blocks against the pretrained ones:
               fresh blocks on a 10x learning rate should move far more, and if they do not,
               they are not learning.

  |W| ratio    ||W_late|| / ||W_early||  on attn.out_proj specifically.
               < 1 and falling across steps = the model is building its own gate and
               closing it. This is the ungated version of "gamma decayed".

  update/wt    the same quantity the engine logs live as the UD ratio, recomputed here so a
               finished run can be audited without the training log.

The decisive test is not a weight statistic at all: ABLATE the blocks at eval and measure
the val AUROC drop. If deleting them costs nothing, they are not being used, whatever the
norms say. `--ablate` prints the command for that.

USAGE
  python runs/others/cross_slice_usage.py --early A_step200.pt --late A_last.pt
  python runs/others/cross_slice_usage.py --early B_step100.pt --late B_last.pt --init-from mricore_vitb_relabel
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def load_sd(path: Path) -> tuple[dict, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model") or ck.get("state_dict") or ck
    meta = {k: ck.get(k) for k in ("step", "epoch") if k in ck}
    if isinstance(ck.get("track"), dict):
        meta["best"] = ck["track"].get("best")
    return sd, meta


def is_cross(name: str, sd: dict) -> bool:
    """A spliced-in block, not a pretrained ViT Block.

    CrossSliceBlock holds `attn.in_proj_weight` (nn.MultiheadAttention packs q,k,v into one
    tensor); timm's Block holds `attn.qkv.weight`. That difference is the reliable marker -
    block INDEX is not, because the indices shift when blocks are inserted.
    """
    if ".blocks." not in name:
        return False
    prefix = name.rsplit(".attn.", 1)[0] if ".attn." in name else name.rsplit(".", 1)[0]
    return f"{prefix}.attn.in_proj_weight" in sd


def norm(t: torch.Tensor) -> float:
    return float(t.detach().float().norm())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--early", required=True, type=Path)
    ap.add_argument("--late", required=True, type=Path)
    ap.add_argument("--init-from", type=Path, default=None,
                    help="the checkpoint the run STARTED from, for a true drift-from-init")
    args = ap.parse_args()

    e_sd, e_meta = load_sd(args.early)
    l_sd, l_meta = load_sd(args.late)
    print(f"early  {args.early.name}   {e_meta}")
    print(f"late   {args.late.name}   {l_meta}\n")

    cross = [k for k in l_sd if is_cross(k, l_sd) and l_sd[k].dtype.is_floating_point]
    trunk = [k for k in l_sd
             if ".blocks." in k and not is_cross(k, l_sd) and l_sd[k].dtype.is_floating_point]
    if not cross:
        print("no CrossSliceBlock tensors found - is this a cross-attention run?")
        return 1
    n_blocks = len({k.split(".attn.")[0] for k in cross if ".attn." in k})
    print(f"{len(cross)} cross-slice tensors across {n_blocks} block(s)   |   "
          f"{len(trunk)} pretrained trunk tensors\n")

    def drift(keys):
        out = {}
        for k in keys:
            if k not in e_sd or e_sd[k].shape != l_sd[k].shape:
                continue
            a, b = e_sd[k].detach().float(), l_sd[k].detach().float()
            na = float(a.norm())
            if na == 0 or not math.isfinite(na):
                continue
            out[k] = (float((b - a).norm()) / na, float(b.norm()) / na)
        return out

    d_cross, d_trunk = drift(cross), drift(trunk)

    # ---- 1. did the NEW blocks move more than the pretrained ones? -------------------
    print("=" * 78)
    print("1. DRIFT  ||W_late - W_early|| / ||W_early||   (did the tensor move at all)")
    print("=" * 78)
    for label, d in (("cross-slice (new)", d_cross), ("trunk (pretrained)", d_trunk)):
        if not d:
            continue
        v = sorted(x[0] for x in d.values())
        med = v[len(v) // 2]
        print(f"  {label:<20} median {med:.4f}   min {v[0]:.4f}   max {v[-1]:.4f}   "
              f"({len(v)} tensors)")
    if d_cross and d_trunk:
        mc = sorted(x[0] for x in d_cross.values())[len(d_cross) // 2]
        mt = sorted(x[0] for x in d_trunk.values())[len(d_trunk) // 2]
        r = mc / mt if mt else float("inf")
        print(f"\n  new blocks moved {r:.1f}x the trunk"
              + ("   <- expected: they are random and on a higher lr" if r > 2 else
                 "   <- SUSPECT: fresh blocks on a higher lr should move much more"))

    # ---- 2. the self-built gate ------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. OUTPUT PROJECTION  ||W_late|| / ||W_early||   (the gate the model can build)")
    print("=" * 78)
    print("   out_proj is what scales h into the residual stream. Falling below 1 and")
    print("   staying there is this architecture's version of 'gamma decayed to zero'.\n")
    op = {k: v for k, v in d_cross.items() if "out_proj.weight" in k}
    if not op:
        print("   (no out_proj tensors found)")
    for k, (dr, ratio) in sorted(op.items()):
        blk = k.split(".blocks.")[1].split(".")[0]
        flag = ("  <-- SHRINKING, the branch is being turned down" if ratio < 0.9 else
                "  <-- growing" if ratio > 1.1 else "")
        print(f"   block {blk:>3}   |W| ratio {ratio:6.3f}   drift {dr:6.3f}{flag}")

    print("\n   per-tensor, the new blocks (a huge drift on a near-zero init tensor -")
    print("   a LayerNorm bias, say - is a small absolute move, not a big one):")
    for k in sorted(d_cross, key=lambda x: -d_cross[x][0]):
        dr, ratio = d_cross[k]
        short = k.split("backbone.blocks.")[-1]
        print(f"     {short:<34} drift {dr:9.3f}   |W| ratio {ratio:6.3f}"
              f"   |W_early| {norm(e_sd[k]):8.4f}")

    tr_op = {k: v for k, v in d_trunk.items() if "proj.weight" in k}
    if tr_op:
        v = sorted(x[1] for x in tr_op.values())
        print(f"\n   for scale, pretrained proj.weight ratios: median {v[len(v) // 2]:.3f}"
              f"   min {v[0]:.3f}   max {v[-1]:.3f}")

    # ---- 3. drift from the true init, when we have it ---------------------------------
    if args.init_from:
        print("\n" + "=" * 78)
        print("3. DRIFT FROM THE RUN'S ACTUAL INIT  (index-mapped)")
        print("=" * 78)
        i_sd, _ = load_sd(args.init_from)

        # Inserting blocks SHIFTS every later index, so model blocks.4 is the checkpoint's
        # blocks.3. Comparing like-for-like index is comparing different modules and gives
        # a meaningless number. Rebuild the map: walk the model's blocks in order and
        # advance a separate counter only on the pretrained ones.
        idx = sorted({int(k.split(".blocks.")[1].split(".")[0])
                      for k in l_sd if ".blocks." in k})
        cross_idx = {int(k.split(".blocks.")[1].split(".")[0])
                     for k in l_sd if is_cross(k, l_sd)}
        mapping, orig = {}, 0
        for i in idx:
            if i in cross_idx:
                continue
            mapping[i] = orig
            orig += 1
        shifted = [f"{i}->{o}" for i, o in mapping.items() if i != o]
        print(f"   new blocks at {sorted(cross_idx)}, so the pretrained ones shift: "
              f"{', '.join(shifted) if shifted else '(none)'}\n")

        moved = {}
        for k in trunk:
            i = int(k.split(".blocks.")[1].split(".")[0])
            if i not in mapping:
                continue
            src = k.replace(f".blocks.{i}.", f".blocks.{mapping[i]}.", 1)
            if src not in i_sd or i_sd[src].shape != l_sd[k].shape:
                continue
            a = i_sd[src].detach().float()
            na = float(a.norm())
            if na > 0 and math.isfinite(na):
                moved[k] = float((l_sd[k].detach().float() - a).norm()) / na
        if not moved:
            print("   no trunk tensors could be matched to the init checkpoint")
        else:
            v = sorted(moved.values())
            med = v[len(v) // 2]
            print(f"   trunk moved from its converged init: median {med:.4f}"
                  f"   min {v[0]:.4f}   max {v[-1]:.4f}   ({len(v)} tensors)")
            print("   large here = the new blocks are DISRUPTING the trunk, not just adding")
            print("   to it. That is the cost side of running ungated from a converged model.")

    print("\n" + "=" * 78)
    print("THE DECISIVE TEST is behavioural, not a weight statistic:")
    print("  zero the cross-slice branches at eval and measure the val AUROC drop.")
    print("  No drop => not used, whatever these norms say.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
