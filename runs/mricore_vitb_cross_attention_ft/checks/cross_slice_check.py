"""
cross_slice_check.py — prove CrossSliceBlock does what its comments claim.

The block does four shape operations in a row (view, permute, reshape, and their inverses)
around an attention call. Every one of them is a chance to silently mix the wrong axes: a
wrong permute would have study A's slices attend to study B's, or have a token attend to
other tokens instead of other slices, and NONE of that raises an error - it just trains
badly and looks like a failed idea.

So this checks the properties, not the shapes alone:

    1  every intermediate shape matches the comment in the code
    2  the permute/reshape round trip is an exact inverse
    3  attention mixes over SLICES, not over tokens
    4  studies are independent - study b never sees study b'
    5  token positions are independent - token t never sees token t'
    6  the block is UNGATED: out is exactly x + attn(norm(x)), no scale anywhere
    7  the branch actually moves the output, and every new tensor gets gradient
    8  the whole forward matches an explicit element-by-element reference

  and for _insert_cross_slice, which rewrites timm's block list:

    9  the new blocks land at the right indices, for several `after` lists
   10  the 12 pretrained blocks are the SAME objects, in the same order - not copies,
       not reordered
   11  their pretrained weights are bit-identical after the rewrite
   12  after=[] is a no-op
   13  the rebuilt backbone still runs, and the spliced blocks change its output
   14  no CrossSliceBlock carries a scale parameter - the gate is gone, not just set to 1

    python runs/mricore_vitb_cross_attention_ft/checks/cross_slice_check.py
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

RUN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN_DIR.parent))


def _load_mod():
    spec = importlib.util.spec_from_file_location("_xs_train", RUN_DIR / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_mod():
    """The run's train.py, imported without running its main."""
    spec = importlib.util.spec_from_file_location("_xs_train", RUN_DIR / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_block():
    return _load_mod().CrossSliceBlock


# small enough to loop over by hand, odd sizes so a transposed axis cannot hide
B, S, T, F, H = 3, 4, 5, 8, 2
CrossSliceBlock = _load_block()


def _fresh(seed=0):
    torch.manual_seed(seed)
    blk = CrossSliceBlock(F, H, S).eval()
    return blk


def _shapes():
    """1, 2 — every intermediate shape, and the round trip."""
    x = torch.randn(B * S, T, F)
    seen = {}
    h = x.view(B, S, T, F);                     seen["view"] = tuple(h.shape)
    h = h.permute(0, 2, 1, 3);                  seen["permute"] = tuple(h.shape)
    h = h.reshape(B * T, S, F);                 seen["reshape"] = tuple(h.shape)
    back = h.view(B, T, S, F).permute(0, 2, 1, 3).reshape(B * S, T, F)
    want = {"view": (B, S, T, F), "permute": (B, T, S, F), "reshape": (B * T, S, F)}
    ok_shape = seen == want
    ok_round = torch.equal(back, x)
    print(f"  {'PASS' if ok_shape else 'FAIL'}  shapes match the comments      {seen}")
    print(f"  {'PASS' if ok_round else 'FAIL'}  permute/reshape round trip is an exact inverse")
    return ok_shape and ok_round


def _mixes_slices_not_tokens():
    """3, 5 — perturbing slice s changes the other slices; perturbing token t does not
    change the other tokens."""
    blk = _fresh()
    x = torch.randn(B * S, T, F)
    base = blk(x)

    # NOT a constant offset: LayerNorm subtracts the mean over F, so adding the same value
    # to all F features is erased before the attention ever sees it and the probe would be
    # invisible. A random perturbation survives the norm.
    torch.manual_seed(99)
    x2 = x.clone()
    x2[0] += torch.randn(T, F) * 5.0           # study 0, slice 0
    d = (blk(x2) - base).abs()
    d = d.view(B, S, T, F)
    other_slices = float(d[0, 1:].max())  # noqa: E501       # same study, different slices
    print(f"  {'PASS' if other_slices > 1e-4 else 'FAIL'}  a slice influences the OTHER "
          f"slices of its study   (max delta {other_slices:.3e})")

    x3 = x.clone()
    x3[:, 0, :] += torch.randn(B * S, F) * 5.0  # token 0 of every slice
    d3 = (blk(x3) - base).abs().view(B, S, T, F)
    other_tokens = float(d3[:, :, 1:].max())   # tokens 1..T-1 must be untouched
    print(f"  {'PASS' if other_tokens < 1e-4 else 'FAIL'}  a token does NOT leak into other "
          f"tokens                (max delta {other_tokens:.3e})")
    return other_slices > 1e-4 and other_tokens < 1e-4


def _studies_independent():
    """4 — the failure that would be invisible: study A attending to study B."""
    blk = _fresh()
    x = torch.randn(B * S, T, F)
    base = blk(x).view(B, S, T, F)
    x2 = x.clone()
    torch.manual_seed(123)
    x2[S:2 * S] += torch.randn(S, T, F) * 5.0  # every slice of study 1 only
    out = blk(x2).view(B, S, T, F)
    leak = max(float((out[0] - base[0]).abs().max()),
               float((out[2] - base[2]).abs().max()))
    moved = float((out[1] - base[1]).abs().max())
    print(f"  {'PASS' if leak < 1e-4 else 'FAIL'}  studies are independent           "
          f"(leak {leak:.3e}, the perturbed study moved {moved:.3e})")
    return leak < 1e-4 and moved > 1e-4


def _ungated():
    """6, 7 — the block is a plain residual branch, and all of it trains.

    There is no longer a gate to open, so the property to hold is the opposite of the old
    one: the branch must contribute from step 1, and every tensor in it must receive
    gradient. mricore_vitb_cross_attention_ft failed precisely because a scale parameter
    sat in front of h and the optimizer drove it to zero.
    """
    blk = _fresh()
    x = torch.randn(B * S, T, F)

    # 1. out is EXACTLY x + attn(norm(x)) - hand-computed, nothing scaling h
    with torch.no_grad():
        got = blk(x)
        h = blk.norm(x).view(B, S, T, F).permute(0, 2, 1, 3).reshape(B * T, S, F)
        h, _ = blk.attn(h, h, h, need_weights=False)
        h = h.view(B, T, S, F).permute(0, 2, 1, 3).reshape(B * S, T, F)
        ref = x + h
    d = float((got - ref).abs().max())
    exact = d == 0.0
    print(f"  {'PASS' if exact else 'FAIL'}  out is exactly x + attn(norm(x))  "
          f"(max delta {d:.3e})")

    # 2. no scale parameter survives anywhere in the block
    scales = [n for n, _ in blk.named_parameters()
              if n.endswith("gamma") or n.endswith("scale") or n.endswith("layer_scale")]
    clean = not scales
    print(f"  {'PASS' if clean else 'FAIL'}  no gate/scale parameter on the block  "
          f"{scales if scales else '(none)'}")

    # 3. the branch moves the output - it cannot be a silent no-op any more
    moved = float((got - x).abs().max())
    live = moved > 1e-3
    print(f"  {'PASS' if live else 'FAIL'}  the branch changes the output     "
          f"(max delta {moved:.3e})")

    # 4. every trainable tensor in the block gets gradient
    blk2 = _fresh()
    blk2(torch.randn(B * S, T, F)).sum().backward()
    dead = [n for n, q in blk2.named_parameters()
            if q.grad is None or float(q.grad.abs().sum()) == 0.0]
    trains = not dead
    print(f"  {'PASS' if trains else 'FAIL'}  every param gets gradient         "
          f"{len(list(blk2.named_parameters())) - len(dead)}"
          f"/{len(list(blk2.named_parameters()))} tensors"
          + (f"  dead: {dead}" if dead else ""))
    return exact and clean and live and trains


def _matches_reference():
    """8 — the whole forward, against a loop that no one could accidentally vectorise wrong."""
    blk = _fresh(seed=7)
    x = torch.randn(B * S, T, F)
    with torch.no_grad():
        got = blk(x)

        # reference: for each study b and token t, run attention over that study's S slices
        h = blk.norm(x)
        ref = torch.empty_like(x)
        for b in range(B):
            for t in range(T):
                seq = torch.stack([h[b * S + s, t] for s in range(S)]).unsqueeze(0)  # (1, S, F)
                a, _ = blk.attn(seq, seq, seq, need_weights=False)                   # (1, S, F)
                for s in range(S):
                    ref[b * S + s, t] = x[b * S + s, t] + a[0, s]
    d = float((got - ref).abs().max())
    ok = d < 1e-5
    print(f"  {'PASS' if ok else 'FAIL'}  matches an explicit per-study, per-token loop "
          f"(max delta {d:.3e})")
    return ok


def _insertion():
    """9-13 — _insert_cross_slice rewrites an nn.Sequential that timm owns. Getting the order
    or the identity of those 12 modules wrong would silently reorder a pretrained trunk, and
    nothing would raise."""
    import timm
    mod = _load_mod()
    ok = True

    for after, want in [([3, 6, 9], [3, 7, 11]), ([6], [6]), ([1, 2], [1, 3]),
                        ([12], [12]), ([], [])]:
        bb = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        orig = list(bb.blocks)
        mod._insert_cross_slice(bb, S, after)
        new_at = [i for i, b in enumerate(bb.blocks)
                  if type(b).__name__ == "CrossSliceBlock"]
        kept = [b for b in bb.blocks if type(b).__name__ != "CrossSliceBlock"]

        pos_ok = new_at == want
        len_ok = len(bb.blocks) == 12 + len(after)
        # SAME objects in the SAME order - identity, not equality
        ident_ok = len(kept) == 12 and all(a is b for a, b in zip(kept, orig))
        weights_ok = all(torch.equal(po, pk)
                         for bo, bk in zip(orig, kept)
                         for po, pk in zip(bo.parameters(), bk.parameters()))
        # the new blocks must know S
        s_ok = all(b.S == S for b in bb.blocks if type(b).__name__ == "CrossSliceBlock")
        good = pos_ok and len_ok and ident_ok and weights_ok and s_ok
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  after={str(after):<10} -> new at "
              f"{str(new_at):<12} len {len(bb.blocks):>2}  pretrained intact: "
              f"{ident_ok and weights_ok}  S set: {s_ok}")

    bb = timm.create_model("vit_base_patch16_224", pretrained=False,
                           num_classes=0, img_size=64)
    torch.manual_seed(3)
    x = torch.randn(S, 3, 64, 64)
    with torch.no_grad():
        ref = bb(x)
        mod._insert_cross_slice(bb, S, [3, 6, 9])
        got = bb(x)
    d = float((ref - got).abs().max())
    # Ungated, so the old "bit-identical at gamma=0" property is now impossible BY DESIGN.
    # The thing worth asserting flipped: the rebuilt backbone must still run, and the new
    # blocks must actually change what comes out of it.
    run_ok = got.shape == ref.shape and d > 1e-4 and torch.isfinite(got).all()
    ok = ok and run_ok
    print(f"  {'PASS' if run_ok else 'FAIL'}  rebuilt backbone runs, spliced blocks change "
          f"its output  (max delta {d:.3e})")

    n_scale = [n for n, _ in bb.named_parameters()
               if n.endswith("gamma") or n.endswith("scale")]
    clean = not n_scale
    ok = ok and clean
    print(f"  {'PASS' if clean else 'FAIL'}  no gate parameter anywhere in the backbone  "
          f"{n_scale if n_scale else '(none)'}")
    return ok


def main():
    print(f"CrossSliceBlock  B={B} S={S} T={T} F={F} heads={H}\n" + "-" * 78)
    r = [_shapes(), _mixes_slices_not_tokens(), _studies_independent(), _ungated(),
         _matches_reference()]
    print("-" * 78 + "\n  _insert_cross_slice")
    r.append(_insertion())
    print("-" * 78)
    print(f"  {'ALL CHECKS PASS' if all(r) else 'FAILURES ABOVE'}")
    return 0 if all(r) else 1


if __name__ == "__main__":
    raise SystemExit(main())
