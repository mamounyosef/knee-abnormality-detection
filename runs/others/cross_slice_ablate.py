"""
cross_slice_ablate.py — the decisive test: does deleting the cross-slice blocks hurt?

Every weight statistic is indirect. A branch can look perfectly healthy — out_proj norm
flat, gradients flowing, weights moving — and still contribute nothing, because attention
can go uniform over the slices and turn the block into a constant offset carrying no
cross-slice information at all. No weight norm shows that.

This measures the only thing that settles it: score the SAME checkpoint twice, once whole
and once with the cross-slice branches forced to return their input untouched, and compare
macro AUROC.

    drop ~ 0.000   decoration. The model routes around them and the experiment is a null
                   result however healthy the weights look.
    drop > 0.005   load-bearing. Cross-slice information is genuinely used.
    drop < 0       removing them HELPS — actively hurting, a different and more
                   interesting failure than being ignored.

Ablating by swapping `forward` (not deleting modules) keeps every index, buffer and shape
identical, so the two passes differ in exactly one thing.

    python -m modal run runs/others/cross_slice_ablate.py --run mricore_vitb_cross_attention
    python -m modal run runs/others/cross_slice_ablate.py --run mricore_vitb_cross_attention_ft --checkpoint best

Writes results/<set>_results.json under an `_ablated` run dir for the second pass, so the
whole-model reports are never overwritten.
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402


def _load_build_model(exp_dir: Path):
    """Import the run's own train.py, so the architecture matches what was trained."""
    sys.path.insert(0, str(exp_dir))
    spec = importlib.util.spec_from_file_location(f"{exp_dir.name}_train",
                                                  exp_dir / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model


def _blocks(model):
    return [m for m in model.modules() if type(m).__name__ == "CrossSliceBlock"]


def _entropy(model, cfg, exp_dir, device, checkpoint: str, n_batches: int = 3):
    """How PEAKED is attention over the S slices?

    Uniform (entropy -> log S) means every slice receives the same mixture, so the block
    adds a constant and no slice-specific signal. This is the failure mode that is
    invisible to every weight norm, which is why it is worth reading directly.
    """
    blks = _blocks(model)
    if not blks:
        return {}
    stats: dict[int, list] = {i: [] for i in range(len(blks))}
    handles = []

    def mk(i):
        def hook(mod, inp, _out):
            x = inp[0]
            BS, T, F = x.shape
            B = max(BS // mod.S, 1)
            h = mod.norm(x).view(B, mod.S, T, F).permute(0, 2, 1, 3).reshape(B * T, mod.S, F)
            _, w = mod.attn(h, h, h, need_weights=True, average_attn_weights=True)
            if w is not None:
                p = w.detach().float().clamp_min(1e-9)
                stats[i].append(float((-(p * p.log()).sum(-1)).mean()))
        return hook

    for i, b in enumerate(blks):
        handles.append(b.register_forward_hook(mk(i)))
    try:
        sc.evaluate_official(cfg, model, exp_dir, device=device, checkpoint=checkpoint,
                             eval_sets=("val",), calibrate_if_missing=False)
    finally:
        for h in handles:
            h.remove()

    S = blks[0].S
    return {i: (sum(v) / len(v), math.log(S)) for i, v in stats.items() if v}


def run_ablation(cfg, exp_dir: Path, checkpoint: str, sets: list, entropy: bool,
                 code_dir: Path | None = None):
    """exp_dir is where RESULTS live; code_dir is where train.py lives.

    On Modal those are two different places - the code is baked into the image under
    /root/runs, the checkpoints ride the volume at /runs - so they cannot be the same
    argument. Locally they coincide.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_model = _load_build_model(Path(code_dir or exp_dir))
    model = build_model(cfg)
    blks = _blocks(model)
    print(f"\n  {len(blks)} CrossSliceBlock(s) found in {Path(exp_dir).name}\n")
    if not blks:
        raise SystemExit("no CrossSliceBlock in this model - nothing to ablate")

    print("=" * 74)
    print("PASS 1 — WHOLE MODEL")
    print("=" * 74)
    whole = sc.evaluate_official(cfg, model, exp_dir, device=device, checkpoint=checkpoint,
                                 eval_sets=tuple(sets), calibrate_if_missing=True)

    ent = _entropy(model, cfg, exp_dir, device, checkpoint) if entropy else {}
    if ent:
        print("\n  attention entropy over the S slices (max = uniform = no selectivity):")
        for i, (e, mx) in sorted(ent.items()):
            frac = e / mx
            note = ("   <-- essentially UNIFORM: a constant offset, not slice information"
                    if frac > 0.98 else "   <-- selective" if frac < 0.90 else "")
            print(f"    block {i}   {e:.3f} / {mx:.3f}   ({frac:.1%} of max){note}")

    # the branch returns its input untouched; nothing else about the model changes
    originals = [b.forward for b in blks]
    for b in blks:
        b.forward = (lambda x, *a, **k: x)

    # second pass writes under a separate run_dir so pass 1's reports survive
    acfg = sc.deep_copy(cfg) if hasattr(sc, "deep_copy") else __import__("copy").deepcopy(cfg)
    acfg["output"]["run_dir"] = cfg["output"]["run_dir"] + "_ablated"
    (Path(exp_dir) / acfg["output"]["run_dir"]).mkdir(parents=True, exist_ok=True)
    ck_src = (Path(exp_dir) / cfg["output"]["run_dir"] / cfg["output"]["checkpoints_dir"])
    ck_dst = (Path(exp_dir) / acfg["output"]["run_dir"] / cfg["output"]["checkpoints_dir"])
    if not ck_dst.exists():
        try:
            ck_dst.symlink_to(ck_src, target_is_directory=True)
        except OSError:
            acfg["output"]["run_dir"] = cfg["output"]["run_dir"]   # fall back, same dir

    print("\n" + "=" * 74)
    print("PASS 2 — CROSS-SLICE BRANCHES ABLATED")
    print("=" * 74)
    try:
        abl = sc.evaluate_official(acfg, model, exp_dir, device=device, checkpoint=checkpoint,
                                   eval_sets=tuple(sets), calibrate_if_missing=False)
    finally:
        for b, f in zip(blks, originals):
            b.forward = f

    def pick(res, s):
        """evaluate_official nests the summary under a 'macro' key, and may key the
        per-set results by name or return a single set's dict directly. Walk for the
        first macro_auroc rather than assuming one shape - guessing wrong here is what
        made the first run print an empty result table."""
        def walk(d, depth=0):
            if not isinstance(d, dict) or depth > 3:
                return None
            if "macro_auroc" in d:
                return d["macro_auroc"]
            for key in (s, "macro", "metrics", "summary"):
                if key in d:
                    got = walk(d[key], depth + 1)
                    if got is not None:
                        return got
            for v in d.values():
                got = walk(v, depth + 1)
                if got is not None:
                    return got
            return None
        return walk(res)

    print("\n" + "=" * 74)
    print("  ABLATION RESULT")
    print("=" * 74)
    for s in sets:
        w, a = pick(whole, s), pick(abl, s)
        if w is None or a is None:
            print(f"  {s}: could not read macro_auroc from the reports "
                  f"(whole={w}, ablated={a}) - check results/ and results_ablated/")
            continue
        d = w - a
        print(f"  {s:<8} whole {w:.4f}   ablated {a:.4f}   DROP {d:+.4f}")
        if abs(d) < 0.001:
            print("           -> DECORATION: removing them costs nothing.")
        elif d < 0:
            print("           -> removing them HELPS: actively hurting, not ignored.")
        else:
            print("           -> LOAD-BEARING: cross-slice information is being used.")
    print("=" * 74)
    return {"whole": whole, "ablated": abl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mricore_vitb_cross_attention")
    ap.add_argument("--checkpoint", default="last")
    ap.add_argument("--sets", nargs="*", default=["val"])
    ap.add_argument("--no-entropy", dest="entropy", action="store_false", default=True)
    a = ap.parse_args()
    exp_dir = PKG_ROOT / a.run
    run_ablation(sc.load_config(exp_dir, verbose=False), exp_dir,
                 a.checkpoint, a.sets, a.entropy)


try:
    import modal

    app = modal.App("cross-slice-ablate")
    _cfg0 = sc.load_config(PKG_ROOT / "mricore_vitb_cross_attention", verbose=False)
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(_cfg0["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(_cfg0["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={_cfg0["modal"]["shards_mount"]: _shards,
                           _cfg0["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(_cfg0))
    def ablate_remote(run: str, checkpoint: str, sets: list, entropy: bool):
        exp_dir = PKG_ROOT / run
        cfg = sc.remote_cfg(sc.load_config(exp_dir, verbose=False))
        out_dir = Path(cfg["modal"]["runs_mount"]) / run
        try:
            run_ablation(cfg, out_dir, checkpoint, sets, entropy, code_dir=exp_dir)
        finally:
            _runs.commit()

    @app.local_entrypoint()
    def modal_main(run: str = "mricore_vitb_cross_attention", checkpoint: str = "last",
                   sets: str = "val", no_entropy: bool = False):
        ablate_remote.remote(run, checkpoint, [s for s in sets.split(",") if s],
                             not no_entropy)

except ImportError:
    pass


if __name__ == "__main__":
    main()
