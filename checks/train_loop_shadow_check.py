"""
train_loop_shadow_check.py - does anything inside the training loop shadow an outer name?

WHY THIS EXISTS. _run_experiment is a very long function: it binds the output directory,
the loggers, the throughput trackers and the closures that validate and checkpoint, and
THEN runs a training loop inside the same scope. So a new local in that loop silently
rebinds the outer name, and the damage surfaces far from the edit.

It has now happened twice in one change, both from the auxiliary-report work:

    roll = torch.roll(tgt, 1, dims=0)   clobbered the 20-step throughput deque and crashed
                                        at step 1 with 'Tensor' object has no attribute
                                        'append'
    out  = model(...)                   clobbered the run's OUTPUT DIRECTORY, which the
                                        validation closure captures, and would have crashed
                                        the first validation on out / "ud_ratio.csv"

Neither is findable by the unit checks, because both are about the surrounding scope rather
than the code under test. A static pass is.

    python checks/train_loop_shadow_check.py
"""

import ast
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "shared_code.py"
FUNCS = ["_run_experiment"]

# Names that are DELIBERATELY rebound each iteration: the loop counters and the values the
# loop exists to recompute. Shadowing is the normal, intended behaviour for these.
ALLOWED = {
    "x", "y", "loss", "logits", "gnorm", "lr", "lr_lo", "lr_hi", "ips", "real_ips",
    "mem", "el", "eta", "d", "ce", "ev", "every", "epoch", "it", "gstep", "seen",
    "data_dt", "wait_pct", "overall_ips", "skip", "smp", "comp_t", "r_i", "r_w", "r_d",
    "win_imgs", "win_t0", "win_data", "win_first", "prev_loss", "l", "_extra", "_pb",
    "retr_prev", "_train_it", "ep_t0", "_clip", "_v", "_md", "_mn", "_gw", "_gs",
    "_since", "_eff", "_ri", "_rb", "_rns", "_pi", "_pns", "_pel", "d_i", "d_b",
    "d_ns", "d_el", "_uo", "_um", "_n", "_q", "_a", "_lbl", "_op", "_gopen", "_pe",
    "_batch", "_fwd",
}


def _direct_names(target):
    """Names a target BINDS. A Subscript or Attribute target mutates, it does not rebind,
    so `clip_seen[0] += 1` must not count as shadowing `clip_seen`."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out = set()
        for e in target.elts:
            out |= _direct_names(e)
        return out
    return set()


def assigned_names(node, skip_nested_defs=True):
    """Every name this node BINDS, skipping nested function bodies by default.

    Nested defs have their own scope, so a name bound only inside a closure is not an
    enclosing-scope binding and must not be reported.
    """
    out, stack = set(), [node]
    while stack:
        n = stack.pop()
        for child in ast.iter_child_nodes(n):
            if skip_nested_defs and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                       ast.Lambda, ast.ClassDef)):
                continue
            stack.append(child)
        if isinstance(n, ast.Assign):
            for t in n.targets:
                out |= _direct_names(t)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            out |= _direct_names(n.target)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            out |= _direct_names(n.target)
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            out |= _direct_names(n.optional_vars)
    return out


tree = ast.parse(io.open(SRC, encoding="utf-8").read())
problems = []
for fname in FUNCS:
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == fname), None)
    if fn is None:
        print(f"  ?? {fname} not found in {SRC.name}")
        continue

    # The training loop is a `while True` NESTED inside `for epoch in ...`, not a
    # top-level statement, so search the whole function (minus nested defs) and pick the
    # while that contains the optimizer step.
    def _walk_no_defs(node):
        stack, seen = [node], []
        while stack:
            n = stack.pop()
            seen.append(n)
            for c in ast.iter_child_nodes(n):
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                                  ast.ClassDef)):
                    continue
                stack.append(c)
        return seen

    whiles = [n for n in _walk_no_defs(fn) if isinstance(n, ast.While)]
    loop = None
    for w in whiles:
        src_w = ast.dump(w)
        if "zero_grad" in src_w or "'step'" in src_w:
            loop = w
            break
    if loop is None:
        print(f"  ?? could not identify the training loop in {fname}")
        continue

    # names bound anywhere in the function EXCEPT inside that loop
    loop_nodes = set(map(id, _walk_no_defs(loop)))
    pre = set()
    for n in _walk_no_defs(fn):
        if id(n) in loop_nodes:
            continue
        if isinstance(n, ast.Assign):
            for t in n.targets:
                pre |= _direct_names(t)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            pre |= _direct_names(n.target)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            pre |= _direct_names(n.target)
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            pre |= _direct_names(n.optional_vars)

    # nested closures capture by reference, so a name they READ is the dangerous kind
    read_by_closures = set()
    for c in [n for n in ast.walk(fn)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn]:
        read_by_closures |= {q.id for q in ast.walk(c)
                             if isinstance(q, ast.Name) and isinstance(q.ctx, ast.Load)}

    inner = assigned_names(loop) - ALLOWED
    for name in sorted((inner & pre) - {"self"}):
        risk = ("is CAPTURED BY A CLOSURE" if name in read_by_closures
                else "is bound before the loop")
        problems.append((fname, name, risk))

print(f"checking {', '.join(FUNCS)} in {SRC.relative_to(ROOT)}\n")
if not problems:
    print("  PASS  no training-loop local shadows an outer name")
    sys.exit(0)
for fname, name, risk in problems:
    print(f"  FAIL  {fname}: loop binds {name!r}, which {risk}")
print("\nRename the loop-local, or add it to ALLOWED if the rebinding is deliberate.")
sys.exit(1)
