"""
lint_notebook.py - catch undefined names in a generated submission notebook.

The notebook is assembled from raw string cells inside shared_code, so a name that
exists in the engine but was never written into the notebook (say a helper added later
for training) parses fine, uploads fine, and only fails on Kaggle, minutes later, with
the traceback swallowed by IPython's own formatter. That happened once and cost a full
push cycle. One second of static checking here replaces it.

The check: walk every cell in order, collect every name the notebook BINDS (imports,
assignments, defs, classes, comprehension and loop targets, function arguments, with/
except targets), then flag every name it LOADS that is neither bound earlier nor a
builtin. Cells run top to bottom in one namespace, so ordering matters and is respected.

    python runs/others/ensembling/lint_notebook.py <path to submit.ipynb>
"""

import ast
import builtins
import json
import sys
from pathlib import Path


def _bound(tree):
    """Every name this cell binds, at any nesting depth."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
            args = getattr(node, "args", None)
            if args:
                for a in (list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs)
                          + [args.vararg, args.kwarg]):
                    if a is not None:
                        out.add(a.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            out.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            out.update(node.names)
    return out


def _loaded(tree):
    """(name, lineno) for every name this cell reads."""
    return [(n.id, n.lineno) for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]


def lint(path):
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    known = set(dir(builtins)) | {"__name__", "__file__", "get_ipython", "display", "_"}
    problems = []
    for ci, cell in enumerate(nb["cells"], 1):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append("cell %d: SyntaxError line %s: %s" % (ci, e.lineno, e.msg))
            continue
        # a cell's own bindings count before its loads are judged: a helper defined at
        # the bottom of a cell is legitimately called by a function defined at the top,
        # and cell bindings stay visible to every later cell
        known |= _bound(tree)
        for name, line in _loaded(tree):
            if name not in known:
                problems.append("cell %d line %d: %r is never defined in the notebook"
                                % (ci, line, name))
    return problems


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent / "kaggle" / "notebook" / "submit.ipynb")
    if not target.exists():
        raise SystemExit("no notebook at %s - build one first" % target)
    bad = lint(target)
    if bad:
        print("%d problem(s) in %s" % (len(bad), target))
        for b in bad:
            print("  " + b)
        raise SystemExit(1)
    print("%s: no undefined names" % target.name)
