"""
plot_results.py — redraw this run's plots from the CSVs, no retraining.

Handy after a sync pulls in new rows, or after tweaking a
plot in shared_code.py.

    python runs/convnext_base_25d_18slices/plot_results.py
    python runs/convnext_base_25d_18slices/plot_results.py --results /runs/convnext_base_25d_18slices/results
"""

import argparse
import sys
from pathlib import Path

EXP_NAME = "convnext_base_25d_mil_emb"


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="results dir (default: this run's)")
    ap.add_argument("--out", default=None, help="subdir for the PNGs (default: plots)")
    args = ap.parse_args()

    cfg = sc.load_config(EXP_DIR, verbose=False)
    results = Path(args.results) if args.results else EXP_DIR / cfg["output"]["run_dir"]
    if not results.exists():
        raise SystemExit(f"no results at {results} — has this run trained yet?")

    made = sc.make_plots(cfg, results, args.out or cfg["output"].get("plots_dir", "plots"))
    print(f"\n📈 {len(made)} plot(s) written")


if __name__ == "__main__":
    main()
