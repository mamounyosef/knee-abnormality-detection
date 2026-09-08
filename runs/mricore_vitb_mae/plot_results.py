"""
plot_results.py — reconstruction loss curves for the MAE run.

    python runs/mricore_vitb_mae/plot_results.py
"""

import sys
from pathlib import Path

EXP_NAME = "mricore_vitb_mae"
PKG_ROOT = next((c for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent)
                 if (c / "shared_code.py").exists()), Path(__file__).resolve().parent.parent)
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    cfg = sc.load_config(EXP_DIR, verbose=False)
    res = EXP_DIR / cfg["output"]["run_dir"]
    log = res / "mae_log.csv"
    if not log.exists():
        raise SystemExit(f"no {log} - has this run trained yet?")
    d = pd.read_csv(log)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(d.epoch, d.train_loss, label="train")
    # the holdout studies are never trained on, so a gap here is real overfitting
    ax[0].plot(d.epoch, d.holdout_loss, label="holdout")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("masked-patch MSE")
    ax[0].set_title("reconstruction loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(d.epoch, d.lr); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("decoder lr")
    ax[1].set_title("schedule"); ax[1].grid(alpha=.3)
    fig.tight_layout()

    out = res / cfg["output"].get("plots_dir", "plots")
    out.mkdir(parents=True, exist_ok=True)
    p = out / "mae_loss.png"
    fig.savefig(p, dpi=120)
    print(f"  \U0001F4C8 {p}")
    print(d.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
