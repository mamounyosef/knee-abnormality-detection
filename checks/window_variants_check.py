"""window_variants_check.py - the 8 Oct _normalise refactor equals the old window code for every existing mode, and the
new modes (zscore, fgref, *_clahe) give valid 0-255 images with blank padding kept blank. CPU, synthetic volumes.

    python checks/window_variants_check.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

fails, n = [], [0]


def check(name, ok, detail=""):
    n[0] += 1
    print("  %-60s %s%s" % (name, "ok" if ok else "FAIL", "  " + str(detail) if detail else ""))
    if not ok:
        fails.append(name)


class Fake:
    pass


rng = np.random.default_rng(0)
for win in ("volume", "slice", "shard"):
    for pct in ([1.0, 99.0], [0.5, 99.5]):
        for padn in (0, 3):
            d = Fake()
            d.window, d.cfg = win, {"input": {"window_pct": pct}}
            vol = (rng.gamma(2.0, 40.0, (18, 64, 64)) * (rng.random((18, 64, 64)) > 0.4)).astype(np.float32)
            pad = np.zeros(18, bool)
            pad[18 - padn:] = True
            vol[pad] = 0
            a = sc.KneeShardDataset._normalise(d, vol.copy(), pad)
            b = sc.KneeShardDataset._normalise_legacy(d, vol.copy(), pad)
            check(f"{win} {pct} pad {padn}: new == legacy", np.array_equal(a, b), f"max diff {np.abs(a - b).max():.2e}")
for win in ("zscore", "fgref", "volume_clahe", "zscore_clahe"):
    d = Fake()
    d.window, d.cfg = win, {"input": {"window_pct": [1.0, 99.0]}}
    vol = (rng.gamma(2.0, 40.0, (18, 64, 64)) * (rng.random((18, 64, 64)) > 0.4)).astype(np.float32)
    pad = np.zeros(18, bool); pad[-2:] = True; vol[pad] = 0
    o = sc.KneeShardDataset._normalise(d, vol.copy(), pad)
    check(f"{win}: in [0, 255], padding blank, not constant",
          o.min() >= 0 and o.max() <= 255 and (o[pad] == 0).all() and o[~pad].std() > 5,
          f"range {o.min():.0f}-{o.max():.0f}, sd {o[~pad].std():.1f}")
print("\n" + ("ALL OK (%d/%d)" % (n[0], n[0]) if not fails else "FAILED: %s" % ", ".join(fails)))
