"""engine_accum_smoke.py - run the REAL training loop on a tiny local set and check its step
accounting: partial last accumulation window stepped, steps per epoch = ceil(micro / accum),
stop_after_steps honoured, no "discarded" tail.

22 studies at batch 2 x accum 4 = 11 micro-batches -> 3 steps per epoch (4 + 4 + 3 partial).
2 epochs -> 6 steps; stop_after_steps 5 must end it at step 5. Writes into a scratch folder.

    python checks/engine_accum_smoke.py [run]
"""
import copy
import importlib.util
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "convnext_base_25d_18slices_mglabels_raw_localcheck"
    sys.path.insert(0, str(ROOT / "runs"))
    spec = importlib.util.spec_from_file_location("train", ROOT / "runs" / run / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    sc = tm.sc
    tmp = Path(tempfile.mkdtemp(prefix="accum_smoke_"))
    data = tmp / "data"
    data.mkdir()
    cfg = copy.deepcopy(tm.cfg)
    tr = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg)).head(22)
    va = pd.read_csv(sc.resolve(cfg["paths"]["val_csv"], cfg)).head(8)
    tr.to_csv(data / "tr.csv", index=False)
    va.to_csv(data / "va.csv", index=False)
    au = pd.read_csv(sc.resolve(cfg["paths"]["audit_csv"], cfg)).head(8)
    au.to_csv(data / "au.csv", index=False)
    cfg["paths"].update(train_csv=str(data / "tr.csv"), val_csv=str(data / "va.csv"),
                        audit_csv=str(data / "au.csv"))
    cfg["training"].update(epochs=2, stop_after_steps=5)
    cfg["training"]["early_stopping"]["enable"] = False
    cfg["evaluation"].update(eval_every_steps_before=1, eval_every_steps_mid=1, eval_every_steps_after=1)
    cfg["output"].update(checkpoint_every_steps=0, keep_top_k=0, upload_checkpoints=None,
                         save_debug_images=0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        sc.run_experiment(cfg, tm.build_model(cfg), tmp)
    txt = buf.getvalue()
    log = pd.read_csv(tmp / cfg["output"]["run_dir"] / "train_log.csv")
    steps = sorted(log.global_step.unique())
    per_epoch = log.groupby("epoch").global_step.nunique().to_dict()
    checks = {
        "steps logged 1..5": steps == [1, 2, 3, 4, 5],
        "epoch 1 has 3 steps (4+4+3 partial micro-batches)": per_epoch.get(1) == 3,
        "no 'discarded trailing micro-batch' message": "discarded" not in txt,
        "stop_after_steps message": "stop_after_steps=5 reached at step 5" in txt,
        "finite loss on every step": log.loss.notna().all() and (log.loss < 10).all(),
    }
    for k, v in checks.items():
        print(f"  {'OK  ' if v else 'FAIL'}  {k}")
    print(f"  steps per epoch: {per_epoch}; scratch: {tmp}")
    ok = all(checks.values())
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
