"""fp16_accuracy_check.py - does pure fp16 inference (shared_code._HalfInfer, KNEE_INFER_HALF=1) change what a member
predicts on REAL studies? Scores audit-58 from the local shards twice per member: today's autocast vs _HalfInfer.
Reports macro AUROC both ways, max |logit diff| and wall time.

    python runs/others/ensembling/fp16_accuracy_check.py <bundle weights dir> [member indices]
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

W = Path(sys.argv[1])
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
import shared_code as sc  # noqa: E402

IDX = [int(i) for i in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
dev = torch.device("cuda:0")
members = json.loads((W / "members.json").read_text())
for mm in members:
    i = mm["i"]
    if IDX is not None and i not in IDX:
        continue
    cfg = json.loads((W / ("m%d_config.json" % i)).read_text())
    spec = importlib.util.spec_from_file_location("m%d_def" % i, W / ("m%d_def.py" % i))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m%d_def" % i] = mod
    spec.loader.exec_module(mod)
    model = mod.build_model(cfg)
    sd = torch.load(W / ("m%d_model.pt" % i), map_location="cpu", weights_only=False)["model"]
    model.load_state_dict({k.replace("_orig_mod.", "", 1): v for k, v in sd.items()})
    model = model.eval().to(dev)
    plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
    df = sc._eval_df(cfg, "05_audit58.csv")
    out = {}
    for mode in ("amp", "fp16"):
        m = model if mode == "amp" else sc._HalfInfer(model)
        col = {}
        torch.cuda.synchronize(); t0 = time.perf_counter()
        y, p, _ = sc._predict_dataframe(cfg, m, df, dev, plan=plan, batch_size=1, num_workers=0, collect=col)
        torch.cuda.synchronize()
        out[mode] = (y, np.asarray(col.get("logits", p)), time.perf_counter() - t0)
    y = out["amp"][0]
    T = cfg["tasks"]
    mac = {k: float(np.mean([roc_auc_score(y[:, j], out[k][1][:, j]) for j in range(len(T)) if 0 < y[:, j].sum() < len(y)]))
           for k in out}
    print(json.dumps({"i": i, "name": mm["name"], "auroc_amp": round(mac["amp"], 5), "auroc_fp16": round(mac["fp16"], 5),
                      "max_logit_diff": float(np.abs(out["amp"][1] - out["fp16"][1]).max()),
                      "sec_amp": round(out["amp"][2], 1), "sec_fp16": round(out["fp16"][2], 1)}), flush=True)
    del model
    torch.cuda.empty_cache()
