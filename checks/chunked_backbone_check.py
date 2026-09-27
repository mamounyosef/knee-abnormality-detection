"""chunked_backbone_check.py - does feeding the backbone N images at a time change the logits?

Same member, same studies, whole stack vs ChunkedBackbone(n) (data_code/19_ens_member_preds.py).
Should agree to fp16 rounding (max |diff| of a few 1e-3), since the backbone treats every image
independently.

    python checks/chunked_backbone_check.py [member] [n]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ens", ROOT / "data_code/19_ens_member_preds.py")
ens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ens)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "medimageinsight_davit_huge_336_fulldata"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    tm = ens.load_member(name)
    sc, cfg = tm.sc, tm.cfg
    ck = torch.load(ROOT / f"runs/{name}/results/checkpoints/best.pt", map_location="cpu", weights_only=False)
    model = tm.build_model(cfg)
    sc._load_model_state(model, ck["model"])
    dev = torch.device("cuda")
    model = model.to(dev).eval()
    aud = pd.read_csv(ROOT / "data/05_audit58.csv").StudyInstanceUID
    df = sc._eval_df(cfg, ens.STUDIES_CSV)
    df = df[df.StudyInstanceUID.isin(aud)].head(8).reset_index(drop=True)
    plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
    outs = []
    for chunk in (0, n):
        if chunk:
            model.backbone = ens.ChunkedBackbone(model.backbone, chunk)
        pk = {}
        sc._predict_dataframe(cfg, model, df, dev, plan=plan, batch_size=1, num_workers=0, collect=pk)
        outs.append(np.concatenate(pk["_lg"]))
    d = np.abs(outs[0] - outs[1])
    ok = d.max() < 0.05
    print(f"{name}: 8 studies, whole stack vs {n} images at a time: max |logit diff| {d.max():.5f}, "
          f"mean {d.mean():.6f}  {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
