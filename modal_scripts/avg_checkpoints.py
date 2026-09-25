"""avg_checkpoints.py - average a run's saved step checkpoints into one model and score it.

Uniform weight average of step<N>.pt files, written as avg_<first>-<last>.pt beside them.
Scores the average AND each listed reference step on val + audit-58 in the same process,
so the reference reproducing the val_log numbers proves the scoring path. Writes nothing
into results/ except the new checkpoint (results/*_predictions.csv stay the best.pt ones).

    python -m modal run modal_scripts/avg_checkpoints.py --run <run> --first 4000 --last 5100 --refs 3900
"""
import importlib.util
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
sys.path.insert(0, "/root/runs")
import shared_code as sc  # noqa: E402

RUN = os.environ.get("AVG_RUN") or next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--run=")), None)
if RUN is None and "--run" in sys.argv:
    RUN = sys.argv[sys.argv.index("--run") + 1]
cfg = sc.load_config(Path(sc.__file__).resolve().parent / RUN, verbose=False)
app = modal.App(f"{RUN}-avg")
_shards = modal.Volume.from_name(cfg["modal"]["shards_volume"])
_runs = modal.Volume.from_name(cfg["modal"]["runs_volume"])


def _train_mod(run_dir: Path):
    sys.path.insert(0, str(run_dir))
    spec = importlib.util.spec_from_file_location(f"train_{run_dir.name}", run_dir / "train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@app.function(image=sc.modal_image(), secrets=[modal.Secret.from_dict({"AVG_RUN": RUN})],
              volumes={cfg["modal"]["shards_mount"]: _shards, cfg["modal"]["runs_mount"]: _runs},
              **sc.modal_resources(cfg))
def avg_remote(run: str, steps: list, refs: list) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    rcfg = sc.remote_cfg(cfg)
    assert rcfg["experiment"]["name"] == run, rcfg["experiment"]["name"]
    ck_dir = Path(cfg["modal"]["runs_mount"]) / run / rcfg["output"]["run_dir"] / rcfg["output"]["checkpoints_dir"]
    tm = _train_mod(Path("/root/runs") / run)
    dev = torch.device("cuda")

    def ck(s):
        f = ck_dir / f"step{s}.pt"
        return f if f.exists() else ck_dir / f"top_step{s}.pt"

    acc, n, keys = None, 0, None
    for s in steps:
        sd = torch.load(ck(s), map_location="cpu", weights_only=False)
        assert int(sd["step"]) == s, (s, sd["step"])
        sd = sd["model"]
        if acc is None:
            keys = list(sd)
            acc = {k: v.double().clone() if v.is_floating_point() else v.clone() for k, v in sd.items()}
        else:
            assert list(sd) == keys
            for k, v in sd.items():
                if v.is_floating_point():
                    acc[k] += v.double()
        n += 1
    bn = [k for k in keys if k.endswith("running_mean")]
    avg = {k: (v / n).to(torch.float32) if v.is_floating_point() else v for k, v in acc.items()}
    out = ck_dir / f"avg_{steps[0]}-{steps[-1]}.pt"
    torch.save({"model": avg, "cfg": rcfg, "epoch": None, "step": steps[-1], "score": None,
                "track": None, "note": f"uniform weight average of steps {steps}"}, out)
    _runs.commit()

    plan = sc.plan_slots(pd.read_csv(sc.resolve(rcfg["paths"]["series_index"], rcfg)), rcfg)
    cut = float(rcfg.get("target", {}).get("hard_threshold", 0.5))
    sets = {"val": rcfg["paths"]["val_csv"], "audit": rcfg["evaluation"]["audit"]["csv"]}
    dfs = {k: sc._eval_df(rcfg, v) for k, v in sets.items()}
    tasks = rcfg["tasks"]

    def score(state):
        model = tm.build_model(rcfg)
        sc._load_model_state(model, state)
        model = model.to(dev).eval()
        r = {}
        for k, df in dfs.items():
            y, p, _ = sc._predict_dataframe(rcfg, model, df, dev, plan=plan, desc=k)
            y = np.asarray(y) > cut
            from sklearn.metrics import roc_auc_score
            per = [roc_auc_score(y[:, j], p[:, j]) for j in range(len(tasks)) if 0 < y[:, j].sum() < len(y)]
            r[k] = float(np.mean(per))
        return r

    res = {f"avg_{steps[0]}-{steps[-1]}": score(avg)}
    for s in refs:
        res[f"step{s}"] = score(torch.load(ck(s), map_location="cpu", weights_only=False)["model"])
    res["_bn_buffers"] = len(bn)
    res["_n"] = n
    res["_file"] = str(out)
    return res


@app.local_entrypoint()
def main(run: str, first: int, last: int, every: int = 100, refs: str = ""):
    steps = list(range(first, last + 1, every))
    r = avg_remote.remote(run, steps, [int(x) for x in refs.split(",") if x])
    for k, v in r.items():
        print(k, v)
