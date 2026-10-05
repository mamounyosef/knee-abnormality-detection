"""resume_stress.py - finish a STRESS publish whose local waiter died while the Kaggle kernel kept running:
waits for the kernel, then verifies and stamps exactly as submit_ensemble.py would have.

    python runs/others/ensembling/resume_stress.py <stress config> [kernel version]
The config's OUT_DIR must still hold the bundle that was uploaded.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
g = {"Path": Path, "__file__": str(HERE / "submit_ensemble.py")}
src = (HERE / "submit_ensemble.py").read_text(encoding="utf-8")
exec(compile(src[:src.index("if len(sys.argv) > 1:")], "submit_ensemble_defaults", "exec"), g)   # its defaults
exec(Path(sys.argv[1]).read_text(encoding="utf-8"), g)                                          # then the config
sys.path.insert(0, str(HERE.parent.parent))
import shared_code as sc  # noqa: E402

out = Path(g["OUT_DIR"])
nb_id = json.loads((out / "notebook" / "kernel-metadata.json").read_text(encoding="utf-8"))["id"].split("/")[-1]
ds_id = json.loads((out / "weights" / "dataset-metadata.json").read_text(encoding="utf-8"))["id"].split("/")[-1]
ok = sc.resume_stress(out, g["KAGGLE_USER"], ds_id, nb_id, int(g["STRESS"]),
                      version=sys.argv[2] if len(sys.argv) > 2 else None)
sys.exit(0 if ok else 1)
