"""check_combined_bundle.py - a built combined bundle vs the stress stamp of the ensemble it extends.

Every weight file of the stamp must be present with the same byte size (so our side is the bundle that already
passed), the notebook must hold exactly one STRESS and one PUBLIC_W line, and the public cell must come before
any of our cells that import torch.

    python runs/others/ensembling/public_side/check_combined_bundle.py C:/temp1/relabel_b2 b4ae8ed14ec0
"""
import json
import re
import sys
from pathlib import Path

out, fp = Path(sys.argv[1]), sys.argv[2]
stamps = list((Path(__file__).resolve().parents[1] / "stress_passes").glob(fp + "*.json"))
assert len(stamps) == 1, stamps
want = json.loads(stamps[0].read_text(encoding="utf-8"))["manifest"]
have = {f.name: f.stat().st_size for f in (out / "weights").iterdir() if f.is_file() and f.name != "dataset-metadata.json"}
bad = [(k, want[k], have.get(k)) for k in want if k != "shared_code.py" and have.get(k) != want[k]]
extra = sorted(set(have) - set(want))
print(f"stamp files {len(want)}, bundle files {len(have)}; differing (engine excluded): {bad or 'none'}; extra: {extra}")
nb = json.loads((out / "notebook" / "submit.ipynb").read_text(encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
lines = [l for c in code for l in c.splitlines()]
print("STRESS lines:", [l for l in lines if re.match(r"STRESS = \d+", l)])
print("PUBLIC_W lines:", [l for l in lines if l.startswith("PUBLIC_W = ")])
pub_i = next(i for i, c in enumerate(code) if "PUBLIC PIPELINE FIRST" in c)
torch_i = next(i for i, c in enumerate(code) if "torch" in c and "import" in c)
print(f"public cell #{pub_i}, first torch import cell #{torch_i}, blend cell last: {'BLEND per finding' in code[-1]}")
meta = json.loads((out / "notebook" / "kernel-metadata.json").read_text(encoding="utf-8"))
print("datasets:", meta["dataset_sources"]); print("models:", meta.get("model_sources"))
ok = not bad and extra == ["public_pipeline.ipynb"] and pub_i < torch_i and "BLEND per finding" in code[-1]
print("OK" if ok else "PROBLEM")
