"""make_fit10_weights.py - write fit10_weights.json from combined_public_fit10.py's MEMBERS.

Every member weight 1 on all 12 findings, except the joint-zoom model: 1 on the 5 joint findings, 0 elsewhere. The
engine renormalises each finding's column over the members that vote on it, which is the offline formula.

    python runs/others/ensembling/configs/make_fit10_weights.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
JOINT = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]
ZOOM = "convnext_base_25d_v5_joint_zoom_030"
g = {"Path": Path}
exec((HERE / "combined_public_fit10.py").read_text(encoding="utf-8"), g)
names = [n for n, _ in g["MEMBERS"]]
assert ZOOM in names
w = {t: {n: ((1.0 if t in JOINT else 0.0) if n == ZOOM else 1.0) for n in names} for t in T}
out = HERE.parent / "fit10_weights.json"
out.write_text(json.dumps({"members": names, "weights_per_class": w,
                           "note": "fit-10: all members 1; the 0.30 mm joint zoom 1 on the 5 joint findings, 0 elsewhere"},
                          indent=1), encoding="utf-8")
for t in T:
    print(f"  {t:17s} zoom weight {w[t][ZOOM]:g}")
print("wrote", out)
