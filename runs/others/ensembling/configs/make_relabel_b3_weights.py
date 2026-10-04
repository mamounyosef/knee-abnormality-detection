"""make_relabel_b3_weights.py - write relabel_b3_weights.json from relabel_b3.py's MEMBERS.

Every full member weight 1 on all 12 findings; each specialist its weight on its own findings and 0 elsewhere (the
engine also ignores members that do not predict a finding). The engine renormalises each finding's column over the
members that predict it, which is exactly the offline formula (sum of the 14 + w x specialist) / (14 + w).

    python runs/others/ensembling/configs/make_relabel_b3_weights.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OWN = {"convnext_base_25d_v5_spec_trauma": ["Fracture", "Contusion"],
       "convnext_base_25d_v5_spec_pfoa": ["PF OA"],
       "convnext_base_25d_v5_spec_fluid": ["Effusion", "Synovitis", "Baker's"]}
g = {"Path": Path}
exec((HERE / "relabel_b3.py").read_text(encoding="utf-8"), g)
names = [n for n, _ in g["MEMBERS"]]
w = {t: {n: (float(g["SPEC_W"][n][1]) if t in OWN[n] else 0.0) if n in OWN else 1.0 for n in names} for t in T}
out = HERE.parent / "relabel_b3_weights.json"
out.write_text(json.dumps({"members": names, "weights_per_class": w,
                           "note": "batch 3: full members 1; trauma 1 on Fracture/Contusion, pfoa 2 on PF OA, fluid 1 on "
                                   "Effusion/Synovitis/Baker's"}, indent=1), encoding="utf-8")
for t in T:
    print(f"  {t:17s} " + ", ".join(f"{n.replace('convnext_base_25d_v5_', '')} {v:g}" for n, v in w[t].items() if v != 1.0 or n in OWN))
print("wrote", out)
