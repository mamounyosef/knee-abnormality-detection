"""laterality_decode_group_check.py - a member must never be scored on another member's mirroring.

THE BUG THIS PINS. predict_dicom_ensemble_grouped buckets members by loader_geometry() and
builds ONE dataset per bucket from that bucket's FIRST member's config. laterality_flip was
not among LOADER_GEOMETRY_KEYS, so convnext_base_25d_poolattn - trained on unmirrored
studies - shared a bucket with convnext_base_25d_all_knees_same_orientation and was scored
on MIRRORED pixels for every left knee. That is 1,938 of 4,407 studies.

It was silent because a mirrored tensor has exactly the same shape. The guard inside
predict_dicom_ensemble exists to catch "member i disagrees on loader geometry" and it could
not see this one, because the key it compares did not include the setting that differed.

Run:  python checks/laterality_decode_group_check.py
"""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

PASS, FAIL = [], []


def ck(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  %s %-60s %s" % ("PASS" if ok else "FAIL", name, detail))


print("\n1. the key itself")
ck("laterality_flip is part of the loader geometry key",
   "laterality_flip" in sc.LOADER_GEOMETRY_KEYS, str(sc.LOADER_GEOMETRY_KEYS[-1]))

print("\n2. two configs that differ ONLY in mirroring must not share a decode pass")
base = sc.load_config(ROOT / "runs" / "convnext_base_25d_poolattn", verbose=False)
mirrored = copy.deepcopy(base)
mirrored["input"]["laterality_flip"] = "left"
ck("identical configs group together",
   sc.loader_geometry(base) == sc.loader_geometry(copy.deepcopy(base)))
ck("mirrored vs unmirrored group APART",
   sc.loader_geometry(base) != sc.loader_geometry(mirrored))

print("\n3. the single-pass guard now refuses the mismatch instead of allowing it")
bad = [k for k in sc.LOADER_GEOMETRY_KEYS
       if base["input"].get(k) != mirrored["input"].get(k)]
ck("the guard's own comparison spots it", bad == ["laterality_flip"], str(bad))

print("\n4. the real shipped members split correctly")
MEMBERS = ["convnext_base_relabel", "mricore_vitb_cross_attention",
           "mricore_vitb_mae_cross_attention", "dinov3_convnext_base",
           "convnext_base_25d_all_knees_same_orientation", "swinv2_base_in22k",
           "convnext_base_25d_poolattn",
           "convnext_base_25d_perfinding_slots_and_pooling"]
groups = {}
for i, name in enumerate(MEMBERS):
    cfg = sc.load_config(ROOT / "runs" / name, verbose=False)
    groups.setdefault(sc.loader_geometry(cfg), []).append(
        (name, cfg["input"].get("laterality_flip")))
print("     %d decode groups" % len(groups))
mixed = []
for g, members in groups.items():
    lats = {lat for _, lat in members}
    print("       %d member(s), laterality %s: %s"
          % (len(members), lats, ", ".join(n[:34] for n, _ in members)))
    if len(lats) > 1:
        mixed.append(members)
ck("no group mixes mirrored and unmirrored members", not mixed,
   "%d mixed" % len(mixed) if mixed else "clean")

print("\n5. and mirroring really does change the pixels, so the split is not cosmetic")
v = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
same = all(np.array_equal(sc._mirror_slot(v, s), v) for s in ("sag_plain", "cor_fluid"))
ck("a mirrored slot differs from the original", not same)

print("\n" + "=" * 74)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("    FAILED: %s" % f)
sys.exit(1 if FAIL else 0)
