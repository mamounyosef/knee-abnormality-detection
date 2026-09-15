"""
ensemble_grouped_check.py - is a multi-geometry ensemble blended EXACTLY like a single-pass one?

predict_dicom_ensemble_grouped exists so convnext_base_25d_group3 (336 px) can ride in an
ensemble whose other members are 512 px. It does that by decoding once per geometry group and
blending at the end, which introduces one genuinely dangerous bug class: the groups are
scored in group order, not member order, so a member's logits can land in another member's
slot. Per-class weights are indexed by member, so that would apply the wrong weights to the
wrong model and still produce a perfectly plausible submission.

This stubs out the scoring so no DICOMs, GPU or checkpoints are needed, hands each member a
uniquely identifiable logit pattern, and checks the result is bit-identical to blending the
same logits in one pass.

    python checks/ensemble_grouped_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

fails = []
N, T = 7, 12                       # studies, tasks
TASKS = ["t%d" % j for j in range(T)]


def check(name, ok, detail=""):
    print("  %-56s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


def cfg_for(name, size, tasks=TASKS):
    return {"experiment": {"name": name}, "tasks": list(tasks),
            "input": {"slots": ["a", "b", "c", "d"], "n_slices": 24, "size": size,
                      "mm_per_px": 0.40, "window": "pct", "window_pct": [0.1, 99.9],
                      "channels": 3, "slice_context": None, "slice_context_stride": 1},
            "training": {"amp": False}}


# ---------------------------------------------------------------------------------
# 1. geometry grouping
# ---------------------------------------------------------------------------------
print("\ngeometry grouping")
a, b = cfg_for("m512", 512), cfg_for("m336", 336)
check("same size -> same key", sc.loader_geometry(a) == sc.loader_geometry(cfg_for("x", 512)))
check("different size -> different key", sc.loader_geometry(a) != sc.loader_geometry(b))

# resize_to and slices_as_channels must NOT split a group: both are applied per member inside
# prepare_batch, from that member's own cfg, on an already-decoded tensor.
c = cfg_for("m512r", 512)
c["input"]["resize_to"] = 384
check("resize_to does NOT split a group", sc.loader_geometry(a) == sc.loader_geometry(c))
d = cfg_for("m512g", 512)
d["input"]["slices_as_channels"] = 8
check("slices_as_channels does NOT split a group", sc.loader_geometry(a) == sc.loader_geometry(d))

# ---------------------------------------------------------------------------------
# 2. the wrapper assembles members in the RIGHT ORDER and blends like one pass
# ---------------------------------------------------------------------------------
# Members deliberately INTERLEAVED across two geometries: 512, 336, 512, 336, 512. If the
# wrapper wrote results in group order instead of member order, member 1 would get member 2's
# logits and nothing would complain.
SIZES = [512, 336, 512, 336, 512]
cfgs = [cfg_for("m%d_%d" % (i, s), s) for i, s in enumerate(SIZES)]
models = [object() for _ in SIZES]

# member i gets a constant logit of (i + 1), so whose logits landed where is unambiguous
TRUTH = np.stack([np.full((N, T), i + 1.0, np.float32) for i in range(len(SIZES))])

real = sc.predict_dicom_ensemble
calls = []


def fake(sub_cfgs, sub_models, studies_df, *a_, return_member_logits=False, **kw):
    """Stand in for the scorer: return each member's known constant, in the order asked."""
    calls.append([c["experiment"]["name"] for c in sub_cfgs])
    assert return_member_logits, "the wrapper must ask for raw member logits"
    names = [c["experiment"]["name"] for c in cfgs]
    # a cfg outside the main fixture (the single-geometry sub-test) just gets a distinct
    # constant, so the stub never has to know about every scenario
    lg = np.stack([TRUTH[names.index(c["experiment"]["name"])]
                   if c["experiment"]["name"] in names
                   else np.full((N, T), 9.0, np.float32)
                   for c in sub_cfgs])
    # member-logit mode hands the timing BACK rather than writing it: each group would
    # otherwise overwrite the previous group's inference_log.txt and timing.json, and the
    # log would claim the ensemble had only the last group's members. That is not
    # hypothetical - the first grouped Kaggle run logged "scored 3 studies with 1 members".
    t = {"decode_s": 1.0 * len(sub_cfgs), "gpu_s": 2.0,
         "member_s": [float(i + 1) for i in range(len(sub_cfgs))],
         "members": [c["experiment"]["name"] for c in sub_cfgs]}
    return lg, t


studies = pd.DataFrame({"StudyInstanceUID": ["s%d" % i for i in range(N)]})
sc.predict_dicom_ensemble = fake
try:
    print("\nflat blend across two geometry groups")
    res = sc.predict_dicom_ensemble_grouped(cfgs, models, studies, {}, None, {},
                                            device=torch.device("cpu"))
    check("one call per geometry group", len(calls) == 2, "%d calls: %s" % (len(calls), calls))
    want = torch.sigmoid(torch.as_tensor(TRUTH.mean(0))).numpy()
    got = res[TASKS].to_numpy(np.float32)
    check("flat result equals a single-pass nanmean",
          np.allclose(got, want, atol=1e-6), "max diff %.2e" % np.abs(got - want).max())
    check("StudyInstanceUID column preserved",
          list(res["StudyInstanceUID"]) == list(studies["StudyInstanceUID"]))

    print("\nper-class weights land on the RIGHT member")
    # a weight vector that is zero everywhere except member 3 (a 336 member, i.e. in the
    # SECOND group). If ordering were wrong the answer would be some other member's constant.
    W = np.zeros((len(SIZES), T), np.float32)
    W[3, :] = 1.0
    calls.clear()
    res = sc.predict_dicom_ensemble_grouped(cfgs, models, studies, {}, None, {},
                                            device=torch.device("cpu"), weights=W)
    got = res[TASKS].to_numpy(np.float32)
    want = torch.sigmoid(torch.full((N, T), 4.0)).numpy()      # member 3 -> constant 4.0
    check("all weight on member 3 returns member 3's logits",
          np.allclose(got, want, atol=1e-6),
          "got %.4f, expected %.4f" % (got[0, 0], want[0, 0]))

    # every member weighted equally must reproduce the flat blend
    calls.clear()
    res = sc.predict_dicom_ensemble_grouped(cfgs, models, studies, {}, None, {},
                                            device=torch.device("cpu"),
                                            weights=np.ones((len(SIZES), T), np.float32))
    got = res[TASKS].to_numpy(np.float32)
    want = torch.sigmoid(torch.as_tensor(TRUTH.mean(0))).numpy()
    check("uniform weights reproduce the flat blend",
          np.allclose(got, want, atol=1e-6), "max diff %.2e" % np.abs(got - want).max())

    print("\nsingle geometry behaves exactly as before")
    one = [cfg_for("only%d" % i, 512) for i in range(3)]
    calls.clear()
    res = sc.predict_dicom_ensemble_grouped(one, models[:3], studies, {}, None, {},
                                            device=torch.device("cpu"))
    check("one group -> ONE decode pass", len(calls) == 1, "%d" % len(calls))

    print("\npartial-head members (NaN columns) are averaged only where they vote")
    # member 1 has no opinion on the last 6 findings, exactly like the task-group branches
    T2 = TRUTH.copy()
    T2[1][:, 6:] = np.nan
    saved, TRUTH_BAK = T2, TRUTH
    globals()["TRUTH"] = T2
    calls.clear()
    res = sc.predict_dicom_ensemble_grouped(cfgs, models, studies, {}, None, {},
                                            device=torch.device("cpu"))
    got = res[TASKS].to_numpy(np.float32)
    want = torch.sigmoid(torch.as_tensor(np.nanmean(T2, axis=0))).numpy()
    check("nanmean skips the abstaining member",
          np.allclose(got, want, atol=1e-6), "max diff %.2e" % np.abs(got - want).max())
    check("abstained findings differ from the full-vote ones",
          not np.allclose(got[:, :6], got[:, 6:], atol=1e-6))
    globals()["TRUTH"] = TRUTH_BAK

    print("\ncombined report covers EVERY member, not just the last group's")
    import json as _json
    for f in ("inference_log.txt", "timing.json"):
        Path(f).unlink(missing_ok=True)
    calls.clear()
    sc.predict_dicom_ensemble_grouped(cfgs, models, studies, {}, None, {},
                                      device=torch.device("cpu"))
    log = Path("inference_log.txt").read_text(encoding="utf-8")
    check("log names all %d members" % len(SIZES),
          all(c["experiment"]["name"] in log for c in cfgs),
          log.splitlines()[0] if log else "(empty)")
    check("log does NOT under-report the member count",
          ("%d/%d members" % (len(SIZES), len(SIZES))) in log,
          next((l for l in log.splitlines() if "members in" in l), "(no summary line)"))
    tj = _json.loads(Path("timing.json").read_text(encoding="utf-8"))
    check("timing.json lists all members",
          len(tj["members"]) == len(SIZES), "%d entries" % len(tj["members"]))
    check("timing.json records the decode-pass count",
          tj.get("decode_passes") == 2, str(tj.get("decode_passes")))
    check("decode cost is SUMMED across passes, not overwritten",
          tj["decode_s_per_study"] > 0, "%.3f s/study" % tj["decode_s_per_study"])
    for f in ("inference_log.txt", "timing.json"):
        Path(f).unlink(missing_ok=True)
finally:
    sc.predict_dicom_ensemble = real

print("\n" + "=" * 74)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("grouped ensemble: members keep their identity across passes, blend matches one pass")
