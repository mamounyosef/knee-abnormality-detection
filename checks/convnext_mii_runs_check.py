"""convnext_mii_runs_check.py - does each ConvNeXt + MII label run train on exactly the labels it claims?

For runs/convnext_base_25d_18slices_oof_convnext_mii_labels_w30 and _w50, independently of
data_code/22_make_convnext_mii_labels.py:
  1. the run loads its OWN config through its train.py; name matches the folder
  2. the train / val CSVs it resolves are the 22_* files of ITS weight (w30 or w50)
  3. labels rebuilt from the raw sources match the file cell by cell:
       v3 = logit(Qwen), with logit(MedGemma) averaged in on the five MedGemma findings
       teacher = mean of the ConvNeXt fold logit and the MII fold logit, each from the ONE fold
                 model whose held-out set contains the study (its best-val prediction file)
       label = sigmoid((1 - w) * v3 + w * teacher)
     and the effective source weights per finding are printed
  4. same studies, same order, same non-label columns as the donor's 17_* files; no audit
     study; no NaN; all labels in (0, 1)
  5. everything else in the merged config equals the donor convnext_base_25d_18slices_mglabels_raw
     except the expected keys (name, the two CSVs, patience 12 -> 7, GPU A100-80GB -> H100, cpu 14 -> 16)
  6. the w30 and w50 files differ, and w50 sits further from v3 than w30
  7. the files are inside data/ and not excluded from the Modal image

    python checks/convnext_mii_runs_check.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
RUNS = ROOT / "runs"
sys.path.insert(0, str(RUNS))
sys.path.insert(0, str(RUNS / "others/ensembling"))
import sweep_new_runs as sw  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
B5 = ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]
DONOR = "convnext_base_25d_18slices_mglabels_raw"
RUN = "convnext_base_25d_18slices_oof_convnext_mii_labels_{}"
WEIGHTS = {"w30": 0.3, "w50": 0.5, "w60": 0.6}
EXPECTED_DIFF = {"experiment.name", "paths.train_csv", "paths.val_csv", "training.early_stopping.patience",
                 "modal.gpu", "modal.cpu"}   # H100 + 16 cpu instead of A100-80GB + 14: hardware only
bad = 0


def check(ok, msg):
    global bad
    bad += not ok
    print(f"  {'OK  ' if ok else 'FAIL'}  {msg}")


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_run(run):
    spec = importlib.util.spec_from_file_location(run, RUNS / run / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    return tm.cfg, tm.sc


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def teacher_logits(pattern, fold_csv):
    """study -> logits from the one fold model that held the study out (its best-val file)."""
    parts = []
    for k in range(5):
        run = pattern.format(k)
        f = sw.best_npz(run, "val")
        z = np.load(f, allow_pickle=True)
        df = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T)
        held = set(pd.read_csv(D / fold_csv.format(k)).StudyInstanceUID.astype(str))
        assert set(df.index) == held, f"{run}: {f.name} is not exactly its held-out fold"
        parts.append(df)
    o = pd.concat(parts)
    assert o.index.is_unique
    return o


print("sources")
Q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID")
M = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID")
CNX = teacher_logits("convnext_base_25d_18slices_f{}", "13_fold{}_val.csv")
MII = teacher_logits("medimageinsight_davit_huge_336_v3_f{}", "21_fold{}_val.csv")
check(set(CNX.index) == set(MII.index) and len(CNX) == 4342, f"both teachers cover the same {len(CNX)} studies")
audit = set(pd.read_csv(D / "05_audit58.csv").StudyInstanceUID.astype(str))

dcfg, dsc = load_run(DONOR)
files = {}
for tag, w in WEIGHTS.items():
    run = RUN.format(tag)
    print(f"\n{run}  (teacher weight {w})")
    cfg, sc = load_run(run)
    check(cfg["experiment"]["name"] == run, "1. loads its own config, experiment.name matches the folder")
    diff = {k for k in set(flat(cfg)) | set(flat(dcfg)) if flat(cfg).get(k) != flat(dcfg).get(k)}
    check(diff == EXPECTED_DIFF, f"5. config differs from {DONOR} only in {sorted(diff)}")
    check(cfg["training"]["early_stopping"]["patience"] == 7 and cfg["training"]["early_stopping"]["enable"],
          "5. early stopping on, patience 7")
    check(cfg["input"].get("laterality_flip") == dcfg["input"].get("laterality_flip"),
          f"5. laterality_flip same as donor ({cfg['input'].get('laterality_flip')})")
    print(f"        GPU {cfg['modal']['gpu']}, cpu {cfg['modal'].get('cpu')}, memory {cfg['modal'].get('memory_gb')} GB, "
          f"epochs {cfg['training']['epochs']}, batch {cfg['dataloader']['batch_size']} x accum "
          f"{cfg['dataloader']['accum_steps']}")
    for part, donor_csv in (("train", "17_train_qwen-mg-raw.csv"), ("val", "17_val_qwen-mg-raw.csv")):
        p = Path(sc.resolve(cfg["paths"][f"{part}_csv"], cfg))
        want = f"22_{part}_qwen-mg-convnext-mii-{tag}.csv"
        check(p.name == want and p.parent == D, f"2. {part}: reads data/{p.name}")
        df = pd.read_csv(p)
        don = pd.read_csv(D / donor_csv)
        files[(tag, part)] = df
        check(list(df.StudyInstanceUID) == list(don.StudyInstanceUID), f"4. {part}: {len(df)} studies, same order as {donor_csv}")
        other = [c for c in don.columns if c not in T]
        check(list(df.columns) == list(don.columns) and df[other].astype(str).equals(don[other].astype(str)),
              f"4. {part}: same columns, non-label columns unchanged")
        ids = df.StudyInstanceUID.astype(str).tolist()
        check(not (set(ids) & audit), f"4. {part}: no audit study")
        v = df[T].values
        check(np.isfinite(v).all() and v.min() > 0 and v.max() < 1, f"4. {part}: no NaN, all labels in (0, 1)")
        v3 = lg(Q.loc[ids, T].values)
        bi = [T.index(t) for t in B5]
        v3[:, bi] = 0.5 * lg(Q.loc[ids, B5].values) + 0.5 * lg(M.loc[ids, B5].values)
        teach = 0.5 * CNX.loc[ids, T].values + 0.5 * MII.loc[ids, T].values
        exp = 1 / (1 + np.exp(-((1 - w) * v3 + w * teach)))
        err = np.abs(exp - v).max()
        check(err < 1e-6, f"3. {part}: rebuilt from Qwen / MedGemma / ConvNeXt folds / MII folds, max |diff| {err:.1e}")
    q5, q7 = (1 - w) / 2, 1 - w
    print(f"        effective weights (logit scale):  MedGemma five: Qwen {q5:.0%} MedGemma {q5:.0%} "
          f"ConvNeXt {w / 2:.0%} MII {w / 2:.0%}  |  other seven: Qwen {q7:.0%} ConvNeXt {w / 2:.0%} MII {w / 2:.0%}")
    ign = ["shards", "raw", "backups", "stats", "sample_dicoms", "pretrained", "_relay_stage"]
    check(not any(x in str(p.relative_to(D)) for x in ign) and p.suffix == ".csv",
          "7. label files travel in the Modal image (data/ CSV, not excluded)")

print("\nw30 vs w50")
for part in ("train", "val"):
    a, b = files[("w30", part)][T].values, files[("w50", part)][T].values
    don = pd.read_csv(D / f"17_{part}_qwen-mg-raw.csv")[T].values
    check(np.abs(a - b).max() > 0.01, f"6. {part}: the two label sets differ (max |diff| {np.abs(a - b).max():.3f})")
    check(np.abs(b - don).mean() > np.abs(a - don).mean(),
          f"6. {part}: mean distance from v3, w30 {np.abs(a - don).mean():.4f} < w50 {np.abs(b - don).mean():.4f}")
print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}")
sys.exit(1 if bad else 0)
