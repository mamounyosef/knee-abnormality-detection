"""
23_make_split8.py - a smaller validation split (8%) that also holds the 58 audit studies.

WHY (user, 2 Oct). The old split holds out 704 studies (16%), which never train. Full-data runs
train on everything but then have nothing to stop on, and audit-58 alone is too noisy to stop on
(runs/others/audit_noise_check.py: from epoch 6 on, its check-to-check moves do not replicate
across halves, reliability 0.06). So: hold out HALF as much (8%), train on the rest, and put the 58
panel-labelled studies INTO validation too, so the stop signal includes the only ground truth.
Audit-58 is still reported on its own (paths.audit_csv stays 05_audit58.csv).

HOW, same as data_code/05_make_splits.py: grouped by report text (a group never straddles the
split), stratified by language + number of positive findings + degenerative/traumatic, seed 42.
Strata are computed from the v3 labels. Among 200 seeds, the one with the smallest largest gap in
label prevalence and language between train and val is kept (only balance is looked at, never a model).

LABELS
  train rows  w50  (22_train_full_qwen-mg-convnext-mii-w50.csv) - what the relabelled members learn
  val rows    v3   (17_train_full_qwen-mg-raw.csv) - NO model predictions in them, so the val score
              does not partly measure "agrees with the teacher models" (the w50 twin printed 0.96
              on its own w50 val labels)
  audit rows  the panel's 0/1 labels (source == "audit" in both files, identical)

    python data_code/23_make_split8.py

Outputs: data/23_train_w50.csv, data/23_val_v3_audit.csv, data/23_split_assignment.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
VAL_FRAC, SEED, NSEEDS = 0.08, 42, 1000
SLOTS = ["sag_plain", "sag_fluid", "cor_plain", "cor_fluid", "ax_plain", "ax_fluid"]


def swap_search(pool, Fv, gidx, vg, proposals=300000, seed=0):
    """Polish the best seed: swap one val group with one train group OF THE SAME STRATUM (so the
    stratification and the report-text grouping both hold), keep the swap only if it lowers the
    worst balance gap (ties: the sum of squared gaps), and keep val at 7.5-8.5% of the pool.
    Pure bookkeeping on group sums, so tens of thousands of proposals take seconds."""
    rng = np.random.default_rng(seed)
    groups, inv = np.unique(gidx, return_inverse=True)
    G = np.zeros((len(groups), Fv.shape[1])); np.add.at(G, inv, Fv)
    size = np.bincount(inv).astype(float)
    strat = pool.groupby("group")["stratum"].first().reindex(groups).values
    inval = np.isin(groups, list(vg))
    tot, N = G.sum(0), size.sum()

    def score(Sv, nv):
        gap = np.abs(Sv / nv - (tot - Sv) / (N - nv)) * 100
        return gap.max(), (gap ** 2).sum()

    Sv, nv = G[inval].sum(0), size[inval].sum()
    cur = score(Sv, nv)
    start = cur[0]
    by_s = {}
    for i, st in enumerate(strat):
        by_s.setdefault(st, []).append(i)
    keys = [k for k, v in by_s.items() if len(v) > 1]
    for _ in range(proposals):
        st = keys[rng.integers(len(keys))]
        members = np.array(by_s[st])
        a_pool, b_pool = members[inval[members]], members[~inval[members]]
        if len(a_pool) == 0 or len(b_pool) == 0:
            continue
        a, b = a_pool[rng.integers(len(a_pool))], b_pool[rng.integers(len(b_pool))]
        Sv2, nv2 = Sv - G[a] + G[b], nv - size[a] + size[b]
        if not 0.075 <= nv2 / N <= 0.085:
            continue
        new = score(Sv2, nv2)
        if new < cur:
            inval[a], inval[b], Sv, nv, cur = False, True, Sv2, nv2, new
    print(f"swap search: worst balance gap {start:.2f} -> {cur[0]:.2f} pts ({proposals} proposals, swaps within a stratum)")
    return set(groups[inval].tolist())


def balance_features(pool):
    """Every property train and val should share, as 0/1 columns whose MEAN is a share (labels'
    soft means are scaled into [0, 1] too). Gaps are compared in percentage points.
      labels     12 prevalences (v3 > 0.5), 12 mean soft v3 labels, 12 mean w50 labels
      burden     number of positive findings: 0 / 1 / 2-3 / 4+
      language   (a proxy for site)
      vendor     the study's main scanner manufacturer
      side       left / right / unknown knee
      views      which of the 6 MRI series slots the study has
      resolution in-plane pixel spacing and slice thickness of its sagittal series, in thirds"""
    ids = pool.StudyInstanceUID
    cols = {}
    hard = (pool[LABELS] > 0.5)
    for t in LABELS:
        cols[f"prev {t}"] = hard[t].values
        cols[f"v3 mean {t}"] = pool[t].values
    w50 = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
    for t in LABELS:
        cols[f"w50 mean {t}"] = w50.loc[ids, t].values
    npos = hard.sum(axis=1)
    for lab, (lo, hi) in {"0": (0, 0), "1": (1, 1), "2-3": (2, 3), "4+": (4, 12)}.items():
        cols[f"positives {lab}"] = ((npos >= lo) & (npos <= hi)).values
    for l in sorted(pool["lang"].astype(str).unique()):
        cols[f"lang {l}"] = (pool["lang"].astype(str) == l).values
    si = pd.read_csv(D / "07_series_index.csv")
    si = si[si.study.isin(set(ids))]
    vendor = si.groupby("study").manufacturer.agg(lambda x: x.value_counts().index[0]).reindex(ids).fillna("unknown")
    # one manufacturer, several spellings: GE / GEHC, and Canon bought Toshiba Medical (CANON_MEC);
    # FUJIFILM bought Hitachi's imaging business
    VEND = {"siemens": "siemens", "ge": "ge", "gehc": "ge", "philips": "philips", "toshiba": "canon/toshiba",
            "canon_mec": "canon/toshiba", "fujifilm": "fujifilm/hitachi", "hitachi": "fujifilm/hitachi"}
    vendor = vendor.str.split().str[0].str.lower().map(lambda x: VEND.get(x, x))
    for vname in sorted(vendor.unique()):
        cols[f"vendor {vname}"] = (vendor == vname).values
    side = pd.read_csv(D / "12_laterality.csv").set_index("StudyInstanceUID").side.reindex(ids).fillna("unknown")
    for sname in sorted(side.unique()):
        cols[f"side {sname}"] = (side == sname).values
    have = si.groupby("study").slot.agg(set).reindex(ids)
    for sl in SLOTS:
        cols[f"has {sl}"] = have.map(lambda x: isinstance(x, set) and sl in x).values
    sag = si[si.plane == "sagittal"].groupby("study").agg(px=("pixel_spacing_r", "median"),
                                                         th=("slice_thickness", "median")).reindex(ids)
    q = sag["px"].quantile([1 / 3, 2 / 3]).values          # pixel spacing is continuous: thirds
    x = sag["px"].values
    cols["pixel spacing low third"] = np.nan_to_num(x <= q[0])
    cols["pixel spacing mid third"] = np.nan_to_num((x > q[0]) & (x <= q[1]))
    cols["pixel spacing high third"] = np.nan_to_num(x > q[1])
    cols["pixel spacing unknown"] = np.isnan(x)
    th = sag["th"].values                                   # slice thickness comes in fixed values: bands
    for lab, (lo, hi) in {"<2.5 mm": (0, 2.45), "2.5-2.9 mm": (2.45, 2.95), "3.0 mm": (2.95, 3.05),
                          "3.1-3.9 mm": (3.05, 3.95), ">=4 mm": (3.95, 99)}.items():
        cols[f"slice thickness {lab}"] = np.nan_to_num((th > lo) & (th <= hi))
    cols["slice thickness unknown"] = np.isnan(th)
    return pd.DataFrame(cols, index=ids.values).astype(float)


def main():
    fails = []

    def check(ok, msg):
        print(("  OK    " if ok else "  FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    w50 = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv")
    v3 = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv")
    check(list(w50.StudyInstanceUID) == list(v3.StudyInstanceUID), "w50 and v3 files list the same studies in the same order")
    aud = set(pd.read_csv(D / "05_audit58.csv").StudyInstanceUID)
    check(set(v3.StudyInstanceUID[v3.source == "audit"]) == aud and len(aud) == 58, "the 58 audit rows are the 05_audit58 studies")
    a1 = w50.set_index("StudyInstanceUID").loc[sorted(aud), LABELS].values
    a2 = v3.set_index("StudyInstanceUID").loc[sorted(aud), LABELS].values
    check(np.array_equal(a1, a2), "audit rows carry identical (panel) labels in both files")

    pool = v3[v3.source != "audit"].copy()
    hard = (pool[LABELS] > 0.5).astype(int)
    degen = hard[["Medial OA", "Lateral OA", "PF OA", "Baker's"]].sum(axis=1)
    trauma = hard[["ACL", "MCL", "Contusion", "Fracture"]].sum(axis=1)
    pool["stratum"] = (pool["lang"].astype(str) + "|"
                       + pd.cut(hard.sum(axis=1), [-1, 0, 1, 3, 12], labels=["0", "1", "2-3", "4+"]).astype(str)
                       + "|" + (degen > trauma).map({True: "deg", False: "tr"}))
    g = pool.groupby("group").agg(stratum=("stratum", "first"), size=("group", "size")).reset_index()

    def draw(seed):
        """per stratum, add groups while that brings the count CLOSER to 8% of the stratum (the 05
        rule always took the first group, which overfills the many tiny strata: 9.4% on the first try)"""
        vg, carry = set(), 0.0
        for _, sub in g.groupby("stratum"):
            idx = sub.sample(frac=1.0, random_state=seed).index
            target, taken = VAL_FRAC * sub["size"].sum() + carry, 0
            for i in idx:
                n = int(g.loc[i, "size"])
                if abs(taken + n - target) < abs(taken - target):
                    vg.add(int(g.loc[i, "group"]))
                    taken += n
            carry = target - taken                  # rounding left over passes to the next stratum
        return vg

    F = balance_features(pool)                       # one row per pool study, every column a share
    Fv = F.values.astype(float)
    gidx = pool["group"].values

    def imbalance(vg):
        m = np.isin(gidx, list(vg))
        gap = np.abs(Fv[m].mean(0) - Fv[~m].mean(0)) * 100
        return gap.max(), m.mean(), gap

    # balance search over seeds: only the data's own properties are looked at, never a model
    best = None
    for seed in range(SEED, SEED + NSEEDS):
        vg = draw(seed)
        imb, frac, _ = imbalance(vg)
        if 0.075 <= frac <= 0.085 and (best is None or imb < best[0]):
            best = (imb, seed, vg)
    print(f"balance search: {NSEEDS} seeds, kept seed {best[1]}: largest gap over all {F.shape[1]} balance columns "
          f"{best[0]:.2f} pts")
    val_groups = swap_search(pool, Fv, gidx, set(best[2]))
    pool["split"] = np.where(pool["group"].isin(val_groups), "val", "train")
    tr_ids = list(pool.StudyInstanceUID[pool.split == "train"])
    va_ids = list(pool.StudyInstanceUID[pool.split == "val"])

    W = w50.set_index("StudyInstanceUID")
    V = v3.set_index("StudyInstanceUID")
    train = W.loc[tr_ids].reset_index()
    train["source"] = "train"
    val = pd.concat([V.loc[va_ids].reset_index().assign(source="val"),
                     V.loc[sorted(aud)].reset_index().assign(source="audit")], ignore_index=True)
    cols = list(v3.columns)
    train, val = train[cols], val[cols]

    print(f"\npool {len(pool)}  ->  train {len(train)} ({len(train) / len(v3):.1%} of all 4,400)   "
          f"val {len(va_ids)} ({len(va_ids) / len(pool):.1%} of the pool) + 58 audit = {len(val)}")
    check(not (set(tr_ids) & set(va_ids)) and not (set(tr_ids) & aud), "train, val and audit are disjoint")
    check(len(tr_ids) + len(va_ids) + 58 == len(v3), "every one of the 4,400 studies is used exactly once")
    tg = set(pool.group[pool.split == "train"]); vg = set(pool.group[pool.split == "val"])
    check(not (tg & vg), f"no report-text group straddles train and val ({len(vg)} val groups)")
    check(0.075 <= len(va_ids) / len(pool) <= 0.085, "val is 7.5-8.5% of the pool")
    check(np.allclose(train[LABELS].values, W.loc[tr_ids, LABELS].values), "train rows carry the w50 labels")
    check(np.allclose(val[val.source == "val"][LABELS].values, V.loc[va_ids, LABELS].values), "val rows carry the v3 labels")
    check(set(val[val.source == "audit"].StudyInstanceUID) == aud, "all 58 audit studies are in val")
    av = val[val.source == "audit"].set_index("StudyInstanceUID").loc[sorted(aud), LABELS].values
    check(np.array_equal(av, a2), "audit rows in val carry the panel's labels")

    print("\nlabel prevalence, fraction above 0.5 (%), v3 labels; old split = 05 (704 val):")
    old_va = set(v3.StudyInstanceUID[v3.source == "val"]); old_tr = set(v3.StudyInstanceUID[v3.source == "train"])
    pr = lambda ids: (V.loc[list(ids), LABELS] > 0.5).mean().mul(100)   # noqa: E731
    tab = pd.DataFrame({"new train": pr(tr_ids), "new val (no audit)": pr(va_ids), "old train": pr(old_tr),
                        "old val": pr(old_va), "audit58": pr(aud)}).round(1)
    tab["new gap"] = (tab["new val (no audit)"] - tab["new train"]).round(1)
    tab["old gap"] = (tab["old val"] - tab["old train"]).round(1)
    print(tab.to_string())
    check(tab["new gap"].abs().max() <= 1.0, f"largest train/val prevalence gap {tab['new gap'].abs().max():.1f} pts (old split: {tab['old gap'].abs().max():.1f})")
    _, _, gap = imbalance(val_groups)
    m = pool["group"].isin(val_groups).values
    rep = pd.DataFrame({"train %": F[~m].mean() * 100, "val %": F[m].mean() * 100, "gap pts": gap}).round(2)
    print("\nEVERY BALANCE COLUMN (train vs new val, audit excluded), largest gaps first:")
    print(rep.sort_values("gap pts", ascending=False).to_string())
    print()
    for grp in ("prev", "v3 mean", "w50 mean", "positives", "lang", "vendor", "side", "has", "pixel spacing", "slice thickness"):
        sub = rep[[c.startswith(grp) for c in rep.index]]
        print(f"  largest gap, {grp:16s} {sub['gap pts'].max():.2f} pts")
    check(rep["gap pts"].max() <= 1.0, f"every one of the {len(rep)} balance columns within 1.0 pt (largest {rep['gap pts'].max():.2f})")
    print("\npositives in the new val (v3 > 0.5), the rarest decide how stable a column's AUROC is:")
    print("  " + ", ".join(f"{t} {int((V.loc[va_ids, t] > 0.5).sum())}" for t in LABELS))
    print("\nlanguage (%):")
    lt = pd.DataFrame({"new train": V.loc[tr_ids, "lang"].value_counts(normalize=True).mul(100),
                       "new val": V.loc[va_ids, "lang"].value_counts(normalize=True).mul(100)}).fillna(0).round(1)
    print(lt.to_string())

    if fails:
        print(f"\n{len(fails)} CHECK(S) FAILED - nothing written")
        return 1
    train.to_csv(D / "23_train_w50.csv", index=False)
    val.to_csv(D / "23_val_v3_audit.csv", index=False)
    asg = pd.DataFrame({"StudyInstanceUID": v3.StudyInstanceUID})
    asg["split"] = asg.StudyInstanceUID.map({**{s: "train" for s in tr_ids}, **{s: "val" for s in va_ids},
                                              **{s: "audit" for s in aud}})
    asg.to_csv(D / "23_split_assignment.csv", index=False)
    print("\nwritten: data/23_train_w50.csv  data/23_val_v3_audit.csv  data/23_split_assignment.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
