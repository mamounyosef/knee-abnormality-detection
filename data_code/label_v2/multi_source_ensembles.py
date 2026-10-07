"""multi_source_ensembles.py - do AVERAGES of many labellers beat the v4 labels on audit-58? (7 Oct)

label_blend_search.py mixes ONE source at a time into a base. This builds committees instead: the logit mean of all
text-reading LLMs, of all image-reading VLMs, of everything, of the top-k by solo audit, then
  (a) the committee alone,
  (b) v4' = v4 with the committee blended in at w (0.1..0.5, logit space), w chosen cross-fitted (choose on 29,
      score on the other 29, 200 splits),
  (c) v4 rebuilt with the committee REPLACING Qwen as the report half (report half = mean(Qwen, committee)).
Only files covering all 58 studies with all 12 findings are used. Base v4 = data_code/22_make_convnext_mii_labels.py
--w 0.5 on audit (0.9453).

    python data_code/label_v2/multi_source_ensembles.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data"
sys.path.insert(0, str(ROOT / "runs/others/ensembling"))
import sweep_new_runs as sw  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
B5 = ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]
IMG_KEYS = ("medgemma", "gemma4-img", "gemma4-fluid", "hulumed")
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
Y = aud.loc[u, ["official_" + t for t in T]].values.astype(int)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda x: 1 / (1 + np.exp(-x))


def auc_cols(pred, idx):
    y, r = Y[idx], rankdata(pred[idx], axis=0)
    n1 = y.sum(0).astype(float)
    n0 = len(idx) - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ((r * y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = np.nan
    return out


mac = lambda p, idx=np.arange(len(u)): np.nanmean(auc_cols(p, idx))
load = lambda f: pd.read_csv(D / f).set_index("StudyInstanceUID").loc[u, T].values.astype(float)
Q = load("03_report_labels_qwen24t-med.csv")
M15 = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, T].values.astype(float)
bi = [T.index(t) for t in B5]
V3 = Q.copy()
V3[:, bi] = sg((lg(Q[:, bi]) + lg(M15[:, bi])) / 2)
tl = {}
for fam, run in (("cnx", "convnext_base_25d_18slices_f{k}"), ("mii", "medimageinsight_davit_huge_336_v3_f{k}")):
    tl[fam] = np.mean([pd.DataFrame((z := np.load(sw.best_npz(run.format(k=k), "audit"), allow_pickle=True))["logits"],
                                    index=[str(s) for s in z["study_ids"]], columns=T).loc[u].values
                       for k in range(5)], axis=0)
TEACH = 0.5 * tl["cnx"] + 0.5 * tl["mii"]
V4 = sg(0.5 * lg(V3) + 0.5 * TEACH)
print(f"v3 {mac(V3):.4f} | teacher {mac(TEACH):.4f} | v4 {mac(V4):.4f}")

src = {}
for f in sorted(D.glob("03_report_labels_*_validate58.csv")):
    n = f.stem[len("03_report_labels_"):-len("_validate58")]
    df = pd.read_csv(f).set_index("StudyInstanceUID")
    if n == "qwen24t-med" or not set(u) <= set(df.index) or not all(t in df for t in T):
        continue
    src[n] = df.loc[u, T].values.astype(float)
solo = {n: mac(x) for n, x in src.items()}
img = [n for n in src if any(k in n for k in IMG_KEYS)]
txt = [n for n in src if n not in img]
print(f"{len(src)} full sources: {len(txt)} text, {len(img)} image")
rank_t = sorted(txt, key=lambda n: -solo[n])
rank_i = sorted(img, key=lambda n: -solo[n])
print("top text:", ", ".join(f"{n} {solo[n]:.4f}" for n in rank_t[:6]))
print("top image:", ", ".join(f"{n} {solo[n]:.4f}" for n in rank_i[:6]))
com = {"all text": txt, "all image": img, "everything": list(src),
       "top3 text": rank_t[:3], "top5 text": rank_t[:5], "top3 image": rank_i[:3], "top5 image": rank_i[:5],
       "top3 text + top3 image": rank_t[:3] + rank_i[:3]}
rng = np.random.default_rng(0)
splits = [rng.permutation(len(u)) for _ in range(200)]
W = (0.1, 0.2, 0.3, 0.4, 0.5)
print(f"\n{'committee':26s} {'alone':>7s} | best w in-sample | cross-fitted w: mean delta, % positive | as report half")
for cn, names in com.items():
    C = np.mean([lg(src[n]) for n in names], axis=0)
    blends = {w: sg((1 - w) * lg(V4) + w * C) for w in W}
    ins = {w: mac(b) for w, b in blends.items()}
    bw = max(ins, key=ins.get)
    d = []
    for p in splits:
        a, b = p[:29], p[29:]
        w = max(W, key=lambda w: mac(blends[w], a) - mac(V4, a))
        d.append(mac(blends[w], b) - mac(V4, b))
    rep = mac(sg(0.5 * (0.5 * lg(V3) + 0.5 * C) + 0.5 * TEACH))
    print(f"{cn:26s} {mac(sg(C)):7.4f} | w={bw} {ins[bw]:.4f} ({ins[bw] - mac(V4):+.4f}) | "
          f"{np.mean(d):+.4f}, {np.mean(np.array(d) > 0):4.0%} | {rep:.4f} ({rep - mac(V4):+.4f})")
