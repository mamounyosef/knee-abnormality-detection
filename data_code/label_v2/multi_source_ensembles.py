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
src["Qwen"] = Q                                    # the report labeller behind v3/v4 (0.9114)
src["MedGemma15"] = M15                            # the image labeller behind v3's five findings
src["v3 (Qwen+MedGemma)"] = V3
solo.update({n: mac(src[n]) for n in ("Qwen", "MedGemma15", "v3 (Qwen+MedGemma)")})
com = {"Qwen + all text": ["Qwen"] + txt, "Qwen + top3 text": ["Qwen"] + rank_t[:3],
       "v3 + all text": ["v3 (Qwen+MedGemma)"] + txt, "v3 + top3 text + top3 image": ["v3 (Qwen+MedGemma)"] + rank_t[:3] + rank_i[:3],
       "Qwen + MedGemma + all": ["Qwen", "MedGemma15"] + list(src)[:-3],
       "Qwen + today's new": ["Qwen"] + [n for n in ("oss20", "oss120-med", "or-apodex_apodex-1.1-mini_free") if n in src],
       "Qwen + Kimi3 + M3-like": ["Qwen"] + [n for n in ("kimi3", "baichuanm3-avg2", "nemotron3ultra") if n in src],
       "MedGemma + all image": ["MedGemma15"] + img,
       "Qwen+MedGemma+best 2 VLMs": ["Qwen", "MedGemma15"] + rank_i[:2],
       "all text": txt, "all image": img, "everything": list(src),
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

print(f"\n=== against v3 (Qwen + MedGemma, NO teacher; {mac(V3):.4f}): committee blended in at w, cross-fitted")
for cn, names in com.items():
    C = np.mean([lg(src[n]) for n in names], axis=0)
    blends = {w: sg((1 - w) * lg(V3) + w * C) for w in W}
    d = []
    for p in splits:
        a_, b_ = p[:29], p[29:]
        w = max(W, key=lambda w: mac(blends[w], a_) - mac(V3, a_))
        d.append(mac(blends[w], b_) - mac(V3, b_))
    bw = max(W, key=lambda w: mac(blends[w]))
    print(f"{cn:26s} in-sample w={bw} {mac(blends[bw]):.4f} ({mac(blends[bw]) - mac(V3):+.4f}) | cross-fitted {np.mean(d):+.4f}, {np.mean(np.array(d) > 0):4.0%}")

# JOINT weights (user, 7 Oct): label = a*v3 + b*teacher + c*committee in logit space, a+b+c = 1, steps of 0.1;
# the (a, b, c) is chosen on 29 studies and scored on the other 29, against v4 (a=0.5, b=0.5, c=0)
print(f"\n=== joint a*v3 + b*teacher(5-fold ConvNeXt+MII) + c*committee; v4 = (0.5, 0.5, 0) = {mac(V4):.4f}")
GRID = [(a / 10, b / 10, round(1 - a / 10 - b / 10, 1)) for a in range(11) for b in range(11 - a)]
for cn in ("everything", "all image", "MedGemma + all image", "Qwen + MedGemma + all", "all text", "top3 text + top3 image"):
    C = np.mean([lg(src[n]) for n in com[cn]], axis=0)
    L = {g: sg(g[0] * lg(V3) + g[1] * TEACH + g[2] * C) for g in GRID}
    full = {g: mac(x) for g, x in L.items()}
    best = max(full, key=full.get)
    d, picks = [], []
    for p in splits:
        a_, b_ = p[:29], p[29:]
        g = max(GRID, key=lambda g: mac(L[g], a_))
        picks.append(g)
        d.append(mac(L[g], b_) - mac(V4, b_))
    top = max(set(picks), key=picks.count)
    print(f"{cn:26s} in-sample best {best} {full[best]:.4f} ({full[best] - mac(V4):+.4f}) | cross-fitted "
          f"{np.mean(d):+.4f}, {np.mean(np.array(d) > 0):4.0%} positive | most picked {top} ({picks.count(top)}/200)")

# NESTED (user, 7 Oct): the LLMs blended into ONE report member R, with or without MedGemma on its five findings
# (as v3), then R and the 5-fold teacher blended: label = (1-t)*R + t*teacher. t=0.5 is v4's own weight (no
# selection, the clean comparison); t chosen cross-fitted from 0.3..0.7 is the tuned one.
others = [n for n in txt if not n.startswith("qwen24t")]
REP = {"Qwen only (v4's report half)": lg(Q),
       "all text LLMs": np.mean([lg(src[n]) for n in txt] + [lg(Q)], axis=0),
       "Qwen 50% + other LLMs 50%": 0.5 * lg(Q) + 0.5 * np.mean([lg(src[n]) for n in others], axis=0),
       "Qwen 50% + all other text 50%": 0.5 * lg(Q) + 0.5 * np.mean([lg(src[n]) for n in txt], axis=0),
       "top3 text LLMs": np.mean([lg(src[n]) for n in rank_t[:3]], axis=0)}
TS = (0.3, 0.4, 0.5, 0.6, 0.7)
print(f"\n=== nested: report member R (+/- MedGemma on {B5}) then (1-t)*R + t*teacher; v4 = {mac(V4):.4f}")
for rn, R in REP.items():
    for mgn in ("no MedGemma", "with MedGemma"):
        Rm = R.copy()
        if mgn == "with MedGemma":
            Rm[:, bi] = 0.5 * R[:, bi] + 0.5 * lg(M15[:, bi])
        L = {t: sg((1 - t) * Rm + t * TEACH) for t in TS}
        d = []
        for p in splits:
            a_, b_ = p[:29], p[29:]
            t = max(TS, key=lambda t: mac(L[t], a_))
            d.append(mac(L[t], b_) - mac(V4, b_))
        r5 = mac(L[0.5])
        print(f"{rn:30s} {mgn:14s} R alone {mac(sg(Rm)):.4f} | t=0.5 {r5:.4f} ({r5 - mac(V4):+.4f}) | "
              f"tuned t cross-fitted {np.mean(d):+.4f}, {np.mean(np.array(d) > 0):4.0%} positive")

# HIERARCHICAL (user, 7 Oct): every model TYPE blended in its own space into ONE member first -
#   TEXT  = the text LLMs, IMAGE = the vision models (MedGemma, Gemma 4, Hulu-Med ...), FOLD = the 5-fold teacher -
# then label = a*TEXT + b*IMAGE + c*FOLD (logit space, a+b+c = 1, steps 0.1). IMAGE votes on all 12 findings, or
# only on MedGemma's five (elsewhere TEXT takes its share). Equal thirds = no selection; the grid is cross-fitted.
G4 = [n for n in img if n.startswith("gemma4")]
MGs = [n for n in img if n.startswith("medgemma")]
TEXTS = {"all text LLMs": np.mean([lg(src[n]) for n in txt] + [lg(Q)], axis=0), "Qwen only": lg(Q),
         "top3 text": np.mean([lg(src[n]) for n in rank_t[:3]], axis=0)}
IMGS = {"MedGemma corpus only": lg(M15), "MedGemma + Gemma 4": np.mean([lg(M15)] + [lg(src[n]) for n in G4], axis=0),
        "all MedGemma modes + Gemma 4": np.mean([lg(M15)] + [lg(src[n]) for n in MGs + G4], axis=0),
        "all vision models": np.mean([lg(M15)] + [lg(src[n]) for n in img], axis=0),
        "top3 vision": np.mean([lg(src[n]) for n in rank_i[:3]], axis=0)}
G3 = [(a / 10, b / 10, round(1 - a / 10 - b / 10, 1)) for a in range(11) for b in range(11 - a)]
print(f"\n=== hierarchical: a*TEXT + b*IMAGE + c*FOLD (each type blended first); v4 = {mac(V4):.4f}")
for tn, TX in TEXTS.items():
    for imn, IM in IMGS.items():
        for scope in ("5 findings", "all 12"):
            def lab(g):
                im = IM.copy()
                if scope == "5 findings":
                    im = TX.copy()
                    im[:, bi] = IM[:, bi]
                return sg(g[0] * TX + g[1] * im + g[2] * TEACH)
            L = {g: lab(g) for g in G3}
            eq = mac(lab((1 / 3, 1 / 3, 1 / 3)))
            full = {g: mac(x) for g, x in L.items()}
            best = max(full, key=full.get)
            d = []
            for p in splits:
                a_, b_ = p[:29], p[29:]
                g = max(G3, key=lambda g: mac(L[g], a_))
                d.append(mac(L[g], b_) - mac(V4, b_))
            print(f"{tn:10s} | {imn:28s} | {scope:10s} | thirds {eq:.4f} ({eq - mac(V4):+.4f}) | in-sample {best} "
                  f"{full[best]:.4f} | cross-fitted {np.mean(d):+.4f}, {np.mean(np.array(d) > 0):4.0%}")
