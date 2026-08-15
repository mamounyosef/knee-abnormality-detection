"""
Do the 58 official labels agree with what the reports actually say?

For each (study, finding) we classify the REPORT into one of three states:
  YES     - the report explicitly states the finding
  NO      - the report explicitly denies it ("no tear", "intact", "normal")
  SILENT  - the report never addresses it
then cross-tabulate against the official LABEL. Contradictions (report YES /
label 0, or report NO / label 1) bound how well ANY report-based labeler can do.

Writes a plain-text report to data/stats/report_vs_label_audit.txt
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(r"d:\My Projects\knee_abnormality_detection")
OUT = ROOT / "data" / "stats" / "report_vs_label_audit.txt"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

# --- vocabulary (multilingual, same spirit as notebook cell 04b) -------------
TEAR = ["tear", "torn", "rupture", "ruptur", "rotura", "desgarro", "yırtık", "yirtik",
        "rüptür", "ruptura", "puknuće", "ρήξη", "riss", "разкъсване", "руптура",
        "scheur", "ruptuur", "déchirure", "lezija", "sprain", "injury", "injuri",
        "esguince", "lesión", "lesion"]
OA = ["osteoarthritis", "osteoarthrosis", "arthrosis", "chondrosis", "chondral loss",
      "cartilage loss", "cartilage defect", "chondromalacia", "chondropathy",
      "chondral defect", "chondral ulcer", "artrosis", "gonartrosis", "condropatía",
      "condromalacia", "úlcera condral", "gonartroz", "osteoartrit", "kondropati",
      "kondromalazi", "artroza", "gonartroza", "hondropatija", "οστεοαρθρίτιδα",
      "γοναρθρίτιδα", "χονδροπάθεια", "arthrose", "gonarthrose", "chondropathie",
      "knorpelschaden", "остеоартрит", "гонартроза", "хондропатия", "artrose",
      "kraakbeenverlies", "osteophyte", "osteofito", "osteofit"]

ANAT = {
    "ACL": ["anterior cruciate", " acl", "ligamento cruzado anterior", " lca",
            "ön çapraz", "öçb", "prednji križni", "πρόσθιο χιαστό", "vordere kreuzband",
            "предна кръстна", "voorste kruisband", "croisé antérieur"],
    "MCL": ["medial collateral", " mcl", "colateral medial", "colateral interno",
            "iç yan bağ", "medial kollateral", "medijalni kolateralni", "έσω πλάγιο",
            "innenband", "медиален колатерален", "mediale collaterale", "collatéral médial"],
    "Medial Meniscus": ["medial meniscus", "menisco interno", "menisco medial",
                        "medial menisküs", "iç menisküs", "medijalni menisk",
                        "έσω μηνίσκο", "innenmeniskus", "medialer meniskus",
                        "медиален менискус", "mediale meniscus", "ménisque interne"],
    "Lateral Meniscus": ["lateral meniscus", "menisco externo", "menisco lateral",
                         "lateral menisküs", "dış menisküs", "lateralni menisk",
                         "έξω μηνίσκο", "außenmeniskus", "lateraler meniskus",
                         "латерален менискус", "laterale meniscus", "ménisque externe"],
    "Medial OA": ["medial compartment", "femorotibial medial", "compartimento medial",
                  "medial femoral condyle", "medial tibial", "medijaln", "έσω",
                  "медиал", "innen", "mediaal"],
    "Lateral OA": ["lateral compartment", "femorotibial lateral", "compartimento lateral",
                   "lateral femoral condyle", "lateral tibial", "lateraln", "έξω",
                   "латерал", "außen", "lateraal"],
    "PF OA": ["patellofemoral", "patelofemoral", "femoropatel", "retropatel", "patellar facet",
              "trochlea", "tróclea", "patella", "rotulian", "επιγονατιδομηριαί", "пателофеморал"],
}
SELF = {   # findings that are their own keyword
    "Effusion": ["effusion", "joint fluid", "derrame", "efüzyon", "eklem içi sıvı",
                 "mayi", "sıvı artış", "izljev", "izliv", "αρθρικό υγρό", "συλλογή υγρού",
                 "erguss", "излив", "gewrichtsvocht", "épanchement", "hemarthrosis",
                 "hemartrosis"],
    "Synovitis": ["synovitis", "sinovitis", "sinovit", "sinovijalna upala", "υμενίτιδα",
                  "synovialitis", "синовит", "synovite", "synovial thickening",
                  "synovial membrane thickening", "engrosamiento sinovial", "sinovyal"],
    "Baker's": ["baker", "popliteal cyst", "quiste poplíteo", "quistes poplíteos",
                "poplit kist", "poplitealna cista", "κύστη baker", "poplitealzyste",
                "bakerzyste", "поплитеална киста", "bakerse cyste", "kyste poplité"],
    "Contusion": ["contusion", "bone bruise", "bone marrow edema", "bone marrow oedema",
                  "marrow edema", "contusión", "edema óseo", "edema de médula ósea",
                  "kontüzyon", "kemik iliği ödem", "kontuzija", "koštani edem",
                  "μυελικό οίδημα", "knochenmarködem", "костен оток", "beenmergoedeem",
                  "œdème osseux", "bone edema"],
    "Fracture": ["fracture", "fractura", "kırık", "fraktür", "fraktura", "prijelom",
                 "κάταγμα", "fraktur", "фрактура", "счупване", "fractuur"],
}
NEG = ["no ", "not ", "without", "intact", "normal", "unremarkable", "absence",
       "negative for", "sin ", "íntegr", "integr", "ausencia", "conservad", "preserved",
       "yok", "doğal", "izlenmedi", "saptanmadı", "bez", "uredn", "nema", "χωρίς",
       "δεν ", "φυσιολογ", "kein", "ohne", "unauffällig", "intakt", "без", "не ",
       "geen", "zonder", "niet ", "pas de", "sans", "aucun", "is not torn", "no evidence"]

SENT = re.compile(r"[.\n;•>]+")


def classify(text, label):
    """-> 'YES' / 'NO' / 'SILENT' for one finding in one report."""
    anat = ANAT.get(label)
    path = OA if label.endswith("OA") else (TEAR if anat else SELF[label])
    if label in SELF:
        anat = None
    saw_topic = False
    for s in SENT.split(str(text).lower()):
        if not s.strip():
            continue
        hit_path = any(p in s for p in path)
        hit_anat = (anat is None) or any(a in s for a in anat)
        if not (hit_path and hit_anat):
            continue
        saw_topic = True
        if not any(n in s for n in NEG):
            return "YES"
    return "NO" if saw_topic else "SILENT"


tr = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
lab = tr[tr[LABELS].notna().all(axis=1)].reset_index(drop=True)

rows = []
for i, r in lab.iterrows():
    for c in LABELS:
        rows.append({"study": i + 1, "finding": c, "label": int(r[c]),
                     "report": classify(r["Report"], c)})
A = pd.DataFrame(rows)

L = []
def w(s=""):
    L.append(str(s)); print(s)

w("=" * 78)
w("DO THE 58 OFFICIAL LABELS MATCH WHAT THE REPORTS SAY?")
w("=" * 78)
w("""
The 58 labels were made by the Society of Skeletal Radiology annotation team
using MD.ai, an IMAGE annotation tool. The reports were written separately, by
the original radiologist at each hospital. So the two can legitimately disagree.

For every (study, finding) the report is classified as:
  YES    - report explicitly states the finding
  NO     - report explicitly denies it ("no tear", "intact", "normal")
  SILENT - report never mentions it
and compared with the official 0/1 label. This bounds how well ANY method that
reads only the report can score.
""")

w("--- overall, 58 studies x 12 findings = 696 decisions ---")
ct = pd.crosstab(A["report"], A["label"])
w(ct.to_string())
w("")
tot = len(A)
agree = ((A.report == "YES") & (A.label == 1)).sum() + ((A.report == "NO") & (A.label == 0)).sum()
contra = ((A.report == "YES") & (A.label == 0)).sum() + ((A.report == "NO") & (A.label == 1)).sum()
silent = (A.report == "SILENT").sum()
w(f"report and label AGREE      : {agree:4d}  ({agree/tot:.1%})")
w(f"report and label CONTRADICT : {contra:4d}  ({contra/tot:.1%})")
w(f"report SILENT               : {silent:4d}  ({silent/tot:.1%})")
w(f"   ...of those silent, label=1 in {((A.report=='SILENT') & (A.label==1)).sum()} cases "
  f"({((A.report=='SILENT') & (A.label==1)).mean()*100:.1f}% of all decisions)")
w("")
w("A 'SILENT + label=1' case is invisible to any report-based labeler: the")
w("finding is real but the radiologist never wrote it down.")

w("\n\n--- per finding ---")
per = []
for c in LABELS:
    s = A[A.finding == c]
    n_pos = (s.label == 1).sum()
    per.append({
        "finding": c,
        "label_pos": n_pos,
        "YES&1": ((s.report == "YES") & (s.label == 1)).sum(),
        "YES&0": ((s.report == "YES") & (s.label == 0)).sum(),
        "NO&1": ((s.report == "NO") & (s.label == 1)).sum(),
        "NO&0": ((s.report == "NO") & (s.label == 0)).sum(),
        "SILENT&1": ((s.report == "SILENT") & (s.label == 1)).sum(),
        "SILENT&0": ((s.report == "SILENT") & (s.label == 0)).sum(),
    })
P = pd.DataFrame(per).set_index("finding")
P["recoverable_%"] = (P["YES&1"] / P["label_pos"].replace(0, 1) * 100).round(0)
w(P.to_string())
w("")
w("recoverable_% = of the positives, how many the report explicitly states.")
w("This is roughly the ceiling on RECALL for a perfect report reader.")

w("\n\n--- worked examples of each contradiction type ---")
for kind, rep_state, lab_val in [("REPORT SAYS YES, LABEL SAYS 0", "YES", 0),
                                 ("REPORT SAYS NO, LABEL SAYS 1", "NO", 1),
                                 ("REPORT SILENT, LABEL SAYS 1", "SILENT", 1)]:
    sub = A[(A.report == rep_state) & (A.label == lab_val)]
    w(f"\n### {kind}  ({len(sub)} cases)")
    for _, row in sub.head(6).iterrows():
        txt = str(lab.loc[row.study - 1, "Report"])
        kw = (OA if row.finding.endswith("OA")
              else (SELF[row.finding] if row.finding in SELF else TEAR))
        frag = ""
        for s in SENT.split(txt.lower()):
            if any(k in s for k in kw):
                frag = s.strip()[:150]
                break
        w(f"  study {row.study:2d}  {row.finding:17s} | report fragment: {frag!r}")

OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"\nwritten -> {OUT}")
