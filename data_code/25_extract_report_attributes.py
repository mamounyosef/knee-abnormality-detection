"""
25_extract_report_attributes.py - structured, GRADED attributes from every report, as extra training targets.

WHY (user, 4 Oct). The 12 labels give each study 12 bits. The report also states WHERE and HOW SEVERE: cartilage grade
per compartment, osteophytes, meniscus tear type, effusion size, synovial thickening... The 13 Sep test regressed the
whole report EMBEDDING and lost (-0.0049): most of an embedding is writing style, not findings. This extracts only
finding-relevant, graded facts, which an auxiliary head (dropped at inference) can learn from.

Each attribute is an integer grade or null ("the report does not say"); null cells are masked out of the loss.
Ollama Cloud (free tier), native /api/chat, JSON format. The API key comes ONLY from the environment of the command that
runs this (OLLAMA_API_KEY, or OLLAMA_API_KEYS=key1,key2 to spread the requests over several keys); nothing is read
from or written to .env. Model: gemma4:31b, the best free text labeller on audit-58 (0.8931 vs Nemotron 3 Ultra 0.8884,
gpt-oss:120b 0.8832).

    $env:OLLAMA_API_KEYS="k1,k2"; python data_code/25_extract_report_attributes.py --limit 3 --show   # smoke test
    $env:OLLAMA_API_KEYS="k1,k2"; python data_code/25_extract_report_attributes.py                   # all 4,407 (resumes)
    python data_code/25_extract_report_attributes.py --build      # jsonl -> data/25_report_attributes.csv (+ checks)

Outputs: data/25_report_attributes_raw.jsonl (one line per study, resumable), data/25_report_attributes.csv
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
EN_CSV = D / "02_reports_en.csv"
RAW = D / "25_report_attributes_raw.jsonl"
OUT = D / "25_report_attributes.csv"
URL = "https://ollama.com/api/chat"

# name -> (max grade, definition). Grades are ordinal; 0 always means "explicitly normal / absent".
ATTRS = {
    # cartilage and osteoarthritis, per compartment (medial tibiofemoral, lateral tibiofemoral, patellofemoral)
    **{f"{c}_cartilage": (3, f"cartilage damage in the {n}: 0 normal; 1 low-grade (softening, fissuring, superficial, "
                              f"<50% thickness, chondromalacia grade 1-2); 2 high-grade (>50% thickness, deep ulcer, "
                              f"grade 3); 3 full-thickness loss / exposed bone (grade 4)")
       for c, n in (("med", "MEDIAL tibiofemoral compartment (medial femoral condyle / medial tibial plateau)"),
                    ("lat", "LATERAL tibiofemoral compartment (lateral femoral condyle / lateral tibial plateau)"),
                    ("pf", "PATELLOFEMORAL compartment (patella / trochlea)"))},
    **{f"{c}_osteophytes": (2, f"osteophytes (bone spurs) in the {n} compartment: 0 none; 1 small/marginal; 2 moderate/large")
       for c, n in (("med", "medial"), ("lat", "lateral"), ("pf", "patellofemoral"))},
    **{f"{c}_subchondral": (1, f"subchondral bone marrow edema, bone marrow lesion or subchondral cyst in the {n} "
                               f"compartment related to degeneration (NOT trauma): 0 no; 1 yes")
       for c, n in (("med", "medial"), ("lat", "lateral"), ("pf", "patellofemoral"))},
    **{f"{c}_oa_stated": (3, f"osteoarthritis / arthrosis / degenerative joint disease STATED for the {n} compartment "
                             f"(also counts if the report says bicompartmental/tricompartmental and includes it): 0 none; "
                             f"1 mild/early/incipient; 2 moderate (or unspecified severity); 3 severe/advanced")
       for c, n in (("med", "medial tibiofemoral"), ("lat", "lateral tibiofemoral"), ("pf", "patellofemoral"))},
    "jsn_med": (1, "medial joint space narrowing stated: 0 no; 1 yes"),
    "jsn_lat": (1, "lateral joint space narrowing stated: 0 no; 1 yes"),
    # menisci
    **{f"{c}_meniscus": (2, f"{n} meniscus: 0 normal; 1 degeneration / intrasubstance signal without a tear; 2 tear")
       for c, n in (("med", "MEDIAL"), ("lat", "LATERAL"))},
    **{f"{c}_meniscus_complex": (1, f"{n} meniscus tear that is complex, radial, root, flap, bucket-handle or displaced: "
                                    f"0 no (or no tear); 1 yes")
       for c, n in (("med", "medial"), ("lat", "lateral"))},
    **{f"{c}_meniscus_extrusion": (1, f"{n} meniscal extrusion / subluxation: 0 no; 1 yes")
       for c, n in (("med", "medial"), ("lat", "lateral"))},
    # ligaments
    "acl": (2, "anterior cruciate ligament: 0 normal; 1 sprain / partial tear / mucoid degeneration; 2 complete tear "
               "(or reconstructed graft torn)"),
    "pcl": (2, "posterior cruciate ligament: 0 normal; 1 sprain / partial tear; 2 complete tear"),
    "mcl": (2, "medial collateral ligament: 0 normal; 1 sprain / grade 1 / periligamentous edema; 2 partial or complete "
               "tear (grade 2-3)"),
    "lcl": (2, "lateral collateral ligament complex / posterolateral corner: 0 normal; 1 sprain; 2 tear"),
    "acl_reconstruction": (1, "prior ACL reconstruction (graft present): 0 no; 1 yes"),
    # fluid and synovium
    "effusion": (3, "joint effusion: 0 none; 1 small/minimal/trace/physiological; 2 moderate (or unspecified size); "
                    "3 large/massive/marked"),
    "synovitis_stated": (2, "synovitis stated by the radiologist: 0 no; 1 mild/minimal; 2 moderate/marked "
                            "(or unspecified)"),
    "synovial_thickening": (1, "synovial thickening / hypertrophy / proliferation described: 0 no; 1 yes"),
    "hoffa": (1, "Hoffa's (infrapatellar) fat pad edema or inflammation: 0 no; 1 yes"),
    "hemarthrosis": (1, "hemarthrosis or lipohemarthrosis (blood in the joint): 0 no; 1 yes"),
    "plica": (1, "a plica described (any): 0 no; 1 yes"),
    "baker": (2, "Baker's (popliteal) cyst: 0 none; 1 small/trace; 2 moderate/large or ruptured/leaking"),
    "loose_body": (1, "intra-articular loose body: 0 no; 1 yes"),
    # bone, trauma
    **{f"contusion_{c}": (1, f"traumatic bone contusion / bone bruise / post-traumatic marrow edema in the {n}: 0 no; 1 yes")
       for c, n in (("lfc", "lateral femoral condyle"), ("ltp", "lateral tibial plateau"),
                    ("mfc", "medial femoral condyle"), ("mtp", "medial tibial plateau"), ("patella", "patella"))},
    "fracture_traumatic": (1, "acute traumatic fracture (any bone, incl. osteochondral impaction): 0 no; 1 yes"),
    "insufficiency_fracture": (1, "subchondral insufficiency fracture / SIFK / SONK / avascular necrosis: 0 no; 1 yes"),
    "osteochondral_lesion": (1, "osteochondral lesion / osteochondritis dissecans / osteochondral defect: 0 no; 1 yes"),
    "patellar_instability": (1, "patellar dislocation/instability signs (MPFL tear, lateral patellar tilt or subluxation, "
                                "medial patellar facet bruise): 0 no; 1 yes"),
    # tendons, other
    "patellar_tendinosis": (1, "patellar or quadriceps tendinosis/tendinopathy/tear: 0 no; 1 yes"),
    "bursitis": (1, "bursitis (prepatellar, pes anserine, suprapatellar or other): 0 no; 1 yes"),
    "postoperative": (1, "any prior knee surgery described (meniscectomy, repair, arthroplasty, hardware): 0 no; 1 yes"),
}
NAMES = list(ATTRS)

SYSTEM = (
    "You are a musculoskeletal radiologist extracting structured facts from a knee MRI report for a research dataset.\n"
    "For EACH attribute below, give an integer grade, or null when the report does not address it at all.\n"
    "Rules:\n"
    "- Use only what the report states. Do not infer from other findings.\n"
    "- 0 means the report explicitly says normal/absent/intact (or clearly describes that structure as normal).\n"
    "- null means the report is silent about it. Silence is NOT 0. A short report that lists only a few abnormal\n"
    "  findings says NOTHING about everything else: every attribute it does not mention is null, not 0. Give 0 only\n"
    "  with an explicit normal statement about that structure (e.g. 'ligaments intact' -> acl 0, pcl 0, mcl 0, lcl 0;\n"
    "  'no bone marrow edema' -> the contusion_* attributes 0; 'no effusion' -> effusion 0).\n"
    "- If the report covers both knees (two reports), use the one that is clearly abnormal; if unclear, the first.\n"
    "- Grades follow the definitions exactly; when the severity is not given, use the grade marked "
    "'(or unspecified)' if there is one, otherwise the lowest positive grade.\n\n"
    "ATTRIBUTES (name: allowed grades and definition):\n"
    + "\n".join(f"- {k}: 0..{m} or null. {d}" for k, (m, d) in ATTRS.items())
    + "\n\nReply with ONE raw JSON object and nothing else (no prose, no fences), with EXACTLY these keys:\n"
    + ", ".join(NAMES))

_lock = threading.Lock()


def keys():
    ks = [k.strip() for k in (os.environ.get("OLLAMA_API_KEYS") or os.environ.get("OLLAMA_API_KEY") or "").split(",") if k.strip()]
    if not ks:
        sys.exit("set OLLAMA_API_KEY (or OLLAMA_API_KEYS=k1,k2) in this command's environment")
    return ks


def ask(text, model, key, timeout=300, retries=6):
    body = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "REPORT:\n" + text}]}
    for a in range(retries):
        try:
            r = requests.post(URL, json=body, timeout=timeout, headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"status {r.status_code}: {r.text[:120]}")
            r.raise_for_status()
            txt = r.json()["message"]["content"]
            i, j = txt.find("{"), txt.rfind("}")
            return json.loads(txt[i:j + 1])
        except Exception as e:                                  # noqa: BLE001 - retry anything
            if a == retries - 1:
                raise
            time.sleep(min(5 * 2 ** a, 90))
            last = e  # noqa: F841


def coerce(obj):
    """Keep only known keys; clip to the allowed range; anything unparseable becomes null."""
    out, bad = {}, 0
    for k, (m, _) in ATTRS.items():
        v = obj.get(k, None)
        if v is None or v == "null":
            out[k] = None
            continue
        try:
            v = int(round(float(v)))
            out[k] = min(max(v, 0), m)
        except (TypeError, ValueError):
            out[k], bad = None, bad + 1
    return out, bad


def run(args):
    rep = pd.read_csv(EN_CSV)
    done = set()
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["StudyInstanceUID"])
            except Exception:                                   # noqa: BLE001 - a torn last line
                pass
    todo = rep[~rep.StudyInstanceUID.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(rep)} reports, {len(done)} done, {len(todo)} to do, model {args.model}, workers {args.workers}", flush=True)
    t0, n, fails = time.time(), 0, 0

    ks = keys()
    print(f"{len(ks)} API key(s); requests alternate between them", flush=True)

    def one(row, k):
        obj = ask(str(row.Report_en), args.model, k)
        attrs, bad = coerce(obj)
        return {"StudyInstanceUID": row.StudyInstanceUID, "model": args.model, "bad": bad, **attrs}

    with ThreadPoolExecutor(max_workers=args.workers * len(ks)) as pool:
        futs = {pool.submit(one, r, ks[i % len(ks)]): r for i, r in enumerate(todo.itertuples())}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except Exception as e:                              # noqa: BLE001
                fails += 1
                print(f"  FAIL {futs[f].StudyInstanceUID[-12:]}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            with _lock:
                with RAW.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            n += 1
            if args.show:
                print(json.dumps({k: v for k, v in rec.items() if v is not None}, indent=None)[:1500], flush=True)
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                eta = el / n * (len(todo) - n)
                print(f"  {n}/{len(todo)} done ({n / len(todo):.0%}), {fails} failed, {el / n:.1f} s/report, "
                      f"elapsed {el / 60:.0f} min, ETA {eta / 60:.0f} min", flush=True)
    print(f"finished: {n} written, {fails} failed (rerun to retry them)")


def build():
    """jsonl -> csv, plus sanity checks against the 12 labels the same reports produced."""
    recs = [json.loads(x) for x in RAW.read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(recs).drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")
    rep = pd.read_csv(EN_CSV)
    print(f"{len(df)} of {len(rep)} studies extracted; missing {len(set(rep.StudyInstanceUID) - set(df.index))}")
    cov = df[NAMES].notna().mean().sort_values()
    print("coverage (share of reports that address each attribute):")
    print("  " + ", ".join(f"{k} {v:.0%}" for k, v in cov.items()))
    lab = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv").set_index("StudyInstanceUID")
    pairs = {"Medial OA": "med_cartilage", "Lateral OA": "lat_cartilage", "PF OA": "pf_cartilage",
             "Medial Meniscus": "med_meniscus", "Lateral Meniscus": "lat_meniscus", "ACL": "acl", "MCL": "mcl",
             "Effusion": "effusion", "Synovitis": "synovitis_stated", "Baker's": "baker", "Fracture": "fracture_traumatic"}
    common = df.index.intersection(lab.index)
    print("rank correlation of an attribute with the matching v3 label (sanity; reported cells only):")
    for t, a in pairs.items():
        s = df.loc[common, a]
        ok = s.notna()
        r = pd.Series(s[ok].astype(float)).corr(lab.loc[common, t][ok], method="spearman")
        print(f"  {t:17s} vs {a:22s} rho {r:+.3f} on {ok.sum()} studies")
    df[["model", "bad"] + NAMES].to_csv(OUT)
    print(f"wrote {OUT}: {len(df)} studies x {len(NAMES)} attributes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--workers", type=int, default=6)   # per key
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    build() if a.build else run(a)
