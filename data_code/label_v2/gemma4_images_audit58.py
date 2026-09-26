"""
gemma4_images_audit58.py - can Gemma 4 31B (Ollama Cloud, vision) label the 58 audit knees from
the MRI IMAGES, with or without the report? Scores it like every other labeller.

One call per study (not one per finding): each series is its own user message, a header naming
the view and sequence plus its slices in order (Ollama's /api/chat takes base64 images per
message), then one message with the task. The task is data_code/03_label_reports.py's SYSTEM_SOFT
prompt (the one that scored 0.9114 with Qwen) with its first paragraph rewritten for images, plus
the same OUTPUT FORMAT block and the same tolerant parser. Slices: medgemma_audit58's "all36"
selection (36 images spread evenly over the real series, left knees mirrored).

Modes: img = images only; img_rep = images + the translated report.

The API key comes ONLY from the environment of the command that runs this (OLLAMA_API_KEY);
nothing is read from or written to .env.

    OLLAMA_API_KEY=... python data_code/label_v2/gemma4_images_audit58.py --modes img --limit 1
    OLLAMA_API_KEY=... python data_code/label_v2/gemma4_images_audit58.py --modes img,img_rep
"""
import argparse
import base64
import importlib.util
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "data_code/label_v2"))
import medgemma_audit58 as mg  # noqa: E402

URL = "https://ollama.com/api/chat"
MODEL = "gemma4:31b"
SLICE_MODE = "all36"
# fluid modes: the three fluid-sensitive series only (36 slices, 12 each) and a prompt that asks
# only about the three fluid findings
FLUID = ["Effusion", "Synovitis", "Baker's"]
mg.MODE_DEF["fluid36"] = (mg.SLOTS_FLUID, False)
mg.MODE_OPT["fluid36"] = {"slices": 36}
FLUID_EXAMPLE = '{"findings": {"Effusion": {"p": 0.05, "evidence": "..."}, "Synovitis": {...}, "Baker\'s": {...}}}'
OUT_RAW = ROOT / "data_code/label_v2/results/gemma4img_audit58"
IMG_INTRO = (
    "You are given the images of a knee MRI study: each series (view and sequence) is shown as\n"
    "its slices in order. Output a probability (0.0-1.0) that each of 12 findings is present in\n"
    "that knee, plus a short note on what you saw.")


def labeller():
    sys.path.insert(0, str(ROOT / "data_code"))
    spec = importlib.util.spec_from_file_location("label_reports", ROOT / "data_code/03_label_reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def system_prompt(lr):
    first = lr.SYSTEM_SOFT.split("\n\n")
    assert first[1].startswith("You are given the full text of a knee MRI report"), first[1][:60]
    body = "\n\n".join([first[0], IMG_INTRO] + first[2:])
    ex = '{"unusable": false, "findings": {"ACL": {"p": 0.05, "evidence": "..."}, ...}}'
    return body + (
        "\n\nOUTPUT FORMAT - follow exactly, no exceptions:\n"
        "Reply with ONE raw JSON object and nothing else. No prose, no explanation,\n"
        "no markdown, no ``` fences, no table. The object must be:\n\n"
        + ex + "\n\nOne entry per finding, using EXACTLY these 12 keys and no "
        "others:\n" + ", ".join(lr.LABELS) +
        "\nEach entry has exactly two fields: 'p' (a number 0.0-1.0) and 'evidence' "
        "(the exact sentence you relied on, or an empty string).")


def fluid_prompt(lr):
    """SYSTEM_SOFT cut down to the three fluid findings: the same opening and critical context,
    and only the Effusion / Synovitis / Baker's definitions, verbatim."""
    s = lr.SYSTEM_SOFT
    parts = s.split("\n\n")
    head = parts[0].replace("labeling knee MRI reports", "labeling knee MRI studies")
    ctx = next(x for x in parts if x.startswith("CRITICAL CONTEXT"))
    defs = s[s.index("Effusion  -"):s.index("Contusion -")].rstrip()
    intro = IMG_INTRO.replace("12 findings", "3 fluid findings")
    rules = "DEFINITIONS AND SCORING RULES (recovered from the reference panel's own labeling):\n\n" + defs
    return "\n\n".join([head, intro, ctx, rules]) + (
        "\n\nOUTPUT FORMAT - follow exactly, no exceptions:\n"
        "Reply with ONE raw JSON object and nothing else. No prose, no explanation,\n"
        "no markdown, no ``` fences, no table. The object must be:\n\n" + FLUID_EXAMPLE +
        "\n\nOne entry per finding, using EXACTLY these 3 keys and no others: " + ", ".join(FLUID) +
        "\nEach entry has exactly two fields: 'p' (a number 0.0-1.0) and 'evidence' "
        "(what you relied on, or an empty string).")


def parse_fluid(lr, text):
    obj = lr._extract_json(text)
    src = obj.get("findings") if isinstance(obj.get("findings"), dict) else obj
    out = {}
    for k in FLUID:
        v = src[k]
        if isinstance(v, dict):
            v = v.get("p", v.get("probability"))
        elif isinstance(v, (list, tuple)):
            v = v[0]
        out[k] = min(1.0, max(0.0, float(v)))
    return out


def png64(a):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def messages(job, sysmsg, with_report, fluid=False):
    with np.load(ROOT / f"data/shards/{job['uid']}.npz") as z:
        groups = mg.volume_slices(z, job, "fluid36" if fluid else SLICE_MODE)
    msgs = [{"role": "system", "content": sysmsg}]
    for slot, sl in groups:
        msgs.append({"role": "user", "content": f"KNEE MRI SERIES: {mg.SLOT_NAME[slot]}, {len(sl)} slices in order.",
                     "images": [png64(x) for x in sl]})
    task = ("Label the 3 fluid findings (Effusion, Synovitis, Baker's) for this knee from the images above."
            if fluid else "Label the 12 findings for this knee from the images above.")
    if with_report:
        task = f"KNEE MRI REPORT:\n\n{job['report']}\n\n" + task + " The report of this study is given too."
    msgs.append({"role": "user", "content": task})
    return msgs, sum(len(sl) for _, sl in groups)


def call(msgs, key, timeout=900, retries=6):
    for a in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=timeout,
                              json={"model": MODEL, "messages": msgs, "stream": False,
                                    "options": {"temperature": 0}})
            if r.status_code == 429:
                time.sleep(min(15 * (a + 1), 90))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if a == retries - 1:
                raise
            time.sleep(min(10 * 2 ** a, 90))
    raise RuntimeError("rate limited on every retry")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="img,img_rep")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    key = os.environ.get("OLLAMA_API_KEY")
    assert key, "set OLLAMA_API_KEY in this command's environment"
    lr = labeller()
    sysmsg = system_prompt(lr)
    sysmsg_fluid = fluid_prompt(lr)
    jobs = mg.build_jobs(reports=True)
    missing = [j["uid"] for j in jobs if not (ROOT / f"data/shards/{j['uid']}.npz").exists()]
    assert not missing, f"{len(missing)} audit shards missing locally"
    if a.limit:
        jobs = jobs[:a.limit]
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    for mode in a.modes.split(","):
        raw = OUT_RAW / f"raw_{mode}{'_first%d' % a.limit if a.limit else ''}.jsonl"
        done = {}
        if raw.exists():
            for line in raw.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if "findings" in r:
                    done[r["uid"]] = r
        todo = [j for j in jobs if j["uid"] not in done]
        print(f"[{mode}] {len(jobs)} studies, {len(done)} already labelled, {len(todo)} to go", flush=True)
        lock = threading.Lock()
        t0 = time.time()

        def one(job):
            fl = mode.startswith("fluid")
            msgs, n_img = messages(job, sysmsg_fluid if fl else sysmsg, mode.endswith("_rep"), fluid=fl)
            resp = call(msgs, key)
            text = resp["message"]["content"]
            rec = {"uid": job["uid"], "n_img": n_img, "text": text,
                   "prompt_tokens": resp.get("prompt_eval_count"), "output_tokens": resp.get("eval_count")}
            try:
                if fl:
                    rec["findings"] = parse_fluid(lr, text)
                else:
                    f = lr._coerce(lr._extract_json(text))
                    rec["findings"] = {k: v["p"] for k, v in f["findings"].items()}
            except Exception as e:                                   # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
            with lock:
                with open(raw, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            return rec

        n = 0
        with ThreadPoolExecutor(a.workers) as ex:
            for fu in as_completed([ex.submit(one, j) for j in todo]):
                n += 1
                try:
                    r = fu.result()
                    msg = (f"ok, {r['n_img']} images, {r['prompt_tokens']} in / {r['output_tokens']} out tokens"
                           if "findings" in r else "PARSE FAIL " + r["error"][:80])
                except Exception as e:                               # noqa: BLE001
                    msg = f"CALL FAIL {type(e).__name__}: {str(e)[:100]}"
                print(f"[{mode}] {n}/{len(todo)} {time.time() - t0:.0f}s  {msg}", flush=True)
        recs = {}
        for line in raw.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "findings" in r:
                recs[r["uid"]] = r["findings"]
        print(f"[{mode}] labelled {len(recs)} of {len(jobs)}", flush=True)
        if a.limit or len(recs) < len(jobs):
            continue
        d = pd.DataFrame([{"StudyInstanceUID": u, "unusable": False, **recs[u]} for u in [j["uid"] for j in jobs]])
        if mode.startswith("fluid"):
            f = ROOT / f"data/03_report_labels_gemma4-{mode.replace('_', '-')}_validate58.csv"
            d[["StudyInstanceUID", "unusable"] + FLUID].to_csv(f, index=False)
            print("wrote", f.relative_to(ROOT), flush=True)
            fluid_score(d)
            continue
        f = ROOT / f"data/03_report_labels_gemma4-{mode.replace('_', '-')}_validate58.csv"
        d[["StudyInstanceUID", "unusable"] + lr.LABELS].to_csv(f, index=False)
        print("wrote", f.relative_to(ROOT), flush=True)
        df = d.rename(columns={"StudyInstanceUID": "uid"}).assign(mode=mode)
        mg.scoreboard(df, [mode])


def fluid_score(d):
    """The three fluid findings only: the model alone, Qwen, the current labels, and the model
    averaged into each (logit space), against the panel."""
    from sklearn.metrics import roc_auc_score
    aud = pd.read_csv(ROOT / "data/05_audit58.csv").set_index("StudyInstanceUID")
    u = list(aud.index)
    Y = aud.loc[u, ["official_" + t for t in FLUID]].values

    def load(f):
        return pd.read_csv(ROOT / "data" / f).set_index("StudyInstanceUID").loc[u, FLUID].values.astype(float)

    def lg(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    q, m15 = load("03_report_labels_qwen24t-med.csv"), load("16_medgemma15_corpus_all.csv")
    g = d.set_index("StudyInstanceUID").loc[u, FLUID].values.astype(float)
    sg = lambda x: 1 / (1 + np.exp(-x))                                     # noqa: E731
    cur = sg((lg(q) + lg(m15)) / 2)
    rows = {"Qwen": q, "current labels (Qwen+MedGemma)": cur, "Gemma alone": g,
            "Qwen + Gemma": sg((lg(q) + lg(g)) / 2), "current + Gemma (1/3)": sg((2 * lg(cur) + lg(g)) / 3)}
    print("\nFLUID findings, audit-58 AUROC:  mean | " + " | ".join(FLUID), flush=True)
    for k, P in rows.items():
        a_ = [roc_auc_score(Y[:, j], P[:, j]) for j in range(3)]
        print(f"  {k:34s} {np.mean(a_):.4f} | " + " | ".join(f"{x:.3f}" for x in a_), flush=True)


if __name__ == "__main__":
    main()
