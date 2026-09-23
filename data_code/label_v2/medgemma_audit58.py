"""
medgemma_audit58.py - can MedGemma 1.5 4B label our knees? Score it on the 58 panel studies.

MedGemma 1.5 reads MRI volumes plus text. Before it relabels anything, this asks it about
each of the 12 findings on the 58 audit studies and writes one CSV per mode in the
compare_labelers.py format (data/03_report_labels_medgemma15-<mode>_validate58.csv).

Input format copied from the MedGemma 1.5 technical report (arXiv:2604.05081, sec 2.3.1 and
Table 13): MRI trained on AXIAL volumes only, <= 85 slices per query, per-volume min-max,
grey copied into R, G and B, 896 px, and one yes/no question per condition ("MRI-US1"
prompt). Our shards are already windowed to 0-255 per series, which is the same min-max.

Series: one REAL series per slot (the engine's plan_slots, duplicate_rule coverage, NO
same-plane fallback, so no series is ever fed twice). The 85 images are split evenly over
the slots present. Left knees are mirrored with the engine's own _mirror_slot, exactly as
the laterality_flip members train (sagittal reverses slice order, coronal/axial reverse width).

Modes (all ask the same 12 questions; the panel labels are only used to SCORE, never input):
  rep       report text only, no images        (is MedGemma a worse report reader than Qwen?)
  ax        axial slots only, no report        (what the images alone give, in distribution)
  ax_rep    axial slots + report               (the labeller we would actually use)
  all_rep   every real slot + report           (side and front views: out of distribution)

The score is P(yes) from the next-token logits after a pre-filled "Final Answer:", so every
cell gets a continuous probability (AUROC needs ranks). The images and report come first
and the finding comes last, so each study's shared prefix is encoded ONCE and its cache is
reused for the 12 questions. checks/medgemma_prefix_cache_check.py proves that equals a full
forward on a tiny random Gemma 3, and the first study of every mode re-checks it on the real
model.

Needs the Hugging Face gate accepted for google/medgemma-1.5-4b-it, and a Modal secret named
`huggingface` holding HF_TOKEN in the workspace it runs on.

    python data_code/label_v2/medgemma_audit58.py --dry     # local: every prompt, no GPU
    python -m modal run data_code/label_v2/medgemma_audit58.py --limit 2
    python -m modal run data_code/label_v2/medgemma_audit58.py
"""
import copy
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "google/medgemma-1.5-4b-it"
MODES = ["rep", "ax", "ax_rep", "all_rep"]
MAX_SLICES = 85
TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# panel conventions from 03_label_reports.py, phrased as the condition MedGemma is asked about
LABEL = {
    "ACL": "a tear of the anterior cruciate ligament (ACL), partial or complete",
    "MCL": "an injury of the medial collateral ligament (MCL), including a sprain or tear",
    "Medial Meniscus": "a tear of the medial meniscus",
    "Lateral Meniscus": "a tear of the lateral meniscus",
    "Medial OA": "osteoarthritis of the medial tibiofemoral compartment with substantial "
                 "cartilage loss",
    "Lateral OA": "osteoarthritis of the lateral tibiofemoral compartment with substantial "
                  "cartilage loss",
    "PF OA": "osteoarthritis of the patellofemoral compartment with substantial cartilage loss",
    "Effusion": "a moderate or large knee joint effusion",
    "Synovitis": "synovitis (synovial thickening or inflammation)",
    "Baker's": "a Baker's (popliteal) cyst",
    "Contusion": "a traumatic bone contusion (bone bruise)",
    "Fracture": "a fracture",
}
TAIL = (' You may write out your argument before stating your final answer "Final Answer: yes" '
        'or "Final Answer: no"')
SLOTS_ALL = ["sag_plain", "sag_fluid", "cor_plain", "cor_fluid", "ax_plain", "ax_fluid"]
SLOTS_AX = ["ax_plain", "ax_fluid"]
SLOT_NAME = {
    "sag_plain": "sagittal (side view), non-fat-suppressed",
    "sag_fluid": "sagittal (side view), fluid-sensitive fat-suppressed",
    "cor_plain": "coronal (front view), non-fat-suppressed",
    "cor_fluid": "coronal (front view), fluid-sensitive fat-suppressed",
    "ax_plain": "axial (cross-section), non-fat-suppressed",
    "ax_fluid": "axial (cross-section), fluid-sensitive fat-suppressed",
}
FLIP_SIDE = "left"                     # same side every laterality_flip member mirrors

app = modal.App("medgemma-audit58")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "transformers==4.57.3", "accelerate", "pillow",
                      "numpy", "pandas", "pyyaml")
         .add_local_file(ROOT / "runs/shared_code.py", "/root/shared_code.py"))
shards = modal.Volume.from_name("knee-shards")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("knee-runs")
REMOTE_DIR = "/runs_vol/labelling/medgemma15_audit58"      # knee-runs volume
LOCAL_DIR = ROOT / "data_code/label_v2/results/medgemma15_audit58"


def _sc():
    for p in ("/root", str(ROOT / "runs")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import shared_code as sc
    return sc


def slot_plan(uids):
    """{study: {slot: series}}: one REAL series per slot, no same-plane fallback."""
    import pandas as pd
    idx = pd.read_csv(ROOT / "data/07_series_index.csv")
    cfg = {"input": {"slots": SLOTS_ALL, "duplicate_rule": "coverage",
                     "fallback_same_plane": False}}
    return _sc().plan_slots(idx[idx.study.isin(set(uids))], cfg)


def build_jobs(uids=None, reports=True):
    """One job per study: report text, side, one series per real slot."""
    import pandas as pd
    if uids is None:
        uids = list(pd.read_csv(ROOT / "data/05_audit58.csv").StudyInstanceUID)
    rep = pd.read_csv(ROOT / "data/02_reports_en.csv").set_index("StudyInstanceUID")
    lat = pd.read_csv(ROOT / "data/12_laterality.csv").set_index("StudyInstanceUID")["side"]
    plan = slot_plan(uids)
    return [{"uid": u, "report": str(rep.loc[u, "Report_en"]).strip() if reports else "",
             "side": lat.get(u), "slots": plan.get(u, {})} for u in uids]


def pick(n, k):
    """k indices spread evenly over n, as the technical report samples a stacked volume."""
    if n <= k:
        return list(range(n))
    return [int(round(i / (k - 1) * (n - 1))) for i in range(k)]


def volume_slices(z, job, mode):
    """[(slot, [uint8 HxW slices])] for this mode; [] for report-only. Left knees mirrored."""
    if mode == "rep":
        return []
    want = SLOTS_AX if mode.startswith("ax") else SLOTS_ALL
    have = [s for s in want if job["slots"].get(s) in z.files]
    per = MAX_SLICES // max(len(have), 1)
    mirror = job.get("side") == FLIP_SIDE
    out = []
    for s in have:
        vol = z[job["slots"][s]]
        vol = vol[pick(vol.shape[0], per)]
        if mirror:
            vol = _sc()._mirror_slot(vol, s)
        out.append((s, list(vol)))
    return out


def prompt_text(job, mode, task):
    q = f"Is there {LABEL[task]} in the MRI volume?"
    if mode == "rep":
        return (f'After reading this knee MRI report "{job["report"]}", is there {LABEL[task]} '
                f'in this knee?' + TAIL)
    if mode == "ax":
        return q + TAIL
    return f'After looking at the patient history "{job["report"]}", {q}' + TAIL


def messages(groups, text):
    """groups = [(slot, n_slices)]; each series is introduced by its view and sequence."""
    # The Gemma 3 template TRIMS each text item and joins items with nothing, so two text
    # items in a row would run together ("SLICE 21KNEE MRI SERIES"). Adjacent text is merged
    # into one item with a newline inside it; images are padded with blank lines by the
    # processor itself.
    content = []
    for slot, n in groups:
        content.append({"type": "text", "text": f"KNEE MRI SERIES: {SLOT_NAME[slot]}, "
                                                f"{n} slices in order."})
        for i in range(n):
            content.append({"type": "image"})
            content.append({"type": "text", "text": f"SLICE {i + 1}"})
    content.append({"type": "text", "text": text})
    merged = []
    for c in content:
        if merged and c["type"] == "text" and merged[-1]["type"] == "text":
            merged[-1] = {"type": "text", "text": merged[-1]["text"] + "\n" + c["text"]}
        else:
            merged.append(c)
    content = merged
    return [{"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "Final Answer:"}]}]


def common_prefix_len(seqs):
    """Tokens shared by every sequence (the images + report part)."""
    n = min(len(s) for s in seqs)
    for i in range(n):
        v = seqs[0][i]
        if any(s[i] != v for s in seqs[1:]):
            return i
    return n


def yes_prob_shared(model, inp_list, yes_ids, no_ids, check=False):
    """P(yes) for each of several prompts that share a prefix: encode the prefix (with all
    its images) once, then run each prompt's remaining tokens on a copy of its cache.

    inp_list: processor outputs, one per question, identical up to the question. check=True
    also runs one prompt in full and returns the largest logit difference.

    Architecturally, the prefix pass is everything: the SigLIP vision tower encodes every
    slice, the projector pools each to 256 tokens and splices them into the text embeddings,
    and the decoder runs over the whole prefix, leaving each layer's keys and values in the
    cache (image tokens attend bidirectionally within their own image, via token_type_ids,
    which only matters in this prefill). Each question then only pushes its own few tokens
    through the decoder, attending to that cache.

    Returns (P(yes) per prompt, P(yes)+P(no) mass per prompt, prefix length, check diff).
    """
    import torch
    ids = [x["input_ids"][0].tolist() for x in inp_list]
    L = common_prefix_len(ids) - 1        # keep >= 1 token of suffix for every prompt
    first = inp_list[0]
    kw = {"input_ids": first["input_ids"][:, :L], "use_cache": True}
    if "attention_mask" in first:
        kw["attention_mask"] = first["attention_mask"][:, :L]
    if "token_type_ids" in first:
        kw["token_type_ids"] = first["token_type_ids"][:, :L]
    if "pixel_values" in first:
        kw["pixel_values"] = first["pixel_values"]
    with torch.inference_mode():
        pre = model(**kw)
        cache = pre.past_key_values
        probs, mass, last = [], [], []
        for x in inp_list:
            suf = x["input_ids"][:, L:]
            n = suf.shape[1]
            skw = {"input_ids": suf, "past_key_values": copy.deepcopy(cache), "use_cache": True,
                   "cache_position": torch.arange(L, L + n, device=suf.device)}
            if "attention_mask" in x:
                skw["attention_mask"] = x["attention_mask"]
            if "token_type_ids" in x:
                skw["token_type_ids"] = x["token_type_ids"][:, L:]
            lg = model(**skw).logits[0, -1].float()
            last.append(lg)
            probs.append(float(torch.sigmoid(torch.logsumexp(lg[yes_ids], 0)
                                             - torch.logsumexp(lg[no_ids], 0))))
            sm = torch.softmax(lg, 0)
            mass.append(float(sm[yes_ids].sum() + sm[no_ids].sum()))
        diff = None
        if check:
            fkw = {k: v for k, v in inp_list[0].items()}
            full = model(**fkw).logits[0, -1].float()
            p_full = float(torch.sigmoid(torch.logsumexp(full[yes_ids], 0)
                                         - torch.logsumexp(full[no_ids], 0)))
            diff = (float((full - last[0]).abs().max()), abs(p_full - probs[0]))
    return probs, mass, L, diff


@app.function(gpu="L40S", image=image, timeout=3 * 3600,
              volumes={"/shards": shards, "/root/.cache/huggingface": hf_cache,
                       "/runs_vol": runs_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def score(jobs, modes, tag):
    """Returns (rows, log text). Also writes log.txt and rows.jsonl to the knee-runs volume
    after EVERY study, so a crash or a stop keeps everything done so far."""
    import os
    import time

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    assert os.environ.get("HF_TOKEN"), "Modal secret `huggingface` has no HF_TOKEN"
    out_dir = f"{REMOTE_DIR}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    logf = open(f"{out_dir}/log.txt", "a", encoding="utf-8")
    rowf = open(f"{out_dir}/rows.jsonl", "a", encoding="utf-8")
    lines = []

    def log(*a, **_):
        m = " ".join(str(x) for x in a)
        print(m, flush=True)
        lines.append(m)
        logf.write(m + "\n")
        logf.flush()

    log(f"remote results: knee-runs volume {out_dir.replace('/runs_vol/', '')}")
    t_load = time.time()
    log(f"GPU {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB; "
          f"{len(jobs)} studies x modes {modes} x {len(TASKS)} findings", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    hf_cache.commit()
    tc = model.config.text_config
    log(f"loaded {MODEL} in {time.time() - t_load:.0f}s: decoder {tc.num_hidden_layers} layers, "
          f"hidden {tc.hidden_size}, heads {tc.num_attention_heads} / kv {tc.num_key_value_heads}, "
          f"sliding window {tc.sliding_window}; {model.config.mm_tokens_per_image} tokens per image; "
          f"GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    tok = proc.tokenizer
    yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0]
                      for w in (" yes", "yes", " Yes", "Yes")})
    no_ids = sorted({tok.encode(w, add_special_tokens=False)[0]
                     for w in (" no", "no", " No", "No")})
    log("yes tokens", [(i, tok.decode([i])) for i in yes_ids],
          "no tokens", [(i, tok.decode([i])) for i in no_ids], flush=True)

    out, shown = [], set()
    t0 = time.time()
    for k, job in enumerate(jobs):
        with np.load(f"/shards/{job['uid']}.npz") as z:
            per_mode = {m: volume_slices(z, job, m) for m in modes}
        for m in modes:
            row = {"uid": job["uid"], "mode": m, "side": job.get("side")}
            try:
                t1 = time.time()
                groups = per_mode[m]
                imgs = [Image.fromarray(np.ascontiguousarray(s)).convert("RGB")
                        for _, sl in groups for s in sl]
                inps = []
                for t in TASKS:
                    msg = messages([(s, len(sl)) for s, sl in groups], prompt_text(job, m, t))
                    text = proc.apply_chat_template(msg, continue_final_message=True,
                                                    tokenize=False)
                    # the template already starts with <bos>; do not let the tokenizer add one
                    x = proc(text=text, images=imgs or None, add_special_tokens=False,
                             return_tensors="pt").to("cuda")
                    ids0 = x["input_ids"][0]
                    assert ids0[0] == tok.bos_token_id and ids0[1] != tok.bos_token_id, "BOS"
                    if imgs:
                        n_it = int((ids0 == model.config.image_token_id).sum())
                        assert n_it == len(imgs) * model.config.mm_tokens_per_image, n_it
                        assert "token_type_ids" in x, "no token_type_ids: image attention causal"
                        x["pixel_values"] = x["pixel_values"].to(torch.bfloat16)
                    inps.append(x)
                first = m not in shown
                probs, mass, L, diff = yes_prob_shared(model, inps, yes_ids, no_ids, check=first)
                if first:
                    shown.add(m)
                    txt = tok.decode(inps[0]["input_ids"][0][L:])
                    log(f"  [{m}] first study: {len(imgs)} images, {L} prefix tokens + "
                          f"{inps[0]['input_ids'].shape[1] - L} question tokens; question part: "
                          f"{txt!r}", flush=True)
                    # bf16 logits of magnitude 32-64 are stored in steps of 0.25, so a max
                    # difference of 0.25-0.4 over 262k vocab entries is one rounding step
                    log(f"  [{m}] CACHE CHECK full pass vs shared prefix: max |logit diff| "
                        f"{diff[0]:.4f} (one bf16 step at this size is 0.25), "
                        f"|P(yes) diff| {diff[1]:.4f} (should be < 0.02)", flush=True)
                row.update({"n_img": len(imgs),
                            "slots": "+".join(f"{s}:{len(sl)}" for s, sl in groups),
                            "prefix_tok": L, "total_tok": int(inps[-1]["input_ids"].shape[1]),
                            "cache_diff": diff[0] if diff else None,
                            "cache_pdiff": diff[1] if diff else None, "yn_mass_min": min(mass),
                            "yn_mass_mean": float(np.mean(mass)), "sec": time.time() - t1})
                row.update(dict(zip(TASKS, probs)))
            except Exception as e:                             # noqa: BLE001
                log(f"  ERROR {job['uid']} {m}: {type(e).__name__}: {e}", flush=True)
                row.update({"error": f"{type(e).__name__}: {e}"})
            out.append(row)
            rowf.write(json.dumps(row) + "\n")
        rowf.flush()
        el = time.time() - t0
        eta = el / (k + 1) * (len(jobs) - k - 1)
        rs = out[-len(modes):]
        log(f"{k + 1}/{len(jobs)} {job['uid'][-8:]} {job.get('side')}  {el:.0f}s elapsed, "
              f"ETA {eta / 60:.1f} min  peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
              flush=True)
        for r in rs:
            if "error" in r:
                continue
            p = " ".join(f"{r[t]:.2f}" for t in TASKS)
            log(f"    {r['mode']:8s} {r['n_img']:3d} img {r['total_tok']:6d} tok "
                  f"{r['sec']:5.1f}s  yes/no mass min {r['yn_mass_min']:.2f}  P(yes): {p}",
                  flush=True)
        runs_vol.commit()                               # persist after every study
    log(f"done: {len(out)} study-modes in {time.time() - t0:.0f}s")
    logf.close()
    rowf.close()
    runs_vol.commit()
    return out, "\n".join(lines)


def dry_run():
    """Local, no model: slot plan, slice picks, mirroring and prompts for real shards."""
    import collections

    import numpy as np
    jobs = build_jobs()
    c = collections.Counter(s for j in jobs for s in j["slots"])
    print(f"{len(jobs)} audit studies; left knees {sum(j['side'] == 'left' for j in jobs)}; "
          f"real slots: " + ", ".join(f"{s} {c[s]}" for s in SLOTS_ALL))
    nslots = collections.Counter(len(j["slots"]) for j in jobs)
    print("  slots per study:", dict(sorted(nslots.items())))
    dup = sum(len(set(j["slots"].values())) != len(j["slots"]) for j in jobs)
    print("  studies feeding one series twice:", dup)
    shards_dir = ROOT / "data/shards"
    local = [p.stem for p in shards_dir.glob("*.npz")]
    for job in build_jobs(local, reports=False):
        with np.load(shards_dir / f"{job['uid']}.npz") as z:
            for m in MODES:
                g = volume_slices(z, job, m)
                n = sum(len(sl) for _, sl in g)
                print(f"  {job['uid'][-8:]} {job['side']:5s} {m:8s} {n:3d} img  "
                      f"{' + '.join(f'{s}:{len(sl)}' for s, sl in g) or '-'}")
            # mirroring really happened on a left knee, on the right axis
            if job["side"] == FLIP_SIDE:
                s = "sag_fluid" if "sag_fluid" in job["slots"] else next(iter(job["slots"]))
                raw = z[job["slots"][s]]
                got = volume_slices(z, job, "all_rep")
                vol = np.stack(dict(got)[s])
                ref = raw[pick(raw.shape[0], len(vol))]
                ok = (np.array_equal(vol, ref[::-1]) if s.startswith("sag")
                      else np.array_equal(vol, ref[:, :, ::-1]))
                print(f"    mirror check {s}: {'OK' if ok else 'FAIL'}")
    j = jobs[0]
    for m in MODES:
        print(f"\n[{m}] {prompt_text(j, m, 'Effusion')[:160]}")
    print("\nmessages:", json.dumps(messages([("sag_fluid", 2), ("ax_fluid", 1)], "q"))[:420])


@app.local_entrypoint()
def main(modes: str = ",".join(MODES), limit: int = 0):
    import contextlib
    import datetime
    import io as _io

    import pandas as pd
    jobs = build_jobs()
    if limit:
        jobs = jobs[:limit]
    ms = [m for m in modes.split(",") if m]
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_first{limit}" if limit else "")
    out = LOCAL_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": MODEL, "modes": ms, "studies": len(jobs), "max_slices": MAX_SLICES,
            "slots_all": SLOTS_ALL, "slots_ax": SLOTS_AX, "flip_side": FLIP_SIDE,
            "labels": LABEL, "remote_dir": f"knee-runs/labelling/medgemma15_audit58/{tag}",
            "jobs": [{k: v for k, v in j.items() if k != "report"} for j in jobs]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("local results folder:", out)

    rows, log_text = score.remote(jobs, ms, tag)
    (out / "log.txt").write_text(log_text + "\n", encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw.csv", index=False)
    if "error" in df:
        print("ERRORS:", int(df["error"].notna().sum()), "study-modes failed")
    for m in ms:
        d = df[df["mode"] == m].rename(columns={"uid": "StudyInstanceUID"})
        d.insert(1, "unusable", False)
        d = d[["StudyInstanceUID", "unusable"] + TASKS]
        d.to_csv(out / f"labels_{m}.csv", index=False)
        if not limit:                   # compare_labelers.py reads these; a smoke test never
            f = ROOT / f"data/03_report_labels_medgemma15-{m.replace('_', '-')}_validate58.csv"
            d.to_csv(f, index=False)    # overwrites them
            print("wrote", f.relative_to(ROOT))
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        scoreboard(df, ms)
    print(buf.getvalue())
    (out / "scoreboard.txt").write_text(buf.getvalue(), encoding="utf-8")
    print("saved:", ", ".join(sorted(p.name for p in out.iterdir())), "in", out)


def scoreboard(df, ms):
    """Every mode, and Qwen + each mode averaged, against the panel on the same studies."""
    import warnings

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    aud = pd.read_csv(ROOT / "data/05_audit58.csv").set_index("StudyInstanceUID")
    qwen = pd.read_csv(ROOT / "data/03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID")
    ok = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()] \
        if "error" in df else df
    uids = sorted(set.intersection(*[set(ok[ok["mode"] == m].uid) for m in ms]))
    if len(uids) < 20:
        # AUROC needs positives AND negatives per finding; a smoke test has neither
        print(f"\nSCOREBOARD skipped: {len(uids)} studies is too few to score (needs >= 20).")
        return
    Y = aud.loc[uids, ["official_" + t for t in TASKS]].values
    preds = {"qwen24t-med": qwen.loc[uids, TASKS].values.astype(float)}
    for m in ms:
        preds[m] = ok[ok["mode"] == m].set_index("uid").loc[uids, TASKS].values.astype(float)
    for m in ms:
        preds[f"qwen+{m} mean"] = 0.5 * preds["qwen24t-med"] + 0.5 * preds[m]

    def macro(P, i=None):
        i = np.arange(len(Y)) if i is None else i
        return np.mean([roc_auc_score(Y[i, j], P[i, j]) for j in range(12)
                        if 0 < Y[i, j].sum() < len(i)])

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(Y), len(Y)) for _ in range(2000)]
    q = preds["qwen24t-med"]
    base = macro(q)
    print(f"\nSCOREBOARD vs the panel, {len(uids)} studies (macro AUROC; delta vs Qwen with "
          f"95% bootstrap CI)")
    for name, P in preds.items():
        d = np.array([macro(P, b) - macro(q, b) for b in boots])
        print(f"  {name:22s} {macro(P):.4f} ({macro(P) - base:+.4f})  "
              f"CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  "
              f"P(better) {100 * (d > 0).mean():3.0f}%  Brier {np.mean((P - Y) ** 2):.4f}")
    print("\nper finding AUROC:")
    print("  " + " ".join(f"{n[:12]:>12s}" for n in preds) + "   finding")
    for j, t in enumerate(TASKS):
        if 0 < Y[:, j].sum() < len(Y):
            print("  " + " ".join(f"{roc_auc_score(Y[:, j], P[:, j]):12.3f}" for P in preds.values())
                  + f"   {t} ({int(Y[:, j].sum())} pos)")


if __name__ == "__main__" and "--dry" in sys.argv:
    dry_run()
