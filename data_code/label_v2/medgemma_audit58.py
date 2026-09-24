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

Modes are MODE_DEF: which slots, and whether the report is in the prompt. All ask the same 12
questions; the panel labels are only used to SCORE, never input. Run 1 (23 Sep) used
rep / ax / ax_rep / all_rep; images alone won on the fluid findings, so the plan continues
images-only, a batch at a time (pass --modes):
  A  ax, fluid, all              do views beyond axial help at all?
  B  sag_fluid, cor_fluid        which single fluid view carries each fluid finding?
  C  sag, cor                    side / front views for menisci+ACL / OA+MCL

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
MAX_SLICES = 85
TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# panel conventions from 03_label_reports.py, phrased as the condition MedGemma is asked about
LABEL = {
    "ACL": "a partial or complete anterior cruciate ligament (ACL) tear",
    "MCL": "a medial collateral ligament (MCL) sprain or tear",
    "Medial Meniscus": "a medial meniscus tear",
    "Lateral Meniscus": "a lateral meniscus tear",
    "Medial OA": "medial tibiofemoral osteoarthritis with high-grade cartilage loss",
    "Lateral OA": "lateral tibiofemoral osteoarthritis with high-grade cartilage loss",
    "PF OA": "patellofemoral osteoarthritis with high-grade cartilage loss",
    "Effusion": "a moderate or large knee joint effusion",
    "Synovitis": "synovitis (synovial thickening)",
    "Baker's": "a Baker's (popliteal) cyst",
    "Contusion": "a traumatic bone contusion (bone bruise)",
    "Fracture": "a fracture",
}
# batch D: the same findings with the panel's own decision rules spelled out (from the
# 03_label_reports.py system prompt); MedGemma was trained on short labels, so this is the test
LABEL_DEF = {
    "ACL": "an anterior cruciate ligament (ACL) tear, partial or complete, including interstitial "
           "tears (mucoid degeneration alone is not a tear)",
    "MCL": "a medial collateral ligament (MCL) injury, including a grade 1-2 sprain (not the "
           "lateral collateral ligament)",
    "Medial Meniscus": "a medial meniscus tear, meaning abnormal signal or a defect reaching the "
                       "articular surface (intrasubstance degenerative signal, extrusion alone, or "
                       "a post-surgical meniscus without a new tear does not count)",
    "Lateral Meniscus": "a lateral meniscus tear, meaning abnormal signal or a defect reaching the "
                        "articular surface (intrasubstance degenerative signal, extrusion alone, or "
                        "a post-surgical meniscus without a new tear does not count)",
    "Medial OA": "medial tibiofemoral osteoarthritis with substantial cartilage loss (grade 3-4 "
                 "chondropathy, a full-thickness defect or bone-on-bone; osteophytes or mild "
                 "fissuring alone do not count)",
    "Lateral OA": "lateral tibiofemoral osteoarthritis with substantial cartilage loss (grade 3-4 "
                  "chondropathy, a full-thickness defect or bone-on-bone; osteophytes or mild "
                  "fissuring alone do not count)",
    "PF OA": "patellofemoral osteoarthritis with substantial cartilage loss of the patella or "
             "trochlea (grade 3-4 chondropathy or a full-thickness defect; mild fissuring alone "
             "does not count)",
    "Effusion": "a knee joint effusion of MODERATE or LARGE size, or a haemarthrosis (trace, "
                "minimal, physiologic, small or mild fluid does not count)",
    "Synovitis": "synovitis: synovial thickening or enhancement, including inflammation of "
                 "Hoffa's fat pad (hoffitis)",
    "Baker's": "a Baker's (popliteal) cyst",
    "Contusion": "a traumatic bone contusion (bone marrow oedema from an injury) or a muscle "
                 "contusion (subchondral oedema next to degenerative cartilage loss does not "
                 "count)",
    "Fracture": "a fracture, including insufficiency and avulsion fractures (an osteochondral "
                "defect alone does not count)",
}
TAIL = (' You may write out your argument before stating your final answer "Final Answer: yes" '
        'or "Final Answer: no"')
SLOTS_ALL = ["sag_plain", "sag_fluid", "cor_plain", "cor_fluid", "ax_plain", "ax_fluid"]
SLOTS_AX = ["ax_plain", "ax_fluid"]
SLOTS_FLUID = ["sag_fluid", "cor_fluid", "ax_fluid"]
# mode -> (slots it may use, whether the report is in the prompt). Only the slots a study
# really has are fed, 85 images split evenly over them, always in SLOTS_ALL order.
MODE_DEF = {
    "rep":       ([], True),                               # report text only
    "ax":        (SLOTS_AX, False),                        # batch A: axial images
    "ax_rep":    (SLOTS_AX, True),
    "all_rep":   (SLOTS_ALL, True),
    "fluid":     (SLOTS_FLUID, False),                     # batch A: every fluid view
    "all":       (SLOTS_ALL, False),                       # batch A: every view
    "sag_fluid": (["sag_fluid"], False),                   # batch B
    "cor_fluid": (["cor_fluid"], False),                   # batch B
    "sag":       (["sag_plain", "sag_fluid"], False),      # batch C1: menisci, ACL
    "cor":       (["cor_plain", "cor_fluid"], False),      # batch C2: OA, MCL
    "all_def":   (SLOTS_ALL, False),                       # batch D: panel definitions
    "all_num":   (SLOTS_ALL, False),                       # batch D: stated probability
    "all_reason": (SLOTS_ALL, False),                      # batch E: reasoning first
}
# how a mode asks: which label wording, and how the answer is read ("token" = P(yes) from the
# next-token logits after "Final Answer:"; "number" = MedGemma writes a probability itself)
MODE_OPT = {"all_def": {"labels": "def"}, "all_num": {"answer": "number"},
            "all_reason": {"answer": "reason"}}
REASON_MAX_NEW = 320          # tokens of argument before "Final Answer:" is forced
REASON_MARK = "Final Answer:"
NUM_PREFILL = "Probability:"
NUM_MAX_NEW = 8
MODES = list(MODE_DEF)
FLUID_TASKS = ["Effusion", "Synovitis", "Baker's"]
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
         .pip_install("torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.3", "accelerate", "pillow",
                      "numpy", "pandas", "pyyaml")
         .add_local_file(ROOT / "runs/shared_code.py", "/root/shared_code.py"))
shards = modal.Volume.from_name("knee-shards")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("knee-runs")
REMOTE_DIR = "/runs_vol/labelling/medgemma15_audit58"      # knee-runs volume
SHARDS_DIR = "/shards"                                     # knee-shards volume
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
    if k == 1:
        return [n // 2]
    return [int(round(i / (k - 1) * (n - 1))) for i in range(k)]


def volume_slices(z, job, mode):
    """[(slot, [uint8 HxW slices])] for this mode; [] for report-only. Left knees mirrored."""
    want = MODE_DEF[mode][0]
    if not want:
        return []
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
    opt = MODE_OPT.get(mode, {})
    lab = (LABEL_DEF if opt.get("labels") == "def" else LABEL)[task]
    slots, with_report = MODE_DEF[mode]
    if opt.get("answer") == "number":
        return (f"What is the probability that there is {lab} in the MRI volume? Answer with a "
                f"single number between 0 and 1.")
    q = f"Is there {lab} in the MRI volume?"
    if not slots:
        return (f'After reading this knee MRI report "{job["report"]}", is there {lab} '
                f'in this knee?' + TAIL)
    if not with_report:
        return q + TAIL
    return f'After looking at the patient history "{job["report"]}", {q}' + TAIL


def prefill(mode):
    ans = MODE_OPT.get(mode, {}).get("answer")
    if ans == "number":
        return NUM_PREFILL
    return "" if ans == "reason" else "Final Answer:"


def messages(groups, text, pre="Final Answer:"):
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
            {"role": "assistant", "content": [{"type": "text", "text": pre}]}]


def common_prefix_len(seqs):
    """Tokens shared by every sequence (the images + report part)."""
    n = min(len(s) for s in seqs)
    for i in range(n):
        v = seqs[0][i]
        if any(s[i] != v for s in seqs[1:]):
            return i
    return n


def encode_prompts(proc, texts, imgs):
    """Token ids for every prompt + ONE pixel tensor for the shared images.

    The processor would resize every slice to 896 px and rebuild pixel_values once PER
    QUESTION (12x the CPU work and 12 host-to-GPU copies of ~0.8 GB). The images are the same
    for all 12, so they are processed once here, and each prompt's text is expanded exactly
    the way Gemma3Processor does it: every <start_of_image> becomes full_image_sequence
    (blank lines, <start_of_image>, 256 image tokens, <end_of_image>, blank lines).
    checks/medgemma_processor_check.py proves this equals the processor byte for byte.
    """
    tok = proc.tokenizer
    pv = None
    if imgs:
        pv = proc.image_processor(images=imgs, do_pan_and_scan=False,
                                  return_tensors="pt")["pixel_values"]
    ids = [tok(t.replace(proc.boi_token, proc.full_image_sequence),
               add_special_tokens=False)["input_ids"] for t in texts]
    return ids, pv


def yes_prob_shared(model, ids_list, pixel_values, image_token_id, yes_ids, no_ids,
                    pad_id=0, check=False):
    """P(yes) for prompts that share a prefix: the prefix (all the images) is encoded ONCE,
    then every prompt's remaining tokens run as ONE right-padded batch on that cache.

    Architecturally the prefix pass is everything: the SigLIP vision tower encodes every
    slice, the projector pools each to 256 tokens and splices them into the text embeddings,
    and the decoder runs over the whole prefix, leaving each layer's keys and values in the
    cache (image tokens attend bidirectionally within their own image, via token_type_ids,
    which only matters in this prefill). The cache is then repeated once per question and the
    questions go through the decoder together, each read at its own last real token.

    Returns (P(yes) per prompt, P(yes)+P(no) mass per prompt, prefix length, check) where
    check is (max |logit diff|, |P(yes) diff|) of prompt 0 against a full pass, or None.
    """
    import torch
    dev = next(model.parameters()).device
    L = common_prefix_len(ids_list) - 1           # keep >= 1 own token per prompt
    pre_ids = torch.tensor([ids_list[0][:L]], device=dev)
    kw = {"input_ids": pre_ids, "attention_mask": torch.ones_like(pre_ids),
          "token_type_ids": (pre_ids == image_token_id).long(), "use_cache": True}
    if pixel_values is not None:
        kw["pixel_values"] = pixel_values.to(dev, torch.bfloat16)

    def p_yes(lg):
        return torch.sigmoid(torch.logsumexp(lg[..., yes_ids], -1)
                             - torch.logsumexp(lg[..., no_ids], -1))

    with torch.inference_mode():
        cache = model(**kw).past_key_values
        B = len(ids_list)
        suf = [x[L:] for x in ids_list]
        n = max(len(x) for x in suf)
        lens = torch.tensor([len(x) for x in suf], device=dev)
        s_ids = torch.full((B, n), pad_id, dtype=torch.long, device=dev)
        s_mask = torch.zeros((B, L + n), dtype=torch.long, device=dev)
        s_mask[:, :L] = 1
        for i, x in enumerate(suf):
            s_ids[i, :len(x)] = torch.tensor(x, device=dev)
            s_mask[i, L:L + len(x)] = 1
        cache.batch_repeat_interleave(B)
        out = model(input_ids=s_ids, attention_mask=s_mask, past_key_values=cache,
                    token_type_ids=torch.zeros_like(s_ids), use_cache=True,
                    cache_position=torch.arange(L, L + n, device=dev)).logits
        lg = out[torch.arange(B, device=dev), lens - 1].float()        # (B, vocab)
        probs = p_yes(lg)
        sm = torch.softmax(lg, -1)
        mass = sm[:, yes_ids].sum(-1) + sm[:, no_ids].sum(-1)
        chk = None
        if check:
            f_ids = torch.tensor([ids_list[0]], device=dev)
            fkw = {"input_ids": f_ids, "attention_mask": torch.ones_like(f_ids),
                   "token_type_ids": (f_ids == image_token_id).long()}
            if pixel_values is not None:
                fkw["pixel_values"] = kw["pixel_values"]
            full = model(**fkw).logits[0, -1].float()
            chk = (float((full - lg[0]).abs().max()), float((p_yes(full) - probs[0]).abs()))
    return probs.tolist(), mass.tolist(), L, chk


def parse_prob(txt):
    """First number in MedGemma's reply -> a probability; 0-100 read as a percentage."""
    import re
    m = re.search(r"(\d+(?:\.\d+)?|\.\d+)\s*(%)?", txt)
    if not m:
        return float("nan")
    v = float(m.group(1))
    if m.group(2) or 1 < v <= 100:
        v /= 100
    return min(max(v, 0.0), 1.0)


def stated_prob_shared(model, ids_list, pixel_values, image_token_id, stop_ids, tok,
                       max_new=NUM_MAX_NEW, check=False):
    """MedGemma WRITES a probability: the shared prefix (all images) is encoded once, then each
    question runs on its own copy of the cache and is decoded greedily for a few tokens.

    Per question, not batched: a right-padded batch would give the shorter questions shifted
    positions during decoding. A few tokens x 12 questions is cheap next to the prefix.
    Returns (values, raw texts, prefix length, check) where check says whether the first
    question's greedy tokens equal model.generate on the full prompt.
    """
    import torch
    dev = next(model.parameters()).device
    L = common_prefix_len(ids_list) - 1
    pre_ids = torch.tensor([ids_list[0][:L]], device=dev)
    kw = {"input_ids": pre_ids, "attention_mask": torch.ones_like(pre_ids),
          "token_type_ids": (pre_ids == image_token_id).long(), "use_cache": True}
    if pixel_values is not None:
        kw["pixel_values"] = pixel_values.to(dev, torch.bfloat16)
    vals, raws, first_toks = [], [], None
    with torch.inference_mode():
        cache = model(**kw).past_key_values
        for qi, ids in enumerate(ids_list):
            c = copy.deepcopy(cache)
            x = torch.tensor([ids[L:]], device=dev)
            pos = L
            out_toks = []
            for _ in range(max_new):
                n = x.shape[1]
                lg = model(input_ids=x, attention_mask=torch.ones((1, pos + n), device=dev,
                                                                 dtype=torch.long),
                           past_key_values=c, use_cache=True,
                           token_type_ids=torch.zeros_like(x),
                           cache_position=torch.arange(pos, pos + n, device=dev)).logits[0, -1]
                pos += n
                t = int(lg.argmax())
                if t in stop_ids:
                    break
                out_toks.append(t)
                x = torch.tensor([[t]], device=dev)
            txt = tok.decode(out_toks)
            raws.append(txt)
            vals.append(parse_prob(txt))
            if qi == 0:
                first_toks = out_toks
        chk = None
        if check:
            f = torch.tensor([ids_list[0]], device=dev)
            gkw = {"input_ids": f, "attention_mask": torch.ones_like(f),
                   "token_type_ids": (f == image_token_id).long(), "max_new_tokens": max_new,
                   "do_sample": False}
            if pixel_values is not None:
                gkw["pixel_values"] = kw["pixel_values"]
            g = model.generate(**gkw)[0, f.shape[1]:].tolist()
            g = g[:next((i for i, t in enumerate(g) if t in stop_ids), len(g))]
            chk = (g == first_toks, tok.decode(g))
    return vals, raws, L, chk


def reason_prob_shared(model, ids_list, pixel_values, image_token_id, yes_ids, no_ids,
                       stop_ids, tok, max_new=None, check=False):
    """MedGemma REASONS first, then answers: the shared prefix (all images) is encoded once;
    each question runs on its own cache copy and is decoded greedily until its text reaches
    "Final Answer:". P(yes) is read from the logits right there. If it stops (end of turn) or
    runs out of tokens first, "Final Answer:" is appended and read the same way (counted as
    forced). Per question, not batched, for the same reason as stated_prob_shared.

    Returns (P(yes), P(yes)+P(no) mass, reasoning texts, n tokens, forced flags, prefix
    length, check) where check = |P(yes)| difference of question 0 against ONE full forward
    pass over prompt + its own reasoning + "Final Answer:".
    """
    import torch
    dev = next(model.parameters()).device
    L = common_prefix_len(ids_list) - 1
    pre_ids = torch.tensor([ids_list[0][:L]], device=dev)
    kw = {"input_ids": pre_ids, "attention_mask": torch.ones_like(pre_ids),
          "token_type_ids": (pre_ids == image_token_id).long(), "use_cache": True}
    if pixel_values is not None:
        kw["pixel_values"] = pixel_values.to(dev, torch.bfloat16)
    max_new = REASON_MAX_NEW if max_new is None else max_new
    mark_ids = tok(" " + REASON_MARK, add_special_tokens=False)["input_ids"]

    def step(c, x, pos):
        n = x.shape[1]
        lg = model(input_ids=x, attention_mask=torch.ones((1, pos + n), device=dev,
                                                         dtype=torch.long),
                   past_key_values=c, use_cache=True, token_type_ids=torch.zeros_like(x),
                   cache_position=torch.arange(pos, pos + n, device=dev)).logits[0, -1].float()
        return lg, pos + n

    def read(lg):
        sm = torch.softmax(lg, 0)
        p = float(torch.sigmoid(torch.logsumexp(lg[yes_ids], 0) - torch.logsumexp(lg[no_ids], 0)))
        return p, float(sm[yes_ids].sum() + sm[no_ids].sum())

    probs, mass, texts, ntok, forced, full0 = [], [], [], [], [], None
    with torch.inference_mode():
        cache = model(**kw).past_key_values
        for qi, ids in enumerate(ids_list):
            c = copy.deepcopy(cache)
            lg, pos = step(c, torch.tensor([ids[L:]], device=dev), L)
            gen, done = [], False
            for _ in range(max_new):
                t = int(lg.argmax())
                if t in stop_ids:
                    break
                gen.append(t)
                lg, pos = step(c, torch.tensor([[t]], device=dev), pos)
                tail = tok.decode(gen[-12:])
                # its OWN answer line only: batch E (24 Sep) showed MedGemma often goes on to
                # write the NEXT question itself, quoting '"Final Answer: yes"', and the old
                # endswith() test stopped inside that quote (94 of 696 answers)
                if tail.rstrip().endswith(REASON_MARK) and (
                        "\n" + REASON_MARK in "\n" + tok.decode(gen).split("\n")[-1]):
                    done = True
                    break
                if "\nIs there" in tok.decode(gen):   # started a new question: answer now
                    break
            if not done:                       # no answer reached: ask for it explicitly
                gen += mark_ids
                lg, pos = step(c, torch.tensor([mark_ids], device=dev), pos)
            p, ms = read(lg)
            probs.append(p)
            mass.append(ms)
            texts.append(tok.decode(gen))
            ntok.append(len(gen))
            forced.append(not done)
            if qi == 0:
                full0 = ids + gen
        chk = None
        if check:
            f = torch.tensor([full0], device=dev)
            fkw = {"input_ids": f, "attention_mask": torch.ones_like(f),
                   "token_type_ids": (f == image_token_id).long()}
            if pixel_values is not None:
                fkw["pixel_values"] = kw["pixel_values"]
            p_full, _ = read(model(**fkw).logits[0, -1].float())
            chk = abs(p_full - probs[0])
    return probs, mass, texts, ntok, forced, L, chk


@app.function(gpu="L40S", image=image, timeout=3 * 3600,
              volumes={"/shards": shards, "/root/.cache/huggingface": hf_cache,
                       "/runs_vol": runs_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def score(jobs, modes, tag, part=0, attn="sdpa"):
    """Returns (rows, log text). Also writes log.txt and rows.jsonl to the knee-runs volume
    after EVERY study, so a crash or a stop keeps everything done so far."""
    import os
    import time

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    assert os.environ.get("HF_TOKEN"), "Modal secret `huggingface` has no HF_TOKEN"
    out_dir = f"{REMOTE_DIR}/{tag}/part{part}"
    os.makedirs(out_dir, exist_ok=True)
    logf = open(f"{out_dir}/log.txt", "a", encoding="utf-8")
    rowf = open(f"{out_dir}/rows.jsonl", "a", encoding="utf-8")
    lines = []

    def log(*a, **_):
        m = " ".join(str(x) for x in a)
        print(f"[w{part}] {m}", flush=True)
        lines.append(m)
        logf.write(m + "\n")
        logf.flush()

    log(f"remote results: knee-runs volume {out_dir.replace('/runs_vol/', '')}")
    t_load = time.time()
    log(f"attention: {attn}")
    log(f"GPU {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB; "
          f"{len(jobs)} studies x modes {modes} x {len(TASKS)} findings", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn).eval()
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
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")}
    log("yes tokens", [(i, tok.decode([i])) for i in yes_ids],
          "no tokens", [(i, tok.decode([i])) for i in no_ids], flush=True)

    def prepare(job):
        """CPU side of one study, every mode: slices, PIL images, prompt texts, token ids and
        the pixel tensor. Runs in a worker thread one study AHEAD of the GPU."""
        t = time.time()
        res = {}
        per_mode = {}
        try:
            with np.load(f"{SHARDS_DIR}/{job['uid']}.npz") as z:
                for m in modes:
                    try:
                        per_mode[m] = volume_slices(z, job, m)
                    except Exception as e:                     # noqa: BLE001
                        per_mode[m] = e
        except Exception as e:                                 # noqa: BLE001 (unreadable shard)
            per_mode = {m: e for m in modes}
        for m in modes:
            try:
                t1 = time.time()
                if isinstance(per_mode[m], Exception):
                    raise per_mode[m]
                groups = per_mode[m]
                imgs = [Image.fromarray(np.ascontiguousarray(x)).convert("RGB")
                        for _, sl in groups for x in sl]
                if prefill(m):
                    texts = [proc.apply_chat_template(
                        messages([(g, len(sl)) for g, sl in groups], prompt_text(job, m, q),
                                 pre=prefill(m)),
                        continue_final_message=True, tokenize=False) for q in TASKS]
                else:
                    # nothing pre-filled (reasoning): the standard generation prompt, which ends
                    # "<start_of_turn>model\n"; an empty continued turn would drop that newline
                    texts = [proc.apply_chat_template(
                        messages([(g, len(sl)) for g, sl in groups],
                                 prompt_text(job, m, q))[:1],
                        add_generation_prompt=True, tokenize=False) for q in TASKS]
                ids_list, pv = encode_prompts(proc, texts, imgs)
                if pv is not None:
                    pv = pv.to(torch.bfloat16).pin_memory()
                res[m] = {"groups": groups, "imgs": imgs, "texts": texts, "ids": ids_list,
                          "pv": pv, "prep_s": time.time() - t1}
            except Exception as e:                             # noqa: BLE001
                res[m] = {"error": f"{type(e).__name__}: {e}"}
        res["_load_s"] = time.time() - t
        return res

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    nxt = pool.submit(prepare, jobs[0]) if jobs else None
    out, shown = [], set()
    t0 = time.time()
    for k, job in enumerate(jobs):
        t_wait = time.time()
        prepared = nxt.result()
        t_wait = time.time() - t_wait                      # > 0 only if the CPU fell behind
        nxt = pool.submit(prepare, jobs[k + 1]) if k + 1 < len(jobs) else None
        for m in modes:
            row = {"uid": job["uid"], "mode": m, "side": job.get("side")}
            try:
                pm = prepared[m]
                if "error" in pm:
                    raise RuntimeError("prepare failed: " + pm["error"])
                t1 = time.time()
                groups, imgs, texts, ids_list, pv = (pm["groups"], pm["imgs"], pm["texts"],
                                                     pm["ids"], pm["pv"])
                first = m not in shown
                if first:
                    # the fast path must equal the processor itself, token for token and pixel
                    # for pixel; checked on the real processor once per mode
                    ref = proc(text=texts[0], images=imgs or None, add_special_tokens=False,
                               return_tensors="pt")
                    assert ref["input_ids"][0].tolist() == ids_list[0], "ids differ from processor"
                    if imgs:
                        assert torch.equal(ref["pixel_values"].to(torch.bfloat16), pv), \
                            "pixels differ from processor"
                        assert ref["token_type_ids"][0].tolist() == [
                            int(v == model.config.image_token_id) for v in ids_list[0]]
                for ids0 in ids_list:
                    assert ids0[0] == tok.bos_token_id and ids0[1] != tok.bos_token_id, "BOS"
                    if imgs:
                        n_it = sum(v == model.config.image_token_id for v in ids0)
                        assert n_it == len(imgs) * model.config.mm_tokens_per_image, n_it
                t_prep = pm["prep_s"]
                torch.cuda.synchronize()
                t2 = time.time()
                number = MODE_OPT.get(m, {}).get("answer") == "number"
                reason = MODE_OPT.get(m, {}).get("answer") == "reason"
                raws = None
                if reason:
                    probs, mass, raws, ntok, forced, L, rchk = reason_prob_shared(
                        model, ids_list, pv, model.config.image_token_id, yes_ids, no_ids,
                        stop_ids, tok, check=first)
                    diff = None
                elif number:
                    probs, raws, L, gchk = stated_prob_shared(
                        model, ids_list, pv, model.config.image_token_id, stop_ids, tok,
                        check=first)
                    mass, diff = [float("nan")], None
                else:
                    probs, mass, L, diff = yes_prob_shared(
                        model, ids_list, pv, model.config.image_token_id, yes_ids, no_ids,
                        pad_id=tok.pad_token_id or 0, check=first)
                torch.cuda.synchronize()
                t_gpu = time.time() - t2
                if first:
                    shown.add(m)
                    txt = tok.decode(ids_list[0][L:])
                    log(f"  [{m}] first study: {len(imgs)} images, {L} prefix tokens + "
                        f"{len(ids_list[0]) - L} question tokens; fast path == processor: OK; "
                        f"question part: {txt!r}", flush=True)
                    if reason:
                        log(f"  [{m}] REASONING CHECK shared-prefix decode vs one full pass: "
                            f"|P(yes) diff| {rchk:.4f} (should be < 0.02); first question "
                            f"reasoned {ntok[0]} tokens{' (answer forced)' if forced[0] else ''}: "
                            f"{raws[0][:400]!r}", flush=True)
                    elif number:
                        log(f"  [{m}] GENERATION CHECK shared-prefix greedy == model.generate: "
                            f"{'OK' if gchk[0] else 'DIFFERENT'} (generate said {gchk[1]!r}); "
                            f"replies: {[r for r in raws]}", flush=True)
                    else:
                        # bf16 logits of magnitude 32-64 are stored in steps of 0.25, so a max
                        # difference of 0.25-0.4 over 262k vocab entries is one rounding step
                        log(f"  [{m}] CACHE CHECK full pass vs shared prefix: max |logit diff| "
                            f"{diff[0]:.4f} (one bf16 step at this size is 0.25), "
                            f"|P(yes) diff| {diff[1]:.4f} (should be < 0.02)", flush=True)
                row.update({"n_img": len(imgs),
                            "slots": "+".join(f"{s}:{len(sl)}" for s, sl in groups),
                            "prefix_tok": L, "total_tok": len(ids_list[-1]),
                            "prep_s": t_prep, "gpu_s": t_gpu, "wait_s": t_wait,
                            "cache_diff": diff[0] if diff else None,
                            "cache_pdiff": diff[1] if diff else None, "yn_mass_min": min(mass),
                            "yn_mass_mean": float(np.mean(mass)),
                            "sec": time.time() - t1})
                row.update(dict(zip(TASKS, probs)))
                if raws is not None and reason:
                    row.update({"reason_" + t: r for t, r in zip(TASKS, raws)})
                    row["reason_tok_mean"] = float(np.mean(ntok))
                    row["forced"] = int(sum(forced))
                    if first:
                        row["reason_check"] = rchk
                elif raws is not None:
                    row.update({"raw_" + t: r for t, r in zip(TASKS, raws)})
                    row["unparsed"] = int(sum(v != v for v in probs))
                    if first:
                        row["gen_check"] = bool(gchk[0])
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
              f"ETA {eta / 60:.1f} min  peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB  "
              f"waited for CPU {t_wait:.1f}s (CPU prep overlapped: {prepared['_load_s']:.1f}s)",
              flush=True)
        for r in rs:
            if "error" in r:
                continue
            p = " ".join(f"{r[t]:.2f}" for t in TASKS)
            log(f"    {r['mode']:8s} {r['n_img']:3d} img {r['total_tok']:6d} tok "
                f"gpu {r['gpu_s']:5.1f}s (cpu prep {r['prep_s']:.1f}s, overlapped)  "
                f"yes/no mass min {r['yn_mass_min']:.2f}  P(yes): {p}",
                  flush=True)
        runs_vol.commit()                               # persist after every study
    pool.shutdown()
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
def main(modes: str = ",".join(MODES), limit: int = 0, workers: int = 4, gpu: str = "L40S",
         attn: str = "sdpa"):
    import contextlib
    import datetime
    import io as _io

    import pandas as pd
    jobs = build_jobs()
    if limit:
        jobs = jobs[:limit]
    ms = [m for m in modes.split(",") if m]
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_first{limit}" if limit else "")
    if gpu != "L40S" or attn != "sdpa":           # a speed benchmark: name the folder by setup
        tag += f"_{gpu.replace(':', '')}_{attn}"
    out = LOCAL_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": MODEL, "modes": ms, "studies": len(jobs), "max_slices": MAX_SLICES,
            "mode_def": {m: MODE_DEF[m] for m in ms}, "flip_side": FLIP_SIDE,
            "labels": LABEL, "remote_dir": f"knee-runs/labelling/medgemma15_audit58/{tag}",
            "jobs": [{k: v for k, v in j.items() if k != "report"} for j in jobs]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("local results folder:", out)

    # N containers at once: the GPU-hours (the cost) are the same, the wall time is ~1/N.
    # Each loads the model from the hf-cache volume (~50 s) and writes its own part<i>.
    workers = max(1, min(workers, len(jobs)))
    chunks = [jobs[i::workers] for i in range(workers)]
    print(f"{len(jobs)} studies over {workers} container(s): {[len(c) for c in chunks]}")
    rows, logs = [], []
    # --gpu and --attn change SPEED only: attn=flex_attention is checked equal to sdpa by
    # checks/medgemma_attn_backend_check.py, and the GPU type never changes the model
    fn = score.with_options(gpu=gpu) if gpu != "L40S" else score
    for r, lg in fn.starmap([(c, ms, tag, i, attn) for i, c in enumerate(chunks)]):
        rows += r
        logs.append(lg)
    log_text = "\n\n".join(logs)
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
        nan = np.isnan(preds[m])
        if nan.any():            # a stated probability MedGemma did not write as a number
            print(f"  {m}: {int(nan.sum())} of {nan.size} answers had no number -> scored as 0.5")
            preds[m] = np.where(nan, 0.5, preds[m])
    for m in ms:
        preds[f"qwen+{m} mean"] = 0.5 * preds["qwen24t-med"] + 0.5 * preds[m]
    fj = [TASKS.index(t) for t in FLUID_TASKS]
    for m in ms:
        p = preds["qwen24t-med"].copy()
        p[:, fj] = 0.5 * p[:, fj] + 0.5 * preds[m][:, fj]
        preds[f"qwen+{m} fluid3"] = p        # Qwen, blended ONLY on Effusion/Synovitis/Baker's

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
