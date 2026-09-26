"""
hulumed_audit58.py - can Hulu-Med (ZJU, Apache-2.0, 3D-capable medical VLM) label our knees?
Scores it on the 58 panel studies, one CSV per mode in the compare_labelers.py format
(data/03_report_labels_hulumed<size>-<mode>_validate58.csv).

Input path is Hulu-Med's OWN documented 3D route (model card, "3D Medical Image (NIfTI)"):
each series is written as a NIfTI file and passed as {"type": "3d", "3d": {"image_path",
"nii_num_slices", "nii_axis": 2}}. Its processor (processing_hulumed.HulumedProcessor.load_nii)
then picks nii_num_slices slices with np.linspace, min-max scales EACH SLICE to 0-255, copies
grey into R, G and B and feeds the stack as a video (merge_size 2). All videos of a prompt
share one visual budget of max_tokens = 16384, so more slices means smaller slices.

The NIfTI is written so that volume[:, :, i] is our slice i exactly as the shards store it
(radiological display: rows top to bottom, columns left to right). Mode suffix "T" stores
each slice TRANSPOSED instead, which is what a naive volume[:, :, i] of a scanner-oriented
NIfTI shows; we do not know which of the two the model saw in training, so both are scored.
Left knees are mirrored first with the engine's _mirror_slot (sagittal reverses slice order,
coronal and axial reverse width), as for MedGemma.

Score: P(yes) from the next-token logits right after "<|im_start|>assistant\\n", with the
question ending "Answer with yes or no." (no thinking: Hulu-Med only reasons when asked to).
The volumes come first and the question last, so each study's prefix is encoded ONCE and its
cache is reused for the 12 questions (checked against a full forward on the first study of
every mode). The first study also logs what the model actually writes when left to generate.

    python data_code/label_v2/hulumed_audit58.py --dry        # local: NIfTI round trip, no GPU
    python -m modal run data_code/label_v2/hulumed_audit58.py --limit 2
    python -m modal run data_code/label_v2/hulumed_audit58.py
"""
import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent.parent
SIZE = os.environ.get("HULUMED_SIZE", "14B")                 # 7B | 14B | 32B
MODEL = f"ZJU-AI4H/Hulu-Med-{SIZE}"
MTAG = f"hulumed{SIZE.lower().rstrip('b')}"
GPU = os.environ.get("HULUMED_GPU", "H100")
TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# the same condition wording MedGemma is asked about (medgemma_audit58.LABEL)
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
SLOTS_ALL = ["sag_plain", "sag_fluid", "cor_plain", "cor_fluid", "ax_plain", "ax_fluid"]
SLOT_NAME = {
    "sag_plain": "sagittal (side view), non-fat-suppressed",
    "sag_fluid": "sagittal (side view), fluid-sensitive fat-suppressed",
    "cor_plain": "coronal (front view), non-fat-suppressed",
    "cor_fluid": "coronal (front view), fluid-sensitive fat-suppressed",
    "ax_plain": "axial (cross-section), non-fat-suppressed",
    "ax_fluid": "axial (cross-section), fluid-sensitive fat-suppressed",
}
# mode -> (total slices over the series present, transpose each slice?, system prompt?).
# 180 = the model's config max_frames and the model card's example; 96 gives each slice
# ~1.8x the tokens. The card's own call, processor(conversation, add_system_prompt=True, ...),
# silently DROPS the system prompt (its kwargs merge keeps add_generation_prompt only; measured
# 25 Sep), so the default modes reproduce exactly that, and "S" restores the template's
# "You are Hulu-Med ..." system turn in case that is what it was trained with
# 4th field: which series; 5th: the report text goes in the prompt (before the question).
# "fluid" = the three fluid-sensitive series only: the 16,384-token visual budget is shared
# by every slice of the prompt, so dropping the plain series gives each slice ~2x the detail
SLOTS_FLUID = ["sag_fluid", "cor_fluid", "ax_fluid"]
MODE_DEF = {"all180": (180, False, False, SLOTS_ALL, False),
            "all180T": (180, True, False, SLOTS_ALL, False),
            "all96": (96, False, False, SLOTS_ALL, False),
            "all96S": (96, False, True, SLOTS_ALL, False),
            "fluid": (180, False, False, SLOTS_FLUID, False),
            "all96R": (96, False, False, SLOTS_ALL, True)}
MODES = list(MODE_DEF)
FLIP_SIDE = "left"
QUESTION = "Is there {lab} in this knee MRI? Answer with yes or no."

app = modal.App(f"{MTAG}-audit58")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("torch==2.6.0", "torchvision==0.21.0", "transformers==4.51.2",
                      "accelerate==1.7.0", "numpy<2", "pandas", "pillow", "nibabel",
                      "decord", "ffmpeg-python", "imageio", "opencv-python-headless",
                      "huggingface_hub",
                      # the model card pins flash-attn 2.7.3; its SDPA fallback builds ONE dense
                      # mask over every patch of every slice (169 GiB at 96 slices, 25 Sep)
                      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/"
                      "flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
         .add_local_file(ROOT / "runs/shared_code.py", "/root/shared_code.py"))
shards = modal.Volume.from_name("knee-shards")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("knee-runs")
REMOTE_ROOT = "/runs_vol/labelling"
SUB = f"{MTAG}_audit58"
LOCAL_ROOT = ROOT / "data_code/label_v2/results"


def _mg():
    sys.path.insert(0, str(ROOT / "data_code/label_v2"))
    import medgemma_audit58 as mg
    return mg


def _mirror(vol, slot):
    for p in ("/root", str(ROOT / "runs")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import shared_code as sc
    return sc._mirror_slot(vol, slot)


def write_series(z, job, mode, tmp):
    """One NIfTI per real slot -> [(slot, path, n_slices_to_take)], left knees mirrored."""
    import nibabel as nib
    import numpy as np
    total, transpose, _, slots, _ = MODE_DEF[mode]
    have = [s for s in slots if job["slots"].get(s) in z.files]
    per = total // max(len(have), 1)
    out = []
    for s in have:
        vol = z[job["slots"][s]]                                   # (S, H, W) uint8
        if job.get("side") == FLIP_SIDE:
            vol = _mirror(vol, s)
        data = vol.transpose(2, 1, 0) if transpose else vol.transpose(1, 2, 0)
        # the MODE is in the name: all modes are written before any is scored, and a shared
        # name let the last mode overwrite the others (all180T scored untransposed, 25 Sep)
        p = f"{tmp}/{job['uid'][-12:]}_{mode}_{s}.nii.gz"
        nib.save(nib.Nifti1Image(np.ascontiguousarray(data).astype(np.float32), np.eye(4)), p)
        out.append((s, p, min(per, vol.shape[0])))
    return out


def conversation(series, question, report=None):
    content = []
    for s, p, k in series:
        content.append({"type": "text", "text": f"KNEE MRI SERIES: {SLOT_NAME[s]}, {k} slices.\n"})
        content.append({"type": "3d", "3d": {"image_path": p, "nii_num_slices": k, "nii_axis": 2}})
    if report:              # before the question, so it stays in the shared prefix
        content.append({"type": "text", "text": f"RADIOLOGY REPORT: {report}\n"})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


@app.function(gpu=GPU, image=image, timeout=3 * 3600, cpu=8, memory=65536,
              volumes={"/shards": shards, "/root/.cache/huggingface": hf_cache,
                       "/runs_vol": runs_vol},
              secrets=[modal.Secret.from_dict({"HULUMED_SIZE": SIZE})])  # containers lack the local env
def score(jobs, modes, tag, part=0, expect_model=None):
    import copy
    import tempfile
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    assert expect_model in (None, MODEL), f"container has {MODEL}, caller asked for {expect_model}"
    out_dir = f"{REMOTE_ROOT}/{SUB}/{tag}/part{part}"
    os.makedirs(out_dir, exist_ok=True)
    logf = open(f"{out_dir}/log.txt", "a", encoding="utf-8")
    rowf = open(f"{out_dir}/rows.jsonl", "a", encoding="utf-8")
    lines = []

    def log(m):
        print(f"[w{part}] {m}", flush=True)
        lines.append(m)
        logf.write(m + "\n")
        logf.flush()

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2").eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    hf_cache.commit()
    tok = proc.tokenizer
    venc = model.get_model().get_vision_encoder()
    assert venc.config._attn_implementation == "flash_attention_2", venc.config._attn_implementation
    # the 12 questions run as ONE right-padded batch on the shared cache, which Qwen3's
    # flash-attention path refuses; the language model reads its backend from its own config at
    # every forward, so it moves to SDPA while the vision encoder (own config) stays on flash
    model.config._attn_implementation = "sdpa"
    assert venc.config._attn_implementation == "flash_attention_2"
    log(f"loaded {MODEL} in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}; "
        f"LLM attn {model.config._attn_implementation}, vision attn "
        f"{venc.config._attn_implementation}; token compression "
        f"{model.config.use_token_compression}; image token {model.config.image_token_index}; "
        f"GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def first_ids(words):
        ids = set()
        for w in words:
            e = tok.encode(w, add_special_tokens=False)
            assert len(e) == 1, (w, e)
            ids.add(e[0])
        return sorted(ids)
    yes_ids = first_ids(["Yes", "yes", " Yes", " yes"])
    no_ids = first_ids(["No", "no", " No", " no"])
    log(f"yes tokens {[(i, tok.decode([i])) for i in yes_ids]}, "
        f"no tokens {[(i, tok.decode([i])) for i in no_ids]}")
    dev = torch.device("cuda")
    tmp = tempfile.mkdtemp()
    out, shown = [], set()
    t_run = time.time()
    for k, job in enumerate(jobs):
        with np.load(f"/shards/{job['uid']}.npz") as z:
            series_by_mode = {m: write_series(z, job, m, tmp) for m in modes}
        for m in modes:
            row = {"uid": job["uid"], "mode": m, "side": job.get("side")}
            try:
                t1 = time.time()
                series = series_by_mode[m]
                rep = job.get("report") if MODE_DEF[m][4] else None
                assert (rep is not None and len(rep) > 20) == MODE_DEF[m][4], "report missing"
                convs = [conversation(series, QUESTION.format(lab=LABEL[t]), rep) for t in TASKS]
                # exactly _process_conversation, with the volumes loaded and resized ONCE
                loaded = proc._load_multimodal_data(convs[0])
                imgs = proc._gather_multimodal_data(loaded)
                img_in = proc.process_images(imgs, merge_size=2, return_tensors="pt")
                ids_list = []
                for c in convs:
                    lc = copy.deepcopy(loaded)
                    lc[0]["content"][-1] = c[0]["content"][-1]
                    text = proc.apply_chat_template(lc, tokenize=False,
                                                    add_system_prompt=MODE_DEF[m][2],
                                                    add_generation_prompt=True)
                    ids_list.append(proc.process_text(text, img_in)["input_ids"])
                pv = img_in["pixel_values"].to(dev, torch.bfloat16)
                gs, ms = img_in["grid_sizes"].to(dev), img_in["merge_sizes"].to(dev)
                modals = img_in["modals"]
                first = m not in shown
                if first:          # the fast path must equal the processor itself
                    ref = proc(conversation=convs[0], add_system_prompt=True,
                               add_generation_prompt=True, return_tensors="pt")
                    r0 = ref["input_ids"][0].tolist()
                    k0 = len(ids_list[0]) - len(r0)          # the system turn, "S" modes only
                    assert ids_list[0][k0:] == r0, "ids differ from processor"
                    assert (k0 > 0) == MODE_DEF[m][2], f"system prompt tokens {k0}"
                    if k0:
                        assert tok.decode(ids_list[0][:k0]).startswith(
                            "<|im_start|>system\nYou are Hulu-Med")
                    assert torch.equal(ref["pixel_values"].to(dev, torch.bfloat16), pv), "pixels"
                    assert ref["grid_sizes"].tolist() == img_in["grid_sizes"].tolist()
                    assert list(ref["modals"]) == list(modals)
                n_vis = sum(v == model.config.image_token_index for v in ids_list[0])
                L = next(i for i in range(min(map(len, ids_list)))
                         if any(x[i] != ids_list[0][i] for x in ids_list)) - 1
                with torch.inference_mode():
                    full0 = torch.tensor([ids_list[0]], device=dev)
                    _, _, _, _, emb, _ = model.prepare_inputs_labels_for_multimodal(
                        input_ids=full0, attention_mask=torch.ones_like(full0),
                        pixel_values=pv, grid_sizes=gs, merge_sizes=ms, modals=modals)
                    pre = model.get_model()(inputs_embeds=emb[:, :L], use_cache=True)
                    cache = pre.past_key_values
                    suf = [x[L:] for x in ids_list]
                    B, n = len(suf), max(len(x) for x in suf)
                    lens = torch.tensor([len(x) for x in suf], device=dev)
                    s_ids = torch.full((B, n), tok.pad_token_id or 0, dtype=torch.long, device=dev)
                    s_mask = torch.zeros((B, L + n), dtype=torch.long, device=dev)
                    s_mask[:, :L] = 1
                    for i, x in enumerate(suf):
                        s_ids[i, :len(x)] = torch.tensor(x, device=dev)
                        s_mask[i, L:L + len(x)] = 1
                    cache.batch_repeat_interleave(B)
                    h = model.get_model()(input_ids=s_ids, attention_mask=s_mask,
                                          past_key_values=cache, use_cache=True,
                                          cache_position=torch.arange(L, L + n, device=dev),
                                          position_ids=torch.arange(L, L + n, device=dev)
                                          .unsqueeze(0).expand(B, -1)).last_hidden_state
                    lg = model.lm_head(h[torch.arange(B, device=dev), lens - 1]).float()
                    del cache, pre
                    sm = torch.softmax(lg, -1)
                    p_yes = torch.sigmoid(torch.logsumexp(lg[:, yes_ids], -1)
                                          - torch.logsumexp(lg[:, no_ids], -1))
                    mass = sm[:, yes_ids].sum(-1) + sm[:, no_ids].sum(-1)
                    if first:
                        hf = model.get_model()(inputs_embeds=emb).last_hidden_state[:, -1]
                        lf = model.lm_head(hf).float()[0]
                        pf = torch.sigmoid(torch.logsumexp(lf[yes_ids], 0)
                                           - torch.logsumexp(lf[no_ids], 0))
                        cdiff = float((lf - lg[0]).abs().max())
                        pdiff = float((pf - p_yes[0]).abs())
                        g = model.generate(input_ids=full0, attention_mask=torch.ones_like(full0),
                                           pixel_values=pv, grid_sizes=gs, merge_sizes=ms,
                                           modals=modals, do_sample=False, max_new_tokens=40,
                                           pad_token_id=tok.eos_token_id)
                        said = tok.decode(g[0], skip_special_tokens=False)
                torch.cuda.synchronize()
                if first:
                    shown.add(m)
                    log(f"  [{m}] first study: {len(series)} series "
                        f"{[(s, kk) for s, _, kk in series]}, grid sizes "
                        f"{img_in['grid_sizes'].tolist()[:2]}..., {n_vis} visual tokens, "
                        f"{len(ids_list[0])} tokens, prefix {L}; fast path == processor: OK; "
                        f"CACHE CHECK max |logit diff| {cdiff:.4f}, |P(yes) diff| {pdiff:.4f}; "
                        f"question part {tok.decode(ids_list[0][L:])!r}; left to generate "
                        f"it says {said!r}")
                row.update({"n_series": len(series), "n_vis": n_vis, "total_tok": len(ids_list[0]),
                            "yn_mass_min": float(mass.min()), "yn_mass_mean": float(mass.mean()),
                            "sec": time.time() - t1,
                            "peak_gb": torch.cuda.max_memory_allocated() / 1e9})
                row.update(dict(zip(TASKS, p_yes.tolist())))
            except Exception as e:                                  # noqa: BLE001
                import traceback
                log(f"  ERROR {job['uid']} {m}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                row["error"] = f"{type(e).__name__}: {e}"
            out.append(row)
            rowf.write(json.dumps(row) + "\n")
            rowf.flush()
        el = time.time() - t_run
        rs = [r for r in out if r["uid"] == job["uid"] and "error" not in r]
        log(f"{k + 1}/{len(jobs)} {job['uid'][-8:]} {job.get('side')} {el:.0f}s, "
            f"ETA {el / (k + 1) * (len(jobs) - k - 1) / 60:.1f} min; " + "; ".join(
                f"{r['mode']} {r['n_vis']} vis tok {r['sec']:.1f}s mass {r['yn_mass_min']:.2f} "
                f"P {' '.join(f'{r[t]:.2f}' for t in TASKS)}" for r in rs))
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        runs_vol.commit()
    logf.close()
    rowf.close()
    runs_vol.commit()
    return out, "\n".join(lines)


@app.function(image=image, cpu=4, memory=16384, timeout=1800,
              volumes={"/shards": shards, "/root/.cache/huggingface": hf_cache})
def debug_ids(job, mode="all96"):
    """CPU only: the fast path vs the full processor for one study, and where they differ."""
    import copy
    import tempfile

    import numpy as np
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    tmp = tempfile.mkdtemp()
    with np.load(f"/shards/{job['uid']}.npz") as z:
        series = write_series(z, job, mode, tmp)
    conv = conversation(series, QUESTION.format(lab=LABEL["ACL"]))
    ref = proc(conversation=conv, add_system_prompt=True, add_generation_prompt=True,
               return_tensors="pt")
    loaded = proc._load_multimodal_data(conv)
    img_in = proc.process_images(proc._gather_multimodal_data(loaded), merge_size=2,
                                 return_tensors="pt")
    text = proc.apply_chat_template(copy.deepcopy(loaded), tokenize=False, add_system_prompt=True,
                                    add_generation_prompt=True)
    ids = proc.process_text(text, img_in)["input_ids"]
    r = ref["input_ids"][0].tolist()
    print("ref", len(r), "fast", len(ids), "grid ref", ref["grid_sizes"].tolist()[:3],
          "fast", img_in["grid_sizes"].tolist()[:3], "merge", ref["merge_sizes"].tolist()[:3],
          img_in["merge_sizes"].tolist()[:3], "modals", list(ref["modals"])[:3], img_in["modals"][:3])
    i = next((k for k in range(min(len(r), len(ids))) if r[k] != ids[k]), None)
    tok = proc.tokenizer
    if i is not None:
        print("first diff at", i, "ref:", repr(tok.decode(r[max(0, i - 5):i + 10])),
              "fast:", repr(tok.decode(ids[max(0, i - 5):i + 10])))
    print("text head:", repr(text[:300]))
    print("ref head:", repr(tok.decode(r[:40])))
    print("ref tail:", repr(tok.decode(r[-40:])))
    print("fast tail:", repr(tok.decode(ids[-40:])))
    ref2 = proc(conversation=conv, return_tensors="pt", add_generation_prompt=True)
    print("ref, generation prompt only, same as ref:", ref2["input_ids"][0].tolist() == r)
    ref3 = proc(conversation=conv, return_tensors="pt")
    print("ref, no flags, tail:", repr(tok.decode(ref3["input_ids"][0].tolist()[-12:])))


def dry_run():
    """Local: the NIfTI written for a real shard, read back through Hulu-Med's own load_nii
    logic (copied verbatim below), must give our slices back, min-max scaled per slice."""
    import tempfile

    import nibabel as nib
    import numpy as np
    mg = _mg()
    local = [p.stem for p in (ROOT / "data/shards").glob("*.npz")]
    jobs = mg.build_jobs(local, reports=False)
    tmp = tempfile.mkdtemp()
    bad = 0
    for job in jobs[:6]:
        with np.load(ROOT / f"data/shards/{job['uid']}.npz") as z:
            # written for EVERY mode first, then read back, exactly as score() does it
            written = {m: write_series(z, job, m, tmp) for m in MODES}
            for m in MODES:
                series = written[m]
                for s, p, k in series:
                    vol = z[job["slots"][s]]
                    if job.get("side") == FLIP_SIDE:
                        vol = _mirror(vol, s)
                    volume = nib.load(p).get_fdata()
                    sl = [volume[:, :, i] for i in range(volume.shape[2])]
                    if k < len(sl):
                        idx = np.linspace(0, len(sl) - 1, k, dtype=int)
                        sl = [sl[i] for i in idx]
                    else:
                        idx = np.arange(len(sl))
                    for i, x in zip(idx, sl):
                        ref = vol[i].T if MODE_DEF[m][1] else vol[i]
                        lo, hi = ref.min(), ref.max()
                        want = ((ref - lo) / (hi - lo) * 255.0).astype(np.uint8) if hi > lo \
                            else np.zeros_like(ref)
                        mn, mx = x.min(), x.max()
                        got = ((x - mn) / (mx - mn) * 255.0).astype(np.uint8) if mx > mn \
                            else np.zeros_like(x).astype(np.uint8)
                        bad += not np.array_equal(got, want)
                print(f"  {job['uid'][-8:]} {job['side']:5s} {m:8s} "
                      + " ".join(f"{s}:{k}/{z[job['slots'][s]].shape[0]}" for s, _, k in series))
    print("NIfTI round trip:", "ALL OK" if not bad else f"{bad} slices differ")
    print("prompt:", json.dumps(conversation([("sag_fluid", "x.nii.gz", 30)],
                                             QUESTION.format(lab=LABEL["Effusion"])))[:400])


@app.local_entrypoint()
def main(modes: str = ",".join(MODES), limit: int = 0, workers: int = 2, tag: str = ""):
    import datetime

    import pandas as pd
    mg = _mg()
    ms = [m for m in modes.split(",") if m]
    jobs = mg.build_jobs(reports=any(MODE_DEF[m][4] for m in ms))
    if not any(MODE_DEF[m][4] for m in ms):
        for j in jobs:
            j.pop("report", None)
    if limit:
        jobs = jobs[:limit]
    tag = tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_first{limit}" if limit else "")
    out = LOCAL_ROOT / SUB / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(
        {"model": MODEL, "modes": {m: MODE_DEF[m] for m in ms}, "question": QUESTION,
         "labels": LABEL, "gpu": GPU, "studies": len(jobs),
         "jobs": [{k: v for k, v in j.items() if k != "report"} for j in jobs]}, indent=1),
        encoding="utf-8")
    workers = max(1, min(workers, len(jobs)))
    chunks = [jobs[i::workers] for i in range(workers)]
    print(f"tag {tag}; {len(jobs)} studies over {workers} container(s)")
    rows, logs = [], []
    for r, lg in score.starmap([(c, ms, tag, i, MODEL) for i, c in enumerate(chunks)]):
        rows += r
        logs.append(lg)
    (out / "log.txt").write_text("\n\n".join(logs) + "\n", encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw.csv", index=False)
    if "error" in df:
        print("ERRORS:", int(df["error"].notna().sum()))
    ok = df[df["error"].isna()] if "error" in df else df
    if not all(t in ok for t in TASKS) or ok.empty:
        print("no successful rows; nothing to score")
        return
    for m in ms:
        d = ok[ok["mode"] == m]
        d = d.rename(columns={"uid": "StudyInstanceUID"})
        d.insert(1, "unusable", False)
        d = d[["StudyInstanceUID", "unusable"] + TASKS]
        d.to_csv(out / f"labels_{m}.csv", index=False)
        if not limit:
            f = ROOT / f"data/03_report_labels_{MTAG}-{m}_validate58.csv"
            d.to_csv(f, index=False)
            print("wrote", f.relative_to(ROOT))
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mg.scoreboard(df.drop(columns=[c for c in ("error",) if c in df and df[c].isna().all()]), ms)
    print(buf.getvalue())
    (out / "scoreboard.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__" and "--dry" in sys.argv:
    dry_run()
