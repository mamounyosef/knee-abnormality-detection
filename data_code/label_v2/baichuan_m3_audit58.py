"""
baichuan_m3_audit58.py - can Baichuan-M3-235B (Baichuan AI, Apache-2.0, medical LLM on
Qwen3-235B-A22B) label our REPORTS better than Qwen? Scores it on the 58 panel studies.

Same task as the Qwen labeller that measured 0.9114: data_code/03_label_reports.py's
SYSTEM_SOFT prompt (v1) plus the OUTPUT FORMAT block that script appends for every endpoint
that does not constrain decoding, the translated report as the user turn, and the same
tolerant parser (_extract_json + _coerce). M3's chat template ALWAYS opens the answer with
"<think>\\n", so it reasons first; the JSON is read after the last "</think>".

Weights: baichuan-inc/Baichuan-M3-235B-FP8 (block-FP8, compressed-tensors, 236 GB), served
with vLLM tensor-parallel over 4 H100 (the model card deploys the FP8 build with --tp 4).
Sampling is the model's own generation_config (temperature 0.6, top_p 0.95, top_k 20).
Two samples per report: sample 1 alone is the single-pass score, and their mean is kept too.

    python -m modal run data_code/label_v2/baichuan_m3_audit58.py::download   # CPU, once
    python -m modal run data_code/label_v2/baichuan_m3_audit58.py --limit 3
    python -m modal run data_code/label_v2/baichuan_m3_audit58.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent.parent
REPO = "baichuan-inc/Baichuan-M3-235B-FP8"
MTAG = "baichuanm3"
GPU = os.environ.get("M3_GPU", "H100:4")
TP = int(GPU.split(":")[1]) if ":" in GPU else 1
SAMPLES = 2
MAX_NEW = 16384
MAX_LEN = 24576

app = modal.App(f"{MTAG}-audit58")
# vLLM's own release image: it ships the CUDA compiler and FlashInfer's JIT cache, which the
# FP8 MoE path compiles against at start-up (a pip install on debian_slim died there with
# "Could not find nvcc", 25 Sep)
image = (modal.Image.from_registry("vllm/vllm-openai:v0.29.0")
         .entrypoint([])
         .run_commands("ln -sf \"$(command -v python3)\" /usr/local/bin/python",   # it has python3 only
                       "python -c 'import vllm, sys; print(vllm.__version__, sys.version)'")
         .pip_install("huggingface_hub[hf_xet]", "pandas")
         .env({"HF_XET_HIGH_PERFORMANCE": "1"}))
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("knee-runs")
REMOTE_ROOT = "/runs_vol/labelling"
SUB = f"{MTAG}_audit58"
LOCAL_ROOT = ROOT / "data_code/label_v2/results"


def _labeller():
    """data_code/03_label_reports.py as a module (its name starts with a digit)."""
    sys.path.insert(0, str(ROOT / "data_code"))
    spec = importlib.util.spec_from_file_location("label_reports", ROOT / "data_code/03_label_reports.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def system_prompt(lr):
    """SYSTEM_SOFT + the OUTPUT FORMAT block, character for character as 03_label_reports.py
    builds it for an endpoint without constrained decoding (soft mode)."""
    ex = '{"unusable": false, "findings": {"ACL": {"p": 0.05, "evidence": "..."}, ...}}'
    return lr.SYSTEM_SOFT + (
        "\n\nOUTPUT FORMAT - follow exactly, no exceptions:\n"
        "Reply with ONE raw JSON object and nothing else. No prose, no explanation,\n"
        "no markdown, no ``` fences, no table. The object must be:\n\n"
        + ex + "\n\nOne entry per finding, using EXACTLY these 12 keys and no "
        "others:\n" + ", ".join(lr.LABELS) +
        "\nEach entry has exactly two fields: 'p' (a number 0.0-1.0) and 'evidence' "
        "(the exact sentence you relied on, or an empty string).")


@app.function(image=image, cpu=8, memory=16384, timeout=3 * 3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def download():
    import time

    from huggingface_hub import snapshot_download
    t = time.time()
    p = snapshot_download(REPO, ignore_patterns=["draft/*"])
    hf_cache.commit()
    n = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    print(f"{REPO} -> {p}: {n / 1e9:.1f} GB in {time.time() - t:.0f}s")


@app.function(gpu=GPU, image=image, timeout=2 * 3600, cpu=16, memory=131072,
              volumes={"/root/.cache/huggingface": hf_cache, "/runs_vol": runs_vol})
def score(items, tag):
    """items = [(uid, messages)]. Returns rows {uid, sample, text, n_gen, finish} and a log."""
    import time

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    out_dir = f"{REMOTE_ROOT}/{SUB}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    t0 = time.time()
    path = snapshot_download(REPO, ignore_patterns=["draft/*"], local_files_only=True)
    llm = LLM(model=path, tensor_parallel_size=TP, max_model_len=MAX_LEN,
              gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    log(f"loaded {REPO} on {GPU} in {time.time() - t0:.0f}s")
    prompts = [tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
               for _, msgs in items]
    assert all(p.endswith("<|im_start|>assistant\n<think>\n") for p in prompts), prompts[0][-80:]
    log(f"prompt tokens: min {min(len(tok.encode(p)) for p in prompts)}, "
        f"max {max(len(tok.encode(p)) for p in prompts)}; prompt tail {prompts[0][-60:]!r}")
    sp = SamplingParams(n=SAMPLES, temperature=0.6, top_p=0.95, top_k=20, max_tokens=MAX_NEW,
                        seed=0)
    t1 = time.time()
    res = llm.generate(prompts, sp)
    el = time.time() - t1
    rows = []
    for (uid, _), r in zip(items, res):
        for k, o in enumerate(r.outputs):
            rows.append({"uid": uid, "sample": k, "text": o.text, "n_gen": len(o.token_ids),
                         "finish": o.finish_reason})
    ng = [x["n_gen"] for x in rows]
    log(f"generated {len(rows)} answers in {el:.0f}s: {sum(ng)} tokens "
        f"({sum(ng) / el:.0f} tok/s), mean {sum(ng) / len(ng):.0f}, max {max(ng)}; "
        f"finish reasons {sorted({x['finish'] for x in rows})}, "
        f"hit max_tokens {sum(x['finish'] == 'length' for x in rows)}")
    with open(f"{out_dir}/rows.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    runs_vol.commit()
    return rows, "\n".join(lines)


def parse(lr, text):
    """The JSON after the last </think> -> 12 probabilities, via 03_label_reports' parser."""
    if "</think>" not in text:
        raise ValueError("no </think>: the reasoning never finished")
    ans = text.rsplit("</think>", 1)[1]
    return lr._coerce(lr._extract_json(ans))


@app.local_entrypoint()
def main(limit: int = 0, tag: str = ""):
    import datetime

    import numpy as np
    import pandas as pd
    lr = _labeller()
    sysmsg = system_prompt(lr)
    aud = pd.read_csv(ROOT / "data/05_audit58.csv")
    rep = pd.read_csv(ROOT / "data/02_reports_en.csv").set_index("StudyInstanceUID")["Report_en"]
    uids = list(aud.StudyInstanceUID)[:limit or None]
    items = [(u, [{"role": "system", "content": sysmsg},
                  {"role": "user", "content": f"KNEE MRI REPORT:\n\n{str(rep[u]).strip()}"}])
             for u in uids]
    tag = tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_first{limit}" if limit else "")
    out = LOCAL_ROOT / SUB / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"repo": REPO, "gpu": GPU, "samples": SAMPLES,
                                               "max_new": MAX_NEW, "system": sysmsg,
                                               "uids": uids}, indent=1), encoding="utf-8")
    rows, log = score.remote(items, tag)
    (out / "log.txt").write_text(log + "\n", encoding="utf-8")
    print(log)
    recs, bad = [], []
    for x in rows:
        try:
            f = parse(lr, x["text"])
            recs.append({"uid": x["uid"], "sample": x["sample"], "unusable": f["unusable"],
                         **{k: v["p"] for k, v in f["findings"].items()}})
        except Exception as e:                                      # noqa: BLE001
            bad.append((x["uid"][-8:], x["sample"], x["finish"], f"{type(e).__name__}: {e}"))
    (out / "raw.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    print(f"parsed {len(recs)} of {len(rows)} answers; failures: {bad}")
    df = pd.DataFrame(recs)
    s1 = df[df["sample"] == 0].drop(columns="sample")
    avg = df.groupby("uid")[lr.LABELS].mean().reset_index()
    avg.insert(1, "unusable", False)
    for name, d in (("s1", s1), ("avg2", avg)):
        d = d.rename(columns={"uid": "StudyInstanceUID"})[["StudyInstanceUID", "unusable"] + lr.LABELS]
        d.to_csv(out / f"labels_{name}.csv", index=False)
        if not limit:
            f = ROOT / f"data/03_report_labels_{MTAG}-{name}_validate58.csv"
            d.to_csv(f, index=False)
            print("wrote", f.relative_to(ROOT))
    if len(s1) >= 20:
        sys.path.insert(0, str(ROOT / "data_code/label_v2"))
        import medgemma_audit58 as mg
        long = pd.concat([s1.assign(mode="s1"), avg.assign(mode="avg2")])
        mg.scoreboard(long, ["s1", "avg2"])
        both = df.pivot_table(index="uid", columns="sample", values=lr.LABELS)
        dis = np.mean([abs(both[(t, 0)] - both[(t, 1)]).mean() for t in lr.LABELS])
        print(f"self-consistency: mean |sample 1 - sample 2| per cell {dis:.3f}")
