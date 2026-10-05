"""infer_speed_bench.py - how fast can each ensemble member run on THIS GPU, and does the output change?

Variants per member, on a fixed random uint8 study of the member's own shape (prepare_batch included, like the notebook):
  amp           today: fp32 weights, torch.autocast fp16, channels_last, cudnn.benchmark
  half          weights AND input cast to fp16 (no autocast)
  amp+halfw     fp16 weights under autocast (the engine's KNEE_INFER_HALF=1 path)
  amp+compile   torch.compile(model) under autocast (needs triton)
  half+compile
Every (member, variant) runs in its OWN process: the caching allocator keeps the previous variant's blocks, which made
the heavy members run out of memory on a 15 GB T4 when variants shared one process. Reports s/study, first-call seconds
(compile time) and max |logit diff| vs amp.

    python runs/others/ensembling/infer_speed_bench.py <bundle weights dir> [member indices, e.g. 0,1,3,7] [variants]
Writes speed_bench.json in the working directory.
"""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = ["amp", "half", "amp+halfw", "amp+compile", "half+compile"]
REPS = 8


def child(W, i, variant):
    import torch
    sys.path.insert(0, str(W))
    import shared_code as sc
    dev = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    spec = importlib.util.spec_from_file_location("m%d_def" % i, W / ("m%d_def.py" % i))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m%d_def" % i] = mod
    spec.loader.exec_module(mod)
    cfg = json.loads((W / ("m%d_config.json" % i)).read_text())
    m = mod.build_model(cfg)
    sd = torch.load(W / ("m%d_model.pt" % i), map_location="cpu", weights_only=False)["model"]
    m.load_state_dict({k.replace("_orig_mod.", "", 1): v for k, v in sd.items()})
    m = m.eval().to(dev).to(memory_format=torch.channels_last)
    inp = cfg["input"]
    if isinstance(inp["n_slices"], dict):
        return {"skipped": "per-slot slice counts"}
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (1, len(inp["slots"]), inp["n_slices"], inp["size"], inp["size"]), dtype=torch.uint8, generator=g)
    half = variant.startswith("half") or variant == "amp+halfw"
    if half:
        m = m.half()
    if "compile" in variant:
        m = torch.compile(m)
    use_amp = variant.startswith("amp")

    def fn():
        xb = sc.prepare_batch(x, cfg, dev)
        if variant.startswith("half"):
            xb = xb.half()
        with torch.autocast("cuda", enabled=use_amp):
            return m(xb)
    with torch.inference_mode():
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); first = time.perf_counter() - t0
        fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            out = fn()
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / REPS
    return {"s": round(t, 4), "first_call_s": round(first, 1), "logits": out.float().cpu().tolist(),
            "peak_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)}


if __name__ == "__main__":
    W = Path(sys.argv[1])
    if len(sys.argv) > 4 and sys.argv[4] == "--child":
        try:
            r = child(W, int(sys.argv[2]), sys.argv[3])
        except Exception as e:  # noqa: BLE001
            r = {"err": f"{type(e).__name__}: {str(e)[:200]}"}
        print("RESULT " + json.dumps(r), flush=True)
        sys.exit(0)
    members = json.loads((W / "members.json").read_text())
    idx = [int(i) for i in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] else [m["i"] for m in members]
    variants = sys.argv[3].split(",") if len(sys.argv) > 3 else VARIANTS
    rows = []
    for mm in members:
        if mm["i"] not in idx:
            continue
        row, ref = {"i": mm["i"], "name": mm["name"]}, None
        for v in variants:
            p = subprocess.run([sys.executable, __file__, str(W), str(mm["i"]), v, "--child"], capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            r = json.loads(line[-1][7:]) if line else {"err": (p.stderr or "no output")[-200:]}
            if "logits" in r:
                import numpy as np
                lg = np.array(r.pop("logits"))
                if v == "amp":
                    ref = lg
                elif ref is not None:
                    r["maxdiff"] = float(np.abs(lg - ref).max())
                    r["speedup"] = round(row["amp"]["s"] / r["s"], 2)
            row[v] = r
        print(json.dumps(row), flush=True)
        rows.append(row)
    import torch
    Path("speed_bench.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                                     "rows": rows}, indent=1))
