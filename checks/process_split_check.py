"""process_split_check.py - the second card gets its OWN interpreter, and returns the same numbers.

WHY PROCESSES. The threaded member split was correct and measured twice as slow PER CARD:
cuda:0 went 7.751 -> 16.618 s per study on its own members and cuda:1 8.624 -> 19.536. Two
threads each at half speed is zero throughput gained, which is the signature of the Global
Interpreter Lock (only one Python thread runs Python at a time). Issuing a CUDA kernel is
Python work that holds that lock, and our members are many small kernels over 96 images per
study, so the second thread could only issue while the first was not. A child PROCESS has
its own interpreter and its own lock.

What must hold, and what would break silently if it did not:
  - a child rebuilds a member from the bundle's own files and gets BIT-IDENTICAL logits to
    the parent running the same member. If it did not, half the ensemble would be scored by
    a subtly different model and nothing would say so.
  - the BUNDLE index (which names m<i>_def.py) is not the LOCAL index (the member's slot in
    a geometry group's stack). predict_dicom_ensemble_grouped renumbers each group 0..k-1,
    so conflating the two loads one member's weights into another member's row.
  - a child that fails must RAISE in the parent. A worker that dies quietly leaves zero
    logits, which sigmoid to exactly 0.5, and the run still announces "9/9 members".
  - the workers are cached across chunks and groups. Rebuilding four members per chunk
    would cost far more than the split saves.
  - shutdown actually stops them: each child holds a card and its own copy of its weights.

Run:  python checks/process_split_check.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

PASS, FAIL = [], []


def ck(n, ok, d=""):
    (PASS if ok else FAIL).append(n)
    print("  %s %-58s %s" % ("PASS" if ok else "FAIL", n, d))


# A member definition shaped exactly like a real one: build_model(cfg) -> nn.Module that
# eats prepare_batch's (B*slots*D, C, H, W) and pools back to one row per study.

MODEL_DEF = '''
import torch, torch.nn as nn


class SliceModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_img = len(cfg["input"]["slots"]) * cfg["input"]["n_slices"]
        self.body = nn.Sequential(nn.Conv2d(cfg["input"]["channels"], 4, 3, padding=1),
                                  nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(4, len(cfg["tasks"]))

    def forward(self, x):
        z = self.body(x)                       # (B*imgs, 4)
        z = z.view(-1, self.n_img, 4).mean(1)  # pool within a study, never across
        return self.head(z)


def build_model(cfg):
    return SliceModel(cfg)
'''

# multiprocessing uses SPAWN, and a spawned child re-imports its parent's __main__. Without
# this guard the child re-runs the whole check and spawns children of its own, which is
# exactly what happened the first time this was run: it hung instead of failing. A Kaggle
# notebook has no __main__.__file__, so nothing is re-imported there.
def main():


    CFG = {
        "tasks": ["a", "b", "c"],
        "input": {"slots": ["sag_plain", "cor_fluid"], "n_slices": 2, "channels": 3,
                  "slices_as_channels": None, "resize_to": None,
                  "norm_mean": [0.485, 0.456, 0.406], "norm_std": [0.229, 0.224, 0.225]},
        "training": {"amp": False},
        "experiment": {"name": "fake_member"},
    }

    BUNDLE = Path(tempfile.mkdtemp(prefix="knee_proc_check_"))
    BUNDLE_I = 7                       # deliberately NOT 0, so a bundle/local mix-up shows up
    (BUNDLE / ("m%d_def.py" % BUNDLE_I)).write_text(MODEL_DEF, encoding="utf-8")
    (BUNDLE / ("m%d_config.json" % BUNDLE_I)).write_text(json.dumps(CFG), encoding="utf-8")

    torch.manual_seed(0)
    sys.path.insert(0, str(BUNDLE))
    ref_model = sc.build_member_from_bundle.__wrapped__ if hasattr(
        sc.build_member_from_bundle, "__wrapped__") else None
    import importlib.util  # noqa: E402
    _ms = importlib.util.spec_from_file_location("ref_def", BUNDLE / ("m%d_def.py" % BUNDLE_I))
    _mod = importlib.util.module_from_spec(_ms)
    _ms.loader.exec_module(_mod)
    ref = _mod.build_model(CFG).eval()
    torch.save({"model": ref.state_dict(), "epoch": 1, "score": 0.5},
               BUNDLE / ("m%d_model.pt" % BUNDLE_I))

    print("\n1. a child rebuilds the member from the bundle's own files")
    spec = {"i": BUNDLE_I, "dir": str(BUNDLE), "cfg": CFG}
    rebuilt = sc.build_member_from_bundle(spec)
    ck("build_member_from_bundle returns a module", isinstance(rebuilt, torch.nn.Module))
    ck("in eval mode", not rebuilt.training)
    same = all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(),
                                                 rebuilt.state_dict().values()))
    ck("every weight matches the original", same)
    ck("the def is imported under its OWN module name, not a shared one",
       "m%d_def" % BUNDLE_I in sys.modules)

    print("\n2. the child's logits are bit-identical to the parent's")
    B, S, D, H, W = 2, 2, 2, 24, 24
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.integers(0, 256, (B, S, D, H, W), dtype=np.uint8))
    with torch.no_grad():
        expect = ref(sc.prepare_batch(x, CFG, torch.device("cpu"))).float().numpy()

    t = time.perf_counter()
    card = sc._CardProcess("cpu", [dict(spec, local=0)])
    t_build = time.perf_counter() - t
    ck("a spawned worker reports ready", card.members == [0], "%.1f s to build" % t_build)
    card.submit(x.numpy())
    got, times = card.result(timeout=300)
    ck("it returns exactly its own member", sorted(got) == [0], str(sorted(got)))
    ck("and a time for it", 0 in times and times[0] >= 0)
    ck("logits are BIT-IDENTICAL to the in-process run",
       np.array_equal(got[0], expect),
       "max abs diff %.3e" % float(np.abs(got[0] - expect).max()))
    ck("shape is (studies, tasks)", got[0].shape == (B, len(CFG["tasks"])), str(got[0].shape))

    print("\n3. a second batch reuses the SAME worker, no rebuild")
    x2 = torch.from_numpy(rng.integers(0, 256, (B, S, D, H, W), dtype=np.uint8))
    with torch.no_grad():
        expect2 = ref(sc.prepare_batch(x2, CFG, torch.device("cpu"))).float().numpy()
    t = time.perf_counter()
    card.submit(x2.numpy())
    got2, _ = card.result(timeout=300)
    t_second = time.perf_counter() - t
    ck("a second batch is also identical", np.array_equal(got2[0], expect2))
    ck("and costs far less than the build", t_second < max(t_build, 0.05),
       "%.3f s vs %.3f s to build" % (t_second, t_build))
    ck("a different batch gives a different answer (the test is not vacuous)",
       not np.array_equal(got2[0], got[0]))

    print("\n4. the LOCAL index, not the bundle index, keys the answer")
    card2 = sc._CardProcess("cpu", [dict(spec, local=3)])
    card2.submit(x.numpy())
    got3, _ = card2.result(timeout=300)
    ck("a member at local slot 3 comes back under key 3", sorted(got3) == [3], str(sorted(got3)))
    ck("and it is still the SAME member's logits", np.array_equal(got3[3], expect))
    card2.close()

    print("\n5. a failing worker RAISES here instead of leaving zeros")
    bad = dict(spec, i=999, local=0)                    # no m999_def.py exists
    try:
        sc._CardProcess("cpu", [bad], build_timeout=180)
        ck("a bad spec raises in the parent", False, "it did not raise")
    except RuntimeError as e:
        ck("a bad spec raises in the parent", "could not build" in str(e),
           str(e).splitlines()[0][:60])
    except Exception as e:
        ck("a bad spec raises in the parent", False, repr(e))

    print("\n6. the workers are cached across chunks, and shut down at the end")
    sc.shutdown_card_processes()
    devs = [torch.device("cpu"), torch.device("cpu")]
    assign = [[1], [0]]
    specs = [dict(spec), dict(spec)]
    c1 = sc._get_card_processes(devs, assign, specs)
    c2 = sc._get_card_processes(devs, assign, specs)
    ck("the same assignment reuses the same worker", c1[0] is c2[0])
    ck("exactly one worker for two extra-card calls", len(sc._CARD_PROCS) == 1,
       "%d cached" % len(sc._CARD_PROCS))
    alive = c1[0].p
    sc.shutdown_card_processes()
    time.sleep(0.5)
    ck("shutdown stops the child", not alive.is_alive())
    ck("and empties the cache", not sc._CARD_PROCS)
    card.close()

    print("\n7. the wiring in shared_code")
    src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
    ck("spawn, never fork (fork after CUDA init is undefined)", '_mp.get_context("spawn")' in src)
    ck("the child carries its own no_grad", "with torch.no_grad():" in src)
    ck("the batch crosses as bytes, not through /dev/shm",
       "x_np.tobytes()" in src and "share_memory_" not in src)
    ck("the parent fires the children BEFORE doing its own members",
       src.find("cp.submit(xnp)") < src.find("for _mi in assign[0]:"))
    ck("a dead member still refuses to be blended as 0.5",
       "refusing to blend them as 0.5" in src)
    ck("the grouped path renumbers the specs with the same map as the models",
       "sub_specs = ([model_specs[i] for i in idx] if model_specs else None)" in src)
    ck("and tears the workers down in its finally",
       src.find("shutdown_card_processes()") > 0 and "finally:\n        # every child holds a card" in src)
    ck("no model_specs means one card, not a crash",
       'if len(devices) > 1 and _mode == "process" and model_specs:' in src)
    ck("a worker that cannot start falls back loudly",
       "could not start a worker process" in src)
    # The generated notebook is a BUILD OUTPUT: build_kaggle_ensemble_assets rewrites it
    # every run. The first process-split attempt edited the OUTPUT, so the specs vanished
    # and the kernel silently fell back to the threaded path - it cost a full Kaggle run to
    # notice. Assert against the GENERATOR.
    gen = src[src.find("def build_kaggle_ensemble_assets"):]
    ck("the notebook GENERATOR builds SPECS next to MODELS",
       "SPECS.append({'i': int(i), 'dir': str(W), 'cfg': cfg})" in gen)
    ck("and passes them to the ensemble", "model_specs=SPECS" in gen)
    ck("SPECS is defined before the try that fills it", gen.find("SPECS = []") < gen.find("MODELS = []"))
    ck("split_mode is configurable", "split_mode" in src
       and "split_mode" in (ROOT / "runs" / "shared_config.yaml").read_text(encoding="utf-8"))

    import shutil  # noqa: E402
    shutil.rmtree(BUNDLE, ignore_errors=True)

    print("\n" + "=" * 72)
    print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
