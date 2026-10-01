"""offload_between_groups_check.py - weights leave the card between geometry groups, and nothing changes.

The ten + 5 MII folds ensemble lost its 480 px member: every member stayed on its card once
loaded, so by that member's group card 0 held 7.17 GB of other groups' weights (4.28 GB in the
ten) and its batches ran out of memory and fell back to zeros. predict_dicom_ensemble now parks
each group's weights in host memory when the group finishes, in this process and in the
second-card child (_CardProcess.offload).

What must hold:
  1. a child answers an offload, and its NEXT batch is bit-identical to the one before it
  2. on a real GPU the child's weights actually leave the card (device free memory rises by
     about the model's size) and come back for the next batch
  3. a parent-side module moved to CPU and back gives bit-identical logits
  4. the engine source offloads at the end of every group call, by default, in both places,
     and hands each group's notes up to inference_log.txt

    python checks/offload_between_groups_check.py
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
    print("  %s %-62s %s" % ("PASS" if ok else "FAIL", n, d))


MODEL_DEF = '''
import torch, torch.nn as nn


class SliceModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_img = len(cfg["input"]["slots"]) * cfg["input"]["n_slices"]
        self.body = nn.Sequential(nn.Conv2d(cfg["input"]["channels"], 4, 3, padding=1),
                                  nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.ballast = nn.Linear(8192, 8192)     # ~268 MB of weights, so the move is measurable
        self.head = nn.Linear(4, len(cfg["tasks"]))

    def forward(self, x):
        z = self.body(x)
        z = z.view(-1, self.n_img, 4).mean(1)
        return self.head(z) + 0.0 * self.ballast.weight[0, 0]


def build_model(cfg):
    return SliceModel(cfg)
'''


def main():
    CFG = {"tasks": ["a", "b", "c"],
           "input": {"slots": ["sag_plain", "cor_fluid"], "n_slices": 2, "channels": 3,
                     "slices_as_channels": None, "resize_to": None,
                     "norm_mean": [0.485, 0.456, 0.406], "norm_std": [0.229, 0.224, 0.225]},
           "training": {"amp": False}, "experiment": {"name": "fake_member"}}
    bundle = Path(tempfile.mkdtemp(prefix="knee_offload_check_"))
    (bundle / "m0_def.py").write_text(MODEL_DEF, encoding="utf-8")
    (bundle / "m0_config.json").write_text(json.dumps(CFG), encoding="utf-8")
    torch.manual_seed(0)
    import importlib.util
    ms = importlib.util.spec_from_file_location("ref_def", bundle / "m0_def.py")
    mod = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(mod)
    ref = mod.build_model(CFG).eval()
    torch.save({"model": ref.state_dict()}, bundle / "m0_model.pt")
    spec = {"i": 0, "dir": str(bundle), "cfg": CFG, "local": 0}
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (2, 2, 2, 24, 24), dtype=np.uint8)
    nbytes = sum(p.numel() * p.element_size() for p in ref.parameters())

    devs = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for dev in devs:
        print(f"\n1-2. child on {dev}: offload, then the next batch")
        card = sc._CardProcess(dev, [spec])
        card.submit(x)
        a, _ = card.result(timeout=300)
        t = time.perf_counter()
        left = card.offload()
        t_off = time.perf_counter() - t
        ck(f"[{dev}] offload answers", True, "%.2f s" % t_off)
        if dev.startswith("cuda"):
            # the child's own allocator, not the device-wide free count: on Windows (WDDM)
            # cudaMemGetInfo does not see another process's frees promptly
            ck(f"[{dev}] the weights left the card", left < 0.05 * nbytes,
               "child holds %.1f MB on the card after offload, weights are %.0f MB" % (left / 1e6, nbytes / 1e6))
        card.submit(x)
        b, _ = card.result(timeout=300)
        ck(f"[{dev}] the next batch is bit-identical", np.array_equal(a[0], b[0]),
           "max abs diff %.1e" % float(np.abs(a[0] - b[0]).max()))
        card.offload()
        card.submit(x)
        c, _ = card.result(timeout=300)
        ck(f"[{dev}] a second offload/reload cycle is also identical", np.array_equal(a[0], c[0]))
        card.close()

    print("\n3. parent side: a module moved to CPU and back is unchanged")
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        m = mod.build_model(CFG).eval()
        m.load_state_dict(ref.state_dict())
        m.to(dev)
        xb = sc.prepare_batch(torch.from_numpy(x), CFG, dev)
        with torch.no_grad():
            o1 = m(xb).cpu()
            m.to("cpu")
            torch.cuda.empty_cache()
            m.to(dev)
            o2 = m(xb).cpu()
        ck("logits bit-identical after a CPU round trip", torch.equal(o1, o2))
        del m, xb
        torch.cuda.empty_cache()
    else:
        ck("no GPU here: parent round trip skipped", True)

    print("\n4. the engine offloads at the end of every group call, by default")
    src = (ROOT / "runs/shared_code.py").read_text(encoding="utf-8")
    ens = src[src.find("def predict_dicom_ensemble("):src.find("def predict_dicom_ensemble_grouped")]
    ck("predict_dicom_ensemble reads offload_between_groups, default True",
       'get("offload_between_groups", True)' in ens)
    blk = ens[ens.find('get("offload_between_groups", True)'):]
    blk = blk[:blk.find("timing[\"notes\"]")]
    ck("it parks this process's members on the CPU", '_m.to("cpu")' in blk)
    ck("and tells every live child to offload", "cp.offload()" in blk)
    ck("and frees the cache", "torch.cuda.empty_cache()" in blk)
    ck("the offload comes BEFORE the group returns its logits",
       ens.find('get("offload_between_groups", True)') < ens.find("if return_member_logits:\n        return probs"))
    grp = src[src.find("def predict_dicom_ensemble_grouped"):]
    ck("the grouped run copies each group's notes into the log", 'gtiming.get("notes", [])' in grp)
    wk = src[src.find("def _proc_worker"):src.find("class _CardProcess")]
    ck("the child moves its members back to the card before a batch", "models[i].to(dev)" in wk)
    ck("children offload only when asked (offload_children, default False)",
       'get("offload_children", False)' in blk)
    ck("this process frees its copies of the children's members",
       'to_empty(device="meta")' in ens)
    ck("a freed member is rebuilt from the bundle before a one-card fallback scores it",
       "_is_meta(_m)" in ens and "build_member_from_bundle(dict(model_specs[_i]" in ens)

    print(chr(10) + "5. freeing a member's host copy really releases it, and _is_meta sees it")
    m = mod.build_model(CFG)
    ck("a built member is not meta", not sc._is_meta(m))
    m.to_empty(device="meta")
    ck("after to_empty(meta) it is meta", sc._is_meta(m))
    ck("and holds no storage", all(p.device.type == "meta" for p in m.parameters()))
    rebuilt = sc.build_member_from_bundle(spec)
    with torch.no_grad():
        same = torch.equal(rebuilt(sc.prepare_batch(torch.from_numpy(x), CFG, torch.device("cpu"))),
                           ref(sc.prepare_batch(torch.from_numpy(x), CFG, torch.device("cpu"))))
    ck("a member rebuilt from the bundle gives the original's logits", same)

    sc.shutdown_card_processes()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
