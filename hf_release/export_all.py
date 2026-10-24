"""export_all.py - build the Hugging Face folders for every released model. Uploads nothing.

Same steps as export_model.py, for each model in MODELS:
    1. get the weights + config (submitted Kaggle bundle, or the Modal backup volume for the label teachers)
    2. model.safetensors
    3. config.json, metrics.json, history csv, modeling.py, README.md (+ the DINOv3 licence where needed)
    4. load the folder back with its own modeling.py: every weight must match

    python hf_release/export_all.py              # all models, one process each
    python hf_release/export_all.py <run name>   # one model
    python hf_release/export_all.py dinov3       # every model whose name contains "dinov3"
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DOWNLOADS = OUT / "_download"
sys.path.insert(0, str(HERE))
from catalog import MODELS, card_for   # noqa: E402  (the model list and the card writer)

K14 = "mamounhussam/knee-ten-mii-5folds"

# the three engine helpers the model code calls (copied from runs/shared_code.py)
SHIM = '''
# ---- helpers from the training engine (runs/shared_code.py), so this file needs no repo code ----
import types


def _slot_slices(cfg, slot):
    n = cfg["input"]["n_slices"]
    return int(n[slot]) if isinstance(n, dict) else int(n)


def _slot_groups(cfg, slot):
    g = cfg["input"].get("slices_as_channels")
    g = int(g[slot]) if isinstance(g, dict) else int(g or 0)
    return g if g > 0 else _slot_slices(cfg, slot)


def _images_per_study(cfg):
    i = cfg["input"]
    if isinstance(i["n_slices"], dict) or isinstance(i.get("slices_as_channels"), dict):
        return sum(_slot_groups(cfg, s) for s in i["slots"])
    g = int(i.get("slices_as_channels") or 0)
    return len(i["slots"]) * (g if g > 0 else int(i["n_slices"]))


sc = types.SimpleNamespace(num_output_logits=lambda cfg: len(cfg["tasks"]), slot_groups=_slot_groups,
                           images_per_study=_images_per_study)
'''

LOADER = '''

def load_model(folder, device="cpu"):
    """Build the model from config.json and load model.safetensors from the same folder."""
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    folder = Path(folder)
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = build_model(cfg)
    model.load_state_dict(load_file(folder / "model.safetensors"))   # strict: every weight must match
    return model.to(device).eval()
'''


# ---------------------------------------------------------------- step 1: sources

def kaggle_api():
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
    os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])
    import kaggle
    return kaggle.api


def get_kaggle(bundle, member, name):
    """Download mN_model.pt / mN_config.json / mN_def.py from a bundle into DOWNLOADS/<bundle>/."""
    d = DOWNLOADS / bundle.split("/")[1]
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{member}_{name}"
    if not f.exists():
        print("   downloading", f.name, "from", bundle)
        kaggle_api().dataset_download_file(bundle, f.name, path=str(d), quiet=True)
        z = d / (f.name + ".zip")
        if z.exists():
            shutil.unpack_archive(z, d)
            z.unlink()
    return f


def get_backup(path):
    """Download one file from the knee-ckpt-backup Modal volume (uses your active Modal login)."""
    f = DOWNLOADS / "modal_backup" / path
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        print("   downloading", path, "from the Modal backup volume")
        env = dict(os.environ, PYTHONUTF8="1")
        env.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
        env.setdefault("SSL_CERT_FILE", env["REQUESTS_CA_BUNDLE"])
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "knee-ckpt-backup", path, str(f)],
                       check=True, env=env)
    return f


# ---------------------------------------------------------------- step 3: model code

def modeling_from_def(src: str, run: str) -> str:
    """A training entry point -> standalone modeling.py: drop the repo plumbing, add the helper shim + loader."""
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("import shared_code as sc")) + 1
    body = lines[start:]
    body = [l for l in body if not l.startswith("cfg = sc.load_config")]
    for i, l in enumerate(body):                    # cut the train / Modal launch code, if present
        if l.startswith("def run_local") or l.startswith("# ---- Modal") or l.startswith("if __name__"):
            body = body[:i]
            break
    head = (f'"""Standalone model definition for {run}, from its training script (no repo code needed).\n\n'
            f'    model = load_model("path/to/{run}")\n"""\n'
            "import re\nimport sys\nfrom pathlib import Path\n\nimport timm\nimport torch\nimport torch.nn as nn\n")
    code = head + SHIM + "\n".join(body).rstrip() + "\n" + LOADER
    return re.sub(r"\n{4,}", "\n\n\n", code)


# ---------------------------------------------------------------- one model

def clean_config(full: dict, run: str, step: int) -> dict:
    keep = ["tasks", "model", "input", "augmentation", "target", "training"]
    cfg = {"experiment": {"name": run, "checkpoint_step": step}, **{k: full[k] for k in keep if k in full}}
    cfg["model"] = {k: v for k, v in cfg["model"].items() if k != "weights"}   # pretrained-file name, unused
    cfg["model"]["pretrained"] = False      # weights come from model.safetensors, never downloaded
    for unused in ["aux_report", "aux_attributes", "log_ud_ratio", "perf"]:
        cfg["training"].pop(unused, None)
    if "dataloader" in full:
        cfg["dataloader"] = {k: full["dataloader"][k] for k in ("batch_size", "accum_steps") if k in full["dataloader"]}
    return cfg


def history(run: str, step: int):
    """(metrics at step, history csv path, kind) from the run's val_log, or the train_log for full-data runs."""
    res = ROOT / "runs" / run / "results"
    val_log = res / "val_log.csv"
    if val_log.exists():
        with open(val_log, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if int(r["step"]) == step]
        if rows:
            r = rows[-1]
            num = lambda v: round(float(v), 4) if v not in ("", None) else None
            m = {"val_macro_auroc": num(r["val_macro_auroc"]),
                 "audit58_macro_auroc": num(r.get("audit_macro_auroc")),
                 "val_auroc_per_finding": {k[10:]: num(v) for k, v in r.items() if k.startswith("val_auroc_")},
                 "audit58_auroc_per_finding": {k[12:]: num(v) for k, v in r.items() if k.startswith("audit_auroc_")}}
            return m, val_log, "val_history.csv"
    return {}, res / "train_log.csv", "train_history.csv"


def export(m: dict):
    run, step = m["run"], m["step"]
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n== {run}")

    # step 1 + 2: weights
    if m["src"][0] == "backup":
        ck = torch.load(get_backup(m["src"][1]), map_location="cpu", weights_only=False)
        full_cfg = ck["cfg"]
        def_src = (ROOT / "runs" / run / "train.py").read_text(encoding="utf-8")
    else:
        bundle, member = m["src"]
        ck = torch.load(get_kaggle(bundle, member, "model.pt"), map_location="cpu", weights_only=False)
        full_cfg = json.loads(get_kaggle(bundle, member, "config.json").read_text(encoding="utf-8"))
        def_src = get_kaggle(bundle, member, "def.py").read_text(encoding="utf-8")
    if step is None:                          # backup best.pt: take its own step
        step = m["step"] = int(ck["step"])
    assert ck["step"] == step, f"{run}: expected step {step}, file has {ck['step']}"
    weights = {k.removeprefix("_orig_mod.").removeprefix("module."): v.contiguous() for k, v in ck["model"].items()}
    save_file(weights, out / "model.safetensors", metadata={"run": run, "step": str(step)})
    n_params = sum(v.numel() for v in weights.values())
    epoch = int(ck.get("epoch", 0))
    del ck, weights                           # free RAM before the load check (MedImageInsight is 1.4 GB)
    import gc
    gc.collect()

    # step 3: config, metrics, history, code, card, licences
    cfg = clean_config(full_cfg, run, step)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    metrics, hist, hist_name = history(run, step)
    metrics = {"step": step, "epoch": epoch, "parameters": n_params,
               **m.get("scores", {}), **metrics}
    metrics = {k: v for k, v in metrics.items() if v is not None}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for old in ("val_history.csv", "train_history.csv"):
        (out / old).unlink(missing_ok=True)
    if hist.exists():
        shutil.copy(hist, out / hist_name)
    if m["family"] == "per_finding_attn":
        shutil.copy(HERE / "modeling_per_finding_attn.py", out / "modeling.py")
    else:
        (out / "modeling.py").write_text(modeling_from_def(def_src, run), encoding="utf-8")
    (out / "README.md").write_text(card_for(m, cfg, metrics, hist_name), encoding="utf-8")
    if "dinov3" in cfg["model"]["name"]:
        shutil.copy(DOWNLOADS / "knee-ten-mii-5folds" / "DINOV3_LICENSE.md", out / "LICENSE_DINOv3.md")

    # step 4: rebuild from the folder alone and match every weight name and shape against the file.
    # The model is built on the "meta" device (shapes only, no memory), and only the file's header is read,
    # so even the 1.4 GB models are checked without loading them into RAM.
    sys.dont_write_bytecode = True
    import importlib.util
    import struct
    spec = importlib.util.spec_from_file_location(f"modeling_{run}", out / "modeling.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with torch.device("meta"):
        model = mod.build_model(json.loads((out / "config.json").read_text(encoding="utf-8")))
    expected = {k: list(v.shape) for k, v in model.state_dict().items()}
    with open(out / "model.safetensors", "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))   # safetensors: 8-byte length + JSON
    in_file = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    missing, extra = expected.keys() - in_file.keys(), in_file.keys() - expected.keys()
    bad_shape = [k for k in expected.keys() & in_file.keys() if expected[k] != in_file[k]]
    assert not (missing or extra or bad_shape), f"{run}: missing {missing}, extra {extra}, shape {bad_shape}"
    print(f"   ok: {n_params / 1e6:.1f}M parameters, all {len(expected)} weights match the model")


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    lic =DOWNLOADS / "knee-ten-mii-5folds" / "DINOV3_LICENSE.md"
    if not lic.exists():
        lic.parent.mkdir(parents=True, exist_ok=True)
        kaggle_api().dataset_download_file(K14, "DINOV3_LICENSE.md", path=str(lic.parent), quiet=True)
    names = [m["run"] for m in MODELS]
    if want in names:                        # an exact name: just that model
        export(MODELS[names.index(want)])
    elif want:                               # a fragment: every match, in this process
        for m in MODELS:
            if want in m["run"]:
                export(m)
    else:                                    # everything, one process per model so RAM is freed in between
        for name in names:
            subprocess.run([sys.executable, __file__, name], check=True)
    print("\nall done")


if __name__ == "__main__":
    main()
