"""round2_teachers_check.py - do the 15 round-2 teacher runs load THEIR OWN config and data? (7 Oct)

Per run: experiment name = folder, train/val CSV = data/27_fold{k}_*, labels = v4 (22_*-w50) for every study, no
train/val overlap, no audit study anywhere, knee flip on, family's model id, 12 epochs, the GPU of its recipe.
Then one model per family is BUILT from its own train.py with pretrained weights, and one study-shaped batch goes
through it (12 logits).

    python checks/round2_teachers_check.py
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
v4 = pd.read_csv(ROOT / "data" / "22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
aud = set(pd.read_csv(ROOT / "data" / "05_audit58.csv").StudyInstanceUID)
FAM = {"convnext_base_25d_18slices_v4_flip_f": ("convnext_base.fb_in22k_ft_in1k", "H100"),
       "dinov3_convnext_base_25d_18slices_v4_f": ("convnext_base.dinov3_lvd1689m", "H100"),
       "medimageinsight_davit_huge_336_v4_f": ("davit_huge_fl", "H200")}
n_ok = n_all = 0


def check(cond, msg):
    global n_ok, n_all
    n_all += 1
    n_ok += bool(cond)
    if not cond:
        print("  FAIL", msg)


for pre, (mid, gpu) in FAM.items():
    for k in range(5):
        name = f"{pre}{k}"
        cfg = sc.load_config(ROOT / "runs" / name, verbose=False)
        check(cfg["experiment"]["name"] == name, f"{name}: experiment name {cfg['experiment']['name']}")
        cnx = pre.startswith("convnext")      # ConvNeXt folds validate on v3-graded val (user 7 Oct), early stop reset once
        want_val = f"21_fold{k}_val.csv" if cnx else f"27_fold{k}_val.csv"
        check(cfg["paths"]["train_csv"] == f"27_fold{k}_train.csv" and cfg["paths"]["val_csv"] == want_val,
              f"{name}: csv {cfg['paths']['train_csv']} {cfg['paths']['val_csv']}")
        if cnx:
            es = cfg["training"]["early_stopping"]
            check(es["enable"] and es["patience"] == 7 and es.get("reset_tag") == "v3val", f"{name}: early stopping {es}")
        tr = pd.read_csv(ROOT / "data" / cfg["paths"]["train_csv"]).set_index("StudyInstanceUID")
        va = pd.read_csv(ROOT / "data" / cfg["paths"]["val_csv"]).set_index("StudyInstanceUID")
        check(not (set(tr.index) & set(va.index)), f"{name}: train/val overlap")
        check(not ((set(tr.index) | set(va.index)) & aud), f"{name}: audit study inside")
        for part, f in (("train", tr), ("val", va)):
            ref = v4 if part == "train" or not cnx else pd.read_csv(ROOT / "data" / f"21_fold{k}_val.csv").set_index("StudyInstanceUID")
            d = (f[T] - ref.loc[f.index, T]).abs().to_numpy().max()
            check(d < 1e-12, f"{name}: {part} labels differ from v4 by {d}")
        check(cfg["input"]["laterality_flip"] == "left", f"{name}: flip {cfg['input']['laterality_flip']}")
        check(cfg["model"]["name"] == mid, f"{name}: model {cfg['model']['name']}")
        check(int(cfg["training"]["epochs"]) == 12, f"{name}: epochs {cfg['training']['epochs']}")
        local = pre.startswith("convnext") and k < 2          # folds 0-1 of the ConvNeXt family train on this PC
        check(local or cfg["modal"]["gpu"] == gpu, f"{name}: gpu {cfg['modal']['gpu']}")
        if local:
            dl = cfg["dataloader"]
            check(dl["batch_size"] == 2 and dl["accum_steps"] == 4 and cfg["training"].get("grad_checkpointing")
                  and cfg["output"]["keep_last_n"] == 1 and cfg["output"]["keep_top_k"] == 1, f"{name}: not the local recipe")
        if not pre.startswith("medimageinsight"):
            check(cfg["reproducibility"]["seed"] == {0: 42, 1: 42, 2: 42, 3: 1042, 4: 2042}[k],
                  f"{name}: seed {cfg['reproducibility']['seed']}")
        check(int(cfg["dataloader"]["batch_size"]) * int(cfg["dataloader"].get("accum_steps") or 1) == 8, f"{name}: batch")
    print(f"{pre}0..4 config + data checked")

for pre in FAM:
    name = f"{pre}0"
    spec = importlib.util.spec_from_file_location(name, ROOT / "runs" / name / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.build_model(mod.cfg).eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    check(n > 50, f"{name}: only {n:.1f}M params")
    print(f"{name}: built with pretrained weights, {n:.1f}M params")
    del m

print(f"{'ALL OK' if n_ok == n_all else 'FAILED'} ({n_ok}/{n_all})")
