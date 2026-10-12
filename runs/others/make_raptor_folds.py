"""make_raptor_folds.py - write convnext_base_v5_raptor96_f1..f4 (Modal) from convnext_base_v5_raptor96_f0 (local), 12 Oct.

Everything that decides the model is copied from f0 unchanged (input, model, optimiser, loss, schedule, epochs, no SAM);
only three things change per fold:
  data         27_fold{k}_train.csv / 21_fold{k}_val.csv
  dataloader   the Modal recipe of convnext_base_v5_raptor96: batch 8 x accum 1 (= f0's 2 x 4: the same 8 studies per
               update), val batch 4, 26 workers, no grad checkpointing (H100 80 GB)
  validations  cheap on an H100 (~80 s for 836 studies x 94 windows): the raptor Modal cadence, every 400 steps to
               epoch 6, then every 250; patience 6 (1,500 steps, = f0's 6 x 400 within a validation)
The .py files are copied with the run name swapped.

    python runs/others/make_raptor_folds.py
"""
import re
from pathlib import Path

R = Path(__file__).resolve().parents[1]
SRC = "convnext_base_v5_raptor96_f0"
src = (R / SRC / "config.yaml").read_text(encoding="utf-8")

DATALOADER = """dataloader:
  batch_size: 8                     # 8 studies x 12 windows = 96 images per step (f0 local: 2 x accum 4, same 8 studies)
  accum_steps: 1
  val_batch_size: 4                 # 4 x 94 = 376 images per eval batch
  num_workers: 26
  val_num_workers: null
  prefetch_factor: 4
  pin_memory: true
  persistent_workers: true
  drop_last: false
"""
EVALUATION = """evaluation:
  eval_every_steps_before: 400      # epochs 1-5
  eval_every_steps_mid: 400         # epoch 6
  eval_every_steps_epoch_mid_threshold: 6
  eval_every_steps_after: 250       # epoch 7 onward (~80 s each on an H100)
  eval_every_steps_epoch_threshold: 7
"""


def block(text, name):
    """the top-level YAML block `name:` up to the next top-level key."""
    m = re.search(rf"^{name}:\n(?:(?:[ #].*)?\n)*?(?=^\S)", text, re.M)
    assert m, name
    return m.group(0)


for k in range(1, 5):
    name = f"convnext_base_v5_raptor96_f{k}"
    c = src.replace(SRC, name)
    c = c.replace("27_fold0_train.csv", f"27_fold{k}_train.csv").replace("21_fold0_val.csv", f"21_fold{k}_val.csv")
    c = c.replace("27_fold0_val", f"27_fold{k}_val").replace("round-2 fold 0", f"round-2 fold {k}")
    c = c.replace(block(c, "dataloader"), DATALOADER + "\n")
    c = c.replace(block(c, "evaluation"), EVALUATION + "\n")
    c = c.replace("modal:                              # unused locally; train.py's Modal app needs it to import", "modal:")
    c = re.sub(r"^  grad_checkpointing: true.*\n", "  grad_checkpointing: false         # H100 80 GB\n", c, flags=re.M)
    c = re.sub(r"^    patience: 6 .*$", "    patience: 6                     # validations: 1,500 steps at every 250", c, flags=re.M)
    head = (f"# {name}\n#\n# RAPTOR96 FOLD {k} of 5 on MODAL (H100), NO SAM (12 Oct, user: folds 1-4 on Modal, fold 0 locally).\n"
            f"# Written by runs/others/make_raptor_folds.py from {SRC}: identical recipe; only the fold's data, the Modal\n"
            f"# dataloader (batch 8 x accum 1, the same 8 studies per update) and the cheaper validation cadence differ.\n"
            f"# Cost ~1.5-2 h H100 incl. validations, ~$7-9.\n#\n# ---- {SRC}'s notes ----\n")
    c = head + "\n".join(l for l in c.splitlines()[2:]) + "\n"
    d = R / name
    d.mkdir(exist_ok=True)
    (d / "config.yaml").write_text(c, encoding="utf-8")
    for f in ("train.py", "evaluate.py", "plot_results.py", "submit.py"):
        (d / f).write_text((R / SRC / f).read_text(encoding="utf-8").replace(SRC, name), encoding="utf-8")
    print("wrote", d)
