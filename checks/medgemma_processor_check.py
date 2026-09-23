"""medgemma_processor_check.py - the prompt path the audit uses, on a real Gemma 3 processor.

MedGemma's own repo is gated, so this loads an ungated copy of the Gemma 3 4B processor
(same tokenizer, chat template and image processor family) and runs exactly what
medgemma_audit58.score does for one fake study per mode: template with a pre-filled
"Final Answer:", a single BOS, 256 image tokens per slice, token_type_ids present, the
yes/no token ids, and a shared prefix that covers every image. Also exercises scoreboard()
on made-up probabilities.

    python checks/medgemma_processor_check.py [processor_repo]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_code" / "label_v2"))
import medgemma_audit58 as mg  # noqa: E402

REPO = sys.argv[1] if len(sys.argv) > 1 else "unsloth/gemma-3-4b-it"
proc = AutoProcessor.from_pretrained(REPO, use_fast=True)
tok = proc.tokenizer
IMG_ID = tok.convert_tokens_to_ids("<image_soft_token>")
fails = 0


def check(cond, msg):
    global fails
    print(("  OK   " if cond else "  FAIL ") + msg)
    fails += not cond


yes = [(tok.encode(w, add_special_tokens=False), w) for w in (" yes", "yes", " Yes", "Yes")]
no = [(tok.encode(w, add_special_tokens=False), w) for w in (" no", "no", " No", "No")]
print("yes/no encodings:", yes, no)
check(all(len(e) == 1 for e, _ in yes + no), "every yes/no variant is ONE token")

job = {"uid": "x", "report": "Moderate joint effusion. Intact ACL.", "side": "left",
       "slots": {}}
rng = np.random.default_rng(0)
for mode, groups in (("rep", []), ("ax", [("ax_plain", 2), ("ax_fluid", 3)]),
                     ("ax_rep", [("ax_fluid", 3)]),
                     ("all_rep", [("sag_plain", 2), ("cor_fluid", 2), ("ax_fluid", 2)])):
    imgs = [Image.fromarray(rng.integers(0, 255, (512, 512), dtype=np.uint8)).convert("RGB")
            for _, n in groups for _ in range(n)]
    ids_all = []
    for t in mg.TASKS:
        msg = mg.messages(groups, mg.prompt_text(job, mode, t))
        text = proc.apply_chat_template(msg, continue_final_message=True, tokenize=False)
        x = proc(text=text, images=imgs or None, add_special_tokens=False, return_tensors="pt")
        ids = x["input_ids"][0]
        ids_all.append(ids.tolist())
    print(f"\n[{mode}] {len(imgs)} images; rendered prompt (Effusion), image tokens collapsed:")
    shown = proc.apply_chat_template(mg.messages(groups, mg.prompt_text(job, mode, "Effusion")),
                                     continue_final_message=True, tokenize=False)
    print("   " + shown.replace("\n", "\n   ")[:900])
    check(ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id, "exactly one BOS")
    check(text.rstrip().endswith("Final Answer:"), "prompt ends on the pre-filled 'Final Answer:'")
    check("<end_of_turn>" not in text.split("<start_of_turn>model")[-1],
          "model turn left open (continue_final_message)")
    n_img_tok = int((ids == IMG_ID).sum())
    check(n_img_tok == 256 * len(imgs), f"{n_img_tok} image tokens = 256 x {len(imgs)} slices")
    if imgs:
        check("token_type_ids" in x and int(x["token_type_ids"].sum()) == n_img_tok,
              "token_type_ids mark exactly the image tokens (bidirectional image attention)")
        check(tuple(x["pixel_values"].shape) == (len(imgs), 3, 896, 896),
              f"pixel_values {tuple(x['pixel_values'].shape)} (896 px)")
    L = mg.common_prefix_len(ids_all) - 1
    last_img = max((i for i, v in enumerate(ids.tolist()) if v == IMG_ID), default=-1)
    check(last_img < L, f"shared prefix {L} tokens covers every image (last image token {last_img})")
    check(all(len(s) > L for s in ids_all), "every question keeps >= 1 own token")
    print("   question part:", repr(tok.decode(ids_all[0][L:])))

# scoreboard on fake predictions for the real audit studies
aud = pd.read_csv(ROOT / "data/05_audit58.csv")
rows = []
for u in aud.StudyInstanceUID:
    for m in ("rep", "ax"):
        r = {"uid": u, "mode": m}
        r.update({t: float(rng.random()) for t in mg.TASKS})
        rows.append(r)
print()
mg.scoreboard(pd.DataFrame(rows), ["rep", "ax"])
print("\nALL OK" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
