"""make_label_twin.py - point a copied run at another label set (train + val CSVs), rewrite its
header, and reset its submit.py (best checkpoint, message from val_log, short Kaggle name).

    python data_code/make_label_twin.py <run> <train_csv> <val_csv> <kaggle_name> "<header line>"
"""
import re
import sys
from pathlib import Path

run, tr, va, kname, header = sys.argv[1:6]
d = Path(__file__).resolve().parent.parent / "runs" / run
p = d / "config.yaml"
s = p.read_text(encoding="utf-8")


def sub(text, pat, rep, flags=re.M):
    t, n = re.subn(pat, rep, text, count=1, flags=flags)
    assert n == 1, f"{pat!r} matched {n} times in {run}"
    return t


s = sub(s, r'^  train_csv: +"[^"]+".*$', f'  train_csv: "{tr}"')
s = sub(s, r'^  val_csv: +"[^"]+".*$', f'  val_csv:   "{va}"')
# replace the replica's own header block (everything above the donor-notes marker)
s = sub(s, r"\A.*?(?=# ---- the donor's notes follow ----)", f"# {run}\n#\n# {header}\n#\n", re.S)
p.write_text(s, encoding="utf-8")
sp = d / "submit.py"
t = sp.read_text(encoding="utf-8")
t = sub(t, r'^CHECKPOINT = .*\n(?:\s+#.*\n)?', 'CHECKPOINT = "best"            # best | last | top1.. | <step> | path\n')
t = sub(t, r'^KAGGLE_NAME = "[^"]*"', f'KAGGLE_NAME = "{kname}"')
sp.write_text(t, encoding="utf-8")
print("ok", run)
