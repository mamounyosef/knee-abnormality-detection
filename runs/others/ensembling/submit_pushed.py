"""submit_pushed.py - finish an already-pushed kernel: wait, check its output, submit.

submit.py / submit_ensemble.py keep every member's weights in memory while they wait hours for
Kaggle, ~2-8 GB of RAM this PC needs for local training. Once a script prints "2/4 notebook ...
pushed version N" it can be stopped and this takes over with the SAME steps as
shared_code.kaggle_publish 3/4 and 4/4: poll until complete, verify_kernel_output (every
member ran, no 0.5 fallbacks), then submit that exact version with the given message.

    python runs/others/ensembling/submit_pushed.py <notebook-slug> <version> <n_members> "<message>"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # <repo>/runs
import shared_code as sc  # noqa: E402

USER = "mamounhussam"
slug, version, n_members, message = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
t0 = time.time()
print(f"watching {USER}/{slug} version {version} ({n_members} member(s))", flush=True)
while True:
    st = sc._kaggle("kernels", "status", f"{USER}/{slug}", quiet=True)
    low = st.lower()
    if "complete" in low:
        print(f"✅ complete after {(time.time() - t0) / 3600:.2f} h of watching", flush=True)
        break
    if "error" in low or "cancel" in low:
        raise SystemExit(f"❌ {st.strip()} - not submitted, no attempt spent")
    print(f"[{time.strftime('%H:%M')}] {st.strip()[:90]}", flush=True)
    time.sleep(120)
bad = sc.verify_kernel_output(USER, slug, n_members=n_members)
if bad:
    print("❌ NOT SUBMITTING - the kernel completed but its output is bad:")
    for b in bad:
        print("   - %s" % b)
    raise SystemExit(1)
print("✅ output looks sound: every member ran, no 0.5 fallbacks", flush=True)
print(sc._kaggle("competitions", "submit", "-c", "rsna-knee-abnormality-detection", "-f",
                 "submission.csv", "-k", f"{USER}/{slug}", "-m", message, "-v", version), flush=True)
print("SUBMITTED", flush=True)
