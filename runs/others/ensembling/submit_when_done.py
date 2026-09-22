"""submit_when_done.py - push several ensembles first, submit each one as its kernel finishes.

submit_ensemble.py builds ONE ensemble and then blocks until Kaggle has run it, which is
four to six hours on the real test set. Four submissions that way is a day of wall clock on
this PC for work that Kaggle is perfectly happy to run in parallel. So: push them all with
WAIT_FOR_RUN = False and SUBMIT = False, then hand their slugs to this.

It also happens to be the safe order for an UNPRICED member. A kernel that breaches the 9 h
cap fails on Kaggle and is never submitted, so it costs a run and not one of the five daily
attempts. Only a kernel that completes gets submitted here.

    python runs/others/ensembling/submit_when_done.py knee-nine-medimageinsight ...

Each argument is a NOTEBOOK slug without the user prefix (the "<slug>-submit" form that
submit_ensemble.py pushes). The message attached to each submission is read back from the
kernel's own metadata, so it cannot drift from what was published.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

USER = "mamounhussam"
COMP = "rsna-knee-abnormality-detection"
POLL_SECS = 120
HARD_LIMIT_H = 10.0          # Kaggle's own cap is 9 h; give it slack before giving up here

os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])


def kaggle(*args, check=True):
    p = subprocess.run([sys.executable, "-m", "kaggle", *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise SystemExit("kaggle %s failed:\n%s" % (" ".join(args), out.strip()[:800]))
    return out


def note_for(slug: str) -> str:
    """The description that was published with the kernel, so the record cannot drift."""
    d = Path(__file__).resolve().parent / "kaggle" / "notebook" / "kernel-metadata.json"
    try:
        meta = json.loads(d.read_text(encoding="utf-8"))
        if str(meta.get("id", "")).endswith(slug):
            return str(meta.get("title") or slug)
    except Exception:
        pass
    return slug


def main(slugs):
    if not slugs:
        raise SystemExit(__doc__)
    pending = {s: time.time() for s in slugs}
    done, failed = [], []
    print("waiting on %d kernel(s): %s" % (len(pending), ", ".join(pending)), flush=True)
    while pending:
        for slug in list(pending):
            st = kaggle("kernels", "status", "%s/%s" % (USER, slug), check=False)
            low = st.lower()
            el_h = (time.time() - pending[slug]) / 3600.0
            if "complete" in low:
                print("\n%s: complete after %.2f h -> submitting" % (slug, el_h), flush=True)
                out = kaggle("competitions", "submit", "-c", COMP, "-f", "submission.csv",
                             "-k", "%s/%s" % (USER, slug), "-m", note_for(slug), check=False)
                print("   ", out.strip()[:300], flush=True)
                # A kernel that produced no submission.csv, or a rejected file, must NOT be
                # recorded as submitted: that is how a lost attempt goes unnoticed.
                (done if "successfully submitted" in out.lower() else failed).append(slug)
                pending.pop(slug)
            elif "error" in low or "cancel" in low:
                print("\n%s: FAILED after %.2f h - not submitted, no attempt spent\n   %s"
                      % (slug, el_h, st.strip()[:200]), flush=True)
                failed.append(slug)
                pending.pop(slug)
            elif el_h > HARD_LIMIT_H:
                print("\n%s: still running after %.1f h, giving up on watching it"
                      % (slug, el_h), flush=True)
                failed.append(slug)
                pending.pop(slug)
            else:
                print("  %-34s %-28s %.2f h" % (slug, st.strip()[:28], el_h), flush=True)
        if pending:
            time.sleep(POLL_SECS)
    print("\nsubmitted: %s" % (", ".join(done) or "none"))
    print("not submitted: %s" % (", ".join(failed) or "none"))
    return 0 if done and not failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
