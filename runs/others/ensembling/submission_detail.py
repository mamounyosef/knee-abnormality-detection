"""submission_detail.py - print every field Kaggle reports for the given submission refs (error text, scores, sizes).

    python runs/others/ensembling/submission_detail.py <ref> [<ref> ...]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])
import kaggle  # noqa: E402

want = {int(r) for r in sys.argv[1:]}
for s in kaggle.api.competition_submissions("rsna-knee-abnormality-detection"):
    if s.ref in want:
        d = {k: v for k, v in vars(s).items() if v not in (None, "", [])}
        print(s.ref, {k.lstrip("_"): (str(v)[:150]) for k, v in d.items() if k.lstrip("_") != "description"})
