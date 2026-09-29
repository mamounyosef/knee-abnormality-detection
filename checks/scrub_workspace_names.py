"""scrub_workspace_names.py - remove workspace-specific names from files that go to git.

Replaces Modal config paths (~/.modal_session<N>.toml) in run configs with the "local" pointer
(shared_code.resolve_upload_modal_config) and rewrites "workspace <N>" mentions to neutral
wording. With --check it only reports what would change (exit 1 if anything is found).

    python checks/scrub_workspace_names.py [--check] <file> [...]
"""
import re
import sys
from pathlib import Path

RULES = [
    (r'modal_config: "~/\.modal_session\d+\.toml"',
     'modal_config: "local"            # path in ~/.knee_upload_modal_config (outside the repo)'),
    (r"workspace \d+'s knee-runs volume", "the knee-runs upload volume"),
    (r"workspace \d+", "the upload workspace"),
    (r"modal_session\d+", "modal_config"),
]
check = "--check" in sys.argv
files = [Path(a) for a in sys.argv[1:] if a != "--check"]
hits = 0
for f in files:
    s = f.read_text(encoding="utf-8")
    t = s
    for pat, rep in RULES:
        t = re.sub(pat, rep, t)
    if t != s:
        hits += 1
        n = sum(len(re.findall(p, s)) for p, _ in RULES)
        print(f"{'would fix' if check else 'fixed'} {f} ({n} match(es))")
        if not check:
            f.write_text(t, encoding="utf-8")
print(f"{hits} file(s) {'with workspace names' if check else 'changed'}")
sys.exit(1 if (check and hits) else 0)
