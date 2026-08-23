"""
smoke.py - one labeling call against whatever .env points at, printed raw.

Endpoints that do not constrain decoding let the model answer in any shape, and the
shape differs per model: gpt-oss:120b returned a markdown TABLE, gemma4:31b alternates
between {"probability","sentence"} and the bare pair [0.05, "sentence"]. Finding that
out from a batch costs a whole quota; finding it out from one call costs one call.

Exit code 0 only if the reply parsed AND coerced into the 12-finding shape.

    python data_code/label_v2/smoke.py
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "data_code"))

spec = importlib.util.spec_from_file_location("lab", ROOT / "data_code" / "03_label_reports.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

from llm_client import _extract_json, chat, load_env  # noqa: E402

REPORT = ("Mild joint effusion in the suprapatellar recess. The anterior cruciate "
          "ligament is intact. No meniscal tear is identified. Moderate patellofemoral "
          "chondrosis with full-thickness cartilage loss at the medial patellar facet.")


def main() -> int:
    cfg = load_env()
    print(f"  endpoint : {cfg['base']}")
    print(f"  model    : {cfg['model']}")
    print(f"  auth     : {', '.join(cfg['headers'])}")

    system = lab.SYSTEM
    if not cfg.get("modal"):
        # same contract main() applies when the endpoint cannot enforce the schema
        system += ("\n\nOUTPUT FORMAT - follow exactly, no exceptions:\n"
                   "Reply with ONE raw JSON object and nothing else. No prose, no "
                   "markdown, no fences, no table.\n"
                   '{"unusable": false, "findings": {"ACL": {"p": 0.05, "evidence": "..."}, ...}}\n'
                   "Use EXACTLY these 12 keys: " + ", ".join(lab.LABELS) +
                   '\nEach entry has exactly two fields: "p" (number 0.0-1.0) and "evidence".')

    try:
        raw = chat([{"role": "system", "content": system},
                    {"role": "user", "content": f"KNEE MRI REPORT:\n\n{REPORT}"}],
                   schema=lab.SCHEMA, schema_name="knee_labels", effort="low",
                   max_tokens=4000, timeout=300)
    except Exception as e:                                   # noqa: BLE001
        print(f"  REQUEST FAILED: {type(e).__name__}: {str(e)[:300]}")
        return 1

    print(f"\n  raw reply ({len(raw)} chars), first 200:")
    print(f"    {raw[:200]!r}")

    try:
        out = lab._coerce(_extract_json(raw))
    except Exception as e:                                   # noqa: BLE001
        print(f"\n  PARSE FAILED: {type(e).__name__}: {str(e)[:300]}")
        return 1

    print("\n  parsed:")
    for k, v in out["findings"].items():
        print(f"    {k:18s} p={v['p']:.2f}  {v['evidence'][:52]}")
    # the two the report qualifies - a labeler that reads wording rather than keywords
    # puts Effusion low (the panel does not count "mild") and PF OA high
    print(f"\n  sanity: Effusion={out['findings']['Effusion']['p']:.2f} (want <0.5, "
          f"report says 'mild'), PF OA={out['findings']['PF OA']['p']:.2f} "
          f"(want >0.6, full-thickness loss)")
    print("  SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
