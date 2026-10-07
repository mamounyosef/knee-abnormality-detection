"""openrouter_models.py - what can our OpenRouter keys label with? (7 Oct)

Prints each key's limits (free tier or not, requests/day) and writes data/stats/openrouter_models.csv: every model
with its prices ($/M tokens), context, input modalities and whether it is free. Keys come ONLY from the environment
(OPENROUTER_API_KEYS=k1,k2); nothing is written to disk but the model table.

    $env:OPENROUTER_API_KEYS="k1,k2"; python data_code/label_v2/openrouter_models.py
"""
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
keys = [k.strip() for k in os.environ.get("OPENROUTER_API_KEYS", "").split(",") if k.strip()]
for i, k in enumerate(keys, 1):
    r = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {k}"}, timeout=30)
    d = r.json().get("data", r.json())
    print(f"key {i}: " + ", ".join(f"{a}={d.get(a)}" for a in ("is_free_tier", "limit", "limit_remaining", "usage",
                                                              "usage_daily", "rate_limit")))
m = requests.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
rows = []
for x in m:
    p = x.get("pricing", {})
    pin, pout = float(p.get("prompt") or 0) * 1e6, float(p.get("completion") or 0) * 1e6
    rows.append({"id": x["id"], "name": x.get("name"), "in_per_M": round(pin, 4), "out_per_M": round(pout, 4),
                 "free": pin == 0 and pout == 0, "context": x.get("context_length"),
                 "input": ",".join(x.get("architecture", {}).get("input_modalities") or []),
                 "output": ",".join(x.get("architecture", {}).get("output_modalities") or []),
                 "created": x.get("created")})
df = pd.DataFrame(rows).sort_values(["free", "out_per_M"], ascending=[False, True])
out = ROOT / "data" / "stats" / "openrouter_models.csv"
df.to_csv(out, index=False)
txt = df[df.output.str.contains("text")]
print(f"{len(df)} models, {int(df.free.sum())} free ({int(txt.free.sum())} with text output); table -> {out}")
