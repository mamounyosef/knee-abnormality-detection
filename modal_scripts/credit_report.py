"""credit_report.py - how much of each workspace's starting credit is left?

Modal has no "balance" command, only `modal billing report`, which returns SPEND over a
date range. Each workspace starts with a fixed grant, so remaining = grant - total spend
since the account was created. Start the range well before the first run and it covers
everything.

    python modal_scripts/credit_report.py 21 102 103
    python modal_scripts/credit_report.py 21 --start 2026-08-01 --grant 30
"""

import argparse
import datetime as dt
import io
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
CERT = str(HOME / ".certs" / "ca-bundle.pem")


def windows(start, end, days=30):
    """Modal refuses a daily report spanning more than 31 days, so walk it in chunks."""
    a = dt.date.fromisoformat(start)
    b = dt.date.fromisoformat(end)
    while a <= b:
        z = min(a + dt.timedelta(days=days - 1), b)
        yield a.isoformat(), z.isoformat()
        a = z + dt.timedelta(days=1)


def billing(session, start, end):
    """Billing JSON for one workspace over one window, or None if the call failed."""
    e = dict(os.environ)
    e["MODAL_CONFIG_PATH"] = str(HOME / (".modal_session%s.toml" % session))
    e["PYTHONIOENCODING"] = "utf-8"
    if os.path.exists(CERT):
        e["REQUESTS_CA_BUNDLE"] = e["SSL_CERT_FILE"] = CERT
    r = subprocess.run([sys.executable, "-m", "modal", "billing", "report",
                        "--start", start, "--end", end, "--json"],
                       capture_output=True, text=True, env=e, timeout=180,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if not out:
        return None, (r.stderr or "").strip()[:200]
    try:
        return json.loads(out), None
    except json.JSONDecodeError:
        # some versions print a banner before the payload
        i = out.find("{")
        j = out.find("[")
        k = min(x for x in (i, j) if x >= 0) if (i >= 0 or j >= 0) else -1
        if k < 0:
            return None, out[:200]
        try:
            return json.loads(out[k:]), None
        except json.JSONDecodeError as ex:
            return None, "%s: %s" % (type(ex).__name__, out[:200])


COST_KEYS = ("cost", "total_cost", "amount", "total", "cost_usd", "spend")
DESC_KEYS = ("description", "object id", "app", "name")


def _num(v):
    """Modal returns Cost as a STRING ('0.15912163'), so accept both."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def total_cost(payload):
    """Sum every dollar figure, and attribute it per app. Keys are case-insensitive."""
    rows, by_app = [], {}

    def walk(o):
        if isinstance(o, dict):
            low = {str(k).lower(): v for k, v in o.items()}
            hit = next((_num(low[k]) for k in COST_KEYS
                        if k in low and _num(low[k]) is not None), None)
            if hit is not None:
                rows.append(hit)
                name = next((str(low[k]) for k in DESC_KEYS if low.get(k)), "?")
                by_app[name] = by_app.get(name, 0.0) + hit
                return            # do not double count children of a costed row
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(payload)
    return sum(rows), len(rows), by_app


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("sessions", nargs="+")
    p.add_argument("--start", default="2026-07-01", help="far enough back to cover everything")
    p.add_argument("--end", default=None, help="defaults to today")
    p.add_argument("--grant", type=float, default=30.0, help="starting credit per workspace")
    p.add_argument("--raw", action="store_true", help="dump the JSON for one session")
    p.add_argument("--detail", action="store_true", help="break the spend down per app")
    a = p.parse_args()

    end = a.end or dt.date.today().isoformat()
    print("%-10s %12s %12s %12s   %s" % ("session", "grant", "spent", "left", "note"))
    print("-" * 64)
    tops = {}
    for s in a.sessions:
        spent, rows, errs, apps = 0.0, 0, [], {}
        for w0, w1 in windows(a.start, end):
            data, err = billing(s, w0, w1)
            if data is None:
                errs.append("%s..%s %s" % (w0, w1, err))
                continue
            if a.raw:
                print(json.dumps(data, indent=2)[:3000])
            c, n, by = total_cost(data)
            spent += c
            rows += n
            for k, v in by.items():
                apps[k] = apps.get(k, 0.0) + v
        if errs and rows == 0:
            print("%-10s %12s %12s %12s   %s" % (s, "-", "-", "-", "ERROR: " + errs[0]))
            continue
        tops[s] = apps
        note = "%d rows" % rows + ("  (%d window(s) failed)" % len(errs) if errs else "")
        print("%-10s %11.2f$ %11.2f$ %11.2f$   %s" % (s, a.grant, spent, a.grant - spent, note))
    print("\nspend is %s..%s; 'left' assumes each workspace started at %.2f$"
          % (a.start, end, a.grant))

    if a.detail:
        for s, apps in tops.items():
            print("\n--- session %s, biggest spenders ---" % s)
            for k, v in sorted(apps.items(), key=lambda kv: -kv[1])[:12]:
                print("  %8.2f$  %s" % (v, k))


if __name__ == "__main__":
    main()
