"""
runtime_budget.py - will this ensemble finish inside the 9 hour cap?

WHY THIS EXISTS
The competition caps a submission at 9 hours of GPU on the hidden test set. As the
ensemble grows the cap, not the validation score, decides what can ship, and a timeout
costs the whole submission. Estimating cost from TRAINING throughput was wrong by an
order of magnitude once already, so nothing here is estimated from training. Everything
is measured on Kaggle's own T4 and written down.

THE COST MODEL
Three parts, and they scale differently. Keeping them apart is the whole point:

    total = fixed + n_studies x (decode_s + sum over members of gpu_s)

    fixed     model load and DICOM index build. Paid once, negligible on a big test set.
    decode_s  per STUDY. Paid ONCE however many members ride along, because every member
              scores the same decoded batch. MEASURED at 3.3s/study on a T4, which is
              only ~15% of a nine-member run: adding members really does cost.
    gpu_s     per study PER MEMBER. The ONLY part that grows when a member is added.

Measured, the split is decode 3.3s/study against 18.1s/study of GPU across nine members,
so GPU dominates and member count really is close to proportional. Do not assume decode
carries the run: measure it.

WHERE THE NUMBERS COME FROM
Every kernel run writes timing.json (shared_code._write_timing_json) holding the measured
decode and per-member seconds per study. The COMMIT run's outputs are downloadable; the
leaderboard rerun's are not. So:

    per-study RATES   <- the commit run's timing.json, fetched from Kaggle
    hidden test size  <- solved for, from an OBSERVED rerun wall time you read off the
                         Kaggle submission page

    python runs/others/ensembling/runtime_budget.py watch <kernel-slug>
    python runs/others/ensembling/runtime_budget.py fetch <kernel-slug>
    python runs/others/ensembling/runtime_budget.py observe <kernel-slug> <hours>
    python runs/others/ensembling/runtime_budget.py calibrate <hours> <run> <run> ...
    python runs/others/ensembling/runtime_budget.py predict <run> <run> ...
    python runs/others/ensembling/runtime_budget.py show
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "runtime_ledger.json"
CAP_HOURS = 9.0
KAGGLE_USER = "mamounhussam"
COMPETITION = "rsna-knee-abnormality-detection"
# Under 25% margin is not worth the risk: the rates come from a handful of studies on a
# shared machine, and a timeout forfeits the whole submission rather than scoring badly.
SAFE_MARGIN = 0.25


def _load():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"runs": {}, "n_test_studies": None}


def _save(d):
    LEDGER.write_text(json.dumps(d, indent=1), encoding="utf-8")


def _ca_env():
    """Avast intercepts HTTPS on this PC, so kaggle needs the local CA bundle."""
    env = dict(os.environ)
    ca = Path.home() / ".certs" / "ca-bundle.pem"
    if ca.exists():
        env["REQUESTS_CA_BUNDLE"] = env["SSL_CERT_FILE"] = str(ca)
    return env


def fetch(slug):
    """Pull a finished kernel's timing.json into the ledger."""
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output",
                            KAGGLE_USER + "/" + slug, "-p", td],
                           capture_output=True, text=True, env=_ca_env())
        f = Path(td) / "timing.json"
        if not f.exists():
            print(r.stdout, r.stderr)
            raise SystemExit(
                slug + " has no timing.json.\n"
                "    Only runs made after the timing instrumentation landed have one.\n"
                "    Re-push the notebook to get it.")
        t = json.loads(f.read_text(encoding="utf-8"))
    d = _load()
    # keyed by study count too: a 3-study commit run and a 60-study STRESS run of the same
    # kernel measure different things and both are worth keeping
    key = "%s@%d" % (slug, t["n_studies"])
    row = d["runs"].setdefault(key, {})
    row.update({"n_studies_measured": t["n_studies"], "gpu": t.get("gpu", ""),
                "slug": slug, "stress": t["n_studies"] > 10,
                "decode_s_per_study": t["decode_s_per_study"],
                "members": {m["name"]: m["gpu_s_per_study"] for m in t["members"]}})
    _save(d)
    print("%s: decode %.2fs/study on %s"
          % (slug, t["decode_s_per_study"], t.get("gpu") or "?"))
    for m in t["members"]:
        print("    %6.2fs/study  %s" % (m["gpu_s_per_study"], m["name"]))


def observe(slug, hours):
    """Record the rerun wall time read off the Kaggle submission page.

    This is the only way to learn how many studies the hidden test set holds: the rates
    come from the commit run, the total comes from here, and the size is the ratio.
    """
    d = _load()
    row = d["runs"].setdefault(slug, {})
    row["observed_rerun_hours"] = float(hours)
    if "decode_s_per_study" in row:
        rate = row["decode_s_per_study"] + sum(row["members"].values())
        n = float(hours) * 3600.0 / rate
        row["implied_n_test_studies"] = n
        known = [r["implied_n_test_studies"] for r in d["runs"].values()
                 if "implied_n_test_studies" in r]
        d["n_test_studies"] = sum(known) / len(known)
        print("%s: %.2f h at %.2fs/study implies %.0f studies in the hidden test set"
              % (slug, hours, rate, n))
        print("    ledger estimate now %.0f studies (mean of %d observation(s))"
              % (d["n_test_studies"], len(known)))
    else:
        print("%s: recorded %.2f h. Run `fetch %s` for the rates." % (slug, hours, slug))
    _save(d)


def calibrate(hours, members):
    """Solve for the hidden test set size from a rerun you timed, and a member list.

    `observe` needs the timed run to be the same one that wrote timing.json. This does
    not: because the cost model is ADDITIVE, rates measured on any run cover any subset
    of its members. So one instrumented commit run of the big ensemble also prices the
    small one you already have a leaderboard time for.
    """
    d = _load()
    decode, rates = _rates(d)
    missing = [m for m in members if m not in rates]
    if missing:
        raise SystemExit("no measured rate for: " + ", ".join(missing))
    rate = decode + sum(rates[m] for m in members)
    n = float(hours) * 3600.0 / rate
    d["n_test_studies"] = n
    d.setdefault("calibration", {})
    d["calibration"] = {"hours": float(hours), "members": list(members),
                        "s_per_study": rate}
    _save(d)
    print("%.2f h at %.2fs/study over %d member(s) implies %.0f studies in the hidden "
          "test set" % (hours, rate, len(members), n))


def _rates(d):
    """(decode_s_per_study, {member: gpu_s_per_study}), taking the WORST seen of each.

    Not the average. A short commit run measures decode honestly but inflates GPU with
    one-time warm-up; a STRESS run measures GPU honestly (warm, and throttled the way a
    real multi-hour run will be) but reads the same few studies repeatedly, so its decode
    comes out of the page cache and is fiction. The maximum per stage keeps whichever run
    saw the true cost and errs high, which is the safe direction for a cap that forfeits
    the submission when breached.
    """
    dec, mem = [], {}
    for row in d["runs"].values():
        if "decode_s_per_study" in row:
            dec.append(row["decode_s_per_study"])
            for k, v in row["members"].items():
                mem.setdefault(k, []).append(v)
    if not dec:
        raise SystemExit("nothing measured yet - run `fetch <slug>` on a finished kernel")
    return max(dec), dict((k, max(v)) for k, v in mem.items())


def predict(members):
    d = _load()
    n = d.get("n_test_studies")
    if not n:
        raise SystemExit(
            "the hidden test set size is unknown - record one rerun first:\n"
            "    python runs/others/ensembling/runtime_budget.py observe <slug> <hours>")
    decode, rates = _rates(d)
    unknown = [m for m in members if m not in rates]
    if unknown:
        print("! never measured on Kaggle, EXCLUDED: " + ", ".join(unknown))
        print("  Their cost is unknown, so this number is a FLOOR, not a prediction.")
    gpu = sum(rates[m] for m in members if m in rates)
    rate = decode + gpu
    hours = n * rate / 3600.0
    print("\n%d member(s) over ~%.0f studies" % (len(members), n))
    print("  decode  %6.2fs/study   %4.1f%%  (shared, does not grow)"
          % (decode, decode / rate * 100))
    print("  gpu     %6.2fs/study   %4.1f%%" % (gpu, gpu / rate * 100))
    for m in members:
        if m in rates:
            print("          %6.2fs/study     %s" % (rates[m], m))
    used = hours / CAP_HOURS
    verdict = ("SAFE" if used <= 1 - SAFE_MARGIN else
               "TIGHT" if used < 1.0 else "OVER THE CAP")
    print("  -> %.2f h of the %.0f h cap (%.0f%%, %.0f%% margin)  [%s]"
          % (hours, CAP_HOURS, used * 100, (1 - used) * 100, verdict))
    if unknown:
        print("  (floor only: the unmeasured members above are not counted)")


def show():
    d = _load()
    if not d["runs"]:
        raise SystemExit("ledger is empty")
    print("hidden test set: %s studies\n"
          % (("%.0f" % d["n_test_studies"]) if d.get("n_test_studies") else "?"))
    print("%-46s%8s%9s%12s" % ("kernel", "members", "s/study", "observed h"))
    for slug, r in d["runs"].items():
        rate = r.get("decode_s_per_study", 0) + sum(r.get("members", {}).values())
        obs = r.get("observed_rerun_hours")
        print("%-46s%8d%9.2f%12s"
              % (slug, len(r.get("members", {})), rate,
                 ("%.2f" % obs) if obs is not None else "-"))


def watch(slug, poll_s=120):
    """Time a leaderboard rerun by polling, so nobody has to read a web page.

    Kaggle's API exposes a submission's submit time and its status, but never its
    duration. Polling closes that gap: the newest submission's `date` is when the rerun
    was queued, and the moment `status` leaves PENDING the rerun is done, so the
    difference is its wall time. That includes queue wait, which makes this an UPPER
    bound - a conservative direction for a cap we must not breach.

    Records the result and re-solves the hidden test set size if the rates are known.
    """
    import datetime as _dt
    import time as _time

    import requests
    from requests.auth import HTTPBasicAuth

    cred = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    auth = HTTPBasicAuth(cred["username"], cred["key"])
    env = _ca_env()
    ca = env.get("REQUESTS_CA_BUNDLE", True)

    def newest():
        r = requests.get("https://www.kaggle.com/api/v1/competitions/submissions/list",
                         params={"id": COMPETITION, "page": 1}, auth=auth,
                         verify=ca, timeout=60)
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else None

    sub = newest()
    if sub is None:
        raise SystemExit("no submissions found")
    started = _dt.datetime.fromisoformat(sub["date"].replace("Z", "+00:00"))
    print("watching submission %s queued %s" % (sub.get("ref"), started))
    while True:
        st = str(sub.get("status", ""))
        el = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds() / 3600.0
        print("  %5.2f h  %s" % (el, st), flush=True)
        if "PENDING" not in st.upper():
            break
        _time.sleep(poll_s)
        sub = newest()
    hours = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds() / 3600.0
    print("done in ~%.2f h (queue included), score %s"
          % (hours, sub.get("publicScore")))
    d = _load()
    row = d["runs"].setdefault(slug, {})
    row["observed_rerun_hours"] = hours
    row["public_score"] = sub.get("publicScore")
    _save(d)
    observe(slug, hours)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "watch":
        watch(args[0], int(args[1]) if len(args) > 1 else 120)
    elif cmd == "fetch":
        fetch(args[0])
    elif cmd == "observe":
        observe(args[0], float(args[1]))
    elif cmd == "calibrate":
        calibrate(float(args[0]), args[1:])
    elif cmd == "predict":
        predict(args)
    elif cmd == "show":
        show()
    else:
        raise SystemExit(__doc__)
