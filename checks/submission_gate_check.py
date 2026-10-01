"""submission_gate_check.py - an ensemble cannot be submitted without a passing STRESS run of the
exact same bundle, and a STRESS run only passes when every chunk, group and member scored.

Why: the ten + 5 MII folds ensemble passed the normal 3-study commit run and then scored 0.500
on the hidden set (1 Oct 2026), because 3 studies are ONE decode chunk and the bug lived
between chunks. Nothing here touches Kaggle: kaggle_publish must refuse BEFORE uploading, and
the log readers are fed fake kernel outputs.

    python checks/submission_gate_check.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

PASS, FAIL = [], []


def ck(n, ok, d=""):
    (PASS if ok else FAIL).append(n)
    print("  %s %-66s %s" % ("PASS" if ok else "FAIL", n, d))


def fake_bundle(n_members=2, stress=0, code="x = 1\n", wsize=100):
    a = Path(tempfile.mkdtemp(prefix="knee_gate_"))
    (a / "weights").mkdir()
    (a / "notebook").mkdir()
    (a / "weights" / "shared_code.py").write_text(code, encoding="utf-8")
    (a / "weights" / "dataset-metadata.json").write_text(json.dumps({"id": "u/fake-ds"}))
    (a / "notebook" / "kernel-metadata.json").write_text(json.dumps({"id": "u/fake-nb"}))
    (a / "notebook" / "submit.ipynb").write_text('{"cells": ["STRESS = %d"]}' % stress, encoding="utf-8")
    for i in range(n_members):
        (a / "weights" / ("m%d_model.pt" % i)).write_bytes(b"0" * wsize)
        (a / "weights" / ("m%d_config.json" % i)).write_text('{"i": %d}' % i)
        (a / "weights" / ("m%d_def.py" % i)).write_text("def build_model(c): pass\n")
    return a


def good_log(n_members, n_studies, sha, chunks=4, ram=24.0):
    names = ["mem%d" % i for i in range(n_members)]
    return "\n".join(["scored %d studies with %d/%d members in 1m over 2 decode pass(es)"
                      % (n_studies, n_members, n_members),
                      "members: " + ", ".join(names), "gpus used: 2",
                      "code sha1 %s" % sha, "decode chunks %d of up to 32 studies" % chunks,
                      "group 0 chunk at 0: group start [mem0]: RAM used 16.3 / 33.7 GB; cuda:0 free 15.1 GB",
                      "group 0 chunk at 0: group end [mem0]: RAM used %.1f / 33.7 GB; cuda:0 free 15.5 GB" % ram,
                      "member logit range -3 .. 5", "", "runtime breakdown (seconds per study)",
                      "  decode     5.0   paid once per study"]
                     + ["  gpu        1.200   %s" % nm for nm in names])


def with_fake_output(log_text, fn, *args, sub_csv=None, **kw):
    """Run fn with sc._kaggle replaced by a writer of a fake kernel output."""
    real = sc._kaggle

    def fake(*a, **k):
        if a[:2] == ("kernels", "output"):
            d = Path(a[a.index("-p") + 1])
            if log_text is not None:
                (d / "inference_log.txt").write_text(log_text, encoding="utf-8")
            if sub_csv is not None:
                (d / "submission.csv").write_text(sub_csv, encoding="utf-8")
            return ""
        raise AssertionError("no other Kaggle call is allowed here: %r" % (a,))
    sc._kaggle = fake
    try:
        return fn(*args, **kw)
    finally:
        sc._kaggle = real


def main():
    print("\n1. the fingerprint")
    a0, a1 = fake_bundle(stress=0), fake_bundle(stress=128)
    ck("a STRESS build and the real build of one bundle match",
       sc.bundle_fingerprint(a0) == sc.bundle_fingerprint(a1))
    ck("changing the engine code changes it", sc.bundle_fingerprint(a0)
       != sc.bundle_fingerprint(fake_bundle(code="x = 2\n")))
    ck("changing a weight file's size changes it", sc.bundle_fingerprint(a0)
       != sc.bundle_fingerprint(fake_bundle(wsize=101)))
    ck("adding a member changes it", sc.bundle_fingerprint(a0)
       != sc.bundle_fingerprint(fake_bundle(n_members=3)))

    print("\n2. kaggle_publish refuses BEFORE any upload")
    real = sc._kaggle
    calls = []
    sc._kaggle = lambda *a, **k: calls.append(a) or ""
    try:
        try:
            sc.kaggle_publish(fake_bundle(code="unique-1\n"), "u", "fake-ds", "fake-nb", "m", submit=True)
            ck("an ensemble with no stress stamp is refused", False, "it went ahead")
        except SystemExit as e:
            ck("an ensemble with no stress stamp is refused", "REFUSING TO SUBMIT" in str(e))
        ck("and nothing was uploaded or pushed", calls == [], str(calls[:1]))
        try:
            sc.kaggle_publish(fake_bundle(stress=128), "u", "fake-ds", "fake-nb", "m",
                              submit=True, stress=128)
            ck("a STRESS build with SUBMIT on is refused", False)
        except SystemExit as e:
            ck("a STRESS build with SUBMIT on is refused", "SUBMIT = False" in str(e))
    finally:
        sc._kaggle = real
    # a solo bundle (one member) keeps the old path: no stamp needed
    solo_fp_needed = True
    try:
        sc._kaggle = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("upload reached"))
        sc.kaggle_publish(fake_bundle(n_members=1), "u", "fake-ds", "fake-nb", "m", submit=True)
    except RuntimeError as e:
        solo_fp_needed = "upload reached" not in str(e)
    except SystemExit:
        solo_fp_needed = True
    finally:
        sc._kaggle = real
    ck("a solo bundle is not blocked by the stress gate", not solo_fp_needed)

    print("\n3. verify_stress_output")
    sha = "a" * 40
    v = lambda log, n=14, s=128: with_fake_output(log, sc.verify_stress_output, "u", "nb", n, s, sha)  # noqa: E731
    ck("a clean multi-chunk log passes", v(good_log(14, 128, sha)) == [], str(v(good_log(14, 128, sha))))
    ck("a ONE-chunk run fails", any("chunk" in p for p in v(good_log(14, 128, sha, chunks=1))))
    ck("a stale engine (another code sha1) fails", any("previous dataset" in p for p in v(good_log(14, 128, "b" * 40))))
    ck("a log with no code sha1 fails", any("old dataset" in p for p in
                                            v(good_log(14, 128, sha).replace("code sha1 " + sha, ""))))
    ck("a dead data loader fails", v(good_log(14, 128, sha) + "\ngroup 2 chunk at 32: LOADER DIED at study 0 (X: y)") != [])
    ck("a failed batch fails", v(good_log(14, 128, sha) + "\nbatch at study 4 failed (OOM) -> 0.5") != [])
    ck("a member contributing nothing fails", v(good_log(14, 128, sha) + "\nMEMBERS CONTRIBUTING NOTHING: mem3") != [])
    ck("a member at 0.000 s fails", v(good_log(14, 128, sha).replace("1.200   mem3", "0.000   mem3")) != [])
    ck("too few studies scored fails", v(good_log(14, 96, sha)) != [])
    ck("13 of 14 members fails", v(good_log(14, 128, sha).replace("14/14", "13/14")) != [])
    ck("no log at all fails", v(None) != [])
    ck("host RAM above 85% fails", any("RAM" in p for p in v(good_log(14, 128, sha, ram=32.7))))
    ck("host RAM at 71% passes", v(good_log(14, 128, sha, ram=24.0)) == [])
    no_ram = chr(10).join(l for l in good_log(14, 128, sha).splitlines() if "RAM used" not in l)
    ck("a log with no RAM figures fails", any("RAM" in p for p in v(no_ram)))

    print("\n4. the normal submit check also demands this bundle's engine")
    csv = "StudyInstanceUID,a,b\ns1,0.1,0.9\ns2,0.8,0.2\n"
    ok = with_fake_output(good_log(14, 3, sha, chunks=1), sc.verify_kernel_output, "u", "nb", 14,
                          sub_csv=csv, code_sha=sha)
    ck("matching engine: no problem", ok == [], str(ok))
    bad = with_fake_output(good_log(14, 3, "c" * 40, chunks=1), sc.verify_kernel_output, "u", "nb", 14,
                           sub_csv=csv, code_sha=sha)
    ck("another engine: refused", any("previous dataset" in p for p in bad), str(bad))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
