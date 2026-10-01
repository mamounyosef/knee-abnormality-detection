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

    print("\n3b. real Kaggle logs and the projections to the 946-study hidden set")
    fx = ROOT / "checks" / "fixtures"
    real_pass = (fx / "stress_pass_ten_mii5_inference_log.txt").read_text(encoding="utf-8")
    real_sha = "97eb16209321d135ba76cc187ad0db1eecde3aea"
    rp = with_fake_output(real_pass, sc.verify_stress_output, "u", "nb", 14, 128, real_sha)
    ck("the REAL passing stress log (1 Oct, ten + 5 MII folds) still passes", rp == [], str(rp))
    real_fail = (fx / "stress_fail_ram_ten_mii5_inference_log.txt").read_text(encoding="utf-8")
    rf = with_fake_output(real_fail, sc.verify_stress_output, "u", "nb", 14, 128, real_sha)
    ck("the REAL failing one (RAM 97%, dead loaders) fails", any("RAM" in p for p in rf)
       and any("loader" in p for p in rf), str([p[:50] for p in rf]))
    slow = good_log(14, 128, sha).replace("in 1m over", "in 2h10m00s over")
    ck("a run projecting past 8 h on the hidden set fails", any("projected" in p for p in v(slow)),
       str([p[:60] for p in v(slow)]))
    steep = good_log(14, 128, sha) + "\n" + "\n".join(
        "group 4 chunk at %d: group end [m]: RAM used %.1f / 33.7 GB; cuda:0 free 15 GB" % (c, r)
        for c, r in ((0, 18.0), (32, 20.0), (64, 22.0), (96, 24.0)))
    ck("RAM rising 2 GB per chunk fails (it would cross 90% on the hidden set)",
       any("still rising" in p for p in v(steep)), str([p[:60] for p in v(steep)]))

    print("\n4. the normal submit check also demands this bundle's engine")
    csv = "StudyInstanceUID,a,b\ns1,0.1,0.9\ns2,0.8,0.2\n"
    ok = with_fake_output(good_log(14, 3, sha, chunks=1), sc.verify_kernel_output, "u", "nb", 14,
                          sub_csv=csv, code_sha=sha)
    ck("matching engine: no problem", ok == [], str(ok))
    bad = with_fake_output(good_log(14, 3, "c" * 40, chunks=1), sc.verify_kernel_output, "u", "nb", 14,
                           sub_csv=csv, code_sha=sha)
    ck("another engine: refused", any("previous dataset" in p for p in bad), str(bad))

    print("\n5. submitting from a stress stamp, with no weights on this PC")
    real_dir = sc.STRESS_PASSES
    tmpd = Path(tempfile.mkdtemp(prefix="knee_stamps_"))
    sc.STRESS_PASSES = tmpd
    fp = "f" * 40
    nbd = tmpd / ("%s_notebook" % fp)
    nbd.mkdir()
    guarded = '{"cells": ["STRESS = 128", "ENGINE MISMATCH %s"]}' % sha
    (nbd / "submit.ipynb").write_text(guarded, encoding="utf-8")
    (nbd / "kernel-metadata.json").write_text('{"id": "u/nb"}')
    manifest = {"m0_model.pt": 100, "m1_model.pt": 200, "shared_code.py": 10}
    (tmpd / ("%s.json" % fp)).write_text(json.dumps({
        "fingerprint": fp, "engine_sha1": sha, "notebook": "u/nb", "dataset": "u/ds",
        "n_members": 2, "manifest": manifest}), encoding="utf-8")
    pushed = {}

    def fake_kaggle(listing):
        def f(*a, **k):
            if a[:2] == ("datasets", "files"):
                return "name,size,creationDate\n" + "".join("%s,%d,2026\n" % kv for kv in listing.items())
            if a[:2] == ("kernels", "push"):
                pushed["nb"] = (Path(a[a.index("-p") + 1]) / "submit.ipynb").read_text(encoding="utf-8")
                return "Kernel version 9 successfully pushed"
            if a[:2] == ("kernels", "status"):
                return 'status "KernelWorkerStatus.COMPLETE"'
            if a[:2] == ("kernels", "output"):
                d = Path(a[a.index("-p") + 1])
                (d / "inference_log.txt").write_text(good_log(2, 3, sha, chunks=1), encoding="utf-8")
                (d / "submission.csv").write_text(csv, encoding="utf-8")
                return ""
            if a[:2] == ("competitions", "submit"):
                pushed["submitted"] = a
                return "Successfully submitted"
            raise AssertionError(a)
        return f
    real = sc._kaggle
    try:
        sc._kaggle = fake_kaggle(dict(manifest, **{"m1_model.pt": 201}))
        try:
            sc.submit_from_stress_stamp(fp[:12], "msg", poll_secs=0)
            ck("a dataset that differs from the stamp is refused", False)
        except SystemExit as e:
            ck("a dataset that differs from the stamp is refused", "not the stress-tested" in str(e))
        ck("and nothing was pushed or submitted", not pushed, str(pushed.keys()))
        sc._kaggle = fake_kaggle(manifest)
        sc.submit_from_stress_stamp(fp[:12], "msg", poll_secs=0)
        ck("an exact match pushes the stamped notebook with STRESS = 0",
           "STRESS = 0" in pushed.get("nb", "") and "STRESS = 128" not in pushed.get("nb", ""))
        ck("and submits that kernel version", "submitted" in pushed and "-v" in pushed["submitted"]
           and "9" in pushed["submitted"])
        pushed.clear()
        bad_log_kaggle = fake_kaggle(manifest)

        def stale(*a, **k):
            if a[:2] == ("kernels", "output"):
                d = Path(a[a.index("-p") + 1])
                (d / "inference_log.txt").write_text(good_log(2, 3, "d" * 40, chunks=1), encoding="utf-8")
                (d / "submission.csv").write_text(csv, encoding="utf-8")
                return ""
            return bad_log_kaggle(*a, **k)
        sc._kaggle = stale
        sc.submit_from_stress_stamp(fp[:12], "msg", poll_secs=0)
        ck("a kernel that ran another engine is NOT submitted", "submitted" not in pushed)
        old = tmpd / ("%s.json" % ("e" * 40))
        old.write_text(json.dumps({"fingerprint": "e" * 40, "engine_sha1": sha, "notebook": "u/nb",
                                   "n_members": 2}), encoding="utf-8")
        try:
            sc.submit_from_stress_stamp("e" * 12, "msg", poll_secs=0)
            ck("a stamp without a manifest is refused", False)
        except SystemExit as e:
            ck("a stamp without a manifest is refused", "predates" in str(e))
        (nbd / "submit.ipynb").write_text('{"cells": ["STRESS = 128"]}', encoding="utf-8")
        pushed.clear()
        sc._kaggle = fake_kaggle(manifest)
        try:
            sc.submit_from_stress_stamp(fp[:12], "msg", poll_secs=0)
            ck("a saved notebook without the engine guard is refused", False)
        except SystemExit as e:
            ck("a saved notebook without the engine guard is refused", "engine guard" in str(e))
        ck("and nothing was pushed", "nb" not in pushed)
    finally:
        sc._kaggle = real
        sc.STRESS_PASSES = real_dir

    print("\n6. Kaggle mounting the PREVIOUS dataset version (stress run 3, 1 Oct)")
    for n_bad, want_ok, want_push in ((1, True, 2), (9, False, 3)):
        n = {"push": 0}

        def mounts_old(*a, **k):
            if a[:2] == ("kernels", "push"):
                n["push"] += 1
                return "Kernel version %d successfully pushed" % n["push"]
            if a[:2] == ("kernels", "status"):
                return 'status "KernelWorkerStatus.ERROR"' if n["push"] <= n_bad else 'status "COMPLETE"'
            if a[:2] == ("kernels", "output"):
                if n["push"] <= n_bad:
                    (Path(a[a.index("-p") + 1]) / "engine_mismatch.txt").write_text("b" * 40)
                return ""
            raise AssertionError(a)
        sc._kaggle = mounts_old
        try:
            ver, ok_ = sc._push_and_wait("u", "nb", Path("."), True, 0, engine_guard=True,
                                         retries=3, retry_wait_s=0)
        finally:
            sc._kaggle = real
        ck("old engine %s: %d push(es), ok=%s" % ("once" if n_bad == 1 else "every time", want_push, want_ok),
           (ok_, n["push"]) == (want_ok, want_push), "got ok=%s pushes=%d" % (ok_, n["push"]))
    # a real kernel error that is NOT the engine guard is not retried
    sc._kaggle = lambda *a, **k: ("Kernel version 1 successfully pushed" if a[1] == "push" else
                                  'status "ERROR"' if a[1] == "status" else "")
    try:
        ver, ok_ = sc._push_and_wait("u", "nb", Path("."), True, 0, engine_guard=True, retry_wait_s=0)
    finally:
        sc._kaggle = real
    ck("an ordinary kernel error is reported once, not retried", ok_ is False)

    print("\n7. the dataset wait demands the NEW version's file sizes, not just its names")
    b = fake_bundle(code="new engine, longer\n")
    w = b / "weights"
    loc = {f.name: f.stat().st_size for f in w.iterdir() if f.name != "dataset-metadata.json"}

    def listing(sizes):
        rows = "".join("%s,%d,2026\n" % kv for kv in sizes.items())
        return lambda *a, **k: "name,size,creationDate\n" + rows
    sc._kaggle = listing(dict(loc, **{"shared_code.py": 5}))
    try:
        sc._wait_for_dataset("u", "ds", tries=2, secs=0, expect_models=2, local_dir=w)
        ck("the old version (same names, old engine size) is NOT accepted", False)
    except SystemExit as e:
        ck("the old version (same names, old engine size) is NOT accepted", "shared_code.py" in str(e))
    finally:
        sc._kaggle = real
    sc._kaggle = listing(loc)
    try:
        sc._wait_for_dataset("u", "ds", tries=2, secs=0, expect_models=2, local_dir=w)
        ck("the new version (every size matches) is accepted", True)
    except SystemExit as e:
        ck("the new version (every size matches) is accepted", False, str(e)[:80])
    finally:
        sc._kaggle = real

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
