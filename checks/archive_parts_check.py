r"""archive_parts_check.py - prove PartsFile is byte-identical to the file it replaces.

Every downstream consumer opens the archive with zipfile and trusts it completely. If the
stitching is off by a byte, or a seek lands in the wrong part, nothing complains here - it
surfaces days later as a corrupt DICOM deep inside preprocessing, on a volume that took
hours to fill. So this builds a real zip, splits it into parts, and checks the stitched view
against the original: random reads, every seek mode, and a full zipfile round trip.

CPU only, no Modal and no network.

    python checks/archive_parts_check.py
"""

import io
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modal_scripts"))
import archive_parts as ap  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print("  %-56s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


COMP = "testcomp"
tmp = tempfile.mkdtemp(prefix="parts_check_")
try:
    # --- build a real zip with awkward sizes -------------------------------------
    whole = os.path.join(tmp, f"{COMP}.zip")
    rng = random.Random(0)
    payloads = {}
    with zipfile.ZipFile(whole, "w", zipfile.ZIP_STORED) as zf:
        for i in range(40):
            # incompressible data, so ZIP_STORED sizes are predictable and large enough
            # to span several parts once PART_SIZE is shrunk below
            body = bytes(rng.getrandbits(8) for _ in range(rng.randint(1000, 40000)))
            payloads[f"m{i:03d}.bin"] = body
            zf.writestr(f"m{i:03d}.bin", body)
    raw = Path(whole).read_bytes()
    print(f"\nbuilt a {len(raw)} byte zip with {len(payloads)} members")

    # shrink PART_SIZE so the test actually spans many parts
    ap.PART_SIZE = 7000
    PS = ap.PART_SIZE

    # --- split it into parts exactly as the downloader would ---------------------
    d = ap.parts_dir(tmp, COMP)
    os.makedirs(d, exist_ok=True)
    n_parts = 0
    for off in range(0, len(raw), PS):
        Path(ap.part_path(tmp, COMP, n_parts)).write_bytes(raw[off:off + PS])
        n_parts += 1
    print(f"split into {n_parts} parts of {PS} bytes")

    # --- 1. bookkeeping ----------------------------------------------------------
    # Remove the single archive FIRST: with both present this is the base+parts layout
    # (covered in section 6), and parts_bytes would correctly count the file twice.
    os.remove(whole)
    print("\nbookkeeping (parts only)")
    check("parts_bytes equals the original length",
          ap.parts_bytes(tmp, COMP) == len(raw),
          "%d vs %d" % (ap.parts_bytes(tmp, COMP), len(raw)))
    check("contiguous_parts finds every part",
          len(ap.contiguous_parts(tmp, COMP)) == n_parts)

    # --- 2. the stitched view is byte-identical ----------------------------------
    print("\nstitched reads (single .zip absent, so parts are used)")
    with ap.open_archive(tmp, COMP) as fh:
        check("open_archive returned the parts reader", isinstance(fh, ap.PartsFile))
        check("sequential read() reproduces the file exactly", fh.read() == raw)

        # random slices, deliberately crossing part boundaries
        bad = 0
        for _ in range(400):
            a = rng.randrange(0, len(raw))
            n = rng.randint(1, 3 * PS)
            fh.seek(a)
            if fh.read(n) != raw[a:a + n]:
                bad += 1
        check("400 random cross-boundary reads match", bad == 0, "%d mismatched" % bad)

        # exact part boundaries are the classic off-by-one
        bad = 0
        for i in range(n_parts + 1):
            for delta in (-1, 0, 1):
                a = i * PS + delta
                if 0 <= a < len(raw):
                    fh.seek(a)
                    if fh.read(64) != raw[a:a + 64]:
                        bad += 1
        check("reads at every part boundary +/- 1 byte match", bad == 0, "%d bad" % bad)

        # whence modes: zipfile reads the central directory with whence=2
        fh.seek(-22, 2)
        check("seek(whence=2) lands at the end-of-central-directory", fh.read() == raw[-22:])
        fh.seek(100)
        fh.seek(50, 1)
        check("seek(whence=1) is relative", fh.read(10) == raw[150:160])
        fh.seek(0, 2)
        check("tell() at the end is the total size", fh.tell() == len(raw))
        fh.seek(len(raw))
        check("reading past the end returns empty", fh.read(10) == b"")

    # --- 3. zipfile accepts it and every member round trips ----------------------
    print("\nzipfile over the parts")
    with ap.open_archive(tmp, COMP) as fh:
        with zipfile.ZipFile(fh) as zf:
            names = zf.namelist()
            check("namelist matches", sorted(names) == sorted(payloads))
            bad = [n for n in names if zf.read(n) != payloads[n]]
            check("every member's bytes match", not bad, "%d wrong" % len(bad))
            check("testzip finds no CRC error", zf.testzip() is None)

    # --- 4. a short part in the middle truncates the view ------------------------
    # This is the safety property that makes resume correct: a part interrupted mid-write
    # must invalidate everything after it, not be silently stitched over.
    print("\ninterrupted-part safety")
    victim = ap.part_path(tmp, COMP, 2)
    Path(victim).write_bytes(raw[2 * PS:2 * PS + 11])       # short, not the last part
    good = ap.contiguous_parts(tmp, COMP)
    check("a short middle part stops the contiguous run there",
          len(good) == 3, "kept %d parts, expected 3" % len(good))
    check("parts_bytes drops to the trustworthy prefix",
          ap.parts_bytes(tmp, COMP) == 2 * PS + 11,
          "%d" % ap.parts_bytes(tmp, COMP))

    # --- 5. a whole .zip and no parts (old volumes keep working) -----------------
    print("\nbackward compatibility: a single .zip, no parts")
    shutil.rmtree(ap.parts_dir(tmp, COMP))
    Path(whole).write_bytes(raw)
    with ap.open_archive(tmp, COMP) as fh:
        check("a single .zip with no parts opens directly",
              not isinstance(fh, ap.PartsFile))
        check("and reads correctly", fh.read() == raw)
    check("archive_size reports the whole file", ap.archive_size(tmp, COMP) == len(raw))

    # --- 6. BASE + parts: the session-23 case ------------------------------------
    # A download that began under the single-file code and finished under the split one.
    # The base must come FIRST and the parts must continue exactly where it ends; get the
    # order or the offset wrong and the zip is silently scrambled.
    print("\nbase archive + parts (a download that spans both designs)")
    BASE = 300123                                   # deliberately not a multiple of PART_SIZE
    Path(whole).write_bytes(raw[:BASE])
    os.makedirs(ap.parts_dir(tmp, COMP), exist_ok=True)
    tail = raw[BASE:]
    n2 = 0
    for off in range(0, len(tail), PS):
        Path(ap.part_path(tmp, COMP, n2)).write_bytes(tail[off:off + PS])
        n2 += 1
    check("base_size is the existing archive", ap.base_size(tmp, COMP) == BASE)
    check("parts_bytes counts base plus parts",
          ap.parts_bytes(tmp, COMP) == len(raw), "%d vs %d"
          % (ap.parts_bytes(tmp, COMP), len(raw)))
    with ap.open_archive(tmp, COMP) as fh:
        check("stitched base+parts reproduces the file exactly", fh.read() == raw)
        bad = 0
        for _ in range(300):
            a = rng.randrange(0, len(raw))
            n = rng.randint(1, 3 * PS)
            fh.seek(a)
            if fh.read(n) != raw[a:a + n]:
                bad += 1
        check("300 random reads across the base/part seam match", bad == 0,
              "%d mismatched" % bad)
        # the seam itself is the dangerous byte
        for delta in (-2, -1, 0, 1, 2):
            fh.seek(BASE + delta)
            if fh.read(32) != raw[BASE + delta:BASE + delta + 32]:
                bad += 1
        check("reads straddling the exact seam match", bad == 0)
    with ap.open_archive(tmp, COMP) as fh:
        with zipfile.ZipFile(fh) as zf:
            check("zipfile reads every member across the seam",
                  not [n for n in zf.namelist() if zf.read(n) != payloads[n]])
            check("testzip finds no CRC error across the seam", zf.testzip() is None)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + "=" * 70)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("all checks passed - the stitched view is byte-identical and zipfile-safe")
