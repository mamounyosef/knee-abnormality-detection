"""shared_decode_check.py - decode-once must return EXACTLY what decoding every time returns.

inference.shared_decode walks the studies in chunks and shares each study's raw DICOM volume
across geometry groups through a scratch-disk cache. It changes the order of work, not the
work, so the arrays must be byte-identical. If they are not, every group after the first
would score different pixels from the first and nothing would say so.

Run:  python checks/shared_decode_check.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

PASS, FAIL = [], []


def ck(n, ok, d=""):
    (PASS if ok else FAIL).append(n)
    print("  %s %-58s %s" % ("PASS" if ok else "FAIL", n, d))


files = sorted((ROOT / "data" / "sample_dicoms").glob("*.dcm"))
ck("sample DICOMs present", len(files) >= 5, "%d files" % len(files))
if len(files) < 5:
    raise SystemExit(1)

print("\n1. without the cache")
os.environ.pop("KNEE_VOLCACHE_DIR", None)
t = time.perf_counter()
ref = sc.dicom_series_volume(files)
t_ref = time.perf_counter() - t
ck("decodes to a (slices, H, W) uint8 volume", ref.ndim == 3 and ref.dtype == np.uint8,
   "%s %s  %.2f s" % (ref.shape, ref.dtype, t_ref))

print("\n2. with the cache: first call decodes and writes, second reads")
d = Path(tempfile.mkdtemp(prefix="knee_volcache_test_"))
os.environ["KNEE_VOLCACHE_DIR"] = str(d)
try:
    t = time.perf_counter()
    first = sc.dicom_series_volume(files)
    t1 = time.perf_counter() - t
    written = list(d.glob("*.npy"))
    ck("the first call writes exactly one cache file", len(written) == 1,
       "%d file(s)" % len(written))
    ck("no torn temp file is left behind", not list(d.glob("*.tmp.npy")))
    t = time.perf_counter()
    second = sc.dicom_series_volume(files)
    t2 = time.perf_counter() - t
    ck("first call is byte-identical to the uncached decode", np.array_equal(first, ref))
    ck("CACHED call is byte-identical to the uncached decode", np.array_equal(second, ref))
    ck("dtype and shape preserved", second.dtype == ref.dtype and second.shape == ref.shape,
       "%s %s" % (second.shape, second.dtype))
    ck("the cached read is faster than decoding", t2 < t1,
       "decode %.3f s vs cached read %.3f s = %.0fx" % (t1, t2, t1 / max(t2, 1e-9)))

    print("\n3. the key is the FILE LIST, so order does not matter but content does")
    third = sc.dicom_series_volume(list(reversed(files)))
    ck("a reordered file list hits the same cache entry", len(list(d.glob("*.npy"))) == 1)
    ck("and returns the same volume", np.array_equal(third, ref))
    sub = sc.dicom_series_volume(files[:-2])
    ck("a DIFFERENT file list gets its own entry", len(list(d.glob("*.npy"))) == 2)
    ck("and is not handed the other series' volume",
       sub.shape[0] != ref.shape[0] or not np.array_equal(sub, ref),
       "%d vs %d slices" % (sub.shape[0], ref.shape[0]))

    print("\n4. a corrupt cache file is recovered from, not trusted")
    victim = next(iter(d.glob("*.npy")))
    victim.write_bytes(b"not a numpy file")
    again = sc.dicom_series_volume(files if victim.name else files)
    ck("a torn cache entry falls back to decoding",
       np.array_equal(again, ref) or np.array_equal(again, sub))
finally:
    os.environ.pop("KNEE_VOLCACHE_DIR", None)
    import shutil
    shutil.rmtree(d, ignore_errors=True)

print("\n5. without the env var nothing is cached")
d2 = Path(tempfile.mkdtemp(prefix="knee_volcache_off_"))
sc.dicom_series_volume(files)
ck("no cache file appears when KNEE_VOLCACHE_DIR is unset", not list(d2.glob("*")))
d2.rmdir()

print("\n6. the grouped blender wiring")
src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
ck("shared_decode is opt-in", 'bool(_inf.get("shared_decode"))' in src)
ck("it only chunks when there is more than one group to share with",
   "and len(group_plan) > 1" in src)
ck("timing ADDS across chunks", 'timing["member_s"][i] += gtiming["member_s"][slot]' in src)
ck("results land at the chunk's own positions", "allm[i, c0:c0 + k] = lg[slot]" in src)
ck("the cache is cleared per chunk and removed at the end",
   'cache_dir.glob("*.npy")' in src and "shutil.rmtree(cache_dir" in src)
# Placement is no longer memoised and does not need to be: nn.Module.to() moves a module
# IN PLACE, so the grouped loop hands back the same objects each chunk and the second chunk
# onward is a no-op. Memoising was only needed by the replicating design, which is gone.
ck("placement is in-place, so chunks do not re-copy models",
   "_shard_models" in src and "_REPLICA_CACHE" not in src)
ck("the split banner prints once, not once per chunk", '_BANNERS.get("shard")' in src)

print("\n" + "=" * 72)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("    FAILED: %s" % f)
sys.exit(1 if FAIL else 0)
