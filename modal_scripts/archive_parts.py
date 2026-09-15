r"""archive_parts.py - the competition archive as many small files instead of one huge one.

WHY THIS EXISTS
---------------
Downloading the 247 GiB archive as a SINGLE file on a Modal volume does not work, and the
failure gets worse the further it gets. Two independent causes, both measured on session 23:

1. `vol.commit()` costs O(FILE SIZE), not O(bytes appended). Modal finds changed blocks by
   hashing, so appending 256 MiB to a 233 GiB file re-hashes 233 GiB. Measured: a single
   commit at 233 GiB ran **over 5 minutes without returning**. Early in a download, when the
   file is small, commits are instant - which is why this looks fine at first and then
   degrades into what appears to be a network problem. It also means a container spends most
   of its life inside commit, unable to answer Modal's cancellation signal, so it gets killed
   ("failed to respond to cancellation for too long: 30 seconds").

2. Kaggle will not sustain an open-ended range response. `Range: bytes=<have>-` spanning the
   last 14 GiB delivered exactly one 32 MiB chunk in 1.0 s and then stalled: 0.06 GiB in
   319 s. The same bytes requested as a BOUNDED range completed at 14.6 MB/s.

Together those cost 54 relaunches and $1.53 to move a download 0.2 GiB.

THE FIX
-------
Fetch bounded byte windows into SEPARATE part files. Only the newest part is modified, so a
commit hashes one part, not the whole archive - the cost stops growing. Each part is fetched
with its own bounded request, so no connection is ever asked to stream more than one part.

Nothing is ever concatenated: `open_archive()` returns a seekable file-like object that
stitches the parts back together on read, which `zipfile.ZipFile` accepts directly. That
avoids writing a second 247 GiB copy (and the one huge commit that would need).

Backward compatible: volumes that already hold a single `<comp>.zip` (sessions 19, 20, 21)
keep working untouched, because `open_archive()` prefers a whole archive when it finds one.

    from archive_parts import open_archive, parts_bytes, PART_SIZE
    with open_archive("/data", "rsna-knee-abnormality-detection") as fh:
        with zipfile.ZipFile(fh) as zf:
            ...
"""

import io
import os
import re

# One part per bounded HTTP request, and one volume commit per part. 2 GiB keeps the number
# of parts manageable (~124 for this competition) while keeping each commit cheap and losing
# at most one part's worth of work if a container dies.
PART_SIZE = 2 * 1024**3

_PART_RE = re.compile(r"^part_(\d{5})\.bin$")


def parts_dir(mount, comp):
    return os.path.join(mount, f"{comp}.parts")


def archive_path(mount, comp):
    """The single-file archive, as older volumes have it."""
    return os.path.join(mount, f"{comp}.zip")


def list_parts(mount, comp):
    """Existing parts in index order. Returns [(index, path, size), ...]."""
    d = parts_dir(mount, comp)
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        m = _PART_RE.match(name)
        if m:
            p = os.path.join(d, name)
            out.append((int(m.group(1)), p, os.path.getsize(p)))
    out.sort()
    return out


def part_path(mount, comp, index):
    return os.path.join(parts_dir(mount, comp), "part_%05d.bin" % index)


def contiguous_parts(mount, comp):
    """Parts from index 0 with no gap, and every one except the last exactly PART_SIZE.

    A short part in the MIDDLE means that part was interrupted mid-write, so everything from
    it onward is untrustworthy even though the files exist. Truncating the list here is what
    makes resume safe: the downloader refetches from the first bad part instead of appending
    past a hole and producing an archive that only fails much later, at unzip time.
    """
    good = []
    for i, (idx, p, size) in enumerate(list_parts(mount, comp)):
        if idx != i:
            break                      # a gap in numbering
        if size == 0:
            break
        good.append((idx, p, size))
        if size != PART_SIZE:
            break                      # short part: only valid if it is the final one
    return good


def base_size(mount, comp):
    """Bytes held in a pre-split single `<comp>.zip`, if this volume has one.

    A volume part-way through the OLD single-file download (session 23 stopped at 232.8 GiB
    of 246.8) must not have that work thrown away. Rather than copy 232 GiB into parts - which
    would need the volume to hold both at once - the existing archive is kept as the BASE and
    new parts simply continue after it. Part i therefore covers
    `base_size + i * PART_SIZE`, and the reader stitches base + parts in that order.
    """
    whole = archive_path(mount, comp)
    return os.path.getsize(whole) if os.path.exists(whole) else 0


def parts_bytes(mount, comp):
    """How many contiguous, trustworthy bytes we already hold, base included."""
    return base_size(mount, comp) + sum(s for _, _, s in contiguous_parts(mount, comp))


class PartsFile(io.RawIOBase):
    """A seekable read-only file over an ordered list of part files.

    zipfile needs seek/tell/read and reads the central directory from the END, so this
    supports whence=2 as well. Only one part is held open at a time: the access pattern is
    long sequential runs punctuated by seeks, so caching a single handle is enough and keeps
    memory flat no matter how many parts there are.
    """

    def __init__(self, paths, sizes):
        self._paths = list(paths)
        self._sizes = list(sizes)
        self._starts = []
        off = 0
        for s in self._sizes:
            self._starts.append(off)
            off += s
        self._size = off
        self._pos = 0
        self._fh = None
        self._fh_i = -1

    # --- plumbing ---------------------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    @property
    def size(self):
        return self._size

    def seek(self, offset, whence=0):
        if whence == 0:
            p = offset
        elif whence == 1:
            p = self._pos + offset
        elif whence == 2:
            p = self._size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        if p < 0:
            raise OSError("negative seek")
        self._pos = p
        return self._pos

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._fh_i = -1
        super().close()

    # --- reading ----------------------------------------------------------------
    def _part_for(self, pos):
        """Index of the part containing `pos`, or -1 past the end."""
        lo, hi = 0, len(self._starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if pos < self._starts[mid]:
                hi = mid - 1
            elif pos >= self._starts[mid] + self._sizes[mid]:
                lo = mid + 1
            else:
                return mid
        return -1

    def _open(self, i):
        if self._fh_i != i:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self._paths[i], "rb")
            self._fh_i = i
        return self._fh

    def readinto(self, b):
        n = len(b)
        if n == 0 or self._pos >= self._size:
            return 0
        got = 0
        view = memoryview(b)
        while got < n and self._pos < self._size:
            i = self._part_for(self._pos)
            if i < 0:
                break
            fh = self._open(i)
            local = self._pos - self._starts[i]
            fh.seek(local)
            want = min(n - got, self._sizes[i] - local)
            chunk = fh.read(want)
            if not chunk:
                break
            view[got:got + len(chunk)] = chunk
            got += len(chunk)
            self._pos += len(chunk)
        return got


def open_archive(mount, comp):
    """A file object for the archive, whichever of the three layouts this volume uses.

      * a single `<comp>.zip` and no parts   - volumes filled before the split, opened directly
      * parts and no `<comp>.zip`            - a clean new download
      * BOTH                                 - a download that began before the split and
                                               finished after it; the zip is the base and the
                                               parts continue after it, in that order

    Raises if there is nothing at all.
    """
    whole = archive_path(mount, comp)
    has_whole = os.path.exists(whole) and os.path.getsize(whole) > 0
    parts = contiguous_parts(mount, comp)
    if has_whole and not parts:
        return open(whole, "rb")
    if not parts:
        raise FileNotFoundError(
            f"no archive on {mount}: neither {whole} nor any part in {parts_dir(mount, comp)}")
    paths = [p for _, p, _ in parts]
    sizes = [s for _, _, s in parts]
    if has_whole:
        paths.insert(0, whole)
        sizes.insert(0, os.path.getsize(whole))
    return PartsFile(paths, sizes)


def archive_size(mount, comp):
    """Total trustworthy bytes held, in whichever layout."""
    return parts_bytes(mount, comp)
