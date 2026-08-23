"""Utilities for a .pt2 archive: making a serialized graph reproducible across runs/machines,
extracting its committed JSON members, and cross-checking a committed graph against a report's
operator counts.

A .pt2 file is a zip: a serialized graph (`models/model.json`), an index of the weight
tensors (`data/weights/model_weights_config.json`), and the raw weight blobs. The graph and
the index are small, text, and describe the architecture exactly as a PT2 backend sees it, so
those are the parts worth extracting and committing; the blobs are not.
"""
import contextlib
import functools
import io
import json
import os
import struct
import sys
import zipfile
from typing import NamedTuple

import yaml

from .catalog import parse_existing_ops, short_op
from .opgraph import DROPPED_NAMESPACES, DROPPED_OPS

# The parts of a .pt2 archive that are worth committing: the graph itself, and the index
# mapping graph tensor names to weight blobs and their shapes/dtypes. Everything else is
# either the blobs (large, and reproducible from the zoo), a pickled sample input, or archive
# bookkeeping including a serialization id that changes every run.
ARCHIVE_JSON = (
    'models/model.json',
    'data/weights/model_weights_config.json',
    'data/constants/model_constants_config.json',
)


# ---------------------------------------------------------------------------- safe_zip_open
#
# Every zip this pipeline opens is, at the point it is opened, a zip this code is trying to
# verify -- a downloaded release archive, a caller-supplied `--model`/`--pt2` path, a nested
# `.pt2` extracted from an outer archive. `zipfile.ZipFile()`'s own constructor eagerly parses
# the *entire* central directory before any of its methods (`infolist()`, `namelist()`) can be
# called, so a bound checked only after construction is not a bound at all: the expensive,
# attacker-controlled parse has already happened by then. `safe_zip_open` is the one place in
# this codebase allowed to construct a `zipfile.ZipFile` on caller-supplied bytes; every other
# zip-touching function in this module and in `scripts/export_pt2.py` goes through it.


class SafeZipError(ValueError):
    """A zip failed one of safe_zip_open's bounds, before or after zipfile's own parse."""


class ZipProfile(NamedTuple):
    """Every limit safe_zip_open enforces for one class of archive, bundled so a call site
    never has to choose limits piecemeal -- see RELEASE_PT2_PROFILE / SELECTED_PT2_PROFILE /
    IMAGES_ARCHIVE_PROFILE / JSON_METADATA_PROFILE below for the four this pipeline uses.
    """
    max_archive_bytes: int             # pre-open, pre-parse: whole-file size cap
    max_central_directory_bytes: int   # pre-open: classic EOCD's own declared cd size cap,
                                        # independent of max_archive_bytes (see module docstring
                                        # below JSON_METADATA_PROFILE for why)
    max_members: int
    max_total_uncompressed: int
    max_compression_ratio: int
    max_member_bytes: int              # per member, for anything not matched as a JSON member


# A small, fixed, weight-independent cap applied to any member whose name ends in `.json`
# (`model.json`, `op_facts.json`, `contract.json`, `manifest.json`, `catalogue.json`,
# `compat-report.json`, `preprocessing.json`, `expected.json`, ...), inside *every* profile --
# not a profile passed to safe_zip_open on its own (nothing in this pipeline opens a
# JSON-only zip). A `.pt2`'s multi-hundred-MB weight blob must not force a named JSON member's
# cap up to meet it, and a legitimate weight blob must not be held to the JSON cap either, so
# the JSON cap is applied per-member inside read_member's own size check, sourced from here.
#
# max_compression_ratio: found too tight at 200 via a real integration test -- inputs.pt (a
# torch.save of solid-color test images, normalized) legitimately compressed at 418:1 under
# plain DEFLATE, and real per-channel-normalized/zero-padded tensor data can compress just as
# well without being adversarial at all. A single (non-nested) DEFLATE stream's own worst-case
# expansion ratio tops out near 1032:1 (RFC 1951's compressed-block format bounds how much a
# single layer can inflate); a nested zip-within-zip is what actually reaches the pathological
# ratios a "zip bomb" implies, and that is caught by this reader rejecting nested archives it
# doesn't expect, plus max_member_bytes/max_total_uncompressed bounding the absolute worst case
# regardless of ratio. 1100 comfortably admits realistic legitimate single-layer compressibility
# while still catching a small member whose declared size is wildly disproportionate to what a
# single DEFLATE layer could actually produce from that many compressed bytes.
_MAX_COMPRESSION_RATIO = 1100

JSON_METADATA_PROFILE = ZipProfile(
    max_archive_bytes=8 * 2**20,
    max_central_directory_bytes=1 * 2**20,
    max_members=64,
    max_total_uncompressed=32 * 2**20,
    max_compression_ratio=_MAX_COMPRESSION_RATIO,
    max_member_bytes=8 * 2**20,
)

# A handful of small sample photos, `labels/`, and `SOURCES.md` -- generous but bounded, no
# weight cap applies at all here.
IMAGES_ARCHIVE_PROFILE = ZipProfile(
    max_archive_bytes=256 * 2**20,
    max_central_directory_bytes=2 * 2**20,
    max_members=4096,
    max_total_uncompressed=512 * 2**20,
    max_compression_ratio=_MAX_COMPRESSION_RATIO,
    max_member_bytes=64 * 2**20,
)


# max_members: a real .pt2 stores every parameter/buffer as its own zip member (plus a handful
# of fixed metadata entries), so this scales with model parameter *count*, not weight_cap_mb --
# rebuilding the actual selected set (100 models, mostly small conv nets with many individually-
# small tensors) measured up to 1965 members (ghostnetv3_050/130's many repeated blocks), so 256
# was rejecting legitimate archives outright. 4096 (matching IMAGES_ARCHIVE_PROFILE, itself many
# small sample files) gives over 2x headroom above that observed real maximum.
_PT2_MAX_MEMBERS = 4096


def _pt2_profile(weight_cap_mb):
    """A PT2-scale profile from one weight-cap number in models-selected.yaml's `selection`
    block. `max_archive_bytes`/`max_total_uncompressed` sit comfortably above the declared
    weight cap (a legitimate archive is always smaller than that headroom);
    `max_central_directory_bytes` is the same small, metadata-scale cap regardless of how large
    the weight cap is, since a legitimate central directory's size does not scale with payload
    size.
    """
    cap_bytes = int(weight_cap_mb * 2**20)
    return ZipProfile(
        max_archive_bytes=cap_bytes * 2,
        max_central_directory_bytes=2 * 2**20,
        max_members=_PT2_MAX_MEMBERS,
        max_total_uncompressed=cap_bytes * 2,
        max_compression_ratio=_MAX_COMPRESSION_RATIO,
        max_member_bytes=cap_bytes * 2,
    )


@functools.lru_cache(maxsize=None)
def _selection_weight_caps(models_selected_path):
    with open(models_selected_path) as f:
        document = yaml.safe_load(f) or {}
    selection = document.get('selection') or {}
    try:
        max_weight_mb, release_max_weight_mb = (
            selection['max_weight_mb'], selection['release_max_weight_mb'])
    except KeyError as e:
        raise SafeZipError(f'{models_selected_path}: selection.{e.args[0]} missing -- '
                            'cannot derive a PT2 zip profile without it') from None

    # `models:` entries reach this weight_mb (the real pretrained checkpoint size) via an
    # explicit `include`, which selection.select() lets bypass max_weight_mb entirely (an
    # explicit choice outranks the heuristic filter) -- so the filter alone cannot be trusted
    # to bound every selected model's actual size. Widening the cap here to the true observed
    # max keeps that filter meaningful as a *selection* knob without also reshuffling which
    # models get auto-selected, which changing max_weight_mb itself would do.
    models = document.get('models') or {}
    observed_max_mb = max((m.get('weight_mb', 0.0) for m in models.values()), default=0.0)
    return max(max_weight_mb, observed_max_mb), release_max_weight_mb


def release_pt2_profile(models_selected_path):
    """Weight-scale caps for a genuine release-tier archive (outer release .zip / inner .pt2):
    derived from `selection.release_max_weight_mb`, read at validation time, not hand-copied.
    """
    _, release_cap_mb = _selection_weight_caps(models_selected_path)
    return _pt2_profile(release_cap_mb)


def selected_pt2_profile(models_selected_path):
    """Weight-scale caps for any selected model's .pt2, release-tier or not: derived from
    `selection.max_weight_mb` widened (if needed) to the largest weight_mb actually selected,
    so a legitimate graph-only model sized between the two caps -- or a heavier model let in via
    an explicit `include` -- is never wrongly rejected by a tighter, filter-only number.
    """
    selected_cap_mb, _ = _selection_weight_caps(models_selected_path)
    return _pt2_profile(selected_cap_mb)


_EOCD_SIGNATURE = b'PK\x05\x06'
_EOCD_FIXED_SIZE = 22  # 4-byte signature + 18 bytes of fixed fields, before the comment
_ZIP64_LOCATOR_SIGNATURE = b'PK\x06\x07'
_ZIP64_LOCATOR_SIZE = 20
_MAX_EOCD_COMMENT_LENGTH = 0xFFFF


class _EocdInfo(NamedTuple):
    total_entries: int
    central_directory_size: int


_ZIP64_EOCD_SIGNATURE = b'PK\x06\x06'
_ZIP64_EOCD_FIXED_SIZE = 56  # signature + size-of-record field + the fixed fields we read


def _read_zip64_eocd(source, locator_bytes, archive_size):
    """Parse the ZIP64 End-Of-Central-Directory locator (already read, 20 bytes immediately
    before the classic EOCD) and record, returning the *real* total_entries/central_directory_
    size -- authoritative over the classic EOCD's own fields whenever a ZIP64 locator is
    present, per the zip format itself, since PyTorch's own .pt2/.pt writer (`torch.export.save`
    / `torch.save`) always emits a ZIP64 locator, even for small archives well under 4GB --
    confirmed empirically: a 4.7MB `.pt2` with 124 members still carries one. Rejecting ZIP64
    outright, as an earlier draft of this reader did, would reject every real .pt2 this
    pipeline produces, not just an oversized one.

    Single-disk archives only (this pipeline never produces or expects a multi-disk archive);
    a multi-disk locator is rejected rather than trusted.
    """
    _disk_no, zip64_eocd_offset, total_disks = struct.unpack('<IQI', locator_bytes[4:])
    if total_disks != 1 or _disk_no != 0:
        raise SafeZipError('multi-disk ZIP64 archives are not supported')
    if not (0 <= zip64_eocd_offset <= archive_size - _ZIP64_EOCD_FIXED_SIZE):
        raise SafeZipError('ZIP64 End-Of-Central-Directory record offset is out of range')

    source.seek(zip64_eocd_offset)
    record = source.read(_ZIP64_EOCD_FIXED_SIZE)
    if record[:4] != _ZIP64_EOCD_SIGNATURE:
        raise SafeZipError('ZIP64 locator points to a record with the wrong signature')
    (_size_of_record, _version_made, _version_needed, disk_no, disk_start,
     _entries_this_disk, total_entries, cd_size, _cd_offset) = struct.unpack('<QHHIIQQQQ', record[4:])
    if disk_no != 0 or disk_start != 0:
        raise SafeZipError('multi-disk ZIP64 archives are not supported')
    return total_entries, cd_size


def _read_eocd(source, archive_size):
    """A minimal, hand-written read of just the End-Of-Central-Directory record (and, for a
    ZIP64 archive, the ZIP64 locator/record too), entirely separate from zipfile's own parser
    -- this is the bounded reader that stands in front of the unbounded one, so it must never
    itself call into `zipfile`.
    """
    scan_size = min(archive_size, _EOCD_FIXED_SIZE + _MAX_EOCD_COMMENT_LENGTH)
    source.seek(archive_size - scan_size)
    tail = source.read(scan_size)
    idx = tail.rfind(_EOCD_SIGNATURE)
    if idx == -1:
        raise SafeZipError('not a zip archive (no End-Of-Central-Directory record found)')
    eocd = tail[idx:idx + _EOCD_FIXED_SIZE]
    if len(eocd) < _EOCD_FIXED_SIZE:
        raise SafeZipError('truncated End-Of-Central-Directory record')

    (_disk_no, _disk_cd_start, entries_this_disk, total_entries,
     cd_size, cd_offset, comment_len) = struct.unpack('<HHHHIIH', eocd[4:])

    eocd_offset = archive_size - scan_size + idx
    # The comment is whatever is left after the fixed part; a real EOCD's comment always runs
    # exactly to the end of the file. A mismatch means this signature match is spurious (e.g.
    # embedded inside another EOCD's comment) -- reject rather than search for another
    # candidate, since a crafted trailing comment is exactly the kind of thing this check
    # exists to catch.
    if eocd_offset + _EOCD_FIXED_SIZE + comment_len != archive_size:
        raise SafeZipError('End-Of-Central-Directory record does not account for the rest of '
                            'the file -- rejecting rather than guessing which signature match '
                            'is the real one')

    is_zip64_sentinel = (entries_this_disk == 0xFFFF or total_entries == 0xFFFF
                          or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF)
    locator_bytes = None
    if eocd_offset >= _ZIP64_LOCATOR_SIZE:
        source.seek(eocd_offset - _ZIP64_LOCATOR_SIZE)
        candidate = source.read(_ZIP64_LOCATOR_SIZE)
        if candidate[:4] == _ZIP64_LOCATOR_SIGNATURE:
            locator_bytes = candidate

    if locator_bytes is not None:
        # The locator is authoritative over the classic EOCD's own fields whenever present --
        # see _read_zip64_eocd's docstring for why this pipeline cannot simply reject it.
        total_entries, cd_size = _read_zip64_eocd(source, locator_bytes, archive_size)
    elif is_zip64_sentinel:
        # Sentinel values with no locator to resolve them against: the classic EOCD's own
        # fields are unusable and there is nothing authoritative to fall back to.
        raise SafeZipError('classic EOCD declares ZIP64 sentinel values but no ZIP64 locator '
                            'record was found -- cannot determine the real entry count or '
                            'central directory size')

    return _EocdInfo(total_entries=total_entries, central_directory_size=cd_size)


class _MemoryZipSource:
    """A zero-copy seekable adapter over `memoryview(data)`, standing in for `io.BytesIO`
    (which copies its input into a second, internally-owned buffer -- doubling resident
    memory for the nested-.pt2 case this pipeline actually exercises, where `data` is already
    an extracted member of an outer archive). Implements only what `zipfile.ZipFile` and
    `_read_eocd` actually call.
    """
    def __init__(self, view):
        self._view = view
        self._pos = 0
        self._closed = False

    def read(self, n=-1):
        if n is None or n < 0:
            n = len(self._view) - self._pos
        chunk = self._view[self._pos:self._pos + n]
        self._pos += len(chunk)
        return bytes(chunk)

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = len(self._view) + offset
        else:
            raise ValueError(f'unsupported whence: {whence}')
        return self._pos

    def tell(self):
        return self._pos

    def seekable(self):
        return True

    def fileno(self):
        raise OSError('a bytes-input archive has no file descriptor')

    def close(self):
        # Idempotent, like a real file object -- _with_validated_source's caller may already
        # have "closed" this (dropped its reference); safe_zip_open's own exit always calls
        # this too, regardless.
        self._closed = True
        self._view = None


class _ValidatedZip:
    """What safe_zip_open yields: a validated archive exposing only named-member reads
    (`read_member`, resolved through an already-duplicate-checked, already-size-checked
    mapping) plus one narrow, internal escape hatch (`_with_validated_source`) for the single
    call site that must hand the whole archive to a third-party loader instead of reading a
    member out of it. Stays open for the lifetime of the `with` block; both the wrapped
    `ZipFile` and the underlying source are closed only on that block's exit.
    """
    def __init__(self, zf, name_to_info, source, proc_fd_path):
        self._zf = zf
        self._name_to_info = name_to_info
        self._source = source
        self._proc_fd_path = proc_fd_path
        self._consumed = False

    def names(self):
        return list(self._name_to_info)

    def member_size(self, name):
        return self._name_to_info[name].file_size

    def read_member(self, name):
        if self._consumed:
            raise SafeZipError('this validated archive was already handed to a validated-source '
                                'loader (_with_validated_source) -- read_member is not supported '
                                'afterward, since the source\'s position/open-state is no longer '
                                'guaranteed')
        try:
            info = self._name_to_info[name]
        except KeyError:
            raise KeyError(f'{name}: not a member of this archive') from None
        return self._zf.read(info)

    def _with_validated_source(self, fn):
        """Internal -- not part of the public wrapper API. See the `worker_pack` /
        `--model <path>` design: the one call site that must hand `torch.export.load` the
        whole validated archive, not a named member, because that API deserializes the
        entire archive itself and has no member-name argument to give it.

        `fn(handle, proc_fd_path)`: `handle` is this archive's own already-open source object
        (the file object for a path input, this `_MemoryZipSource` for a bytes input), seeked
        to 0; `proc_fd_path` is `f"/proc/self/fd/{handle.fileno()}"` for a path input, `None`
        for a bytes input (no fd exists for an in-memory buffer). Ownership of `handle` passes
        to `fn` for the duration of the call -- it may read, seek, or close it freely.

        After `fn` returns (or raises), this wrapper is consumed: `read_member`/`names`/
        `member_size` are no longer supported. This is a single terminal hand-off, not a
        resumable one. `safe_zip_open`'s own exit still attempts to close the underlying
        source regardless, tolerating it already being closed by `fn`.
        """
        self._consumed = True
        self._source.seek(0)
        return fn(self._source, self._proc_fd_path)


@contextlib.contextmanager
def safe_zip_open(path_or_bytes, *, profile):
    """Open a zip archive -- a path, or an in-memory `bytes`/`bytearray`/`memoryview` already
    extracted from another archive -- under `profile`'s bounds, yielding a `_ValidatedZip`.

    Every check below runs before `zipfile.ZipFile(...)` is ever constructed except the final
    duplicate-member/per-member/total-uncompressed checks, which run against `infolist()`
    immediately after construction and before any member's bytes are read -- so nothing this
    pipeline treats as untrusted ever has its full central directory parsed, let alone a
    member's bytes decompressed, before every bound has been checked.

    For a path input, the file is opened exactly once and every check -- size, EOCD, the final
    `ZipFile` construction -- runs against that one descriptor, never a fresh
    `os.path.getsize(path)` or a second `open(path)`: POSIX guarantees an open descriptor keeps
    referring to the same underlying file content regardless of what later happens to the
    path, closing the replacement-race window a path-based re-check on every step would leave
    open. For a bytes input there is no such window (already one immutable, already-in-memory
    object), so the zero-copy `_MemoryZipSource` adapter is used directly.
    """
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        archive_size = len(path_or_bytes)
        if archive_size > profile.max_archive_bytes:
            raise SafeZipError(f'{archive_size} bytes exceeds max_archive_bytes='
                                f'{profile.max_archive_bytes}')
        view = path_or_bytes if isinstance(path_or_bytes, memoryview) else memoryview(path_or_bytes)
        source = _MemoryZipSource(view)
    else:
        path = os.fspath(path_or_bytes)
        fd = os.open(path, os.O_RDONLY)
        source = os.fdopen(fd, 'rb')  # takes ownership of fd; source.close() closes it
        archive_size = os.fstat(source.fileno()).st_size
        if archive_size > profile.max_archive_bytes:
            source.close()
            raise SafeZipError(f'{path}: {archive_size} bytes exceeds max_archive_bytes='
                                f'{profile.max_archive_bytes}')

    try:
        eocd = _read_eocd(source, archive_size)
        if eocd.total_entries > profile.max_members:
            raise SafeZipError(f'declared entry count {eocd.total_entries} exceeds '
                                f'max_members={profile.max_members}')
        if eocd.central_directory_size > profile.max_central_directory_bytes:
            raise SafeZipError(f'declared central directory size {eocd.central_directory_size} '
                                f'exceeds max_central_directory_bytes='
                                f'{profile.max_central_directory_bytes}')

        source.seek(0)
        zf = zipfile.ZipFile(source)
        try:
            infos = zf.infolist()
            if len(infos) != eocd.total_entries:
                raise SafeZipError(f'parsed {len(infos)} central directory entries but the '
                                    f'End-Of-Central-Directory record declared '
                                    f'{eocd.total_entries} -- archive directory does not match '
                                    'its own header')
            if len(infos) > profile.max_members:
                raise SafeZipError(f'{len(infos)} members exceeds max_members='
                                    f'{profile.max_members}')

            name_to_info = {}
            total_uncompressed = 0
            for info in infos:
                if info.filename in name_to_info:
                    raise SafeZipError(f'duplicate member name: {info.filename!r}')
                name_to_info[info.filename] = info

                cap = (JSON_METADATA_PROFILE.max_member_bytes if info.filename.endswith('.json')
                       else profile.max_member_bytes)
                if info.file_size > cap:
                    raise SafeZipError(f'{info.filename!r}: declared uncompressed size '
                                        f'{info.file_size} exceeds member cap {cap}')
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > profile.max_compression_ratio:
                    raise SafeZipError(f'{info.filename!r}: compression ratio {ratio:.0f} '
                                        f'exceeds max_compression_ratio='
                                        f'{profile.max_compression_ratio}')
                total_uncompressed += info.file_size

            if total_uncompressed > profile.max_total_uncompressed:
                raise SafeZipError(f'total uncompressed size {total_uncompressed} exceeds '
                                    f'max_total_uncompressed={profile.max_total_uncompressed}')

            proc_fd_path = None
            if hasattr(source, 'fileno'):
                try:
                    proc_fd_path = f'/proc/self/fd/{source.fileno()}'
                except OSError:
                    proc_fd_path = None  # the bytes-input _MemoryZipSource case

            yield _ValidatedZip(zf, name_to_info, source, proc_fd_path)
        finally:
            zf.close()
    finally:
        source.close()


def graphs(exported):
    """Every graph in the program, the top-level one and any a higher-order op carries.

    An undecomposed ATen graph keeps its higher-order ops -- `wrap_with_autocast` around an
    autocast region, control flow -- and each holds a nested graph of its own, serialized in
    full alongside the rest. Anything that walks "the graph" to make it portable has to reach
    those too.
    """
    for module in exported.graph_module.modules():
        graph = getattr(module, 'graph', None)
        if graph is not None:
            yield graph


def make_portable(exported):
    """Strip everything from the graph that describes the machine rather than the model.

    A committed graph has to depend only on the architecture: it is regenerated in two
    different CI jobs and diffed against what is in git, so anything reflecting where or when
    it was built makes that check impossible to pass. Two fields do:

      stack_trace     a third of the serialized graph, every frame naming an absolute path
                      inside the venv, which differs between a uv checkout and the devcontainer
      from_node       provenance, tagged with id(node.graph) -- a Python object address, so a
                      different value on every single run

    Applied to every graph in the program, not just the top-level one: see `graphs()`.

    Done before saving rather than after extracting, so the released .pt2 is as portable as
    the committed JSON, and so the committed bytes stay the serializer's own output.
    """
    for graph in graphs(exported):
        for node in graph.nodes:
            node.meta.pop('stack_trace', None)
    _canonicalize_provenance(exported)


def _canonicalize_provenance(exported):
    """Renumber the graph ids inside `from_node` so the same model always serializes the same.

    `from_node` records where each node came from, and tags every entry with the
    graph it came from -- as `id(node.graph)`, a Python object address. That address is
    different on every run, so the serialized graph would never be byte-identical twice and
    the "regenerate and diff" check these files exist for could never pass.

    The addresses are only ever compared for equality (did these two nodes come from the same
    graph?), so replacing them with 0, 1, 2... in order of first appearance keeps everything
    the field is used for and drops the only part that was never meaningful. Older torch
    versions numbered them this way to begin with.

    One numbering spans the whole program rather than one per graph, so that "same source
    graph" still means the same thing across a higher-order op's boundary.
    """
    ids = {}
    sources = 0

    def visit(source):
        nonlocal sources
        info = getattr(source, 'node_info', None)
        if info is not None:
            sources += 1
            info.graph_id = ids.setdefault(info.graph_id, len(ids))
        # to_dict() memoizes; the stale copy would otherwise be what gets serialized.
        source._dict = None
        for parent in getattr(source, 'from_node', ()) or ():
            visit(parent)

    seen_from_node = False
    for graph in graphs(exported):
        for node in graph.nodes:
            for source in node.meta.get('from_node') or ():
                seen_from_node = True
                visit(source)

    # Everything above reaches into torch internals that are explicitly not backward
    # compatible (NodeSource is @compatibility(is_backward_compatible=False), and `_dict` is
    # a private memo). If a rename ever makes this a no-op, the symptom is a graph that
    # silently differs on every run and a CI failure showing one changed line of minified
    # JSON. Fail here instead, where the cause is stated.
    if seen_from_node and not sources:
        raise RuntimeError(
            'from_node metadata is present but carries no node_info: torch has changed '
            'NodeSource and graph ids are no longer being canonicalized. The committed '
            'graphs would not be reproducible -- update _canonicalize_provenance.')


def assert_portable(pt2_path, profile):
    """Fail if a saved archive still carries machine-specific provenance.

    `make_portable` only strips what it walks, and a higher-order op nests a graph it may not
    reach. Such a `stack_trace` is invisible on the machine that wrote it -- every path matches
    -- and shows up as absolute venv paths everywhere else, so it is worth one zip read to be
    sure before the JSON is committed.

    `profile` is `selected_pt2_profile(...)` for a just-built .pt2 of any selected model (this
    runs in `worker_convert`, before release-tier status is even known) -- reachable on an
    arbitrary caller-supplied path via `export_pt2.py extract`, so it goes through
    `safe_zip_open` like every other .pt2 boundary rather than a raw `zipfile.ZipFile(...)`.
    """
    with safe_zip_open(pt2_path, profile=profile) as z:
        members = [name for name in z.names() if name.endswith('models/model.json')]
        for member in members:
            raw = z.read_member(member)
            if b'"stack_trace"' in raw:
                raise RuntimeError(
                    f'{pt2_path}: {member} still contains stack_trace after make_portable -- '
                    'some graph was not walked (a new higher-order op nesting one?). The '
                    'committed JSON would embed absolute venv paths and differ per machine.')


def load_manifest(path):
    """Read a selection manifest (e.g. models-selected.yaml) into {name: entry}, failing
    loudly if it is missing."""
    if not os.path.exists(path):
        sys.exit(f'{path}: not found -- run `make models.select` first')
    with open(path) as f:
        document = yaml.safe_load(f) or {}
    models = document.get('models') or {}
    if not models:
        sys.exit(f'{path}: no models listed')
    return models


def release_names(models, only=None):
    """The release tier, optionally narrowed to `only`. One reader of the manifest's schema.

    `only` is checked against the whole manifest rather than silently intersected, so a typo
    or a graph-only model is an error at the point it was named instead of an empty run that
    reports success.
    """
    tier = [name for name, entry in sorted(models.items()) if entry.get('release')]
    if not only:
        return tier
    unknown = [name for name in only if name not in models]
    if unknown:
        sys.exit(f'not in the manifest: {", ".join(unknown)}')
    graph_only = [name for name in only if name not in tier]
    if graph_only:
        sys.exit('not release-tier (no fetchable weights, or over the weight cap): '
                 f'{", ".join(graph_only)}')
    return list(only)


def single_root(names, archive_label):
    """The one top-level directory every member of a .pt2 archive nests under, or a named
    error if there are zero or multiple candidates (a malformed or multi-root archive).

    Shared by `extract()` and `compare_embedded_graph()`'s nested-.pt2 handling, so a
    multi-root archive is rejected identically -- and with the same message shape -- wherever
    this pipeline needs to find the one root.
    """
    roots = {name.split('/', 1)[0] for name in names}
    if len(roots) != 1:
        raise ValueError(f'{archive_label}: expected a single top-level directory, got {sorted(roots)}')
    return roots.pop()


def extract(pt2_path, name, models_dir, profile):
    """Copy the JSON members of a .pt2 into models/<name>/, byte for byte.

    The archive nests everything under a directory named after the file stem; that prefix is
    stripped so the committed layout is stable regardless of where the .pt2 was built. The
    bytes are the serializer's own output -- nothing is re-encoded here, so a diff in these
    files always means the graph changed, never that the formatting did.

    `profile` is `selected_pt2_profile(...)`: this runs on any selected model in `cmd_build`,
    release-tier or not, and is also reachable directly on an arbitrary caller-supplied path
    via the public `export_pt2.py extract --pt2 <path>` command -- exactly the kind of
    unbounded, duplicate-blind zip boundary `safe_zip_open` exists to close off.
    """
    target = os.path.join(models_dir, name)
    written = []
    with safe_zip_open(pt2_path, profile=profile) as z:
        root = single_root(z.names(), pt2_path)
        for relative in ARCHIVE_JSON:
            member = f'{root}/{relative}'
            try:
                data = z.read_member(member)
            except KeyError:
                continue  # constants config is absent for models with no constant tensors
            path = os.path.join(target, relative)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(data)
            written.append(relative)
    if not written:
        raise ValueError(f'{pt2_path}: no JSON members found')
    return written


def compare_embedded_graph(release_zip_path, name, models_dir, profile):
    """Byte-compare the committed `models/<name>/models/model.json` against the graph
    actually embedded in a release archive's inner `<name>.pt2` -- the check that gives the
    committed JSON its meaning as "the graph in the shipped archive," not just "a graph this
    pipeline produced at some point."

    Both zip layers -- the outer release `.zip` and the inner `.pt2` its `<name>.pt2` member
    decompresses to -- go through `safe_zip_open`, the inner one via the zero-copy
    `memoryview` adapter over the bytes `read_member()` already extracted (never a second
    disk read, never `io.BytesIO`'s copy). `single_root` is reused rather than duplicated, so
    a malformed or multi-root inner `.pt2` is rejected identically here and in `extract()`.

    Raises `ValueError` naming both the outer and inner member paths on any mismatch.
    """
    committed_path = os.path.join(models_dir, name, 'models', 'model.json')
    if not os.path.exists(committed_path):
        raise ValueError(f'{name}: no committed graph at {committed_path}')
    with open(committed_path, 'rb') as f:
        committed = f.read()

    with safe_zip_open(release_zip_path, profile=profile) as outer:
        pt2_member = f'{name}.pt2'
        try:
            pt2_bytes = outer.read_member(pt2_member)
        except KeyError:
            raise ValueError(f'{release_zip_path}: no {pt2_member!r} member') from None

        with safe_zip_open(pt2_bytes, profile=profile) as inner:
            root = single_root(inner.names(), f'{release_zip_path}!{pt2_member}')
            inner_member = f'{root}/models/model.json'
            try:
                embedded = inner.read_member(inner_member)
            except KeyError:
                raise ValueError(
                    f'{release_zip_path}!{pt2_member}: no {inner_member!r} member') from None

    if embedded != committed:
        raise ValueError(
            f'{name}: committed {committed_path} does not byte-match the graph embedded in '
            f'{release_zip_path}!{pt2_member}!{inner_member}')


def graph_op_counts(model_json_path):
    """{aten target: node count} from a committed graph, for cross-checking against a report's
    ops matrix.

    Higher-order targets are left out for the same reason `collect_ops` leaves them out of the
    report: they carry no schema, so there is nothing to describe and nothing to compare. A
    live graph tests `hasattr(target, '_schema')`; here only the target string survives, so the
    rule is expressed as the namespace it amounts to.
    """
    with open(model_json_path) as f:
        document = json.load(f)
    counts = {}
    for node in document['graph_module']['graph']['nodes']:
        target = node['target'].replace('torch.ops.', '')
        if target.split('.', 1)[0] in DROPPED_NAMESPACES:
            continue
        counts[target] = counts.get(target, 0) + 1
    return counts


def graph_differences(models, models_dir, ops_path):
    """({name: 'op=reported/committed ...'}, [problem, ...]) comparing graphs with a report's
    operator matrix. Both must describe the same dialect for the comparison to mean anything.

    `_assert_tensor_metadata` is dropped on the committed side because the report already
    drops it as export bookkeeping rather than computation -- DROPPED_OPS is that rule, so a
    second entry there does not silently become spurious diffs here.
    """
    ops_by_model, _, _ = parse_existing_ops(ops_path)
    differences, problems = {}, []

    for name in sorted(models):
        model_json = os.path.join(models_dir, name, 'models', 'model.json')
        if not os.path.exists(model_json):
            problems.append(f'{name}: no committed graph')
            continue
        try:
            committed = graph_op_counts(model_json)
        except Exception as e:
            problems.append(f'{name}: unreadable graph ({e})')
            continue
        for op in DROPPED_OPS:
            committed.pop(op, None)

        reported = {}
        for op, _, count in ops_by_model.get(name, []):
            reported[op] = reported.get(op, 0) + count
        if not reported:
            problems.append(f'{name}: not in {os.path.basename(ops_path)}')
            continue

        if committed != reported:
            differences[name] = ' '.join(
                f'{short_op(op)}={reported.get(op, 0)}/{committed.get(op, 0)}'
                for op in sorted(set(committed) | set(reported))
                if committed.get(op, 0) != reported.get(op, 0))

    return differences, problems


def render_differences(differences, total, ops_name='ops-func.yaml'):
    """The models whose committed graph disagrees with the ops report, as a file to commit."""
    lines = [
        f'# Where a committed graph disagrees with the operator counts in {ops_name}.',
        '#',
        '# Generated by `make models.differences`, do not edit by hand. `make models.verify`',
        '# holds the tree to this list: a model that starts or stops diverging, or diverges',
        '# differently, is a failure until it is regenerated here and reviewed in the diff.',
        '#',
        '# The two artifacts are exported on different devices: the report traces on `meta`,',
        '# while a .pt2 carries real weight blobs and traces on CPU. Functional ATen is normally',
        '# device-independent, except that attention-result strides can change whether a reshape',
        '# becomes a view or clone + _unsafe_view. Any differences are pinned here for review.',
        '#',
        f'# Values are `op={ops_name}/graph`. {len(differences)} of {total} models differ.',
        '',
        'models:',
    ]
    for name, delta in sorted(differences.items()):
        lines.append(f'  {name}: {delta}')
    lines.append('')
    return '\n'.join(lines)


def read_differences(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        document = yaml.safe_load(f) or {}
    return {name: str(delta) for name, delta in (document.get('models') or {}).items()}
