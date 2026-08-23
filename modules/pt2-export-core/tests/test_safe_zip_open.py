import os
import struct
import zipfile

import pytest

from pt2_export_core import archive
from pt2_export_core.archive import (
    SafeZipError, ZipProfile, _MemoryZipSource, safe_zip_open,
)

GENEROUS = ZipProfile(
    max_archive_bytes=50 * 2**20,
    max_central_directory_bytes=2 * 2**20,
    max_members=100,
    max_total_uncompressed=50 * 2**20,
    max_compression_ratio=200,
    max_member_bytes=20 * 2**20,
)


def make_zip(path, members, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, 'w', compression) as z:
        for name, data in members.items():
            z.writestr(name, data)


def make_zip_bytes(members, compression=zipfile.ZIP_STORED):
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def find_eocd_offset(data):
    idx = data.rfind(b'PK\x05\x06')
    assert idx != -1
    return idx


def patch_eocd_field(data, *, total_entries=None, cd_size=None):
    """Bit-patch the classic EOCD's declared entry count and/or central-directory-size
    fields, leaving everything else (including the real central directory) untouched."""
    data = bytearray(data)
    idx = find_eocd_offset(bytes(data))
    # offsets relative to the signature: entries_this_disk@8, total_entries@10, cd_size@14
    if total_entries is not None:
        struct.pack_into('<H', data, idx + 8, total_entries)
        struct.pack_into('<H', data, idx + 10, total_entries)
    if cd_size is not None:
        struct.pack_into('<I', data, idx + 12, cd_size)
    return bytes(data)


# ---------------------------------------------------------------------------- roundtrip


def test_roundtrip_path_input(tmp_path):
    path = tmp_path / 'a.zip'
    make_zip(path, {'models/model.json': b'{"a": 1}', 'data/weights/x.json': b'{}'})
    with safe_zip_open(str(path), profile=GENEROUS) as z:
        assert set(z.names()) == {'models/model.json', 'data/weights/x.json'}
        assert z.read_member('models/model.json') == b'{"a": 1}'
        assert z.member_size('models/model.json') == len(b'{"a": 1}')


def test_roundtrip_bytes_input_uses_zero_copy_adapter():
    data = make_zip_bytes({'models/model.json': b'{"a": 1}'})
    with safe_zip_open(data, profile=GENEROUS) as z:
        assert z.read_member('models/model.json') == b'{"a": 1}'
        # The zero-copy contract: a memoryview-backed adapter, never io.BytesIO (which would
        # copy `data` into a second internally-owned buffer).
        assert isinstance(z._source, _MemoryZipSource)


def test_roundtrip_memoryview_input():
    data = make_zip_bytes({'models/model.json': b'{"a": 1}'})
    with safe_zip_open(memoryview(data), profile=GENEROUS) as z:
        assert z.read_member('models/model.json') == b'{"a": 1}'


# ---------------------------------------------------------------------------- duplicates


def test_duplicate_member_name_rejected_before_any_read(tmp_path):
    path = tmp_path / 'dup.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('images/cat.jpg', b'one')
        z.writestr('images/cat.jpg', b'two')
    with pytest.raises(SafeZipError, match='duplicate member'):
        with safe_zip_open(str(path), profile=GENEROUS) as z:
            z.read_member('images/cat.jpg')  # must never be reached


# ---------------------------------------------------------------------------- pre-open size cap


def test_archive_over_max_archive_bytes_rejected_without_constructing_zipfile(tmp_path, monkeypatch):
    path = tmp_path / 'big.zip'
    make_zip(path, {'a.json': b'{}' * 1000})
    tiny_profile = GENEROUS._replace(max_archive_bytes=4)

    calls = []
    real_init = zipfile.ZipFile.__init__

    def spy_init(self, *a, **kw):
        calls.append(1)
        return real_init(self, *a, **kw)

    monkeypatch.setattr(zipfile.ZipFile, '__init__', spy_init)
    with pytest.raises(SafeZipError, match='max_archive_bytes'):
        with safe_zip_open(str(path), profile=tiny_profile):
            pass
    assert not calls, 'zipfile.ZipFile must never be constructed once the pre-open size cap fails'


def test_bytes_over_max_archive_bytes_rejected():
    data = make_zip_bytes({'a.json': b'{}' * 1000})
    tiny_profile = GENEROUS._replace(max_archive_bytes=4)
    with pytest.raises(SafeZipError, match='max_archive_bytes'):
        with safe_zip_open(data, profile=tiny_profile):
            pass


def test_path_opened_exactly_once(tmp_path, monkeypatch):
    path = tmp_path / 'a.zip'
    make_zip(path, {'a.json': b'{}'})
    opens = []
    real_open = os.open

    def spy_open(p, flags, *a, **kw):
        opens.append(p)
        return real_open(p, flags, *a, **kw)

    monkeypatch.setattr(os, 'open', spy_open)
    with safe_zip_open(str(path), profile=GENEROUS) as z:
        z.read_member('a.json')
    assert opens == [str(path)]


# ---------------------------------------------------------------------------- EOCD-level bounds


def test_eocd_declared_entry_count_exceeds_max_members_rejected_before_full_parse(tmp_path, monkeypatch):
    path = tmp_path / 'small.zip'
    make_zip(path, {'a.json': b'{}', 'b.json': b'{}'})
    corrupted = patch_eocd_field(path.read_bytes(), total_entries=50000)
    path.write_bytes(corrupted)

    calls = []
    real_init = zipfile.ZipFile.__init__
    monkeypatch.setattr(zipfile.ZipFile, '__init__',
                        lambda self, *a, **kw: (calls.append(1), real_init(self, *a, **kw))[1])

    with pytest.raises(SafeZipError, match='max_members'):
        with safe_zip_open(str(path), profile=GENEROUS):
            pass
    assert not calls, 'the full zipfile parse must never run once the EOCD-declared count fails'


def test_eocd_declared_central_directory_size_exceeds_cap_rejected(tmp_path):
    path = tmp_path / 'small.zip'
    make_zip(path, {'a.json': b'{}'})
    corrupted = patch_eocd_field(path.read_bytes(), cd_size=10 * 2**20)
    path.write_bytes(corrupted)
    tight = GENEROUS._replace(max_central_directory_bytes=1024)
    with pytest.raises(SafeZipError, match='max_central_directory_bytes'):
        with safe_zip_open(str(path), profile=tight):
            pass


def test_zip64_sentinel_entry_count_without_locator_rejected(tmp_path):
    # Sentinel values with no ZIP64 locator to resolve them against: unusable, no fallback.
    path = tmp_path / 'small.zip'
    make_zip(path, {'a.json': b'{}'})
    corrupted = patch_eocd_field(path.read_bytes(), total_entries=0xFFFF)
    path.write_bytes(corrupted)
    with pytest.raises(SafeZipError, match='ZIP64'):
        with safe_zip_open(str(path), profile=GENEROUS):
            pass


def test_zip64_locator_with_valid_record_is_parsed_not_rejected(tmp_path):
    # Real .pt2/.pt archives (torch.export.save/torch.save) always carry a ZIP64 locator, even
    # when small -- confirmed empirically. A well-formed one must be parsed, not rejected.
    path = tmp_path / 'small.zip'
    make_zip(path, {'a.json': b'{}', 'b.json': b'{}'})
    original = path.read_bytes()
    idx = find_eocd_offset(original)
    eocd = original[idx:idx + 22]
    _disk, _start, entries_this_disk, total_entries, cd_size, cd_offset, _comment = \
        struct.unpack('<HHHHIIH', eocd[4:])
    zip64_record = (
        b'PK\x06\x06' + struct.pack('<Q', 44) + struct.pack('<HH', 45, 45)
        + struct.pack('<II', 0, 0) + struct.pack('<QQQQ', entries_this_disk, total_entries,
                                                  cd_size, cd_offset)
    )
    zip64_eocd_offset = idx
    zip64_locator = b'PK\x06\x07' + struct.pack('<IQI', 0, zip64_eocd_offset, 1)
    patched = original[:idx] + zip64_record + zip64_locator + original[idx:]
    path.write_bytes(patched)
    with safe_zip_open(str(path), profile=GENEROUS) as z:
        assert z.read_member('a.json') == b'{}'


def test_zip64_locator_pointing_out_of_range_rejected(tmp_path):
    path = tmp_path / 'small.zip'
    make_zip(path, {'a.json': b'{}'})
    original = path.read_bytes()
    idx = find_eocd_offset(original)
    bogus_locator = b'PK\x06\x07' + struct.pack('<IQI', 0, len(original) + 10_000, 1)
    patched = original[:idx] + bogus_locator + original[idx:]
    path.write_bytes(patched)
    with pytest.raises(SafeZipError, match='out of range'):
        with safe_zip_open(str(path), profile=GENEROUS):
            pass


def test_zip64_locator_multi_disk_rejected(tmp_path):
    path = tmp_path / 'small.zip'
    make_zip(path, {'a.json': b'{}'})
    original = path.read_bytes()
    idx = find_eocd_offset(original)
    multi_disk_locator = b'PK\x06\x07' + struct.pack('<IQI', 0, 0, 2)  # total_disks=2
    patched = original[:idx] + multi_disk_locator + original[idx:]
    path.write_bytes(patched)
    with pytest.raises(SafeZipError, match='multi-disk'):
        with safe_zip_open(str(path), profile=GENEROUS):
            pass


# ---------------------------------------------------------------------------- per-member bounds


def test_member_size_cap_rejected(tmp_path):
    path = tmp_path / 'big_member.zip'
    make_zip(path, {'blob.bin': b'\x00' * (2 * 2**20)})
    tight = GENEROUS._replace(max_member_bytes=1024)
    with pytest.raises(SafeZipError, match='member cap'):
        with safe_zip_open(str(path), profile=tight):
            pass


def test_json_member_cap_applies_independently_of_general_member_cap(tmp_path):
    path = tmp_path / 'json_member.zip'
    # Small enough to satisfy the general (large) per-member cap, but bigger than
    # JSON_METADATA_PROFILE's own cap.
    big_json = b'{"x": "' + b'a' * (archive.JSON_METADATA_PROFILE.max_member_bytes + 1) + b'"}'
    make_zip(path, {'contract.json': big_json})
    with pytest.raises(SafeZipError, match='member cap'):
        with safe_zip_open(str(path), profile=GENEROUS):
            pass


def test_compression_ratio_bomb_rejected(tmp_path):
    path = tmp_path / 'bomb.zip'
    # All-zero content compresses to a tiny fraction of its size under deflate.
    make_zip(path, {'blob.bin': b'\x00' * (4 * 2**20)}, compression=zipfile.ZIP_DEFLATED)
    tight_ratio = GENEROUS._replace(max_compression_ratio=10)
    with pytest.raises(SafeZipError, match='compression ratio'):
        with safe_zip_open(str(path), profile=tight_ratio):
            pass


def test_total_uncompressed_cap_rejected(tmp_path):
    path = tmp_path / 'many.zip'
    make_zip(path, {f'{i}.bin': b'\x00' * (10 * 1024) for i in range(10)})
    tight = GENEROUS._replace(max_total_uncompressed=1024)
    with pytest.raises(SafeZipError, match='max_total_uncompressed'):
        with safe_zip_open(str(path), profile=tight):
            pass


# ---------------------------------------------------------------------------- _with_validated_source


def test_with_validated_source_yields_seeked_handle_and_proc_fd_path(tmp_path):
    path = tmp_path / 'a.zip'
    make_zip(path, {'a.json': b'{}'})
    seen = {}

    def loader(handle, proc_fd_path):
        seen['tell'] = handle.tell()
        seen['proc_fd_path'] = proc_fd_path
        seen['first_bytes'] = handle.read(4)
        return 'ok'

    with safe_zip_open(str(path), profile=GENEROUS) as z:
        result = z._with_validated_source(loader)
    assert result == 'ok'
    assert seen['tell'] == 0
    assert seen['proc_fd_path'].startswith('/proc/self/fd/')
    assert seen['first_bytes'] == b'PK\x03\x04'


def test_with_validated_source_bytes_input_has_no_proc_fd_path():
    data = make_zip_bytes({'a.json': b'{}'})
    seen = {}

    def loader(handle, proc_fd_path):
        seen['proc_fd_path'] = proc_fd_path
        return 'ok'

    with safe_zip_open(data, profile=GENEROUS) as z:
        z._with_validated_source(loader)
    assert seen['proc_fd_path'] is None


def test_read_member_unsupported_after_with_validated_source(tmp_path):
    path = tmp_path / 'a.zip'
    make_zip(path, {'a.json': b'{}'})
    with safe_zip_open(str(path), profile=GENEROUS) as z:
        z._with_validated_source(lambda handle, proc_fd_path: None)
        with pytest.raises(SafeZipError, match='not supported afterward'):
            z.read_member('a.json')


def test_with_validated_source_closing_handle_does_not_raise_on_exit(tmp_path):
    path = tmp_path / 'a.zip'
    make_zip(path, {'a.json': b'{}'})
    with safe_zip_open(str(path), profile=GENEROUS) as z:
        z._with_validated_source(lambda handle, proc_fd_path: handle.close())
    # No exception on context exit even though the handle was already closed inside the callback.


# ---------------------------------------------------------------------------- replacement race


def test_open_descriptor_immune_to_path_replacement_after_open(tmp_path):
    path = tmp_path / 'a.zip'
    make_zip(path, {'a.json': b'{"original": true}'})
    replacement = tmp_path / 'b.zip'
    make_zip(replacement, {'a.json': b'{"replacement": true}'})

    with safe_zip_open(str(path), profile=GENEROUS) as z:
        os.replace(str(replacement), str(path))  # swap the file at `path` mid-`with`
        assert z.read_member('a.json') == b'{"original": true}'
