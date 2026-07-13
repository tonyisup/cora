import io
import zipfile

from cora_infans.io.remote_zip import (
    UINT32_MODULUS,
    RemoteZipIndex,
    _resolve_wrapped_central_offset,
    _unwrap_member_offset,
)


def test_remote_zip_reads_only_requested_verified_members() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nested/a.txt", b"alpha")
        archive.writestr("nested/b.txt", b"beta" * 100)
    payload = stream.getvalue()
    ranges = []

    def fetch(start, end):
        ranges.append((start, end))
        return payload[start : end + 1]

    remote = RemoteZipIndex.build(archive_size=len(payload), fetch_range=fetch)

    assert set(remote.members) == {"nested/a.txt", "nested/b.txt"}
    assert remote.read("nested/b.txt") == b"beta" * 100
    assert sum(end - start + 1 for start, end in ranges) < len(payload) * 2


def test_remote_zip_rejects_corrupt_member_bytes() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("a.txt", b"alpha")
    payload = bytearray(stream.getvalue())

    def fetch(start, end):
        data = bytes(payload[start : end + 1])
        if len(data) == 5 and data == b"alpha":
            return b"omega"
        return data

    remote = RemoteZipIndex.build(archive_size=len(payload), fetch_range=fetch)
    try:
        remote.read("a.txt")
    except ValueError as exc:
        assert "CRC mismatch" in str(exc)
    else:
        raise AssertionError("corrupt member must fail CRC verification")


def test_remote_zip_resolves_classic_offsets_wrapped_after_four_gib() -> None:
    central_size = 162_010
    actual_central_offset = 5_781_983_669
    declared_offset = actual_central_offset - UINT32_MODULUS

    assert _resolve_wrapped_central_offset(
        declared_offset,
        central_size=central_size,
        eocd_offset=actual_central_offset + central_size,
    ) == actual_central_offset

    before_wrap, epoch = _unwrap_member_offset(
        4_294_601_280,
        previous_raw_offset=None,
        offset_epoch=0,
        wrapped_archive=True,
    )
    after_wrap, epoch = _unwrap_member_offset(
        2_853_104,
        previous_raw_offset=before_wrap,
        offset_epoch=epoch,
        wrapped_archive=True,
    )

    assert before_wrap == 4_294_601_280
    assert after_wrap == UINT32_MODULUS + 2_853_104
    assert epoch == UINT32_MODULUS
