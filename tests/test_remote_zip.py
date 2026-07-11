import io
import zipfile

from cora_infans.io.remote_zip import RemoteZipIndex


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
