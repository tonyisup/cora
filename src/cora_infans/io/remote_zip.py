"""Bounded byte-range access to selected members of a remote ZIP archive."""

from __future__ import annotations

import binascii
import struct
import zlib
from dataclasses import dataclass
from typing import Callable


RangeFetcher = Callable[[int, int], bytes]
EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"


@dataclass(frozen=True)
class RemoteZipMember:
    name: str
    compression: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


class RemoteZipIndex:
    """ZIP central directory plus verified, selective member extraction."""

    def __init__(
        self,
        *,
        archive_size: int,
        fetch_range: RangeFetcher,
        members: dict[str, RemoteZipMember],
        central_directory: bytes,
    ) -> None:
        self.archive_size = archive_size
        self.fetch_range = fetch_range
        self.members = members
        self.central_directory = central_directory

    @classmethod
    def build(
        cls, *, archive_size: int, fetch_range: RangeFetcher
    ) -> "RemoteZipIndex":
        if archive_size < 22:
            raise ValueError("remote ZIP is too small")
        tail_size = min(archive_size, 65557)
        tail_start = archive_size - tail_size
        tail = fetch_range(tail_start, archive_size - 1)
        if len(tail) != tail_size:
            raise ValueError("remote ZIP tail range has the wrong length")
        eocd_offset = tail.rfind(EOCD_SIGNATURE)
        if eocd_offset < 0 or len(tail) - eocd_offset < 22:
            raise ValueError("remote ZIP end-of-central-directory record not found")
        (
            _signature,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
            comment_length,
        ) = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
        if disk_number != 0 or central_disk != 0 or disk_entries != total_entries:
            raise ValueError("multi-disk ZIP archives are not supported")
        if comment_length != len(tail) - eocd_offset - 22:
            raise ValueError("remote ZIP EOCD comment length is inconsistent")
        if central_offset + central_size > archive_size:
            raise ValueError("remote ZIP central directory is out of bounds")
        central = fetch_range(central_offset, central_offset + central_size - 1)
        if len(central) != central_size:
            raise ValueError("remote ZIP central-directory range has the wrong length")

        members: dict[str, RemoteZipMember] = {}
        offset = 0
        for _ in range(total_entries):
            if central[offset : offset + 4] != CENTRAL_SIGNATURE:
                raise ValueError("invalid ZIP central-directory member signature")
            fields = struct.unpack_from("<4s6H3L5H2L", central, offset)
            (
                _signature,
                _made_by,
                _needed,
                flags,
                compression,
                _time,
                _date,
                crc32,
                compressed_size,
                uncompressed_size,
                name_length,
                extra_length,
                comment_length,
                _disk_start,
                _internal,
                _external,
                local_header_offset,
            ) = fields
            start = offset + 46
            name_bytes = central[start : start + name_length]
            encoding = "utf-8" if flags & 0x800 else "cp437"
            name = name_bytes.decode(encoding)
            if name in members:
                raise ValueError(f"duplicate ZIP member: {name}")
            members[name] = RemoteZipMember(
                name=name,
                compression=compression,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
            )
            offset = start + name_length + extra_length + comment_length
        if offset != len(central):
            raise ValueError("ZIP central directory has trailing or missing bytes")
        return cls(
            archive_size=archive_size,
            fetch_range=fetch_range,
            members=members,
            central_directory=central,
        )

    def read(self, name: str) -> bytes:
        member = self.members[name]
        header = self.fetch_range(
            member.local_header_offset, member.local_header_offset + 29
        )
        if len(header) != 30 or header[:4] != LOCAL_SIGNATURE:
            raise ValueError(f"invalid local ZIP header: {name}")
        fields = struct.unpack("<4s5H3L2H", header)
        flags = fields[2]
        compression = fields[3]
        name_length = fields[9]
        extra_length = fields[10]
        if flags & 0x1:
            raise ValueError(f"encrypted ZIP member is unsupported: {name}")
        if compression != member.compression:
            raise ValueError(f"ZIP compression metadata mismatch: {name}")
        data_start = member.local_header_offset + 30 + name_length + extra_length
        compressed = self.fetch_range(
            data_start, data_start + member.compressed_size - 1
        )
        if len(compressed) != member.compressed_size:
            raise ValueError(f"short ZIP member range: {name}")
        if member.compression == 0:
            payload = compressed
        elif member.compression == 8:
            payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
        else:
            raise ValueError(f"unsupported ZIP compression method: {member.compression}")
        if len(payload) != member.uncompressed_size:
            raise ValueError(f"ZIP member size mismatch: {name}")
        if binascii.crc32(payload) & 0xFFFFFFFF != member.crc32:
            raise ValueError(f"ZIP member CRC mismatch: {name}")
        return payload
