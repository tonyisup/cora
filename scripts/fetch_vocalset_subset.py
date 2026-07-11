#!/usr/bin/env python3
"""List or fetch selected VocalSet members without downloading the 2.1 GB ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path, PurePosixPath

from cora_infans.io.remote_zip import RemoteZipIndex


DEFAULT_URL = "https://zenodo.org/api/records/1193957/files/VocalSet.zip/content"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--list-contains")
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument("--out-dir")
    parser.add_argument("--manifest")
    args = parser.parse_args()

    with urllib.request.urlopen(urllib.request.Request(args.url, method="HEAD")) as response:
        archive_size = int(response.headers["Content-Length"])

    def fetch_range(start: int, end: int) -> bytes:
        request = urllib.request.Request(
            args.url, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request) as response:
            if response.status != 206:
                raise RuntimeError(f"server did not honor byte range: {response.status}")
            return response.read()

    archive = RemoteZipIndex.build(
        archive_size=archive_size, fetch_range=fetch_range
    )
    if args.list_contains is not None:
        for name in sorted(archive.members):
            if args.list_contains in name:
                print(name)

    if args.member:
        if not args.out_dir or not args.manifest:
            parser.error("--member requires --out-dir and --manifest")
        output = Path(args.out_dir)
        output.mkdir(parents=True, exist_ok=True)
        records = []
        for name in args.member:
            posix = PurePosixPath(name)
            if posix.is_absolute() or ".." in posix.parts:
                raise ValueError(f"unsafe archive member path: {name}")
            payload = archive.read(name)
            destination = output.joinpath(*posix.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            member = archive.members[name]
            records.append(
                {
                    "member": name,
                    "bytes": len(payload),
                    "crc32": f"{member.crc32:08x}",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "local_path": str(destination.resolve()),
                }
            )
        manifest = {
            "source": "VocalSet 1.0",
            "doi": "10.5281/zenodo.1193957",
            "license": "CC BY 4.0",
            "archive_url": args.url,
            "archive_size": archive_size,
            "central_directory_sha256": hashlib.sha256(
                archive.central_directory
            ).hexdigest(),
            "members": records,
        }
        Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
