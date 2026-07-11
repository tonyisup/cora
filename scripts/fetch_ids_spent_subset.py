#!/usr/bin/env python3
"""Fetch only hash-pinned, already-spent Music Lab clips by ZIP byte range."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cora_infans.io.ids_corpus import collect_spent_ids_hashes
from cora_infans.io.remote_zip import RemoteZipIndex


DEFAULT_URL = (
    "https://zenodo.org/api/records/5525161/files/"
    "IDS-corpus-edited.zip/content"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", default="docs/evidence")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    expected = collect_spent_ids_hashes(args.evidence_dir)
    with urllib.request.urlopen(
        urllib.request.Request(args.url, method="HEAD")
    ) as response:
        archive_size = int(response.headers["Content-Length"])

    downloaded_bytes = 0
    download_lock = threading.Lock()

    def fetch_range(start: int, end: int) -> bytes:
        nonlocal downloaded_bytes
        request = urllib.request.Request(
            args.url, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request) as response:
            if response.status != 206:
                raise RuntimeError(
                    f"server did not honor byte range: {response.status}"
                )
            payload = response.read()
        with download_lock:
            downloaded_bytes += len(payload)
        return payload

    archive = RemoteZipIndex.build(
        archive_size=archive_size, fetch_range=fetch_range
    )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    def fetch_one(item: tuple[str, str]) -> dict[str, object]:
        name, expected_sha256 = item
        member_name = f"IDS-corpus-edited/{name}"
        if member_name not in archive.members:
            raise ValueError(f"spent IDS member is missing from archive: {name}")
        destination = output / name
        if destination.exists():
            existing = destination.read_bytes()
            if hashlib.sha256(existing).hexdigest() == expected_sha256:
                return {
                    "name": name,
                    "sha256": expected_sha256,
                    "bytes": len(existing),
                    "reused_local_file": True,
                }
        payload = archive.read(member_name)
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"spent IDS hash mismatch: {name}")
        destination.write_bytes(payload)
        return {
            "name": name,
            "sha256": actual_sha256,
            "bytes": len(payload),
            "reused_local_file": False,
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(fetch_one, expected.items()))

    manifest = {
        "mode": "hash_pinned_spent_ids_subset",
        "doi": "10.5281/zenodo.5525161",
        "license": "CC BY-NC-SA 4.0",
        "archive_url": args.url,
        "archive_size": archive_size,
        "central_directory_sha256": hashlib.sha256(
            archive.central_directory
        ).hexdigest(),
        "clip_count": len(records),
        "reused_local_clip_count": sum(
            bool(record["reused_local_file"]) for record in records
        ),
        "downloaded_range_bytes": downloaded_bytes,
        "reserved_fieldsites_accessed": False,
        "clips": records,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
