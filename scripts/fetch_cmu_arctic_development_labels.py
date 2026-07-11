#!/usr/bin/env python3
"""Fetch a bounded prefix of labels for the frozen CMU Arctic dev speakers."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cora_infans.cmu_arctic import parse_label_index


DEVELOPMENT_SPEAKERS = ("bdl", "slt", "clb", "jmk", "ksp")
BASE = "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_{speaker}_arctic"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-labels", type=int, default=300)
    args = parser.parse_args()
    if args.max_labels <= 0:
        raise ValueError("--max-labels must be positive")
    root = Path(args.out)
    records = []
    for speaker in DEVELOPMENT_SPEAKERS:
        base = BASE.format(speaker=speaker)
        index_url = f"{base}/lab/"
        index_bytes = fetch(index_url)
        names = parse_label_index(index_bytes.decode("utf-8"))[: args.max_labels]
        label_dir = root / speaker / "lab"
        label_dir.mkdir(parents=True, exist_ok=True)

        def get(name: str) -> tuple[str, str, bytes]:
            url = f"{base}/lab/{name}"
            return name, url, fetch(url)

        with ThreadPoolExecutor(max_workers=8) as executor:
            fetched = list(executor.map(get, names))
        for name, url, payload in fetched:
            path = label_dir / name
            path.write_bytes(payload)
            records.append(
                {
                    "speaker": speaker,
                    "kind": "label",
                    "name": name,
                    "url": url,
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        records.append(
            {
                "speaker": speaker,
                "kind": "label_index",
                "url": index_url,
                "listed_label_count": len(parse_label_index(index_bytes.decode("utf-8"))),
                "fetched_label_count": len(names),
                "sha256": hashlib.sha256(index_bytes).hexdigest(),
            }
        )
    manifest = {
        "mode": "cmu_arctic_development_label_prefix",
        "development_speakers": list(DEVELOPMENT_SPEAKERS),
        "confirmation_audio_or_labels_accessed": False,
        "max_labels_per_speaker": args.max_labels,
        "records": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"root": str(root), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
