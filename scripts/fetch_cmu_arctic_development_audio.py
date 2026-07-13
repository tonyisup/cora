#!/usr/bin/env python3
"""Fetch only WAVs selected from the frozen CMU Arctic development labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cora_infans.cmu_arctic import select_tokens


DEVELOPMENT_SPEAKERS = ("bdl", "slt", "clb", "jmk", "ksp")
BASE = "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_{speaker}_arctic"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--per-vowel", type=int, default=20)
    parser.add_argument("--minimum-duration-ms", type=float, default=120.0)
    parser.add_argument("--exclude-label", action="append", default=[])
    args = parser.parse_args()
    root = Path(args.root)
    label_manifest = json.loads((root / "manifest.json").read_text())
    if label_manifest.get("confirmation_audio_or_labels_accessed") is not False:
        raise ValueError("development label manifest has invalid confirmation boundary")
    if tuple(label_manifest.get("development_speakers", ())) != DEVELOPMENT_SPEAKERS:
        raise ValueError("development speaker split drift")

    all_tokens = []
    wav_records = []
    exclusions: dict[str, set[str]] = {speaker: set() for speaker in DEVELOPMENT_SPEAKERS}
    for value in args.exclude_label:
        speaker, separator, name = value.partition(":")
        if separator != ":" or speaker not in exclusions or not name.endswith(".lab"):
            raise ValueError("--exclude-label must be frozen-speaker:label.lab")
        exclusions[speaker].add(name)
    minimum_duration_seconds = args.minimum_duration_ms / 1000.0
    for speaker in DEVELOPMENT_SPEAKERS:
        tokens = select_tokens(
            root / speaker / "lab",
            per_vowel=args.per_vowel,
            minimum_duration_seconds=minimum_duration_seconds,
            excluded_label_names=exclusions[speaker],
        )
        for token in tokens:
            token["speaker"] = speaker
        all_tokens.extend(tokens)
        wav_names = sorted({str(token["wav_name"]) for token in tokens})
        wav_dir = root / speaker / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        base = BASE.format(speaker=speaker)

        def get(name: str) -> tuple[str, str, bytes]:
            url = f"{base}/wav/{name}"
            return name, url, fetch(url)

        with ThreadPoolExecutor(max_workers=8) as executor:
            fetched = list(executor.map(get, wav_names))
        for name, url, payload in fetched:
            path = wav_dir / name
            path.write_bytes(payload)
            wav_records.append(
                {
                    "speaker": speaker,
                    "name": name,
                    "url": url,
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    output = {
        "mode": "cmu_arctic_development_audio_subset",
        "development_speakers": list(DEVELOPMENT_SPEAKERS),
        "confirmation_audio_or_labels_accessed": False,
        "selection": {
            "per_vowel_per_speaker": args.per_vowel,
            "minimum_duration_ms": args.minimum_duration_ms,
            "ordering": "label filename lexical, then segment order",
            "source_integrity_exclusions": {
                speaker: sorted(names)
                for speaker, names in exclusions.items()
                if names
            },
        },
        "tokens": all_tokens,
        "wav_files": wav_records,
    }
    (root / "audio_manifest.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "root": str(root),
                "token_count": len(all_tokens),
                "wav_count": len(wav_records),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
