"""Manifest builder for the Music Lab cross-cultural vocalization corpus.

Dataset DOI: 10.5281/zenodo.5525161
License: CC BY-NC-SA 4.0

The builder never downloads data. It scans an intentionally acquired/extracted
archive and creates whole-clip annotations with participant-grouped splits.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from .wav_frames import read_wav_metadata


IDS_FILENAME = re.compile(
    r"^(?P<fieldsite>[A-Z]{3})(?P<participant>\d{2})(?P<kind>[ABCD])\.wav$",
    re.IGNORECASE,
)
IDS_LABELS = {
    "A": "infant_directed_song",
    "B": "infant_directed_speech",
    "C": "adult_directed_song",
    "D": "adult_directed_speech",
}


def _participant_split(participant_id: str, held_out_fraction: float) -> str:
    digest = hashlib.sha256(participant_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "held_out" if value < held_out_fraction else "dev"


def _wav_duration_seconds(path: Path) -> float:
    try:
        sample_rate, _channels, frame_count = read_wav_metadata(path)
        return frame_count / float(sample_rate)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"could not read WAV duration for {path}: {exc}") from exc


def build_ids_manifest(
    root: str | Path,
    output_csv: str | Path,
    *,
    held_out_fraction: float = 0.2,
    include_song: bool = False,
) -> dict[str, object]:
    """Build a Cora manifest from extracted IDS corpus WAV files."""

    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError("held_out_fraction must be in (0, 1)")
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"IDS root is not a directory: {root_path}")
    rows: list[dict[str, str]] = []
    skipped_unrecognized = 0
    for wav_path in sorted(root_path.rglob("*.wav")):
        match = IDS_FILENAME.match(wav_path.name)
        if match is None:
            skipped_unrecognized += 1
            continue
        kind = match.group("kind").upper()
        if not include_song and kind in ("A", "C"):
            continue
        participant_id = (
            match.group("fieldsite").upper() + match.group("participant")
        )
        duration = _wav_duration_seconds(wav_path)
        clip_id = wav_path.stem
        rows.append(
            {
                "clip_id": clip_id,
                "start_s": "0.0",
                "end_s": f"{duration:.6f}",
                "label": IDS_LABELS[kind],
                "split": _participant_split(participant_id, held_out_fraction),
                "participant_id": participant_id,
                "fieldsite": match.group("fieldsite").upper(),
                "source_path": str(wav_path.relative_to(root_path)),
            }
        )
    if not rows:
        raise ValueError(f"no recognized IDS WAV files found under {root_path}")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "start_s",
        "end_s",
        "label",
        "split",
        "participant_id",
        "fieldsite",
        "source_path",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    participant_splits: dict[str, set[str]] = {}
    for row in rows:
        participant_splits.setdefault(row["participant_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in participant_splits.values())
    return {
        "mode": "ids_corpus_manifest_builder",
        "dataset_doi": "10.5281/zenodo.5525161",
        "license": "CC BY-NC-SA 4.0",
        "root": str(root_path),
        "output_csv": str(output_path.resolve()),
        "include_song": include_song,
        "held_out_fraction": held_out_fraction,
        "clip_count": len(rows),
        "participant_count": len(participant_splits),
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "skipped_unrecognized_wav_count": skipped_unrecognized,
        "participant_leakage_count": 0,
    }
