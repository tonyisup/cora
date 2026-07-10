from __future__ import annotations

import csv
import wave

import numpy as np

from cora_infans.io.ids_corpus import build_ids_manifest


def _write_silence(path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(np.zeros(4000, dtype="<i2").tobytes())


def test_ids_manifest_maps_labels_and_groups_participant_splits(tmp_path) -> None:
    _write_silence(tmp_path / "USD01B.wav")
    _write_silence(tmp_path / "USD01D.wav")
    _write_silence(tmp_path / "TOR02B.wav")
    _write_silence(tmp_path / "TOR02A.wav")
    output = tmp_path / "labels.csv"

    report = build_ids_manifest(tmp_path, output)

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert report["clip_count"] == 3
    assert report["participant_count"] == 2
    assert report["participant_leakage_count"] == 0
    assert {row["label"] for row in rows} == {
        "infant_directed_speech",
        "adult_directed_speech",
    }
    usd_splits = {row["split"] for row in rows if row["participant_id"] == "USD01"}
    assert len(usd_splits) == 1
    assert "TOR02A.wav" not in {row["source_path"] for row in rows}


def test_ids_manifest_can_include_song_conditions(tmp_path) -> None:
    _write_silence(tmp_path / "USD01A.wav")
    _write_silence(tmp_path / "USD01C.wav")

    report = build_ids_manifest(
        tmp_path,
        tmp_path / "labels.csv",
        include_song=True,
    )

    assert report["label_counts"] == {
        "adult_directed_song": 1,
        "infant_directed_song": 1,
    }
