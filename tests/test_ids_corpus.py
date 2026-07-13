from __future__ import annotations

import csv
import json
import wave

import numpy as np
import pytest

from cora_infans.io.ids_corpus import build_ids_manifest, collect_spent_ids_hashes


def _write_silence(path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(np.zeros(4000, dtype="<i2").tobytes())


def test_ids_manifest_maps_labels_and_groups_participant_splits(tmp_path) -> None:
    _write_silence(tmp_path / "BEJ01B.wav")
    _write_silence(tmp_path / "BEJ01D.wav")
    _write_silence(tmp_path / "MEN02B.wav")
    _write_silence(tmp_path / "MEN02A.wav")
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
    bej_splits = {row["split"] for row in rows if row["participant_id"] == "BEJ01"}
    assert len(bej_splits) == 1
    assert "MEN02A.wav" not in {row["source_path"] for row in rows}


def test_ids_manifest_can_include_song_conditions(tmp_path) -> None:
    _write_silence(tmp_path / "BEJ01A.wav")
    _write_silence(tmp_path / "BEJ01C.wav")

    report = build_ids_manifest(
        tmp_path,
        tmp_path / "labels.csv",
        include_song=True,
    )

    assert report["label_counts"] == {
        "adult_directed_song": 1,
        "infant_directed_song": 1,
    }


def test_ids_manifest_rejects_protected_fieldsite_before_metadata_read(tmp_path) -> None:
    (tmp_path / "USD01B.wav").write_bytes(b"not a wav")

    with pytest.raises(ValueError, match="Stage 2 fieldsite is protected"):
        build_ids_manifest(tmp_path, tmp_path / "labels.csv")


def test_spent_ids_hash_collection_enforces_exact_frozen_boundary(tmp_path) -> None:
    spent = [
        "ACO", "ARA", "BEJ", "HAD", "JEN", "KRA", "LIM", "MBE", "MEN",
        "MES", "NYA", "PNG", "QUE", "SPA", "TOP", "TSI", "TUR", "VAN",
    ]
    records = {}
    index = 0
    while len(records) < 200:
        fieldsite = spent[index % len(spent)]
        participant = (index // (len(spent) * 4)) + 1
        condition = "ABCD"[(index // len(spent)) % 4]
        records[f"{fieldsite}{participant:02d}{condition}.wav"] = "a" * 64
        index += 1
    (tmp_path / "result.json").write_text(
        json.dumps({"selected_file_sha256": records})
    )

    assert collect_spent_ids_hashes(tmp_path) == dict(sorted(records.items()))

    records["WEL01B.wav"] = "b" * 64
    (tmp_path / "result.json").write_text(
        json.dumps({"selected_file_sha256": records})
    )
    with pytest.raises(ValueError, match="reserved IDS fieldsite"):
        collect_spent_ids_hashes(tmp_path)
