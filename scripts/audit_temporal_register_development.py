#!/usr/bin/env python3
"""Evaluate the temporal register candidate on only already-spent IDS clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cora_infans.io.ids_corpus import IDS_FILENAME, IDS_LABELS
from cora_infans.io.offline_eval import Annotation, CorpusClip
from cora_infans.io.wav_frames import read_wav_metadata
from cora_infans.temporal_register import (
    build_temporal_register_example,
    evaluate_temporal_leave_one_fieldsite_out,
    select_temporal_register_thresholds,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    manifest_path = Path(args.manifest)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("mode") != "hash_pinned_spent_ids_subset":
        raise ValueError("temporal audit requires the spent IDS manifest")
    if manifest.get("clip_count") != 200:
        raise ValueError("temporal audit requires exactly 200 spent clips")
    if manifest.get("reserved_fieldsites_accessed") is not False:
        raise ValueError("temporal audit crossed the reserved fieldsite boundary")

    clips = []
    for record in manifest["clips"]:
        name = str(record["name"])
        match = IDS_FILENAME.fullmatch(name)
        if match is None:
            raise ValueError(f"invalid IDS filename in manifest: {name}")
        path = root / name
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"spent IDS hash mismatch during audit: {name}")
        sample_rate, _channels, frame_count = read_wav_metadata(path)
        duration = frame_count / sample_rate
        label = IDS_LABELS[match.group("kind").upper()]
        clips.append(
            CorpusClip(
                clip_id=path.stem,
                wav_path=path,
                annotations=(Annotation(0.0, duration, label),),
                split="dev",
                sample_rate=sample_rate,
            )
        )

    examples = [build_temporal_register_example(clip) for clip in clips]
    report = evaluate_temporal_leave_one_fieldsite_out(examples)
    threshold_selection = select_temporal_register_thresholds(report["predictions"])
    threshold_metrics = threshold_selection["metrics"]
    threshold_song_rate = (
        threshold_selection["song_false_positive_count"]
        / threshold_selection["song_count"]
    )
    threshold_minimum_fieldsite_f1 = min(
        float(value["f1"])
        for value in threshold_selection["per_fieldsite"].values()
    )
    report.update(
        {
            "evidence_scope": "development_only_18_already_spent_fieldsites",
            "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "threshold_selection": threshold_selection,
            "acceptance": {
                "minimum_pooled_f1": 0.80,
                "maximum_pooled_false_positive_rate": 0.15,
                "maximum_song_false_positive_rate": 0.15,
                "minimum_each_fieldsite_f1": 0.50,
                "minimum_vocal_recall": 0.95,
            },
        }
    )
    report["development_candidate_ready"] = (
        float(threshold_metrics["f1"]) >= 0.80
        and float(threshold_metrics["false_positive_rate"]) <= 0.15
        and float(threshold_song_rate) <= 0.15
        and threshold_minimum_fieldsite_f1 >= 0.50
        and float(report["vocal_recall"]) >= 0.95
    )
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
