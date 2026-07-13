#!/usr/bin/env python3
"""Run the frozen nonlinear predictive codebook development audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cora_infans.io.ids_corpus import IDS_FILENAME, IDS_LABELS
from cora_infans.io.offline_eval import Annotation, CorpusClip
from cora_infans.io.wav_frames import read_wav_metadata
from cora_infans.nonlinear_predictive_codebook import (
    NonlinearPredictiveCodebookModel,
    nonlinear_codebook_leave_one_fieldsite_out,
    select_nonlinear_codebook_thresholds,
)
from cora_infans.predictive_register_dynamics import build_prosodic_sequence_example


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
        raise ValueError("nonlinear audit requires the spent IDS manifest")
    if manifest.get("clip_count") != 200:
        raise ValueError("nonlinear audit requires exactly 200 spent clips")
    if manifest.get("reserved_fieldsites_accessed") is not False:
        raise ValueError("nonlinear audit crossed the reserved fieldsite boundary")

    clips = []
    for record in manifest["clips"]:
        name = str(record["name"])
        match = IDS_FILENAME.fullmatch(name)
        if match is None:
            raise ValueError(f"invalid IDS filename in manifest: {name}")
        path = root / name
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"spent IDS hash mismatch during nonlinear audit: {name}")
        sample_rate, _channels, frame_count = read_wav_metadata(path)
        duration = frame_count / sample_rate
        clips.append(
            CorpusClip(
                clip_id=path.stem,
                wav_path=path,
                annotations=(
                    Annotation(0.0, duration, IDS_LABELS[match.group("kind").upper()]),
                ),
                split="dev",
                sample_rate=sample_rate,
            )
        )

    examples = [build_prosodic_sequence_example(clip) for clip in clips]
    if not all(bool(np.all(np.isfinite(example.states))) for example in examples):
        raise ValueError("nonlinear audit contains non-finite sequences")
    report = nonlinear_codebook_leave_one_fieldsite_out(examples)
    threshold_selection = select_nonlinear_codebook_thresholds(
        report["predictions"]
    )
    threshold_metrics = threshold_selection["metrics"]
    song_rate = (
        threshold_selection["song_false_positive_count"]
        / threshold_selection["song_count"]
    )
    minimum_fieldsite_f1 = min(
        float(value["f1"])
        for value in threshold_selection["per_fieldsite"].values()
    )
    candidate_ready = (
        float(threshold_metrics["f1"]) >= 0.80
        and float(threshold_metrics["false_positive_rate"]) <= 0.15
        and float(song_rate) <= 0.15
        and minimum_fieldsite_f1 >= 0.50
        and float(report["predicted_vocal_count"]) / len(examples) >= 0.95
        and float(report["minimum_codebook_utilization_fraction"]) >= 0.75
        and bool(report["all_fold_quantization_errors_nonincreasing"])
        and int(report["fieldsite_count"]) == 18
        and int(report["prediction_count"]) == 200
    )
    full_model = NonlinearPredictiveCodebookModel.fit(examples)
    report.update(
        {
            "evidence_scope": "development_only_18_already_spent_fieldsites",
            "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "all_clip_hashes_verified": True,
            "all_sequences_finite": True,
            "threshold_selection": threshold_selection,
            "minimum_selected_fieldsite_f1": minimum_fieldsite_f1,
            "selected_song_false_positive_rate": song_rate,
            "development_candidate_ready": candidate_ready,
            "confirmation_audio_or_metadata_accessed": False,
            "promotion_allowed": False,
            "full_development_model": full_model.to_dict(),
        }
    )
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
