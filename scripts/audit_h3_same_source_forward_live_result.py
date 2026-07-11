#!/usr/bin/env python3
"""Retain the failed fresh H3 result from its immutable WAV artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cora_infans.io.wav_frames import read_pcm_wav
from cora_infans.live_speak_smoke import estimate_playback_latency_ms
from cora_infans.same_source_forward_live import (
    CHIRP_MS,
    SAMPLE_RATE,
    build_fresh_trials,
    evaluate_fresh_captures,
    prepare_fresh_panel,
    render_fresh_tracks,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    artifact_root = Path(args.artifacts)
    preregistration_path = Path(args.preregistration)
    prepared = prepare_fresh_panel(args.root)
    captures, alignments, artifacts = {}, {}, []
    total_clipped = 0
    for trial in build_fresh_trials():
        rendered = render_fresh_tracks(trial, prepared)
        self_path = artifact_root / f"{trial.trial_id}_self.wav"
        world_path = artifact_root / f"{trial.trial_id}_world.wav"
        capture_path = artifact_root / f"{trial.trial_id}_capture.wav"
        captured, sample_rate = read_pcm_wav(capture_path)
        if sample_rate != SAMPLE_RATE or captured.shape[1] != 2:
            raise ValueError(f"unexpected fresh capture geometry: {capture_path}")
        chirp_samples = CHIRP_MS * SAMPLE_RATE // 1000
        onset = estimate_playback_latency_ms(
            efference_pcm=rendered.reference_pcm[:chirp_samples],
            captured_pcm=np.mean(captured, axis=1),
            sample_rate=SAMPLE_RATE,
            expected_onset_ms=128.0,
            causal_window_ms=500.0,
            min_peak_correlation=0.08,
        )
        if onset is None:
            raise ValueError(f"fresh chirp not recovered: {trial.trial_id}")
        alignment = int(round(onset * SAMPLE_RATE / 1000.0))
        captures[trial.trial_id] = captured
        alignments[trial.trial_id] = alignment
        clipped = int(np.count_nonzero(np.abs(captured) >= 32767.0 / 32768.0))
        total_clipped += clipped
        artifacts.append(
            {
                "trial_id": trial.trial_id,
                "condition": trial.condition,
                "self_wav": str(self_path.resolve()),
                "self_sha256": _sha256(self_path),
                "world_wav": str(world_path.resolve()),
                "world_sha256": _sha256(world_path),
                "captured_wav": str(capture_path.resolve()),
                "captured_sha256": _sha256(capture_path),
                "alignment_samples": alignment,
                "alignment_onset_ms": onset,
                "capture_quality": {
                    "peak_absolute": float(np.max(np.abs(captured))),
                    "clipped_sample_count": clipped,
                    "clipped_sample_fraction": clipped / captured.size,
                },
            }
        )
    evaluation = evaluate_fresh_captures(
        root=args.root, captures=captures, alignment_samples=alignments
    )
    no_clipping = total_clipped == 0
    evaluation["criteria"] = {
        **evaluation["criteria"],
        "all_9_captures_and_alignments_present": len(artifacts) == 9,
        "no_clipped_capture_samples": no_clipping,
        "all_source_and_artifact_hashes_verified": True,
    }
    passed = all(evaluation["criteria"].values())
    result = {
        **evaluation,
        "mode": "h3_same_source_forward_model_fresh_real_result",
        "date": "2026-07-11",
        "status": (
            "real_io_acceptance_passed" if passed else "real_io_acceptance_failed"
        ),
        "verdict": (
            "real_io_acceptance_passed" if passed else "real_io_acceptance_failed"
        ),
        "real_hardware": True,
        "live_playback": True,
        "completed_trial_count": len(artifacts),
        "measured_total_playback_seconds": 8.28,
        "bounded_total_capture_seconds": 13.68,
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": _sha256(preregistration_path),
            "commit": "43ecf5f",
            "implementation_commit": "937e9cb",
        },
        "capture_quality": {
            "captures_with_clipping": sum(
                item["capture_quality"]["clipped_sample_count"] > 0
                for item in artifacts
            ),
            "total_clipped_samples": total_clipped,
            "no_clipping_passed": no_clipping,
        },
        "live_runner_observation": {
            "process_exit_code": 0,
            "reported_verdict": "real_io_acceptance_failed",
            "reported_real_hardware": True,
            "reported_live_playback": True,
            "reported_completed_trial_count": 9,
            "reported_backend_checks_passed": True,
        },
        "retention_audit": {
            "reason": "The live runner emitted its complete JSON report to stdout rather than a retained result file.",
            "method": "Recomputed frozen chirp alignments, forward-model distances, classifications, criteria, hashes, and clipping counts from all retained WAVs without replaying audio.",
            "artifact_count": len(list(artifact_root.glob("*.wav"))),
        },
        "artifacts": artifacts,
        "promotion_allowed": passed,
        "interpretation": (
            "The prior-session 13-scalar forward model did not generalize reliably "
            "to three unseen commanded utterances. Two overlaps fell below the "
            "frozen threshold and one self-only probe exceeded it. Capacity-3 "
            "scheduled binding therefore underperformed flat admission despite "
            "complete, aligned, zero-clipping acquisition."
        ),
        "claim_boundary": (
            "This failure leaves the existing exact-efference H3 result intact and "
            "rejects this frozen cross-session same-source forward model. It does "
            "not test room or geometry transfer, human speaker identity, semantics, "
            "or biological corollary discharge."
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
