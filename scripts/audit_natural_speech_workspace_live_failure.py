#!/usr/bin/env python3
"""Retain a completed natural-speech H3 run after evaluator gap failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import cora_infans.natural_speech_workspace_binding as workspace
from cora_infans.io.wav_frames import read_pcm_wav
from cora_infans.live_speak_smoke import estimate_playback_latency_ms


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
    preregistration = json.loads(preregistration_path.read_text())
    prepared = workspace.prepare_natural_speech_panel(args.root)
    evidence: dict[str, float] = {}
    artifacts = []
    total_clipped = 0
    captures_with_clipping = 0

    for trial in workspace.build_natural_speech_trials():
        rendered = workspace.render_natural_speech_tracks(trial, prepared)
        self_path = artifact_root / f"{trial.trial_id}_self.wav"
        world_path = artifact_root / f"{trial.trial_id}_world.wav"
        capture_path = artifact_root / f"{trial.trial_id}_capture.wav"
        captured, sample_rate = read_pcm_wav(capture_path)
        if sample_rate != workspace.SAMPLE_RATE or captured.shape[1] != 2:
            raise ValueError(f"unexpected capture geometry: {capture_path}")
        chirp_samples = workspace.CHIRP_MS * workspace.SAMPLE_RATE // 1000
        # The live runner lost its per-trial raw frame offset when scoring raised.
        # Re-run the same frozen chirp correlation in its 500 ms causal window,
        # anchored at the nominal two-block (128 ms) callback preroll.  This is a
        # recovery audit only and is never eligible to promote the failed run.
        onset_ms = estimate_playback_latency_ms(
            efference_pcm=rendered.reference_pcm[:chirp_samples],
            captured_pcm=np.mean(captured, axis=1),
            sample_rate=workspace.SAMPLE_RATE,
            expected_onset_ms=128.0,
            causal_window_ms=500.0,
            min_peak_correlation=0.08,
        )
        if onset_ms is None:
            raise ValueError(f"could not reconstruct chirp onset: {trial.trial_id}")
        alignment = int(round(onset_ms * workspace.SAMPLE_RATE / 1000.0))
        ratio = workspace._residual_ratio(rendered, captured, alignment)
        evidence[trial.trial_id] = ratio
        clipped = int(np.count_nonzero(np.abs(captured) >= (32767.0 / 32768.0)))
        total_clipped += clipped
        captures_with_clipping += int(clipped > 0)
        artifacts.append(
            {
                "trial_id": trial.trial_id,
                "condition": trial.condition,
                "chirp_source": rendered.chirp_source,
                "self_wav": str(self_path.resolve()),
                "self_sha256": _sha256(self_path),
                "world_wav": str(world_path.resolve()),
                "world_sha256": _sha256(world_path),
                "captured_wav": str(capture_path.resolve()),
                "captured_sha256": _sha256(capture_path),
                "reconstructed_alignment_samples": alignment,
                "reconstructed_alignment_onset_ms": onset_ms,
                "residual_ratio": ratio,
                "capture_quality": {
                    "peak_absolute": float(np.max(np.abs(captured))),
                    "clipped_sample_count": clipped,
                    "clipped_sample_fraction": clipped / captured.size,
                },
            }
        )

    trials = workspace.build_natural_speech_trials()
    development_self = [
        evidence[trial.trial_id]
        for trial in trials
        if trial.split == "development" and not trial.expected_world_usable
    ]
    development_world = [
        evidence[trial.trial_id]
        for trial in trials
        if trial.split == "development" and trial.expected_world_usable
    ]
    maximum_self = max(development_self)
    minimum_world = min(development_world)
    gap = minimum_world - maximum_self
    result = {
        "mode": "natural_speech_workspace_binding_real_failure_audit",
        "date": "2026-07-11",
        "status": "completed_live_playback_capture_failed_frozen_evaluation",
        "verdict": "real_io_acceptance_failed",
        "real_hardware": True,
        "live_playback": True,
        "completed_trial_count": len(artifacts),
        "expected_artifact_count": 45,
        "retained_artifact_count": len(list(artifact_root.glob("*.wav"))),
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": _sha256(preregistration_path),
            "commit": "20d8fa6",
            "implementation_commit": preregistration["implementation_commit"],
        },
        "failure": {
            "runtime_error": "ValueError: natural-speech residual evidence has no development gap",
            "maximum_development_self_only_residual_ratio": maximum_self,
            "minimum_development_world_only_residual_ratio": minimum_world,
            "development_gap": gap,
            "clean_development_gap_required": True,
            "clean_development_gap_passed": gap > 0,
            "threshold_defined": False,
            "held_out_classification_scored": False,
        },
        "capture_quality": {
            "captures_with_clipping": captures_with_clipping,
            "total_clipped_samples": total_clipped,
            "no_clipping_passed": total_clipped == 0,
        },
        "world_evidence": evidence,
        "recovery_audit": {
            "reason": "The live CLI did not serialize its in-memory artifacts after the frozen evaluator raised on the missing development gap.",
            "method": "Recomputed the same frozen chirp correlation and residual ratio from the retained WAVs only, using a nominal 128 ms callback offset because the per-trial raw byte-count offset was not serialized.",
            "promotion_allowed": False,
        },
        "criteria": {
            "all_15_captures_present": len(artifacts) == 15,
            "clean_development_gap": gap > 0,
            "no_clipped_capture_samples": total_clipped == 0,
            "classification_gate": False,
        },
        "all_acceptance_gates_passed": False,
        "promotion_allowed": False,
        "artifacts": artifacts,
        "interpretation": "All 15 preregistered trials played and produced 45 retained WAVs, but the chirp-conditioned FIR residual did not separate development self-only from world-only speech. The largest self-only residual ratio exceeded the normalized world-only value, so the frozen threshold was undefined and held-out overlap was not scored. Three captures also contained clipped samples. The natural-speech extension therefore fails without changing the structured-tone H3 result.",
        "claim_boundary": "This recovery audit preserves a failed fixed-geometry run. Its reconstructed alignments lack the lost per-trial raw byte-count offsets and cannot promote H3 under any metric. It does not justify post-hoc thresholding, level changes, or a claim about natural-speech binding.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
