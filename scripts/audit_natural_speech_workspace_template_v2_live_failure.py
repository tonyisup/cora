#!/usr/bin/env python3
"""Retain the completed failed template-v2 natural-speech workspace run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import cora_infans.natural_speech_workspace_template_v2 as workspace
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
    prepared = workspace.prepare_template_panel(args.root)
    captures: dict[str, np.ndarray] = {}
    alignments: dict[str, int] = {}
    artifacts = []
    total_clipped = 0
    captures_with_clipping = 0

    for trial in workspace.build_template_trials():
        rendered = workspace.render_template_tracks(trial, prepared)
        self_path = artifact_root / f"{trial.trial_id}_self.wav"
        world_path = artifact_root / f"{trial.trial_id}_world.wav"
        capture_path = artifact_root / f"{trial.trial_id}_capture.wav"
        captured, sample_rate = read_pcm_wav(capture_path)
        if sample_rate != workspace.SAMPLE_RATE or captured.shape[1] != 2:
            raise ValueError(f"unexpected capture geometry: {capture_path}")
        chirp_samples = workspace.CHIRP_MS * workspace.SAMPLE_RATE // 1000
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
        captures[trial.trial_id] = captured
        alignments[trial.trial_id] = alignment
        clipped = int(np.count_nonzero(np.abs(captured) >= (32767.0 / 32768.0)))
        total_clipped += clipped
        captures_with_clipping += int(clipped > 0)
        artifacts.append(
            {
                "trial_id": trial.trial_id,
                "condition": trial.condition,
                "scored": trial.scored,
                "chirp_source": rendered.chirp_source,
                "self_wav": str(self_path.resolve()),
                "self_sha256": _sha256(self_path),
                "world_wav": str(world_path.resolve()),
                "world_sha256": _sha256(world_path),
                "captured_wav": str(capture_path.resolve()),
                "captured_sha256": _sha256(capture_path),
                "reconstructed_alignment_samples": alignment,
                "reconstructed_alignment_onset_ms": onset_ms,
                "capture_quality": {
                    "peak_absolute": float(np.max(np.abs(captured))),
                    "clipped_sample_count": clipped,
                    "clipped_sample_fraction": clipped / captured.size,
                },
            }
        )

    evaluation = workspace.evaluate_template_captures(
        root=args.root, captures=captures, alignment_samples=alignments
    )
    result = {
        **evaluation,
        "mode": "natural_speech_workspace_template_v2_real_failure_audit",
        "date": "2026-07-11",
        "status": "completed_live_playback_capture_failed_frozen_evaluation",
        "verdict": "real_io_acceptance_failed",
        "real_hardware": True,
        "live_playback": True,
        "completed_trial_count": len(artifacts),
        "measured_total_playback_seconds": 19.32,
        "bounded_total_capture_seconds": 31.92,
        "expected_artifact_count": 63,
        "retained_artifact_count": len(list(artifact_root.glob("*.wav"))),
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": _sha256(preregistration_path),
            "commit": "c5b2e4f",
            "implementation_commit": "db19e7a",
        },
        "capture_quality": {
            "captures_with_clipping": captures_with_clipping,
            "total_clipped_samples": total_clipped,
            "no_clipping_passed": total_clipped == 0,
        },
        "criteria": {
            **evaluation["criteria"],
            "no_clipped_capture_samples": total_clipped == 0,
        },
        "all_acceptance_gates_passed": False,
        "promotion_allowed": False,
        "recovery_audit": {
            "reason": "The complete JSON report was emitted to stdout, but the invoking tool truncated that single long line before it could be retained as a file.",
            "method": "Recomputed the frozen chirp alignments, template gain scalars, and residual ratios from the retained WAVs only. The recovered development gap and every residual ratio exactly match the values visible in the live stdout result.",
            "nominal_alignment_anchor_ms": 128.0,
            "promotion_allowed": False,
        },
        "artifacts": artifacts,
        "interpretation": "All 21 preregistered trials completed. RMS normalization and software-level compensation reduced the missing development gap from v1's -0.441549 to -0.009262, and all three held-out overlap residual ratios exceeded 1.0. However, one development self-probe residual ratio was 1.009262, so the frozen clean-gap rule still failed and no threshold or held-out classifier was selected. One overlap capture contained 18 clipped samples. Template-v2 therefore fails without changing the structured-tone H3 result.",
        "claim_boundary": "This is a failed fixed-geometry exact-efference replay task. Held-out residuals are retained diagnostically but were not threshold-scored. The result cannot support post-hoc thresholding, a natural-speech H3 extension, generalized self-voice prediction, or calibrated-level claims.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
