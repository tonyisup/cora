"""Contracts for offline bounded vocal-imitation product-session reports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from cora_infans.product_session import (
    build_product_session_report,
    score_product_session_directory,
    validate_product_session_report,
)


_SHA256 = "a" * 64


def _artifacts() -> dict[str, dict[str, str]]:
    return {
        "source_human_wav": {"path": "private/source_human.wav", "sha256": _SHA256},
        "target_captured_wav": {"path": "private/target_captured.wav", "sha256": _SHA256},
        "agent_emitted_wav": {"path": "private/agent_emitted.wav", "sha256": _SHA256},
        "response_captured_wav": {"path": "private/response_captured.wav", "sha256": _SHA256},
        "self_prediction_wav": {"path": "private/self_prediction.wav", "sha256": _SHA256},
        # A report cannot self-attest its final SHA-256 without a circular digest.
        # A later sibling manifest may attest this completed report instead.
        "report_json": {"path": "report.json"},
    }


def _scores() -> dict[str, dict[str, float]]:
    return {
        "imitation_fidelity": {"mel_cepstral_distance": 0.24},
        "self_model_fidelity": {"mel_cepstral_distance": 0.11},
    }


def _write_mono_wav(path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _write_replay_fixture(root) -> None:
    sample_rate = 16000
    duration_seconds = 0.8
    time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    samples = (0.25 * np.sin(2.0 * np.pi * 180.0 * time)).astype(np.float32)
    for filename in (
        "source_human.wav",
        "target_captured.wav",
        "agent_emitted.wav",
        "response_captured.wav",
        "self_prediction.wav",
    ):
        _write_mono_wav(root / filename, samples, sample_rate)


def test_synthetic_fixture_report_has_required_artifacts_and_separate_scores() -> None:
    report = build_product_session_report(
        evidence_class="synthetic_fixture",
        artifacts=_artifacts(),
        scores=_scores(),
        association_frozen_before_evaluation=True,
    )

    assert report["evidence_class"] == "synthetic_fixture"
    assert report["v4_evidence_mutated"] is False
    assert set(report["artifacts"]) == set(_artifacts())
    assert set(report["scores"]) == {"imitation_fidelity", "self_model_fidelity"}
    assert report["association_frozen_before_evaluation"] is True
    assert validate_product_session_report(report) == report


def test_real_authorized_session_requires_authorization_and_operating_point() -> None:
    with pytest.raises(ValueError, match="authorization"):
        build_product_session_report(
            evidence_class="real_authorized_session",
            artifacts=_artifacts(),
            scores=_scores(),
            association_frozen_before_evaluation=True,
        )

    report = build_product_session_report(
        evidence_class="real_authorized_session",
        artifacts=_artifacts(),
        scores=_scores(),
        association_frozen_before_evaluation=True,
        authorization={"recording_authorized": True, "protocol_version": "product-session-v1"},
        operating_point={"input_device": "XVF3800 :0", "sample_rate_hz": 16000},
    )

    assert report["authorization"]["recording_authorized"] is True
    assert report["operating_point"]["sample_rate_hz"] == 16000


def test_report_rejects_missing_artifact_or_unfrozen_association() -> None:
    missing = _artifacts()
    missing.pop("self_prediction_wav")
    with pytest.raises(ValueError, match="required artifacts"):
        build_product_session_report(
            evidence_class="synthetic_fixture",
            artifacts=missing,
            scores=_scores(),
            association_frozen_before_evaluation=True,
        )

    with pytest.raises(ValueError, match="frozen"):
        build_product_session_report(
            evidence_class="synthetic_fixture",
            artifacts=_artifacts(),
            scores=_scores(),
            association_frozen_before_evaluation=False,
        )


def test_offline_replay_refuses_unfrozen_association(tmp_path) -> None:
    _write_replay_fixture(tmp_path)

    with pytest.raises(ValueError, match="frozen"):
        score_product_session_directory(
            tmp_path,
            evidence_class="synthetic_fixture",
            association_frozen_before_evaluation=False,
        )


def test_offline_replay_scores_identical_wavs_and_writes_manifest(tmp_path) -> None:
    _write_replay_fixture(tmp_path)

    report = score_product_session_directory(
        tmp_path,
        evidence_class="synthetic_fixture",
        association_frozen_before_evaluation=True,
    )

    imitation = report["scores"]["imitation_fidelity"]
    self_model = report["scores"]["self_model_fidelity"]
    assert imitation == pytest.approx(
        {
            "onset_offset_alignment_ms": 0.0,
            "duration_difference_ms": 0.0,
            "rms_envelope_distance": 0.0,
            "pitch_contour_distance_hz": 0.0,
            "cepstral_distance": 0.0,
        },
        abs=1e-9,
    )
    assert self_model == pytest.approx(imitation, abs=1e-9)
    assert report["scoring_version"] == "offline-replay-v1"
    assert (tmp_path / "report.json").is_file()
    manifest = json.loads((tmp_path / "session-manifest.json").read_text())
    assert len(manifest["report_sha256"]) == 64
    assert "sha256" not in report["artifacts"]["report_json"]


def test_offline_replay_distinguishes_changed_pitch_and_spectrum(tmp_path) -> None:
    _write_replay_fixture(tmp_path)
    sample_rate = 16000
    time = np.arange(int(sample_rate * 0.8)) / sample_rate
    changed_response = (0.25 * np.sin(2.0 * np.pi * 240.0 * time)).astype(np.float32)
    _write_mono_wav(tmp_path / "response_captured.wav", changed_response, sample_rate)

    report = score_product_session_directory(
        tmp_path,
        evidence_class="synthetic_fixture",
        association_frozen_before_evaluation=True,
    )

    imitation = report["scores"]["imitation_fidelity"]
    assert imitation["pitch_contour_distance_hz"] > 0.0
    assert imitation["cepstral_distance"] > 0.0


def test_offline_replay_cli_scores_existing_session_without_live_io(tmp_path) -> None:
    _write_replay_fixture(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--score-vocal-imitation-session",
            "--session-dir",
            str(tmp_path),
            "--evidence-class",
            "synthetic_fixture",
            "--association-frozen-before-evaluation",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    report = json.loads(completed.stdout)
    assert report["evidence_class"] == "synthetic_fixture"
    assert report["artifacts"]["report_json"]["path"] == "report.json"


def test_replay_script_scores_existing_session_without_live_io(tmp_path) -> None:
    _write_replay_fixture(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    script = Path(__file__).parents[1] / "scripts" / "run_bounded_vocal_imitation_replay.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--session-dir",
            str(tmp_path),
            "--evidence-class",
            "synthetic_fixture",
            "--association-frozen-before-evaluation",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert json.loads(completed.stdout)["scoring_version"] == "offline-replay-v1"
