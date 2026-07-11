from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from cora_infans.cli import structured_prediction_cli
from cora_infans.structured_prediction import (
    CALIBRATED_V2_DEVELOPMENT_SEEDS,
    CALIBRATED_V2_HELD_OUT_SEEDS,
    CALIBRATED_V2_STD_FLOOR,
    DEVELOPMENT_SEEDS,
    DROPOUT_V4_DEVELOPMENT_SEEDS,
    DROPOUT_V4_HELD_OUT_SEEDS,
    DROPOUT_V5_DEVELOPMENT_SEEDS,
    DROPOUT_V5_HELD_OUT_SEEDS,
    HELD_OUT_SEEDS,
    ROBUST_V3_DEVELOPMENT_SEEDS,
    ROBUST_V3_HELD_OUT_SEEDS,
    LocalAcousticCodebook,
    StructuredTrialPlan,
    build_dropout_v4_trial_conditions,
    build_structured_protocol_manifest,
    build_robust_v3_trial_conditions,
    render_structured_trial,
    run_structured_prediction_protocol,
)
from cora_infans.io.motor_plan import write_wav
from cora_infans.io.wav_frames import read_pcm_wav


def test_structured_manifest_has_disjoint_fixed_trials() -> None:
    manifest = build_structured_protocol_manifest()

    assert set(DEVELOPMENT_SEEDS).isdisjoint(HELD_OUT_SEEDS)
    assert manifest["trial_count"] == 12
    assert manifest["maximum_total_playback_seconds"] < 90.0
    assert manifest["real_io_requires_explicit_authorization"] is True
    for trial in manifest["trials"]:
        sequence = trial["grammar_sequence"]
        assert len(sequence) == 20
        assert all(
            current != following
            for current, following in zip(sequence, sequence[1:])
        )


def test_calibrated_v2_manifest_uses_fresh_disjoint_seeds() -> None:
    prior = set(DEVELOPMENT_SEEDS) | set(HELD_OUT_SEEDS)
    fresh = set(CALIBRATED_V2_DEVELOPMENT_SEEDS) | set(
        CALIBRATED_V2_HELD_OUT_SEEDS
    )
    manifest = build_structured_protocol_manifest(
        development_seeds=CALIBRATED_V2_DEVELOPMENT_SEEDS,
        held_out_seeds=CALIBRATED_V2_HELD_OUT_SEEDS,
        protocol_variant="calibrated_v2",
        calibration_normalization_std_floor=CALIBRATED_V2_STD_FLOOR,
    )

    assert prior.isdisjoint(fresh)
    assert set(CALIBRATED_V2_DEVELOPMENT_SEEDS).isdisjoint(
        CALIBRATED_V2_HELD_OUT_SEEDS
    )
    assert manifest["protocol_variant"] == "calibrated_v2"
    assert manifest["representation"]["standard_deviation_floor"] == 0.30


def test_robust_v3_manifest_has_fresh_named_conditions() -> None:
    prior = (
        set(DEVELOPMENT_SEEDS)
        | set(HELD_OUT_SEEDS)
        | set(CALIBRATED_V2_DEVELOPMENT_SEEDS)
        | set(CALIBRATED_V2_HELD_OUT_SEEDS)
    )
    fresh = set(ROBUST_V3_DEVELOPMENT_SEEDS) | set(ROBUST_V3_HELD_OUT_SEEDS)
    conditions = build_robust_v3_trial_conditions()
    manifest = build_structured_protocol_manifest(
        development_seeds=ROBUST_V3_DEVELOPMENT_SEEDS,
        held_out_seeds=ROBUST_V3_HELD_OUT_SEEDS,
        protocol_variant="robustness_v3",
        calibration_normalization_std_floor=CALIBRATED_V2_STD_FLOOR,
        trial_conditions=conditions,
    )

    assert prior.isdisjoint(fresh)
    assert len(conditions) == 12
    assert all(trial["condition"]["name"] for trial in manifest["trials"])
    held_names = {
        trial["condition"]["name"]
        for trial in manifest["trials"]
        if trial["family"] == "vowels" and trial["split"] == "held_out"
    }
    assert held_names == {
        "speaker_shift_f0_200",
        "speaker_shift_f0_200_noise_10db",
        "speaker_shift_f0_200_grammar_dropout_10pct",
    }


def test_robust_v3_dropout_leaves_calibration_clean_and_masks_grammar() -> None:
    plan = StructuredTrialPlan(
        trial_id="vowels_held_out_13023",
        family="vowels",
        split="held_out",
        seed=13023,
        calibration_sequence=(0, 1, 2, 3),
        grammar_sequence=(0, 1, 2, 3, 0, 1, 2, 3),
    )
    clean = render_structured_trial(
        plan, stimulus_condition={"vowel_f0_hz": 200.0}
    )
    dropout = render_structured_trial(
        plan,
        stimulus_condition=build_robust_v3_trial_conditions()[plan.trial_id],
    )

    for start, end in clean.token_windows[:4]:
        assert np.array_equal(clean.pcm[start:end], dropout.pcm[start:end])
    assert all(bool(np.all(mask)) for mask in dropout.token_valid_masks[:4])
    assert any(
        not np.array_equal(clean.pcm[start:end], dropout.pcm[start:end])
        for start, end in clean.token_windows[4:]
    )
    assert any(not bool(np.all(mask)) for mask in dropout.token_valid_masks[4:])


def test_frozen_protocol_and_implementation_hashes_match_source() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root / "docs/evidence/2026-07-10-structured-prediction-protocol.json"
    )
    fake_result_path = (
        root / "docs/evidence/2026-07-10-structured-prediction-fake-result.json"
    )
    preregistration_path = root / (
        "docs/evidence/"
        "2026-07-10-structured-prediction-real-preregistration-draft.json"
    )
    retry_preregistration_path = root / (
        "docs/evidence/"
        "2026-07-10-structured-prediction-real-attempt-2-preregistration.json"
    )
    stored = json.loads(protocol_path.read_text())
    generated = json.loads(json.dumps(build_structured_protocol_manifest()))

    for key, value in generated.items():
        assert stored[key] == value
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    fake_result = json.loads(fake_result_path.read_text())
    preregistration = json.loads(preregistration_path.read_text())
    retry_preregistration = json.loads(retry_preregistration_path.read_text())
    assert fake_result["protocol_sha256"] == protocol_hash
    assert preregistration["protocol_sha256"] == protocol_hash
    historical_implementation_hash = (
        "51307274e9582885470b2ed81950da137a93cebfc7e5add5aa9ec42c1707093c"
    )
    # Historical artifacts remain immutable after the pre-playback I/O fix.
    assert fake_result["implementation_sha256"] == historical_implementation_hash
    assert preregistration["implementation_sha256"] == historical_implementation_hash
    assert retry_preregistration["protocol_sha256"] == protocol_hash
    assert retry_preregistration["predictor_implementation_sha256"] == (
        "04d40123872c9af3d73697f39d3915002366d08a3ef061efbd16a125e126d264"
    )
    assert retry_preregistration["io_runner_sha256"] == (
        "1cf10155c49dc32406b4936504f5af452be335ea73ee08f2d7a6fc7437f3f34b"
    )


def test_calibrated_v2_preregistration_matches_frozen_code_and_protocol(
    tmp_path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/evidence"
    protocol_path = evidence / "2026-07-10-calibrated-structured-protocol.json"
    audit_path = evidence / (
        "2026-07-10-calibrated-structured-representation-audit.json"
    )
    fake_path = evidence / "2026-07-10-calibrated-structured-fake-result.json"
    preregistration = json.loads(
        (
            evidence
            / "2026-07-10-calibrated-structured-real-preregistration.json"
        ).read_text()
    )
    generated = run_structured_prediction_protocol(
        out_dir=tmp_path,
        development_seeds=CALIBRATED_V2_DEVELOPMENT_SEEDS,
        held_out_seeds=CALIBRATED_V2_HELD_OUT_SEEDS,
        protocol_variant="calibrated_v2",
        calibration_normalization_std_floor=CALIBRATED_V2_STD_FLOOR,
        acceptance_profile="h2_predictive_advantage_v2",
    )

    assert json.loads(protocol_path.read_text()) == json.loads(json.dumps(generated))
    for field, path in (
        ("design_audit_sha256", audit_path),
        ("protocol_sha256", protocol_path),
        ("fake_result_sha256", fake_path),
    ):
        assert preregistration[field] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert preregistration["predictor_implementation_sha256"] == (
        "417bb4482871a7314ffa5199874752aaa3ab875655610569462ac86ad3b93b61"
    )
    assert preregistration["cli_implementation_sha256"] == (
        "3ed88d19be7608423d1714af820a47d85688d3c0eb585fb98a8c884d817dc803"
    )
    assert preregistration["io_runner_sha256"] == (
        "1cf10155c49dc32406b4936504f5af452be335ea73ee08f2d7a6fc7437f3f34b"
    )


def test_robust_v3_preregistration_matches_frozen_code_and_protocol(
    tmp_path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/evidence"
    protocol_path = evidence / "2026-07-10-structured-robustness-protocol.json"
    audit_path = evidence / "2026-07-10-structured-robustness-design-audit.json"
    fake_path = evidence / "2026-07-10-structured-robustness-fake-result.json"
    preregistration = json.loads(
        (
            evidence
            / "2026-07-10-structured-robustness-real-preregistration.json"
        ).read_text()
    )
    generated = run_structured_prediction_protocol(
        out_dir=tmp_path,
        development_seeds=ROBUST_V3_DEVELOPMENT_SEEDS,
        held_out_seeds=ROBUST_V3_HELD_OUT_SEEDS,
        protocol_variant="robustness_v3",
        calibration_normalization_std_floor=CALIBRATED_V2_STD_FLOOR,
        acceptance_profile="h2_condition_robustness_v3",
        trial_conditions=build_robust_v3_trial_conditions(),
    )

    assert json.loads(protocol_path.read_text()) == json.loads(json.dumps(generated))
    for field, path in (
        ("design_audit_sha256", audit_path),
        ("protocol_sha256", protocol_path),
        ("fake_result_sha256", fake_path),
    ):
        assert preregistration[field] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert preregistration["predictor_implementation_sha256"] == (
        "898a7a2de2b8177c682e12032a62de29bd98232722b068ea662c147be3c4b4b7"
    )
    assert preregistration["cli_implementation_sha256"] == (
        "f3f5fb2ff1b32d29059b4df01f83361311c8f8cccafe1f549e6eef836a8616a1"
    )
    assert preregistration["io_runner_sha256"] == (
        "1cf10155c49dc32406b4936504f5af452be335ea73ee08f2d7a6fc7437f3f34b"
    )


def test_retained_real_result_has_complete_hashed_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / (
        "docs/evidence/"
        "2026-07-10-structured-prediction-real-attempt-2-result.json"
    )
    artifact_root = root / (
        "docs/evidence/"
        "2026-07-10-structured-prediction-real-attempt-2-artifacts"
    )
    result = json.loads(result_path.read_text())

    assert result["verdict"] == "real_io_acceptance_failed"
    assert result["all_acceptance_gates_passed"] is False
    assert len(result["artifacts"]) == 12
    assert all(result["acceptance"]["tones"].values())
    assert not all(result["acceptance"]["vowels"].values())
    for artifact in result["artifacts"]:
        assert artifact["playback"]["backend"] == "ffmpeg_audiotoolbox"
        assert artifact["capture"]["backend"] == "ffmpeg_avfoundation"
        assert artifact["io_timing"]["stream_position_verified"] is True
        for kind in ("efference", "captured"):
            retained = artifact_root / Path(artifact[f"{kind}_wav"]).name
            assert retained.exists()
            assert hashlib.sha256(retained.read_bytes()).hexdigest() == artifact[
                f"{kind}_sha256"
            ]


def test_calibrated_v2_retained_real_result_passes_with_hashed_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/evidence"
    result = json.loads(
        (evidence / "2026-07-10-calibrated-structured-real-result.json").read_text()
    )
    artifact_root = evidence / "2026-07-10-calibrated-structured-real-artifacts"

    assert result["verdict"] == "real_io_acceptance_passed"
    assert result["all_acceptance_gates_passed"] is True
    assert result["protocol_variant"] == "calibrated_v2"
    assert len(result["artifacts"]) == 12
    assert all(
        all(family_gates.values())
        for family_gates in result["acceptance"].values()
    )
    assert result["family_reports"]["vowels"]["accuracy_improvement"] >= 0.20
    for artifact in result["artifacts"]:
        assert artifact["io_timing"]["stream_position_verified"] is True
        for kind in ("efference", "captured"):
            retained = artifact_root / Path(artifact[f"{kind}_wav"]).name
            assert retained.exists()
            assert hashlib.sha256(retained.read_bytes()).hexdigest() == artifact[
                f"{kind}_sha256"
            ]


def test_robust_v3_retained_failure_preserves_condition_and_artifact_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/evidence"
    result = json.loads(
        (evidence / "2026-07-10-structured-robustness-real-result.json").read_text()
    )
    artifact_root = evidence / "2026-07-10-structured-robustness-real-artifacts"

    assert result["verdict"] == "real_io_acceptance_failed"
    assert result["all_acceptance_gates_passed"] is False
    assert all(result["acceptance"]["tones"].values())
    assert result["acceptance"]["vowels"]["each_condition_occupancy_passed"] is False
    vowel_reports = result["family_reports"]["vowels"]["reports"]
    assert all(trial["accuracy_improvement"] >= 0.15 for trial in vowel_reports)
    dropout = next(
        trial
        for trial in vowel_reports
        if trial["condition"]["name"].endswith("dropout_10pct")
    )
    assert dropout["state_occupancy"] == 3
    assert len(result["artifacts"]) == 12
    for artifact in result["artifacts"]:
        for kind in ("efference", "captured"):
            retained = artifact_root / Path(artifact[f"{kind}_wav"]).name
            assert retained.exists()
            assert hashlib.sha256(retained.read_bytes()).hexdigest() == artifact[
                f"{kind}_sha256"
            ]


def test_real_runner_reports_whether_playback_actually_started(tmp_path) -> None:
    playback_calls = 0

    def playback(path: str) -> dict[str, object]:
        nonlocal playback_calls
        playback_calls += 1
        return {"backend": "injected_playback"}

    def capture(path: str, **kwargs) -> dict[str, object]:
        raise RuntimeError("capture unavailable")

    report = run_structured_prediction_protocol(
        out_dir=tmp_path,
        real_io=True,
        bounded_playback_authorized=True,
        playback_fn=playback,
        capture_fn=capture,
        geometry={"test_double": True},
        real_hardware_claim=False,
    )

    assert report["verdict"] == "real_io_failed"
    assert report["live_playback"] is False
    assert playback_calls == 0


def test_structured_protocol_default_and_real_gate_touch_no_io(tmp_path) -> None:
    planned = run_structured_prediction_protocol(out_dir=tmp_path)
    blocked = run_structured_prediction_protocol(
        out_dir=tmp_path,
        real_io=True,
        bounded_playback_authorized=False,
    )

    assert planned["verdict"] == "planned_no_io"
    assert blocked["verdict"] == "blocked_bounded_live_io_not_authorized"
    assert list(tmp_path.iterdir()) == []
    assert planned["live_playback"] is False
    assert blocked["live_playback"] is False


def test_cli_requires_geometry_before_authorized_real_io(tmp_path) -> None:
    blocked = structured_prediction_cli(
        [
            "--evaluate-structured-prediction",
            "--out-dir",
            str(tmp_path),
            "--real-io",
        ]
    )
    assert blocked["verdict"] == "blocked_bounded_live_io_not_authorized"

    with pytest.raises(ValueError, match="speaker-distance-cm"):
        structured_prediction_cli(
            [
                "--evaluate-structured-prediction",
                "--out-dir",
                str(tmp_path),
                "--real-io",
                "--authorize-bounded-playback",
            ]
        )

    assert list(tmp_path.iterdir()) == []


def test_fake_structured_protocol_exercises_measured_transition_path(tmp_path) -> None:
    report = run_structured_prediction_protocol(out_dir=tmp_path, fake_io=True)

    assert report["verdict"] == "fake_io_plumbing_passed"
    assert report["evidence_class"] == "synthetic_fake_io"
    assert report["promotion_allowed"] is False
    assert report["real_hardware"] is False
    assert len(report["artifacts"]) == 12
    assert len(list(tmp_path.glob("*.wav"))) == 24
    for artifact in report["artifacts"]:
        assert Path(artifact["efference_wav"]).exists()
        assert Path(artifact["captured_wav"]).exists()
        assert len(artifact["captured_sha256"]) == 64
        assert artifact["capture_provenance"] == "synthetic_fake_acoustic_channel"
    for family in report["family_reports"].values():
        assert family["held_out_codebook_occupancy"] == 4
        assert family["post_hoc_token_cluster_purity"] == 1.0
        assert family["accuracy_improvement"] >= 0.2
        assert family["top_down_prediction_consumed"] is True
        assert family["bounded_state_scalars"] == 68


def test_calibrated_v2_fake_path_uses_fresh_manifest_and_predictive_gates(
    tmp_path,
) -> None:
    report = structured_prediction_cli(
        [
            "--evaluate-structured-prediction",
            "--out-dir",
            str(tmp_path),
            "--variant",
            "calibrated_v2",
            "--fake-io",
        ]
    )

    assert report["verdict"] == "fake_io_plumbing_passed"
    assert report["protocol_variant"] == "calibrated_v2"
    assert report["development_seeds"] == list(CALIBRATED_V2_DEVELOPMENT_SEEDS)
    assert report["held_out_seeds"] == list(CALIBRATED_V2_HELD_OUT_SEEDS)
    for family in report["family_reports"].values():
        assert family["held_out_codebook_occupancy"] == 4
        assert family["accuracy_improvement"] >= 0.20
        assert family["minimum_trial_improvement"] >= 0.15


def test_robust_v3_fake_path_passes_every_named_condition(tmp_path) -> None:
    report = structured_prediction_cli(
        [
            "--evaluate-structured-prediction",
            "--out-dir",
            str(tmp_path),
            "--variant",
            "robustness_v3",
            "--fake-io",
        ]
    )

    assert report["verdict"] == "fake_io_plumbing_passed"
    assert report["protocol_variant"] == "robustness_v3"
    assert report["development_seeds"] == list(ROBUST_V3_DEVELOPMENT_SEEDS)
    assert report["held_out_seeds"] == list(ROBUST_V3_HELD_OUT_SEEDS)
    for family in report["family_reports"].values():
        assert family["minimum_trial_improvement"] >= 0.15
        assert all(trial["state_occupancy"] == 4 for trial in family["reports"])
        assert all(trial["condition"]["name"] for trial in family["reports"])


def test_dropout_v4_fake_design_failure_blocks_real_io(tmp_path) -> None:
    fake = structured_prediction_cli(
        [
            "--evaluate-structured-prediction",
            "--out-dir",
            str(tmp_path / "fake"),
            "--variant",
            "dropout_v4",
            "--fake-io",
        ]
    )

    assert fake["verdict"] == "fake_io_gate_failed"
    assert fake["development_seeds"] == list(DROPOUT_V4_DEVELOPMENT_SEEDS)
    assert fake["held_out_seeds"] == list(DROPOUT_V4_HELD_OUT_SEEDS)
    assert fake["family_reports"]["vowels"]["accuracy_improvement"] < 0.20
    with pytest.raises(ValueError, match="design gate.*real I/O is prohibited"):
        structured_prediction_cli(
            [
                "--evaluate-structured-prediction",
                "--out-dir",
                str(tmp_path / "real"),
                "--variant",
                "dropout_v4",
                "--real-io",
                "--authorize-bounded-playback",
                "--speaker-distance-cm",
                "50",
                "--speaker-orientation",
                "180deg",
                "--quiet-room-confirmed",
            ]
        )


def test_mask_aware_dropout_signature_passes_spent_v4_matrix(tmp_path) -> None:
    report = run_structured_prediction_protocol(
        out_dir=tmp_path,
        fake_io=True,
        development_seeds=DROPOUT_V4_DEVELOPMENT_SEEDS,
        held_out_seeds=DROPOUT_V4_HELD_OUT_SEEDS,
        protocol_variant="dropout_mask_design_spent_v4",
        calibration_normalization_std_floor=CALIBRATED_V2_STD_FLOOR,
        acceptance_profile="h2_condition_robustness_v3",
        trial_conditions=build_dropout_v4_trial_conditions(),
        signature_mode="efference_masked_mean_30_5",
        representation_selection_boundary="spent dropout-v4 synthetic matrix only",
    )

    assert report["verdict"] == "fake_io_plumbing_passed"
    vowels = report["family_reports"]["vowels"]
    assert all(trial["state_occupancy"] == 4 for trial in vowels["reports"])
    assert vowels["accuracy_improvement"] >= 0.20
    assert vowels["minimum_trial_improvement"] >= 0.15


def test_dropout_v5_plan_is_fresh_and_real_io_stays_blocked(tmp_path) -> None:
    prior = set(DROPOUT_V4_DEVELOPMENT_SEEDS) | set(DROPOUT_V4_HELD_OUT_SEEDS)
    planned = structured_prediction_cli(
        [
            "--evaluate-structured-prediction",
            "--out-dir",
            str(tmp_path / "plan"),
            "--variant",
            "dropout_v5",
        ]
    )

    assert planned["verdict"] == "planned_no_io"
    assert prior.isdisjoint(DROPOUT_V5_DEVELOPMENT_SEEDS)
    assert prior.isdisjoint(DROPOUT_V5_HELD_OUT_SEEDS)
    assert planned["development_seeds"] == list(DROPOUT_V5_DEVELOPMENT_SEEDS)
    assert planned["held_out_seeds"] == list(DROPOUT_V5_HELD_OUT_SEEDS)
    assert planned["representation"]["signature_mode"] == (
        "efference_masked_mean_30_5"
    )
    with pytest.raises(ValueError, match="fresh synthetic gate is pending"):
        structured_prediction_cli(
            [
                "--evaluate-structured-prediction",
                "--out-dir",
                str(tmp_path / "real"),
                "--variant",
                "dropout_v5",
                "--real-io",
                "--authorize-bounded-playback",
                "--speaker-distance-cm",
                "50",
                "--speaker-orientation",
                "180deg",
                "--quiet-room-confirmed",
            ]
        )


def test_retained_dropout_v5_fresh_synthetic_gate_passed() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-10-dropout-v5-fresh-synthetic-result.json"
        ).read_text()
    )

    assert result["report"]["promotion_allowed"] is False
    assert result["gate_summary"]["all_gates_passed"] is True
    assert result["gate_summary"]["tones"]["condition_occupancies"] == [4, 4, 4]
    assert result["gate_summary"]["vowels"]["condition_occupancies"] == [4, 4, 4]
    assert result["gate_summary"]["vowels"]["minimum_trial_improvement"] >= 0.15


def test_local_codebook_updates_only_nearest_bounded_centroid() -> None:
    initial = np.eye(4, dtype=np.float64)
    codebook = LocalAcousticCodebook(initial)
    before = codebook.centroids.copy()

    state = codebook.observe(np.asarray([0.9, 0.1, 0.0, 0.0]))

    assert state == 0
    assert codebook.state_scalars == 20
    assert not np.array_equal(codebook.centroids[0], before[0])
    assert np.array_equal(codebook.centroids[1:], before[1:])


def test_injected_real_runner_uses_synchronized_capture_and_alignment(tmp_path) -> None:
    playback_ready = Event()
    shared: dict[str, object] = {}

    def playback(path: str) -> dict[str, object]:
        pcm, sample_rate = read_pcm_wav(path)
        shared["pcm"] = np.mean(pcm, axis=1)
        shared["sample_rate"] = sample_rate
        playback_ready.set()
        return {"backend": "injected_playback", "completed": True}

    def capture(
        path: str,
        *,
        sample_rate: int,
        duration_ms: int,
        ready_event,
        capture_timeline,
        **kwargs,
    ) -> dict[str, object]:
        playback_offset = int(0.1 * sample_rate)
        acoustic_delay = int(0.03 * sample_rate)
        capture_timeline["frame_position_fn"] = lambda: playback_offset
        capture_timeline["position_source"] = "synthetic_frame_count"
        ready_event.set()
        assert playback_ready.wait(timeout=2.0)
        efference = np.asarray(shared["pcm"], dtype=np.float32)
        playback_ready.clear()
        captured = np.zeros(
            playback_offset + acoustic_delay + len(efference) + sample_rate // 5,
            dtype=np.float32,
        )
        start = playback_offset + acoustic_delay
        captured[start : start + len(efference)] = efference * 0.5
        write_wav(path, captured, sample_rate)
        time.sleep(0.15)
        return {
            "backend": "injected_capture",
            "frames": len(captured),
            "completed": True,
        }

    report = run_structured_prediction_protocol(
        out_dir=tmp_path,
        real_io=True,
        bounded_playback_authorized=True,
        playback_fn=playback,
        capture_fn=capture,
        geometry={"test_double": True},
        real_hardware_claim=False,
    )

    assert report["error"] is None, report["error"]
    assert report["verdict"] == "injected_real_runner_test_complete"
    assert report["real_hardware"] is False
    assert all(
        abs(float(artifact["alignment_onset_ms"]) - 130.0) <= 1.0
        for artifact in report["artifacts"]
    ), json.dumps(report["artifacts"], indent=2)
    assert report["all_acceptance_gates_passed"] is True, json.dumps(
        report["family_reports"], indent=2
    )
    assert len(report["artifacts"]) == 12
    assert all(
        artifact["capture_provenance"] == "injected_io_test_double"
        for artifact in report["artifacts"]
    )
    assert all(
        artifact["io_timing"]["stream_position_verified"] is True
        for artifact in report["artifacts"]
    )
