"""Tests for the M9 temporal-sequence smoke helpers."""

from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import numpy as np

from m9_sound.controller import ControllerParams
from m9_sound.temporal import (
    generate_tone_sequence,
    render_param_sequence,
    run_temporal_smoke_report,
    timing_alignment_score,
    write_temporal_smoke_wav,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = PROJECT_ROOT / "scripts" / "m9_temporal_train.py"
TRAINER_SPEC = importlib.util.spec_from_file_location("m9_temporal_train", TRAINER_PATH)
assert TRAINER_SPEC is not None and TRAINER_SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(TRAINER_SPEC)
TRAINER_SPEC.loader.exec_module(TRAINER)

SWEEP_PATH = PROJECT_ROOT / "scripts" / "m9_temporal_family_sweep.py"
SWEEP_SPEC = importlib.util.spec_from_file_location("m9_temporal_family_sweep", SWEEP_PATH)
assert SWEEP_SPEC is not None and SWEEP_SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SWEEP_SPEC)
SWEEP_SPEC.loader.exec_module(SWEEP)


class TestTemporalNoiseRobustness:
    def test_smoke_report_handles_room_noise(self):
        report = run_temporal_smoke_report(input_noise_rms=0.02, noise_seed=7)

        assert report["input_noise_rms"] == 0.02
        assert report["noise_seed"] == 7
        assert report["decoded_transition_count"] == 2
        assert report["rendered_frame_count"] == 12


class TestTemporalDeviceVariation:
    def test_smoke_report_handles_device_highpass_profile(self):
        report = run_temporal_smoke_report(
            device_profile="highpass",
            device_gain=0.8,
            input_noise_rms=0.01,
            noise_seed=13,
        )

        assert report["device_profile"] == "highpass"
        assert report["device_gain"] == 0.8
        assert report["decoded_transition_count"] == 2
        assert report["rendered_frame_count"] == 12
        assert report["max_boundary_delta"] < 0.12


class TestTemporalToneGeneration:
    def test_generate_tone_sequence_is_frame_aligned(self):
        tones = [(220.0, 0.12), (330.0, 0.12), (440.0, 0.12)]
        audio = generate_tone_sequence(tones, sr=16000)

        assert audio.dtype == np.float32
        assert len(audio) == 5760
        assert np.all(np.isfinite(audio))


class TestTemporalRendering:
    def test_render_param_sequence_returns_continuous_report(self):
        params = [
            ControllerParams(freq_hz=220.0, amplitude=0.5, phase=0.0, waveform=0),
            ControllerParams(freq_hz=330.0, amplitude=0.5, phase=0.0, waveform=0),
            ControllerParams(freq_hz=440.0, amplitude=0.5, phase=0.0, waveform=0),
        ]

        audio, report = render_param_sequence(params, sr=16000, frame_size=480)

        assert audio.dtype == np.float32
        assert len(audio) == 1440
        assert report["frame_count"] == 3
        assert report["max_boundary_delta"] < 0.6
        assert report["min_boundary_delta"] >= 0.0

    def test_timing_alignment_score_rewards_correct_transition_order(self):
        params = [
            ControllerParams(freq_hz=220.0, amplitude=0.5, phase=0.0, waveform=0),
            ControllerParams(freq_hz=330.0, amplitude=0.5, phase=0.0, waveform=0),
            ControllerParams(freq_hz=440.0, amplitude=0.5, phase=0.0, waveform=0),
        ]

        score = timing_alignment_score(
            params,
            expected_transitions=2,
            expected_order=[(290.0, 370.0, 330.0), (400.0, 480.0, 440.0)],
        )

        assert score["transition_count"] == 2
        assert score["order_coverage"] == 1.0
        assert score["transition_recall"] == 1.0
        assert score["transition_precision"] == 1.0
        assert score["timing_error_frames"] == 0.0


class TestTemporalFamilyExpansion:
    def test_smoke_report_supports_diphthong_family(self):
        report = run_temporal_smoke_report(
            family="diphthong",
            input_noise_rms=0.01,
            noise_seed=11,
        )

        assert report["family"] == "diphthong"
        assert report["input_transition_count"] == 4
        assert report["decoded_transition_count"] == 4
        assert report["stimulus_frame_count"] == 10
        assert report["rendered_frame_count"] == 10
        assert report["max_boundary_delta"] < 0.15
        assert len(report["stimulus_tones"]) == 5

    def test_smoke_report_supports_wide_sweep_family(self):
        report = run_temporal_smoke_report(
            family="wide_sweep",
            input_noise_rms=0.01,
            noise_seed=19,
            device_profile="flat",
            device_gain=1.0,
        )

        assert report["family"] == "wide_sweep"
        assert report["input_transition_count"] == 6
        assert report["decoded_transition_count"] >= 6
        assert report["stimulus_frame_count"] == 21
        assert report["rendered_frame_count"] == 21
        assert report["max_boundary_delta"] < 0.16
        assert len(report["stimulus_tones"]) == 7


class TestTemporalWavExport:
    def test_write_temporal_smoke_wav_creates_listenable_wav(self, tmp_path):
        output_path = tmp_path / "m9-temporal-wide-sweep.wav"

        report = write_temporal_smoke_wav(
            output_path,
            family="wide_sweep",
            input_noise_rms=0.01,
            noise_seed=19,
            device_profile="flat",
            device_gain=1.0,
        )

        assert report["output_wav"] == str(output_path)
        assert output_path.exists()

        with wave.open(str(output_path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 16000
            assert wav_file.getnframes() == report["rendered_samples"]


class TestTemporalTrainerLocalEvaluation:
    def test_local_reward_changes_when_controller_is_perturbed(self):
        producer = TRAINER.load_producer_ckpt(
            PROJECT_ROOT / "data" / "m9_controller_temporal_v3_migrated.ckpt.json"
        )
        tones = TRAINER.TEMPORAL_FAMILIES["default"]
        base_reward, base_report = TRAINER.evaluate_candidate(
            producer,
            tones=tones,
            seed=43,
            noise_rms=0.0,
            device_profile="flat",
            device_gain=1.0,
        )

        zero_weights = {k: np.zeros_like(v) for k, v in producer.controller.get_weights().items()}
        producer.controller.set_weights(zero_weights)
        perturbed_reward, perturbed_report = TRAINER.evaluate_candidate(
            producer,
            tones=tones,
            seed=43,
            noise_rms=0.0,
            device_profile="flat",
            device_gain=1.0,
        )

        assert base_reward != perturbed_reward
        assert base_report["decoded_transitions"] != perturbed_report["decoded_transitions"]
        assert base_report["mean_boundary_delta"] != perturbed_report["mean_boundary_delta"]

    def test_heldout_family_changes_reward_surface(self):
        producer = TRAINER.load_producer_ckpt(
            PROJECT_ROOT / "data" / "m9_controller_temporal_v3_migrated.ckpt.json"
        )
        primary_reward, primary_report = TRAINER.evaluate_candidate(
            producer,
            tones=TRAINER.TEMPORAL_FAMILIES["default"],
            seed=43,
            noise_rms=0.0,
            device_profile="flat",
            device_gain=1.0,
        )
        heldout_reward, heldout_report = TRAINER.evaluate_candidate(
            producer,
            tones=TRAINER.TEMPORAL_FAMILIES["wide_sweep"],
            seed=1043,
            noise_rms=0.0,
            device_profile="flat",
            device_gain=1.0,
        )

        assert primary_reward != heldout_reward
        assert primary_report["decoded_transitions"] != heldout_report["decoded_transitions"]
        assert primary_report["rendered_samples"] != heldout_report["rendered_samples"]


class TestTemporalTrainerGeneralizationWeights:
    def test_auto_heldout_weight_grows_with_measured_family_gap(self):
        producer = TRAINER.load_producer_ckpt(
            PROJECT_ROOT / "data" / "m9_controller_temporal_v3_migrated.ckpt.json"
        )
        primary_reward, _ = TRAINER.evaluate_candidate(
            producer,
            tones=TRAINER.TEMPORAL_FAMILIES["default"],
            seed=43,
            noise_rms=0.0,
            device_profile="flat",
            device_gain=1.0,
        )
        diphthong_reward, _ = TRAINER.evaluate_candidate(
            producer,
            tones=TRAINER.TEMPORAL_FAMILIES["diphthong"],
            seed=1043,
            noise_rms=0.0,
            device_profile="flat",
            device_gain=1.0,
        )
        wide_sweep_reward, _ = TRAINER.evaluate_candidate(
            producer,
            tones=TRAINER.TEMPORAL_FAMILIES["wide_sweep"],
            seed=2043,
            noise_rms=0.0,
            device_profile="flat",
            device_gain=1.0,
        )

        diphthong_weight = TRAINER.derive_heldout_weight(primary_reward, diphthong_reward)
        wide_sweep_weight = TRAINER.derive_heldout_weight(primary_reward, wide_sweep_reward)

        assert 0.15 <= diphthong_weight <= 0.45
        assert 0.15 <= wide_sweep_weight <= 0.45
        assert wide_sweep_weight > diphthong_weight

    def test_explicit_heldout_weight_override_wins_over_auto_calibration(self):
        resolved = TRAINER.resolve_heldout_weight(
            "0.33",
            calibration_primary_reward=0.8,
            calibration_heldout_reward=0.7,
        )
        auto_resolved = TRAINER.resolve_heldout_weight(
            "auto",
            calibration_primary_reward=0.8,
            calibration_heldout_reward=0.7,
        )

        assert resolved == 0.33
        assert auto_resolved == TRAINER.derive_heldout_weight(0.8, 0.7)

    def test_auto_heldout_weight_prefers_sweep_summary_when_available(self):
        resolved = TRAINER.resolve_heldout_weight(
            "auto",
            calibration_primary_reward=0.8,
            calibration_heldout_reward=0.7,
            sweep_summary={"mean_heldout_weight": 0.23787318072611038},
        )

        assert resolved == 0.23787318072611038

    def test_family_pair_weight_summary_aggregates_pair_weights(self):
        producer = TRAINER.load_producer_ckpt(
            PROJECT_ROOT / "data" / "m9_controller_temporal_v3_migrated.ckpt.json"
        )
        families = ["default", "diphthong"]
        summary = TRAINER.summarize_family_pair_weights(
            producer,
            families=families,
            seed=43,
            noise_rms=0.0,
            device_profile="flat",
            device_gain=1.0,
        )

        manual_weights = []
        manual_primary = []
        manual_heldout = []
        for primary_index, primary_family in enumerate(families):
            for heldout_index, heldout_family in enumerate(families):
                pair_seed = 43 + primary_index * 100 + heldout_index
                primary_reward, _ = TRAINER.evaluate_candidate(
                    producer,
                    tones=TRAINER.TEMPORAL_FAMILIES[primary_family],
                    seed=pair_seed,
                    noise_rms=0.0,
                    device_profile="flat",
                    device_gain=1.0,
                )
                heldout_reward, _ = TRAINER.evaluate_candidate(
                    producer,
                    tones=TRAINER.TEMPORAL_FAMILIES[heldout_family],
                    seed=pair_seed + 1000,
                    noise_rms=0.0,
                    device_profile="flat",
                    device_gain=1.0,
                )
                manual_primary.append(float(primary_reward))
                manual_heldout.append(float(heldout_reward))
                manual_weights.append(TRAINER.derive_heldout_weight(primary_reward, heldout_reward))

        assert summary["mean_heldout_weight"] == float(np.mean(np.asarray(manual_weights, dtype=np.float32)))
        assert summary["mean_primary_reward"] == float(np.mean(np.asarray(manual_primary, dtype=np.float32)))
        assert summary["mean_heldout_reward"] == float(np.mean(np.asarray(manual_heldout, dtype=np.float32)))


class TestTemporalFamilySweep:
    def test_family_pair_sweep_reports_all_ordered_pairs(self):
        report = SWEEP.build_family_pair_sweep_report(
            base_ckpt=PROJECT_ROOT / "data" / "m9_controller_temporal_v3_migrated.ckpt.json",
            families=("default", "diphthong"),
            seed=43,
            noise_rms=0.0,
            device_profile="flat",
            device_gain=1.0,
        )

        assert report["families"] == ["default", "diphthong"]
        assert report["pair_count"] == 4
        assert len(report["pairs"]) == 4
        assert report["pairs"][0]["primary_family"] == "default"
        assert report["pairs"][0]["heldout_family"] == "default"
        assert report["pairs"][1]["heldout_family"] == "diphthong"
        assert report["pairs"][1]["heldout_weight"] >= report["pairs"][0]["heldout_weight"]
        assert report["summary"]["mean_primary_reward"] >= 0.0
        assert report["summary"]["mean_heldout_reward"] >= 0.0
        assert report["summary"]["mean_heldout_weight"] >= 0.15
