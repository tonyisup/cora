"""Tests for the M9 temporal-sequence smoke helpers."""

from __future__ import annotations

import numpy as np

from m9_sound.controller import ControllerParams
from m9_sound.temporal import (
    generate_tone_sequence,
    render_param_sequence,
    run_temporal_smoke_report,
)


class TestTemporalNoiseRobustness:
    def test_smoke_report_handles_room_noise(self):
        report = run_temporal_smoke_report(input_noise_rms=0.02, noise_seed=7)

        assert report["input_noise_rms"] == 0.02
        assert report["noise_seed"] == 7
        assert report["decoded_transition_count"] == 2
        assert report["rendered_frame_count"] == 12


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
