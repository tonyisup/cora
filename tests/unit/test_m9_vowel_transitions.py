"""Tests for the M9 vowel-transition family helpers."""

from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path

import numpy as np

from m9_sound.controller import ControllerParams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "packages" / "m9_sound" / "vowel_transitions.py"
SPEC = importlib.util.spec_from_file_location("m9_sound.vowel_transitions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VT)


class TestVowelTransitionFamilies:
    def test_default_family_report_is_deterministic(self):
        report1 = VT.build_vowel_transition_report(family="default", noise_rms=0.0, noise_seed=7)
        report2 = VT.build_vowel_transition_report(family="default", noise_rms=0.0, noise_seed=7)

        assert report1 == report2
        assert report1["family"] == "default"
        assert report1["input_transition_count"] == 2
        assert report1["decoded_transition_count"] == 2
        assert report1["stimulus_frame_count"] == 12
        assert report1["rendered_frame_count"] == 12
        assert report1["checkpoint_paths"]["substrate"]

    def test_longer_family_has_expected_transition_count(self):
        report = VT.build_vowel_transition_report(family="wide_sweep", noise_rms=0.01, noise_seed=13)

        assert report["family"] == "wide_sweep"
        assert report["input_transition_count"] == 6
        assert report["decoded_transition_count"] >= 6
        assert report["stimulus_frame_count"] == 21
        assert report["rendered_frame_count"] == 21
        assert len(report["stimulus_vowels"]) == 7

    def test_heldout_family_changes_reward_surface(self):
        base = VT.build_vowel_transition_report(family="default", noise_rms=0.0, noise_seed=7)
        heldout = VT.build_vowel_transition_report(
            family="wide_sweep",
            heldout_family="diphthong",
            noise_rms=0.0,
            noise_seed=7,
        )

        assert base["reward"] != heldout["reward"]
        assert base["heldout_reward"] != heldout["heldout_reward"]
        assert base["heldout_family"] == "diphthong"


class TestVowelTransitionArtifacts:
    def test_write_vowel_transition_wav_creates_pcm16_artifact(self, tmp_path):
        output_path = tmp_path / "m9-vowel-transition.wav"

        report = VT.write_vowel_transition_wav(
            output_path,
            family="default",
            noise_rms=0.0,
            noise_seed=7,
        )

        assert report["output_wav"] == str(output_path)
        assert output_path.exists()

        with wave.open(str(output_path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 16000
            assert wav_file.getnframes() == report["rendered_samples"]


class TestVowelTransitionControllerDecoding:
    def test_rendered_report_includes_controller_shape_metadata(self):
        report = VT.build_vowel_transition_report(family="default", noise_rms=0.0, noise_seed=7)

        assert isinstance(report["controller_hidden_dim"], int)
        assert report["controller_hidden_dim"] == 32
        assert isinstance(report["sample_rate"], int)
        assert report["sample_rate"] == 16000
        assert isinstance(report["rendered_audio"], list)
        assert len(report["rendered_audio"]) == report["rendered_samples"]
