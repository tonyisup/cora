"""Tests for the M9 vowel-transition family helpers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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

    def test_default_family_contains_real_transition_segments(self):
        report = VT.build_vowel_transition_report(family="default", noise_rms=0.0, noise_seed=7)

        assert report["transition_segments"] == [
            {"start_vowel": "i", "end_vowel": "i", "frame_count": 4},
            {"start_vowel": "i", "end_vowel": "E", "frame_count": 4},
            {"start_vowel": "E", "end_vowel": "u", "frame_count": 4},
        ]
        assert report["segment_frame_count"] == 4
        assert report["transition_segment_count"] == 3

    def test_phase_triad_family_forms_a_clean_three_transition_chain(self):
        report = VT.build_vowel_transition_report(family="phase_triad", noise_rms=0.0, noise_seed=11)

        assert report["family"] == "phase_triad"
        assert report["input_transition_count"] == 3
        assert report["transition_segment_count"] == 4
        assert report["segment_frame_count"] == 6
        assert report["stimulus_frame_count"] == 24
        assert report["rendered_frame_count"] == 24
        assert report["transition_segments"] == [
            {"start_vowel": "i", "end_vowel": "i", "frame_count": 6},
            {"start_vowel": "i", "end_vowel": "E", "frame_count": 6},
            {"start_vowel": "E", "end_vowel": "a", "frame_count": 6},
            {"start_vowel": "a", "end_vowel": "i", "frame_count": 6},
        ]

    def test_longer_family_has_expected_transition_count(self):
        report = VT.build_vowel_transition_report(family="wide_sweep", noise_rms=0.01, noise_seed=13)

        assert report["family"] == "wide_sweep"
        assert report["input_transition_count"] == 6
        assert report["decoded_transition_count"] >= 6
        assert report["stimulus_frame_count"] == 21
        assert report["rendered_frame_count"] == 21
        assert len(report["stimulus_vowels"]) == 7

    def test_long_arc_family_uses_a_longer_transition_chain(self):
        report = VT.build_vowel_transition_report(family="long_arc", noise_rms=0.0, noise_seed=13)

        assert report["family"] == "long_arc"
        assert report["input_transition_count"] == 8
        assert report["transition_segment_count"] == 9
        assert report["segment_frame_count"] == 2
        assert report["stimulus_frame_count"] == 18
        assert report["rendered_frame_count"] == 18
        assert report["transition_segments"][0] == {"start_vowel": "i", "end_vowel": "i", "frame_count": 2}
        assert report["transition_segments"][-1] == {"start_vowel": "E", "end_vowel": "i", "frame_count": 2}

    def test_formant_rich_family_spans_all_vowels(self):
        report = VT.build_vowel_transition_report(family="formant_rich", noise_rms=0.0, noise_seed=21)

        assert report["family"] == "formant_rich"
        assert report["input_transition_count"] == 12
        assert report["transition_segment_count"] == 13
        assert report["segment_frame_count"] == 2
        assert report["stimulus_frame_count"] == 26
        assert report["rendered_frame_count"] == 26
        assert report["stimulus_unique_vowel_count"] == 5
        assert report["transition_segments"][0] == {"start_vowel": "i", "end_vowel": "i", "frame_count": 2}
        assert report["transition_segments"][-1] == {"start_vowel": "O", "end_vowel": "i", "frame_count": 2}

    def test_dense_cycle_family_extends_formant_rich_coverage(self):
        report = VT.build_vowel_transition_report(family="dense_cycle", noise_rms=0.0, noise_seed=21)

        assert report["family"] == "dense_cycle"
        assert report["input_transition_count"] == 16
        assert report["transition_segment_count"] == 17
        assert report["segment_frame_count"] == 2
        assert report["stimulus_frame_count"] == 34
        assert report["rendered_frame_count"] == 34
        assert report["stimulus_unique_vowel_count"] == 5
        assert report["transition_segments"][0] == {"start_vowel": "i", "end_vowel": "i", "frame_count": 2}
        assert report["transition_segments"][-1] == {"start_vowel": "E", "end_vowel": "i", "frame_count": 2}

    def test_ultra_dense_family_pushes_beyond_dense_cycle(self):
        report = VT.build_vowel_transition_report(family="ultra_dense", noise_rms=0.0, noise_seed=21)

        assert report["family"] == "ultra_dense"
        assert report["input_transition_count"] == 21
        assert report["transition_segment_count"] == 22
        assert report["segment_frame_count"] == 2
        assert report["stimulus_frame_count"] == 44
        assert report["rendered_frame_count"] == 44
        assert report["stimulus_unique_vowel_count"] == 5
        assert report["transition_segments"][0] == {"start_vowel": "i", "end_vowel": "i", "frame_count": 2}
        assert report["transition_segments"][-1] == {"start_vowel": "i", "end_vowel": "i", "frame_count": 2}

    def test_hyper_dense_family_pushes_beyond_ultra_dense(self):
        report = VT.build_vowel_transition_report(family="hyper_dense", noise_rms=0.0, noise_seed=21)

        assert report["family"] == "hyper_dense"
        assert report["input_transition_count"] == 27
        assert report["transition_segment_count"] == 28
        assert report["segment_frame_count"] == 2
        assert report["stimulus_frame_count"] == 56
        assert report["rendered_frame_count"] == 56
        assert report["stimulus_unique_vowel_count"] == 5
        assert report["transition_segments"][0] == {"start_vowel": "i", "end_vowel": "i", "frame_count": 2}
        assert report["transition_segments"][-1] == {"start_vowel": "i", "end_vowel": "i", "frame_count": 2}

    def test_mega_dense_family_pushes_beyond_hyper_dense(self):
        report = VT.build_vowel_transition_report(family="mega_dense", noise_rms=0.0, noise_seed=21)

        assert report["family"] == "mega_dense"
        assert report["input_transition_count"] == 33
        assert report["transition_segment_count"] == 34
        assert report["segment_frame_count"] == 2
        assert report["stimulus_frame_count"] == 68
        assert report["rendered_frame_count"] == 68
        assert report["stimulus_unique_vowel_count"] == 5
        assert report["transition_segments"][0] == {"start_vowel": "i", "end_vowel": "i", "frame_count": 2}
        assert report["transition_segments"][-1] == {"start_vowel": "a", "end_vowel": "i", "frame_count": 2}

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

    def test_cli_smoke_writes_transition_segments_json(self, tmp_path):
        output_json = tmp_path / "m9-vowel-transition.json"

        script = PROJECT_ROOT / "scripts" / "m9_vowel_transition_smoke.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--family",
                "default",
                "--output",
                str(output_json),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        payload = json.loads(output_json.read_text())
        assert payload["family"] == "default"
        assert payload["transition_segments"] == [
            {"start_vowel": "i", "end_vowel": "i", "frame_count": 4},
            {"start_vowel": "i", "end_vowel": "E", "frame_count": 4},
            {"start_vowel": "E", "end_vowel": "u", "frame_count": 4},
        ]

    def test_cli_smoke_accepts_long_arc_family(self, tmp_path):
        output_json = tmp_path / "m9-vowel-transition-long-arc.json"

        script = PROJECT_ROOT / "scripts" / "m9_vowel_transition_smoke.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--family",
                "long_arc",
                "--output",
                str(output_json),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        payload = json.loads(output_json.read_text())
        assert payload["family"] == "long_arc"
        assert payload["transition_segment_count"] == 9
        assert payload["segment_frame_count"] == 2

    def test_demo_script_writes_listenable_montage(self, tmp_path):
        output_dir = tmp_path / "demo"
        manifest_path = output_dir / "m9-vowel-transition-demo.json"

        script = PROJECT_ROOT / "scripts" / "m9_vowel_transition_demo.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        manifest = json.loads(manifest_path.read_text())
        assert manifest["families"] == ["long_arc", "formant_rich", "dense_cycle", "ultra_dense", "hyper_dense", "mega_dense"]
        assert manifest["montage_wav"] == str(output_dir / "m9-vowel-transition-demo.wav")
        assert len(manifest["sections"]) == 6
        assert (output_dir / "m9-vowel-transition-long_arc.wav").exists()
        assert (output_dir / "m9-vowel-transition-formant_rich.wav").exists()
        assert (output_dir / "m9-vowel-transition-dense_cycle.wav").exists()
        assert (output_dir / "m9-vowel-transition-ultra_dense.wav").exists()
        assert (output_dir / "m9-vowel-transition-hyper_dense.wav").exists()
        assert (output_dir / "m9-vowel-transition-mega_dense.wav").exists()
        assert (output_dir / "m9-vowel-transition-demo.wav").exists()

        with wave.open(str(output_dir / "m9-vowel-transition-demo.wav"), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 16000
            assert wav_file.getnframes() == manifest["montage_sample_count"]


class TestVowelTransitionControllerDecoding:
    def test_rendered_report_includes_controller_shape_metadata(self):
        report = VT.build_vowel_transition_report(family="default", noise_rms=0.0, noise_seed=7)

        assert isinstance(report["controller_hidden_dim"], int)
        assert report["controller_hidden_dim"] == 32
        assert isinstance(report["sample_rate"], int)
        assert report["sample_rate"] == 16000
        assert isinstance(report["rendered_audio"], list)
        assert len(report["rendered_audio"]) == report["rendered_samples"]
