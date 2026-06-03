"""Regression tests for the M9 live-capture retrain smoke harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path("/Users/juicebox/Projects/cora/scripts/m9_live_retrain.py")

spec = importlib.util.spec_from_file_location("m9_live_retrain", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

VOWELS = ["i", "E", "a", "O", "u"]


def make_capture_dir(tmp_path: Path) -> Path:
    capture_dir = tmp_path / "m9_live_captures"
    capture_dir.mkdir()
    for idx, vowel in enumerate(VOWELS):
        sample = {
            "audio_rms": 0.3 + idx * 0.01,
            "mel_frames": (np.full((2, module.N_MEL), 0.1 * (idx + 1), dtype=np.float32)).tolist(),
            "n_frames": 2,
        }
        (capture_dir / f"{vowel}_000.json").write_text(json.dumps(sample), encoding="utf-8")
    return capture_dir


def test_load_captures_accepts_override_directory(tmp_path: Path):
    capture_dir = make_capture_dir(tmp_path)

    captures = module.load_captures(capture_dir)

    assert set(captures) == set(VOWELS)
    assert all(len(captures[v]) == 1 for v in VOWELS)
    assert captures["i"][0].shape == (2, module.N_MEL)


def test_run_smoke_retrain_reports_live_capture_counts(tmp_path: Path):
    capture_dir = make_capture_dir(tmp_path)

    report = module.run_smoke_retrain(capture_dir=capture_dir, n_synthetic=1, epochs=1)

    assert report["trained"] is True
    assert report["live_captured"] == len(VOWELS)
    assert report["vowel_counts"] == {v: 1 for v in VOWELS}
    assert report["dataset_size"] == len(VOWELS) * 2
    assert report["capture_dir"] == str(capture_dir)
