"""Regression tests for the M9 TTS vowel-transition speaker harness."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m9_tts_vowel_transition_harness.py"
SPEC = importlib.util.spec_from_file_location("m9_tts_vowel_transition_harness", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)

SR = 16_000


def _fake_tts_audio_fn(*, text: str, output_wav: Path, voice: str, rate: str, pitch: str) -> np.ndarray:
    # Build a deterministic vowel-ish carrier so the harness has a real PCM
    # artifact to persist, but keep the test independent of macOS TTS.
    duration_s = 0.75 + 0.05 * len(text.split())
    sample_count = int(SR * duration_s)
    t = np.linspace(0.0, duration_s, sample_count, endpoint=False, dtype=np.float32)
    base_freq = 180.0 + 7.5 * len(text)
    pcm = 0.12 * np.sin(2.0 * np.pi * base_freq * t).astype(np.float32)
    HARNESS.write_pcm16_wav(output_wav, pcm, sr=SR)
    return pcm


def _fake_monitor_audio_fn(*, pcm: np.ndarray, output_device: int) -> dict[str, object]:
    return {"output_device": output_device, "sample_count": int(len(pcm)), "label": "captured"}


def test_parse_phrases_uses_nonempty_values():
    phrases = HARNESS._parse_phrases(["  eee ahh  ", "ooo ohh", "", "   "])
    assert phrases == ["eee ahh", "ooo ohh"]


def test_tts_vowel_transition_harness_dry_run_writes_report_and_artifacts(tmp_path: Path):
    output_dir = tmp_path / "tts-harness"
    output = tmp_path / "report.json"

    report = HARNESS.run_tts_vowel_transition_harness(
        output_dir=output_dir,
        output=output,
        phrases=["eee ehh ahh ohh ooo", "ooo ohh ahh ehh eee"],
        repeats=2,
        dry_run=True,
        tts_audio_fn=_fake_tts_audio_fn,
        monitor_audio_fn=_fake_monitor_audio_fn,
        threshold_transitions=0,
        threshold_continuity=0.0,
    )

    assert report["dry_run"] is True
    assert report["phrases"] == ["eee ehh ahh ohh ooo", "ooo ohh ahh ehh eee"]
    assert report["repeats"] == 2
    assert report["summary"]["total"] == 4
    assert report["summary"]["pass_count"] == 4
    assert report["summary"]["verdict"] is True
    assert math.isclose(report["summary"]["pass_rate"], 1.0)
    assert math.isclose(report["summary"]["threshold_transitions"], 0)
    assert math.isclose(report["summary"]["threshold_continuity"], 0.0)

    assert output.exists()
    persisted = output.read_text(encoding="utf-8")
    assert '"output_dir"' in persisted
    assert '"runs"' in persisted
    assert (output_dir / "repeat-01" / "phrase-01" / "tts.wav").exists()
    assert (output_dir / "repeat-01" / "phrase-01" / "captured.wav").exists()
    assert (output_dir / "repeat-02" / "phrase-02" / "tts.wav").exists()
    assert (output_dir / "repeat-02" / "phrase-02" / "captured.wav").exists()

    first_run = report["runs"][0]
    assert first_run["monitor"] == {"output_device": 3, "sample_count": 16000, "label": "captured"}
    assert first_run["phrase"] == "eee ehh ahh ohh ooo"
    assert first_run["verdict"]["passed"] is True
    assert first_run["decoded"]["continuity"] >= 0.0
    assert first_run["decoded"]["decoded_transition_count"] >= 0
