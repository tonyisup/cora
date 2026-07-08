"""Tests for the DoA geometry experiment helpers + stereo stimulus routing."""

import numpy as np
import wave

from cora_infans.io import doa_experiment
from cora_infans.io.doa_experiment import SpeechStimulus
from cora_infans.io.hardware_tuner import DoAReading
from cora_infans.io.live_listener import ParenteseStimulus


def _read_channels(path):
    with wave.open(path, "rb") as w:
        n = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return n, data.reshape(-1, n)


def test_stereo_pan_routes_to_correct_channel():
    stim = ParenteseStimulus()
    # LEFT only (pan=-1): channel 0 has signal, channel 1 silent.
    lp = "/tmp/_stim_l.wav"
    stim.synthesize_wav(lp, pan=-1.0, channels=2)
    n, ch = _read_channels(lp)
    assert n == 2
    assert ch[:, 0].astype(float).max() > 1000  # left has signal
    assert ch[:, 1].astype(float).max() == 0     # right silent

    # RIGHT only (pan=+1): channel 1 has signal, channel 0 silent.
    rp = "/tmp/_stim_r.wav"
    stim.synthesize_wav(rp, pan=1.0, channels=2)
    n, ch = _read_channels(rp)
    assert ch[:, 1].astype(float).max() > 1000   # right has signal
    assert ch[:, 0].astype(float).max() == 0     # left silent


def test_both_pan_fills_both_channels():
    stim = ParenteseStimulus()
    bp = "/tmp/_stim_b.wav"
    stim.synthesize_wav(bp, pan=0.0, channels=2)
    n, ch = _read_channels(bp)
    assert ch[:, 0].astype(float).max() > 1000
    assert ch[:, 1].astype(float).max() > 1000


def test_speech_stimulus_builds_panned_stereo_wav_via_say_and_ffmpeg(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check, stdout=None, stderr=None):
        calls.append(cmd)
        if "ffmpeg" in cmd[0]:
            # Write a minimal valid stereo WAV at the requested output path so
            # callers can treat synthesis as completed.
            out = cmd[-1]
            with wave.open(out, "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(48000)
                w.writeframes(np.zeros((16, 2), dtype=np.int16).tobytes())

    monkeypatch.setattr("subprocess.run", fake_run)
    out = tmp_path / "speech.wav"

    SpeechStimulus().synthesize_wav(str(out), side="left")

    assert out.exists()
    assert calls[0][0] == "/usr/bin/say"
    assert "hello baby" in " ".join(calls[0]).lower()
    assert "ffmpeg" in calls[1][0]
    assert "pan=stereo|c0=c0|c1=0*c0" in calls[1]


def test_speech_stimulus_uses_right_pan_filter(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check, stdout=None, stderr=None):
        calls.append(cmd)
        if "ffmpeg" in cmd[0]:
            out = cmd[-1]
            with wave.open(out, "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(48000)
                w.writeframes(np.zeros((16, 2), dtype=np.int16).tobytes())

    monkeypatch.setattr("subprocess.run", fake_run)

    SpeechStimulus().synthesize_wav(str(tmp_path / "speech-r.wav"), side="right")

    assert "pan=stereo|c0=0*c0|c1=c0" in calls[1]



def test_geometry_report_includes_processed_source_doa_when_available(monkeypatch):
    class FakeTuner:
        def read_doa(self):
            return DoAReading(azimuth_deg=10.0, speech_detected=True)

        def read_selected_azimuths(self):
            # processed source DoA, auto-selected beam; radians.
            return (np.pi / 2, np.pi)

    class FakeProc:
        def wait(self, timeout):
            return 0

        def kill(self):
            pass

    class FakeStimulus:
        def synthesize_wav(self, *args, **kwargs):
            pass

        def play(self, path):
            return FakeProc()

    monkeypatch.setattr(doa_experiment, "HardwareTuner", FakeTuner)
    monkeypatch.setattr(doa_experiment, "ParenteseStimulus", FakeStimulus)

    result = doa_experiment.run_geometry_test(
        "both", stimulus="tone", duration_s=0.01, sample_hz=1000
    )

    assert result["selected_source_samples"] >= 1
    assert result["selected_source_mean_deg"] == 90.0
    assert result["selected_beam_mean_deg"] == 180.0
