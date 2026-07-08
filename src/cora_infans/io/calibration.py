"""Linearity calibration probe for the XVF3800.

Measures captured RMS at several stimulus SPL levels (approximated by tone
amplitude) BEFORE and AFTER applying the RAW_LINEAR profile, to verify that
disabling AGC/NS yields a more linear channel-energy <-> acoustic-pressure
relationship. Uses the live capture path + External-Headphones stimulus, so it
is a real hardware measurement, not a simulation.

All firmware writes are volatile (revert on reboot); nothing is persisted.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
import wave

import numpy as np

from .hardware_tuner import HardwareTuner
from .live_mic import LiveMicArray
from .live_listener import ParenteseStimulus


def _make_tone(path: str, freq_hz: float, amp: float, sr: int = 48000, dur: float = 1.0) -> None:
    t = np.arange(int(dur * sr)) / sr
    pcm = (amp * np.sin(2 * np.pi * freq_hz * t) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _rms_of_capture(wav_path: str) -> list[float]:
    with wave.open(wav_path, "rb") as w:
        n = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if data.size == 0:
        return [0.0] * n
    mat = data.reshape(-1, n).astype(np.float32) / 32768.0
    return [float(np.sqrt(np.mean(col ** 2))) for col in mat.T]


def calibrate_linearity(
    levels: tuple[float, ...] = (0.2, 0.4, 0.7, 1.0),
    freq_hz: float = 500.0,
) -> dict[str, object]:
    """Play graded tones, capture RMS with factory vs raw-linear firmware.

    Returns per-level RMS for both profiles so the caller can assess linearity.
    """
    tuner = HardwareTuner()
    array = LiveMicArray()
    stim = ParenteseStimulus()

    def measure() -> list[float]:
        # Measure happens under the CURRENT firmware state.
        rows: list[float] = []
        for amp in levels:
            tone = tempfile.mktemp(suffix=".wav")
            _make_tone(tone, freq_hz, amp)
            proc = stim.play(tone)  # non-blocking to External Headphones
            time.sleep(0.1)
            frame = array.capture_frame(duration_ms=700)
            proc.wait(timeout=3)
            rows.append(max(frame.channel_energies))
        return rows

    # Factory state first.
    factory = measure()
    # Apply raw-linear (volatile) and re-measure.
    tuner.apply_raw_linear()
    raw_linear = measure()
    # Restore factory-ish: NS back on so we don't leave the board altered.
    tuner.write("PP_MIN_NS", [0.15])
    tuner.write("PP_MIN_NN", [0.51])

    return {
        "levels": list(levels),
        "freq_hz": freq_hz,
        "factory_rms": factory,
        "raw_linear_rms": raw_linear,
    }
