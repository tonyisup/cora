"""Tests for the DoA geometry experiment helpers + stereo stimulus routing."""

import numpy as np
import wave

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
