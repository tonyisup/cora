"""Smoke tests for M9 sound package."""

from __future__ import annotations

import numpy as np
import pytest

from m9_sound.oscillator import OscillatorBank
from m9_sound.substrate import AcousticSubstrate
from m9_sound.controller import NeuralController, ControllerParams
from m9_sound.similarity import AcousticSimilarity, SimilarityResult
from m9_sound.producer import AcousticProducer


class TestOscillatorBank:
    def test_sine_basic(self):
        osc = OscillatorBank(sr=16000)
        tone = osc.sine(440, 1.0, amplitude=0.5)
        assert len(tone) == 16000
        assert tone.dtype == np.float32
        assert np.max(np.abs(tone)) <= 0.5 + 1e-6

    def test_sine_frequency(self):
        osc = OscillatorBank(sr=16000)
        tone = osc.sine(440, 0.1, amplitude=1.0)
        # Count zero crossings — should be ~88 per 0.1s (440 Hz = 440 crosses per second, both directions)
        zc = np.sum(np.abs(np.diff(np.signbit(tone).astype(float))) > 0)
        assert 70 < zc < 110  # rough tolerance

    def test_triangle(self):
        osc = OscillatorBank(sr=16000)
        tri = osc.triangle(220, 0.5, amplitude=0.3)
        assert len(tri) == 8000
        assert np.max(np.abs(tri)) <= 0.3 + 1e-6

    def test_mix(self):
        osc = OscillatorBank(sr=16000)
        # mix expects (waveform, freq, amplitude, phase)
        mixed = osc.mix([("sine", 440, 0.3, 0.0), ("triangle", 880, 0.2, 0.0)])
        assert len(mixed) == 32000  # 2s default
        assert np.max(np.abs(mixed)) <= 1.0 + 1e-6

    def test_generate(self):
        osc = OscillatorBank(sr=16000)
        audio = osc.generate({
            "waveform": "sine", "freq_hz": 880,
            "duration_s": 0.5, "amplitude": 0.4,
        })
        assert len(audio) == 8000
        assert np.max(np.abs(audio)) <= 0.4 + 1e-6


class TestAcousticSubstrate:
    def test_init(self):
        sub = AcousticSubstrate(hidden_dim=32, sr=16000)
        assert sub.hidden_dim == 32
        assert len(sub.hidden_state) == 32

    def test_step(self):
        sub = AcousticSubstrate(hidden_dim=16, sr=16000, frame_size=480)
        frame = np.sin(2 * np.pi * 440 * np.arange(480) / 16000).astype(np.float32)
        h = sub.step(frame)
        assert len(h) == 16
        assert np.all(np.isfinite(h))

    def test_reset(self):
        sub = AcousticSubstrate(hidden_dim=16, sr=16000)
        frame = np.random.randn(480).astype(np.float32) * 0.1
        sub.step(frame)
        sub.reset()
        assert np.allclose(sub.hidden_state, 0.0)

    def test_process(self):
        sub = AcousticSubstrate(hidden_dim=16, sr=16000)
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        h = sub.process(audio)
        assert len(h) == 16
        assert np.all(np.isfinite(h))


class TestNeuralController:
    def test_init(self):
        ctrl = NeuralController(hidden_dim=32, sr=16000)
        assert ctrl.n_params == 4 * 32 + 4  # W + b

    def test_decode(self):
        ctrl = NeuralController(hidden_dim=32, sr=16000)
        h = np.zeros(32, dtype=np.float32)
        params = ctrl.decode(h)
        assert isinstance(params, ControllerParams)
        assert 80.0 <= params.freq_hz <= 2000.0
        assert 0.05 <= params.amplitude <= 1.0
        assert 0 <= params.waveform < 4

    def test_perturb(self):
        ctrl = NeuralController(hidden_dim=32, sr=16000)
        w_before = ctrl._W.copy()
        ctrl.perturb(scale=0.5)
        assert not np.allclose(ctrl._W, w_before)

    def test_forward(self):
        ctrl = NeuralController(hidden_dim=16, sr=16000)
        h = np.random.randn(16).astype(np.float32) * 0.1
        out = ctrl.forward(h)
        assert len(out) == 4
        assert np.all(np.isfinite(out))


class TestAcousticSimilarity:
    def test_identical_signals(self):
        sim = AcousticSimilarity(sr=16000)
        tone = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        result = sim.detailed_score(tone, tone)
        assert result.overall >= 0.99  # identical signals should score ~1.0
        assert result.mfcc_distance < 0.01

    def test_different_frequencies(self):
        sim = AcousticSimilarity(sr=16000)
        tone_a = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        tone_b = np.sin(2 * np.pi * 880 * np.arange(16000) / 16000).astype(np.float32)
        result = sim.detailed_score(tone_a, tone_b)
        assert result.overall < 0.9  # different frequencies should score lower
        assert result.pitch_distance > 0.1

    def test_different_durations(self):
        sim = AcousticSimilarity(sr=16000)
        tone_a = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        tone_b = np.sin(2 * np.pi * 440 * np.arange(8000) / 16000).astype(np.float32) * 0.5
        result = sim.detailed_score(tone_a, tone_b)
        assert result.duration_distance > 0.3  # 2x duration difference

    def test_silence_vs_tone(self):
        sim = AcousticSimilarity(sr=16000)
        silence = np.zeros(16000, dtype=np.float32)
        tone = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        result = sim.detailed_score(silence, tone)
        assert result.overall < 0.5  # silence vs tone should score low


class TestAcousticProducer:
    def test_produce_basic(self):
        producer = AcousticProducer(sr=16000, hidden_dim=16)
        audio = producer.produce(0.5)
        # Producer generates n_frames * frame_size samples
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) <= 1.0 + 1e-6

    def test_produce_and_score(self):
        producer = AcousticProducer(sr=16000, hidden_dim=16)
        target = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        result = producer.produce_and_score(target)
        assert len(result.audio) > 0
        assert 0.0 <= result.similarity.overall <= 1.0
        assert isinstance(result.params, ControllerParams)

    def test_reset(self):
        producer = AcousticProducer(sr=16000, hidden_dim=16)
        producer.produce(0.5)
        h_before = producer.substrate.hidden_state.copy()
        producer.reset()
        assert np.allclose(producer.substrate.hidden_state, 0.0)

    def test_produce_multiple_durations(self):
        producer = AcousticProducer(sr=16000, hidden_dim=16)
        for dur in [0.1, 0.5, 1.0, 2.0]:
            audio = producer.produce(dur)
            assert len(audio) > 0
            assert np.all(np.isfinite(audio))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
