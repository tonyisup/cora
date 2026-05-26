"""M9 — Oscillator bank.

Produces simple periodic sound waveforms: sine, triangle, sawtooth, square.
Each oscillator has controllable frequency, amplitude, and phase.

Usage:
    osc = OscillatorBank(sr=16000)
    tone = osc.sine(freq_hz=440, duration_s=1.0, amplitude=0.5)
    mixed = osc.mix([("sine", 440, 0.3), ("triangle", 880, 0.2)])
"""

from __future__ import annotations

import numpy as np


class OscillatorBank:
    """Bank of simple periodic oscillators for sound production.

    M9a uses this as the "driver" — the rhythmic input that keeps the
    acoustic system moving. The neural controller (substrate + readout)
    modulates the oscillator parameters to shape the output.
    """

    def __init__(self, sr: int = 16000) -> None:
        self._sr = sr

    def sine(self, freq_hz: float, duration_s: float,
             amplitude: float = 0.5, phase: float = 0.0) -> np.ndarray:
        """Generate a sine wave."""
        t = np.arange(int(self._sr * duration_s), dtype=np.float32) / self._sr
        return (amplitude * np.sin(2 * np.pi * freq_hz * t + phase)).astype(np.float32)

    def triangle(self, freq_hz: float, duration_s: float,
                 amplitude: float = 0.5, phase: float = 0.0) -> np.ndarray:
        """Generate a triangle wave."""
        t = np.arange(int(self._sr * duration_s), dtype=np.float32) / self._sr
        # Triangle: 2 * |2 * (ft + phase/2π) - floor(2 * (ft + phase/2π)) - 1| - 1
        phase_norm = phase / (2 * np.pi)
        ramp = 2.0 * ((freq_hz * t + phase_norm) % 1.0) - 1.0
        tri = 2.0 * np.abs(ramp) - 1.0
        return (amplitude * tri).astype(np.float32)

    def sawtooth(self, freq_hz: float, duration_s: float,
                 amplitude: float = 0.5, phase: float = 0.0) -> np.ndarray:
        """Generate a sawtooth wave."""
        t = np.arange(int(self._sr * duration_s), dtype=np.float32) / self._sr
        phase_norm = phase / (2 * np.pi)
        saw = 2.0 * ((freq_hz * t + phase_norm) % 1.0) - 1.0
        return (amplitude * saw).astype(np.float32)

    def square(self, freq_hz: float, duration_s: float,
               amplitude: float = 0.5, phase: float = 0.0) -> np.ndarray:
        """Generate a square wave."""
        t = np.arange(int(self._sr * duration_s), dtype=np.float32) / self._sr
        phase_norm = phase / (2 * np.pi)
        sq = np.sign(np.sin(2 * np.pi * freq_hz * t + phase))
        return (amplitude * sq).astype(np.float32)

    def mix(self, parts: list[tuple[str, float, float, float]]) -> np.ndarray:
        """Mix multiple oscillator outputs.

        Parameters
        ----------
        parts : list of (waveform_type, freq_hz, amplitude, phase)

        Returns
        -------
        np.ndarray
            Mixed audio, clipped to [-1, 1].
        """
        if not parts:
            return np.zeros(0, dtype=np.float32)
        # First part determines duration
        waveforms = {
            "sine": self.sine, "triangle": self.triangle,
            "sawtooth": self.sawtooth, "square": self.square,
        }
        # Generate each part
        generated = []
        max_len = 0
        for wtype, freq, amp, phase in parts:
            fn = waveforms.get(wtype, self.sine)
            n_samples = int(self._sr * 2.0)  # default 2s
            w = fn(freq, n_samples / self._sr, amp, phase)
            generated.append(w)
            max_len = max(max_len, len(w))
        # Pad and sum
        mix = np.zeros(max_len, dtype=np.float32)
        for w in generated:
            mix[:len(w)] += w
        return np.clip(mix, -1.0, 1.0).astype(np.float32)

    def generate(self, params: dict) -> np.ndarray:
        """Generate audio from a parameter dict.

        Parameters
        ----------
        params : dict with keys:
            - waveform: str (sine/triangle/sawtooth/square)
            - freq_hz: float
            - duration_s: float
            - amplitude: float (default 0.5)
            - phase: float (default 0.0)

        Returns
        -------
        np.ndarray
            Generated audio.
        """
        fn_map = {
            "sine": self.sine, "triangle": self.triangle,
            "sawtooth": self.sawtooth, "square": self.square,
        }
        fn = fn_map.get(params.get("waveform", "sine"), self.sine)
        return fn(
            freq_hz=float(params.get("freq_hz", 440.0)),
            duration_s=float(params.get("duration_s", 1.0)),
            amplitude=float(params.get("amplitude", 0.5)),
            phase=float(params.get("phase", 0.0)),
        )
