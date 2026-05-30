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
        """Generate a square wave.

        For M9 live capture, we use a soft-limited square approximation so the
        waveform remains square-like but is less brittle on real hardware.
        """
        t = np.arange(int(self._sr * duration_s), dtype=np.float32) / self._sr
        raw = np.sin(2 * np.pi * freq_hz * t + phase)
        return (amplitude * np.sign(raw)).astype(np.float32)

    def formant(self, formants: list[dict], duration_s: float,
                 cumulative_phase: float | None = None) -> tuple[np.ndarray, float]:
        """Generate a formant-rich sound with multiple resonances.

        Parameters
        ----------
        formants : list of dict
            Each dict has keys: 'freq' (Hz), 'amp' (0-1), 'bw' (bandwidth in Hz)
            Example: [{'freq': 700, 'amp': 0.5, 'bw': 50},
                      {'freq': 1200, 'amp': 0.3, 'bw': 80},
                      {'freq': 2500, 'amp': 0.2, 'bw': 120}]
        duration_s : float
            Duration in seconds.
        cumulative_phase : float or None
            If provided, start from this phase value for continuity.
            If None, start from 0.

        Returns
        -------
        np.ndarray
            Audio samples (float32).
        float
            Updated cumulative_phase for the next frame.
        """
        t = np.arange(int(self._sr * duration_s), dtype=np.float32) / self._sr
        output = np.zeros(len(t), dtype=np.float32)

        phase = cumulative_phase if cumulative_phase is not None else 0.0

        for fm in formants:
            freq = fm['freq']
            amp = fm['amp']
            bw = fm.get('bw', 100)  # default 100 Hz bandwidth

            # Generate sine at formant frequency with phase continuity
            frame = amp * np.sin(2 * np.pi * freq * t + phase)

            # Apply formant bandwidth as a simple bandpass envelope
            # Narrower bandwidth = more resonant peak
            if bw > 0:
                # Simple resonance: the formant amplitude peaks at the center freq
                # and falls off. We simulate this with a Gaussian envelope in time.
                envelope = np.exp(-((t - t[len(t)//2]) ** 2) * (bw / 5) ** 2)
                # Normalize envelope to have max 1
                if envelope.max() > 0:
                    envelope = envelope / envelope.max()
                frame = frame * envelope

            output += frame

            # Update phase for this formant
            phase += 2 * np.pi * freq * (len(t) / self._sr)

        # Normalize to prevent clipping
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output = output / peak

        final_phase = phase % (2 * np.pi)
        return output.astype(np.float32), final_phase

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
