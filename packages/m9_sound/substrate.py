"""M9 — Recurrent acoustic substrate.

Holds short-term sound state as a hidden vector. Updated each timestep
by the oscillator input. The controller readout maps this state to
acoustic output parameters.

This is the "memory" of the motor-learning loop — it captures temporal
patterns in the sound being produced.

Usage:
    substrate = AcousticSubstrate(hidden_dim=32, sr=16000)
    for sample in audio_chunk:
        state = substrate.step(sample)
    output = controller(state)
"""

from __future__ import annotations

import numpy as np


class AcousticSubstrate:
    """Recurrent state holder for temporal sound patterns.

    A simple Elman-style RNN: h_t = tanh(W_h * h_{t-1} + W_x * x_t + b).
    The hidden state captures temporal context for the controller to shape.
    """

    def __init__(self, hidden_dim: int = 32, sr: int = 16000,
                 frame_size: int = 480) -> None:
        self._hidden_dim = hidden_dim
        self._sr = sr
        self._frame_size = frame_size
        self._h = np.zeros(hidden_dim, dtype=np.float32)
        # Random init for weights (will be learned in M9b+)
        scale = 0.1
        self._W_h = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self._W_x = np.random.randn(hidden_dim, frame_size).astype(np.float32) * scale
        self._b = np.zeros(hidden_dim, dtype=np.float32)

    def reset(self) -> None:
        """Reset hidden state to zero."""
        self._h = np.zeros(self._hidden_dim, dtype=np.float32)

    def step(self, frame: np.ndarray) -> np.ndarray:
        """Process one audio frame, return updated hidden state.

        Parameters
        ----------
        frame : np.ndarray
            Audio samples, length should match frame_size.
            Will be truncated or zero-padded if different.

        Returns
        -------
        np.ndarray
            Hidden state vector (hidden_dim,).
        """
        x = np.zeros(self._frame_size, dtype=np.float32)
        n = min(len(frame), self._frame_size)
        x[:n] = frame[:n]
        # Normalize input
        rms = np.sqrt(np.mean(x ** 2))
        if rms > 1e-6:
            x = x / rms
        # RNN step
        self._h = np.tanh(self._W_h @ self._h + self._W_x @ x + self._b)
        return self._h.copy()

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process full audio, return final hidden state."""
        self.reset()
        for i in range(0, len(audio), self._frame_size):
            frame = audio[i:i + self._frame_size]
            self.step(frame)
        return self._h.copy()

    @property
    def hidden_state(self) -> np.ndarray:
        return self._h.copy()

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
