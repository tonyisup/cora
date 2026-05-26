"""M9 — Neural controller / readout.

Maps the substrate's hidden state to oscillator parameters (frequency,
amplitude, waveform selection). This is the "trainable" part of the
motor-learning loop.

M9a: fixed random weights (baseline)
M9b: trained via REINFORCE or similar to maximize acoustic similarity

Usage:
    controller = NeuralController(hidden_dim=32, sr=16000)
    params = controller.decode(hidden_state)
    # params: {"freq_hz": 440.0, "amplitude": 0.5, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ControllerParams:
    """Output parameters for the oscillator."""
    freq_hz: float = 440.0
    amplitude: float = 0.5
    phase: float = 0.0
    waveform: int = 0  # 0=sine, 1=triangle, 2=sawtooth, 3=square


class NeuralController:
    """Neural readout from substrate hidden state to oscillator params.

    Simple linear readout: params = W @ h + b
    W and b are the trainable parameters.
    """

    def __init__(self, hidden_dim: int = 32, sr: int = 16000,
                 n_waveforms: int = 4) -> None:
        self._hidden_dim = hidden_dim
        self._sr = sr
        self._n_waveforms = n_waveforms

        # Output: [freq, amplitude, phase, waveform_logits]
        # Frequency range: 80-2000 Hz -> normalized to [0, 1]
        self._freq_min = 80.0
        self._freq_max = 2000.0

        # Initialize weights (these are the trainable params)
        self._W = np.random.randn(4, hidden_dim).astype(np.float32) * 0.1
        self._b = np.array([0.5, 0.3, 0.0, 0.0], dtype=np.float32)

    def decode(self, hidden_state: np.ndarray) -> ControllerParams:
        """Map hidden state to oscillator parameters."""
        h = hidden_state.flatten()[:self._hidden_dim]
        if len(h) < self._hidden_dim:
            h = np.pad(h, (0, self._hidden_dim - len(h)))

        raw = self._W @ h + self._b

        # Frequency: sigmoid to [0, 1], then map to [freq_min, freq_max]
        freq_norm = float(1.0 / (1.0 + np.exp(-raw[0])))  # sigmoid
        freq_hz = self._freq_min + freq_norm * (self._freq_max - self._freq_min)

        # Amplitude: sigmoid to [0, 1]
        amplitude = float(1.0 / (1.0 + np.exp(-raw[1])))

        # Phase: tanh to [-1, 1] -> map to [-pi, pi]
        phase = float(np.tanh(raw[2]) * np.pi)

        # Waveform: argmax of logits
        waveform_logits = raw[3] + np.zeros(self._n_waveforms, dtype=np.float32)
        waveform_logits[0] = raw[3]  # sine
        waveform_logits[1] = raw[3] * 0.5  # triangle (slightly less likely)
        waveform_logits[2] = raw[3] * 0.3  # sawtooth
        waveform_logits[3] = raw[3] * 0.2  # square
        waveform = int(np.argmax(waveform_logits))

        return ControllerParams(
            freq_hz=float(np.clip(freq_hz, self._freq_min, self._freq_max)),
            amplitude=float(np.clip(amplitude, 0.05, 1.0)),
            phase=phase,
            waveform=int(waveform),
        )

    def forward(self, hidden_state: np.ndarray) -> np.ndarray:
        """Return raw output vector (before decoding)."""
        h = hidden_state.flatten()[:self._hidden_dim]
        if len(h) < self._hidden_dim:
            h = np.pad(h, (0, self._hidden_dim - len(h)))
        return self._W @ h + self._b

    def perturb(self, scale: float = 0.1) -> None:
        """Add random noise to weights (for exploration)."""
        self._W += np.random.randn(*self._W.shape).astype(np.float32) * scale
        self._b += np.random.randn(*self._b.shape).astype(np.float32) * scale * 0.5

    def clip_weights(self, max_norm: float = 5.0) -> None:
        """Clip weights to prevent explosion."""
        w_norm = np.linalg.norm(self._W)
        if w_norm > max_norm:
            self._W = self._W * (max_norm / w_norm)
        self._b = np.clip(self._b, -max_norm, max_norm)

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return self._W.size + self._b.size

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {"W": self._W.copy(), "b": self._b.copy()}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self._W = weights["W"].astype(np.float32).reshape(self._W.shape)
        self._b = weights["b"].astype(np.float32).reshape(self._b.shape)
