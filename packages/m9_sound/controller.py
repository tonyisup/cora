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

    Two-layer MLP: params = W2 @ tanh(W1 @ h + b1) + b2
    The hidden layer lets the controller learn multiple frequency mappings
    without catastrophic forgetting.
    """

    def __init__(self, hidden_dim: int = 32, sr: int = 16000,
                 n_waveforms: int = 4, controller_hidden_dim: int = 16) -> None:
        self._hidden_dim = hidden_dim
        self._sr = sr
        self._n_waveforms = n_waveforms
        self._controller_hidden = controller_hidden_dim

        self._freq_min = 80.0
        self._freq_max = 2000.0

        # Layer 1: hidden_state → controller hidden
        self._W1 = np.random.randn(controller_hidden_dim, hidden_dim).astype(np.float32) * 0.2
        self._b1 = np.zeros(controller_hidden_dim, dtype=np.float32)

        # Layer 2: controller hidden → output params
        self._W2 = np.random.randn(4 + n_waveforms, controller_hidden_dim).astype(np.float32) * 0.1
        self._b2 = np.array([0.5, 0.3, 0.0, 0.0] + [0.0] * n_waveforms, dtype=np.float32)

    def _forward_all(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass returning (raw_output, hidden_activation)."""
        z1 = self._W1 @ h + self._b1
        a1 = np.tanh(z1)
        raw = self._W2 @ a1 + self._b2
        return raw, a1

    def decode(self, hidden_state: np.ndarray) -> ControllerParams:
        """Map hidden state to oscillator parameters."""
        h = hidden_state.flatten()[:self._hidden_dim]
        if len(h) < self._hidden_dim:
            h = np.pad(h, (0, self._hidden_dim - len(h)))
        raw, _ = self._forward_all(h)

        freq_norm = float(1.0 / (1.0 + np.exp(-raw[0])))
        freq_hz = self._freq_min + freq_norm * (self._freq_max - self._freq_min)
        amplitude = float(1.0 / (1.0 + np.exp(-raw[1])))
        phase = float(np.tanh(raw[2]) * np.pi)
        waveform_logits = raw[3:3 + self._n_waveforms]
        waveform = int(np.argmax(waveform_logits))

        return ControllerParams(
            freq_hz=float(np.clip(freq_hz, self._freq_min, self._freq_max)),
            amplitude=float(np.clip(amplitude, 0.05, 1.0)),
            phase=phase,
            waveform=waveform,
        )

    def forward(self, hidden_state: np.ndarray) -> np.ndarray:
        """Return raw output vector (before decoding)."""
        h = hidden_state.flatten()[:self._hidden_dim]
        if len(h) < self._hidden_dim:
            h = np.pad(h, (0, self._hidden_dim - len(h)))
        raw, _ = self._forward_all(h)
        return raw

    def get_hidden_activation(self, hidden_state: np.ndarray) -> np.ndarray:
        """Return hidden layer activation (for REINFORCE)."""
        h = hidden_state.flatten()[:self._hidden_dim]
        if len(h) < self._hidden_dim:
            h = np.pad(h, (0, self._hidden_dim - len(h)))
        _, a1 = self._forward_all(h)
        return a1

    def perturb(self, scale: float = 0.1) -> None:
        """Add random noise to weights (for exploration)."""
        self._W1 += np.random.randn(*self._W1.shape).astype(np.float32) * scale
        self._b1 += np.random.randn(*self._b1.shape).astype(np.float32) * scale * 0.5
        self._W2 += np.random.randn(*self._W2.shape).astype(np.float32) * scale
        self._b2 += np.random.randn(*self._b2.shape).astype(np.float32) * scale * 0.5

    def clip_weights(self, max_norm: float = 5.0) -> None:
        """Clip weights to prevent explosion."""
        for W in [self._W1, self._W2]:
            w_norm = np.linalg.norm(W)
            if w_norm > max_norm:
                W[:] = W * (max_norm / w_norm)
        self._b1 = np.clip(self._b1, -max_norm, max_norm)
        self._b2 = np.clip(self._b2, -max_norm, max_norm)

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return self._W1.size + self._b1.size + self._W2.size + self._b2.size

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self._W1.copy(), "b1": self._b1.copy(),
            "W2": self._W2.copy(), "b2": self._b2.copy(),
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self._W1 = weights["W1"].astype(np.float32).reshape(self._W1.shape)
        self._b1 = weights["b1"].astype(np.float32).reshape(self._b1.shape)
        self._W2 = weights["W2"].astype(np.float32).reshape(self._W2.shape)
        self._b2 = weights["b2"].astype(np.float32).reshape(self._b2.shape)
