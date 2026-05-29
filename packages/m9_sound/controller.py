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

    def train_supervised(
        self,
        hidden_states: np.ndarray,
        target_params: np.ndarray,
        lr: float = 0.01,
        n_epochs: int = 100,
    ) -> list[float]:
        """Train controller via supervised learning.

        Parameters
        ----------
        hidden_states : np.ndarray, shape (n_samples, hidden_dim)
            Substrate hidden state for each sample.
        target_params : np.ndarray, shape (n_samples, 4)
            Target [freq_norm, amplitude, phase_norm, waveform_onehot].
        lr : float
            Learning rate.
        n_epochs : int
            Number of training epochs.

        Returns
        -------
        list[float]
            Loss history.
        """
        history = []
        n_samples = hidden_states.shape[0]

        for epoch in range(n_epochs):
            total_loss = 0.0

            # Shuffle samples
            indices = np.random.permutation(n_samples)

            for idx in indices:
                h = hidden_states[idx]
                target = target_params[idx]  # [freq_norm, amp, phase_norm, wf0, wf1, wf2, wf3]

                # Forward pass
                raw, a1 = self._forward_all(h)

                # Compute loss and gradients for each output
                # Frequency (sigmoid output vs target)
                freq_sig = 1.0 / (1.0 + np.exp(-raw[0]))
                err_freq = freq_sig - target[0]
                derr_freq = err_freq * freq_sig * (1.0 - freq_sig)

                # Amplitude (sigmoid output vs target)
                amp_sig = 1.0 / (1.0 + np.exp(-raw[1]))
                err_amp = amp_sig - target[1]
                derr_amp = err_amp * amp_sig * (1.0 - amp_sig)

                # Phase (tanh output vs target)
                phase_tanh = np.tanh(raw[2])
                err_phase = phase_tanh - target[2]
                derr_phase = err_phase * (1.0 - phase_tanh ** 2)

                # Waveform (softmax cross-entropy)
                wf_logits = raw[3:3 + self._n_waveforms]
                wf_softmax = np.exp(wf_logits - np.max(wf_logits))
                wf_softmax /= wf_softmax.sum()
                target_wf = target[3:3 + self._n_waveforms]
                err_wf = wf_softmax - target_wf  # gradient of cross-entropy

                # Accumulate output gradients
                delta = np.zeros(4 + self._n_waveforms, dtype=np.float32)
                delta[0] = derr_freq
                delta[1] = derr_amp
                delta[2] = derr_phase
                delta[3:3 + self._n_waveforms] = err_wf

                # Loss for tracking
                mse_loss = err_freq ** 2 + err_amp ** 2 + err_phase ** 2 + np.sum(err_wf ** 2)
                total_loss += mse_loss

                # Backprop through output layer (W2, b2)
                # raw = W2 @ a1 + b2
                # d(L)/d(W2) = delta @ a1.T
                dW2 = np.outer(delta, a1)
                db2 = delta

                # Backprop through hidden layer
                # a1 = tanh(z1), z1 = W1 @ h + b1
                da1 = self._W2.T @ delta
                z1 = self._W1 @ h + self._b1
                dz1 = da1 * (1.0 - np.tanh(z1) ** 2)

                dW1 = np.outer(dz1, h)
                db1 = dz1

                # Gradient clipping
                max_grad = 5.0
                for g in [dW2, db2, dW1, db1]:
                    gn = np.linalg.norm(g)
                    if gn > max_grad:
                        g *= max_grad / gn

                # Update weights
                self._W2 -= lr * dW2
                self._b2 -= lr * db2
                self._W1 -= lr * dW1
                self._b1 -= lr * db1

            avg_loss = total_loss / n_samples
            history.append(float(avg_loss))

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:3d}/{n_epochs}  loss={avg_loss:.6f}")

        return history

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
