"""M9 — Acoustic producer.

Combines oscillator bank + substrate + controller to produce sound.
The producer is the "output" of the motor-learning loop.

M9a: fixed random controller weights (baseline — no learning)
M9b: controller weights updated via REINFORCE to maximize similarity

Usage:
    producer = AcousticProducer(sr=16000)
    target = producer.oscillator.sine(440, 1.0)
    produced = producer.produce(target_duration_s=1.0)
    score = producer.similarity.overall_score(target, produced)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from m9_sound.oscillator import OscillatorBank
from m9_sound.substrate import AcousticSubstrate
from m9_sound.controller import NeuralController, ControllerParams
from m9_sound.similarity import AcousticSimilarity, SimilarityResult


@dataclass
class ProductionResult:
    """Result of a sound production attempt."""
    audio: np.ndarray
    target: np.ndarray
    similarity: SimilarityResult
    params: ControllerParams
    n_frames: int


class AcousticProducer:
    """Produces sound by combining oscillator + substrate + controller.

    The production loop:
    1. Controller reads substrate hidden state -> oscillator params
    2. Oscillator generates audio frame
    3. Substrate updates hidden state with new audio
    4. Repeat for duration
    """

    def __init__(self, sr: int = 16000, hidden_dim: int = 32,
                 frame_size: int = 480) -> None:
        self._sr = sr
        self._frame_size = frame_size
        self.oscillator = OscillatorBank(sr=sr)
        self.substrate = AcousticSubstrate(hidden_dim=hidden_dim, sr=sr,
                                           frame_size=frame_size)
        self.controller = NeuralController(hidden_dim=hidden_dim, sr=sr)
        self.similarity = AcousticSimilarity(sr=sr)

    def reset(self) -> None:
        """Reset substrate hidden state."""
        self.substrate.reset()

    def save_checkpoint(self, path: str) -> None:
        """Save controller and substrate weights to a JSON checkpoint."""
        import json as _json
        data = {
            "controller": {
                "W1": self.controller._W1.tolist(),
                "b1": self.controller._b1.tolist(),
                "W2": self.controller._W2.tolist(),
                "b2": self.controller._b2.tolist(),
            },
            "substrate": {
                "W_h": self.substrate._W_h.tolist(),
                "W_x": self.substrate._W_x.tolist(),
                "b": self.substrate._b.tolist(),
            },
            "config": {
                "sr": self._sr,
                "hidden_dim": self.controller._hidden_dim,
                "frame_size": self._frame_size,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(_json.dumps(data))

    def load_checkpoint(self, path: str) -> None:
        """Load controller and substrate weights from a JSON checkpoint."""
        import json as _json
        data = _json.loads(Path(path).read_text())
        # Substrate weights (same format)
        self.substrate._W_h = np.array(data["substrate"]["W_h"], dtype=np.float32)
        self.substrate._W_x = np.array(data["substrate"]["W_x"], dtype=np.float32)
        self.substrate._b = np.array(data["substrate"]["b"], dtype=np.float32)
        # Controller weights: support both old (W, b) and new (W1, b1, W2, b2) format
        c = data["controller"]
        if "W1" in c:
            # New two-layer format
            self.controller._W1 = np.array(c["W1"], dtype=np.float32)
            self.controller._b1 = np.array(c["b1"], dtype=np.float32)
            self.controller._W2 = np.array(c["W2"], dtype=np.float32)
            self.controller._b2 = np.array(c["b2"], dtype=np.float32)
        else:
            # Old single-layer format — convert to two-layer
            old_W = np.array(c["W"], dtype=np.float32)
            old_b = np.array(c["b"], dtype=np.float32)
            # Initialize new layers with old weights mapped to output layer
            self.controller._W2 = np.zeros((4 + self.controller._n_waveforms, self.controller._controller_hidden), dtype=np.float32)
            self.controller._W2[:, :old_W.shape[1]] = old_W[:self.controller._W2.shape[0], :]
            self.controller._b2[:old_b.shape[0]] = old_b

    def produce(self, duration_s: float,
                 init_params: Optional[ControllerParams] = None) -> np.ndarray:
        """Produce sound for a given duration.

        Parameters
        ----------
        duration_s : float
            How long to produce sound.
        init_params : ControllerParams, optional
            Initial oscillator parameters. If None, uses controller's
            default decode from zero hidden state.

        Returns
        -------
        np.ndarray
            Produced audio.
        """
        self.reset()
        n_frames = int(np.ceil(self._sr * duration_s / self._frame_size))
        total_samples = int(round(self._sr * duration_s))
        output = np.zeros(n_frames * self._frame_size, dtype=np.float32)

        # Initial params from controller
        if init_params is None:
            init_params = self.controller.decode(self.substrate.hidden_state)

        waveform_map = ["sine", "triangle", "sawtooth", "square"]

        for i in range(n_frames):
            # Use the provided initial params on the first frame, then read
            # the controller on subsequent frames.
            params = init_params if i == 0 and init_params is not None else self.controller.decode(self.substrate.hidden_state)

            # Generate one frame of audio
            t_frame = np.arange(self._frame_size, dtype=np.float32) / self._sr
            freq = params.freq_hz
            amp = params.amplitude
            phase = params.phase
            wf = waveform_map[params.waveform]

            if wf == "sine":
                frame = amp * np.sin(2 * np.pi * freq * t_frame + phase)
            elif wf == "triangle":
                ramp = 2.0 * ((freq * t_frame + phase / (2 * np.pi)) % 1.0) - 1.0
                frame = amp * (2.0 * np.abs(ramp) - 1.0)
            elif wf == "sawtooth":
                frame = amp * (2.0 * ((freq * t_frame + phase / (2 * np.pi)) % 1.0) - 1.0)
            else:  # square
                frame = amp * np.sign(np.sin(2 * np.pi * freq * t_frame + phase))

            frame = frame.astype(np.float32)
            output[i * self._frame_size:(i + 1) * self._frame_size] = frame

            # Update substrate with produced frame
            self.substrate.step(frame)

        return output[:total_samples]


    def produce_and_score(self, target: np.ndarray) -> ProductionResult:
        """Produce sound and score it against a target.

        Parameters
        ----------
        target : np.ndarray
            Target audio to imitate.

        Returns
        -------
        ProductionResult
            Produced audio, similarity score, and parameters used.
        """
        duration_s = len(target) / self._sr
        produced = self.produce(duration_s)
        sim = self.similarity.detailed_score(target, produced)
        params = self.controller.decode(self.substrate.hidden_state)
        return ProductionResult(
            audio=produced,
            target=target,
            similarity=sim,
            params=params,
            n_frames=int(self._sr * duration_s / self._frame_size),
        )
