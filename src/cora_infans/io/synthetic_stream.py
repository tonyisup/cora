"""Deterministic synthetic frame stream for offline corpus evaluation.

The offline harness deliberately does not decode real WAV files (that belongs
to the separate recording project). This helper produces reproducible
frame-level energies from a clip's labeled intervals so the scoreboard can be
exercised end-to-end without hardware or audio I/O.

For every label interval it emits one ON frame (all channels at a fixed
salient level) and one quiet frame in the unlabeled gap, deterministically.
This is a test/stub provider, not a model of real acoustics.
"""

from __future__ import annotations

from cora_infans.io.offline_eval import CorpusClip

SALIENT_LEVEL = 0.8


def build_synthetic_corpus_frames(
    clip: CorpusClip, frame_ms: int
) -> list[tuple[int, list[float]]]:
    """Build deterministic (timestamp_ms, channel_energies) frames for a clip."""
    frames: list[tuple[int, list[float]]] = []
    if not clip.annotations:
        return frames

    for ann in clip.annotations:
        start_ms = int(ann.start_s * 1000)
        # One salient frame at the interval start, energy concentrated on ch3.
        on = [0.02, 0.02, 0.02, SALIENT_LEVEL]
        frames.append((start_ms, on))
        # One quiet frame mid-interval to exercise off-periods.
        mid_ms = start_ms + max(frame_ms, 1)
        frames.append((mid_ms, [0.01, 0.01, 0.01, 0.01]))
    frames.sort(key=lambda f: f[0])
    return frames
