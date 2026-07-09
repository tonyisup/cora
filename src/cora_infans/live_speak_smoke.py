"""Bounded SPEAK smoke report helper with injectable I/O.

This module is the first implementation slice after the live SPEAK-state design
doc. It does not perform real playback by default. Callers must inject playback
and capture functions; otherwise the report is blocked after generating the
planned efference artifact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from .io.motor_plan import Vowel, synthesize_vowel, write_wav

PlaybackFn = Callable[[str], dict[str, object]]
CaptureFn = Callable[..., dict[str, object]]


def _efference_id(*, vowel: str, sample_rate: int, duration_ms: int) -> str:
    raw = f"{vowel}:{sample_rate}:{duration_ms}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _frame_tags(
    *,
    efference_id: str,
    duration_ms: int,
    max_capture_ms: int,
    latency_aligned_ms: float | None,
) -> list[dict[str, object]]:
    # Always give RECOVERING a positive-width interval so frame tagging never
    # collapses to a zero-length tag, even when capture barely exceeds the
    # speak window.
    post_end = max(duration_ms, min(max_capture_ms, duration_ms + 150))
    recovering_start = post_end
    recovering_end = max(recovering_start + 1, max_capture_ms)
    windows = [
        ("READY_TO_SPEAK", 0, 0, False),
        ("SPEAKING", 0, duration_ms, True),
        ("POST_SPEAK_SUPPRESSION", duration_ms, post_end, True),
        ("RECOVERING", recovering_start, recovering_end, False),
    ]
    tags: list[dict[str, object]] = []
    for state, start, end, expected_self in windows:
        if end < start:
            end = start
        tags.append(
            {
                "speak_state": state,
                "start_ms": start,
                "end_ms": end,
                "speaker_output_active": state == "SPEAKING",
                "efference_id": efference_id,
                "expected_self_audio": expected_self,
                "self_suppression_applied": False,
                "latency_aligned_ms": latency_aligned_ms,
                "contamination_policy": "tag_only",
                "usable_for_world_learning": not expected_self,
            }
        )
    return tags


def run_live_speak_smoke_trial(
    *,
    vowel: Vowel = "a",
    out_dir: str,
    playback_fn: PlaybackFn | None = None,
    capture_fn: CaptureFn | None = None,
    sample_rate: int = 16000,
    duration_ms: int = 300,
    max_capture_ms: int = 1500,
) -> dict[str, object]:
    """Build a bounded live-SPEAK smoke report using injected I/O only.

    With no injected functions, this intentionally returns a blocked report after
    writing the efference WAV. That gives tests and future callers the artifact
    path and report shape without touching speakers or microphones.
    """
    if duration_ms <= 0 or max_capture_ms <= 0:
        raise ValueError("duration_ms and max_capture_ms must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eid = _efference_id(vowel=vowel, sample_rate=sample_rate, duration_ms=duration_ms)
    efference_wav = out / f"efference_{eid}_{vowel}.wav"
    captured_wav = out / f"captured_{eid}_{vowel}.wav"

    pcm = synthesize_vowel(vowel, sample_rate=sample_rate, duration_ms=duration_ms)
    write_wav(str(efference_wav), pcm, sample_rate)

    base: dict[str, object] = {
        "mode": "bounded_live_speak_smoke",
        "verdict": "blocked_missing_injected_io",
        "real_hardware": False,
        "live_playback": False,
        "live_playback_allowed": False,
        "contamination_policy": "tag_only",
        "efference": {
            "id": eid,
            "vowel": vowel,
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
            "samples": len(pcm),
        },
        "artifacts": {
            "efference_wav": str(efference_wav),
            "captured_wav": None,
        },
        "frame_tags": _frame_tags(
            efference_id=eid,
            duration_ms=duration_ms,
            max_capture_ms=max_capture_ms,
            latency_aligned_ms=None,
        ),
        "playback": None,
        "capture": None,
    }
    if playback_fn is None or capture_fn is None:
        return base

    playback = playback_fn(str(efference_wav))
    capture = capture_fn(str(captured_wav), sample_rate=sample_rate, duration_ms=max_capture_ms)
    base["verdict"] = "fake_io_smoke_complete"
    base["artifacts"] = {
        "efference_wav": str(efference_wav),
        "captured_wav": str(captured_wav),
    }
    base["playback"] = playback
    base["capture"] = capture
    return base
