"""Live capture helpers for the M9 temporal smoke path.

This module bridges the hardware-free temporal smoke harness with a real mic/
speaker capture loop. It keeps the render path reusable while allowing tests to
inject a fake capture function.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from m9_sound.temporal import (
    DEFAULT_CONTROLLER_CKPT,
    DEFAULT_SUBSTRATE_CKPT,
    SR,
    build_temporal_smoke_artifact,
    write_pcm16_wav,
)

CaptureFn = Callable[..., np.ndarray]


def _capture_audio_with_sounddevice(
    playback_audio: np.ndarray,
    *,
    sr: int = SR,
    input_device: int | None = None,
    output_device: int | None = None,
    lead_in_s: float = 0.25,
    tail_s: float = 0.25,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """Play playback_audio through the speaker while recording the mic input."""
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - exercised only on real hardware
        raise RuntimeError(
            "sounddevice is required for live capture; install the optional audio deps"
        ) from exc

    playback_audio = np.asarray(playback_audio, dtype=np.float32).reshape(-1)
    lead_in_samples = max(0, int(round(sr * lead_in_s)))
    tail_samples = max(0, int(round(sr * tail_s)))
    play_buffer = np.concatenate(
        [np.zeros(lead_in_samples, dtype=np.float32), playback_audio, np.zeros(tail_samples, dtype=np.float32)]
    )
    capture_samples = len(play_buffer)
    captured = np.zeros(capture_samples, dtype=np.float32)

    play_pos = [0]
    capture_pos = [0]

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"  Stream status: {status}")
        pp = play_pos[0]
        ep = min(pp + frames, len(play_buffer))
        outdata[:, 0] = 0.0
        if pp < len(play_buffer):
            n = ep - pp
            outdata[:n, 0] = play_buffer[pp:ep]
            play_pos[0] = ep
        cp = capture_pos[0]
        ce = min(cp + frames, len(captured))
        if cp < len(captured):
            n = ce - cp
            captured[cp:ce] = indata[:n, 0]
            capture_pos[0] = ce

    blocksize = max(1, int(sr * 30 / 1000))
    with sd.Stream(
        samplerate=sr,
        blocksize=blocksize,
        dtype="float32",
        channels=1,
        callback=callback,
        device=(input_device, output_device),
    ):
        sleep_fn(len(play_buffer) / sr + 0.5)

    return captured


def run_live_temporal_capture_report(
    output_dir: str | Path,
    *,
    family: str = "wide_sweep",
    tones: list[tuple[float, float]] | None = None,
    controller_ckpt_path: str | Path | None = None,
    substrate_ckpt_path: str | Path | None = None,
    sr: int = SR,
    frame_size: int = 480,
    input_noise_rms: float = 0.0,
    noise_seed: int | None = None,
    device_profile: str = "flat",
    device_gain: float = 1.0,
    input_device: int | None = None,
    output_device: int | None = None,
    lead_in_s: float = 0.25,
    tail_s: float = 0.25,
    capture_audio_fn: CaptureFn | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Render the temporal smoke tone and capture it through real or fake hardware."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_audio, smoke_report = build_temporal_smoke_artifact(
        family=family,
        tones=tones,
        controller_ckpt_path=controller_ckpt_path or DEFAULT_CONTROLLER_CKPT,
        substrate_ckpt_path=substrate_ckpt_path or DEFAULT_SUBSTRATE_CKPT,
        sr=sr,
        frame_size=frame_size,
        input_noise_rms=input_noise_rms,
        noise_seed=noise_seed,
        device_profile=device_profile,
        device_gain=device_gain,
    )

    rendered_wav = output_dir / "rendered_temporal.wav"
    rendered_meta = write_pcm16_wav(rendered_wav, rendered_audio, sr=sr)

    if capture_audio_fn is None:
        capture_audio_fn = _capture_audio_with_sounddevice

    live_capture = capture_audio_fn(
        rendered_audio,
        sr=sr,
        input_device=input_device,
        output_device=output_device,
        lead_in_s=lead_in_s,
        tail_s=tail_s,
        sleep_fn=sleep_fn,
    )
    live_capture = np.asarray(live_capture, dtype=np.float32).reshape(-1)

    captured_wav = output_dir / "captured_temporal.wav"
    captured_meta = write_pcm16_wav(captured_wav, live_capture, sr=sr)

    report: dict[str, object] = dict(smoke_report)
    report.update(
        {
            "output_dir": str(output_dir),
            "rendered_wav": str(rendered_meta["path"]),
            "captured_wav": str(captured_meta["path"]),
            "report_json": str(output_dir / "report.json"),
            "live_capture_samples": int(len(live_capture)),
            "live_capture_rms": float(np.sqrt(np.mean(live_capture**2))) if len(live_capture) else 0.0,
            "lead_in_s": float(lead_in_s),
            "tail_s": float(tail_s),
            "input_device": input_device,
            "output_device": output_device,
        }
    )

    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
