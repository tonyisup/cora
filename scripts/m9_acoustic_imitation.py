#!/usr/bin/env python3
"""M9 Acoustic Imitation Loop — self-listening diagnostic.

Simulates the real-world loop:
1. Play a "human" stimulus sound from device A (speaker)
2. Capture it through mic → stimulus_capture
3. Producer generates an imitation
4. Play imitation from device B (headphones)
5. Capture it through mic → imitation_capture
6. Score imitation_capture against stimulus_capture

This validates that M9 can "hear" an external sound and compare its
own production against it — the foundation for acoustic imitation learning.

No weight updates in this version. Pure diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound import AcousticProducer, AcousticSimilarity, OscillatorBank

SR = 16000


def play_and_capture(
    pcm: np.ndarray,
    *,
    input_device: int,
    output_device: int,
    sr: int = SR,
    lead_in_s: float = 0.25,
    tail_s: float = 0.25,
) -> dict[str, Any]:
    """Play audio on output_device and capture from input_device."""
    import sounddevice as sd

    if pcm.ndim != 1:
        pcm = np.asarray(pcm).reshape(-1)
    pcm = pcm.astype(np.float32)

    input_info = sd.query_devices(input_device)
    input_sr = int(input_info["default_samplerate"] or sr)
    input_name = str(input_info["name"])
    output_name = str(sd.query_devices(output_device)["name"])

    playback_duration = len(pcm) / sr
    total_duration = lead_in_s + playback_duration + tail_s
    total_samples = int(total_duration * input_sr)
    captured = np.zeros(total_samples, dtype=np.float32)

    lead_in_samples = int(lead_in_s * input_sr)
    pos = 0
    playback_started_at: float | None = None
    capture_started_at = time.perf_counter()

    stream = sd.InputStream(
        device=input_device,
        channels=1,
        samplerate=input_sr,
        dtype="float32",
        blocksize=1024,
    )
    stream.start()

    try:
        while pos < total_samples:
            if playback_started_at is None and pos >= lead_in_samples:
                sd.play(pcm, samplerate=sr, device=output_device, blocking=False)
                playback_started_at = time.perf_counter()
            data, _ = stream.read(min(1024, total_samples - pos))
            n = min(len(data), total_samples - pos)
            captured[pos : pos + n] = data[:n, 0]
            pos += n
    finally:
        stream.stop()
        stream.close()
        sd.wait()

    if input_sr != sr:
        from scipy.signal import resample_poly
        captured = resample_poly(captured, sr, input_sr).astype(np.float32)

    return {
        "captured_pcm": captured.astype(np.float32),
        "input_device_name": input_name,
        "output_device_name": output_name,
        "capture_duration_s": round(total_duration, 4),
        "playback_duration_s": round(playback_duration, 4),
        "capture_elapsed_s": round(time.perf_counter() - capture_started_at, 4),
        "playback_started": playback_started_at is not None,
    }


def generate_stimulus(
    freq_hz: float,
    duration_s: float,
    waveform: str = "sine",
    sr: int = SR,
) -> np.ndarray:
    """Generate a pure-tone stimulus sound."""
    osc = OscillatorBank(sr=sr)
    fn = {
        "sine": osc.sine,
        "triangle": osc.triangle,
        "sawtooth": osc.sawtooth,
        "square": osc.square,
    }[waveform]
    return fn(freq_hz, duration_s, amplitude=0.5).astype(np.float32)


def run_imitation_loop(
    *,
    stimulus_freq_hz: float,
    stimulus_waveform: str,
    stimulus_device: int,
    imitation_device: int,
    input_device: int,
    duration_s: float,
    sr: int = SR,
) -> dict[str, Any]:
    """Run one iteration of the acoustic imitation loop."""
    similarity = AcousticSimilarity(sr=sr)

    # 1. Generate stimulus (the "human" sound)
    stimulus_pcm = generate_stimulus(stimulus_freq_hz, duration_s, stimulus_waveform, sr)

    # 2. Play stimulus on device A, capture through mic
    stimulus_result = play_and_capture(
        stimulus_pcm,
        input_device=input_device,
        output_device=stimulus_device,
        sr=sr,
    )
    stimulus_capture = stimulus_result["captured_pcm"]

    # 3. Producer generates imitation (random weights for now)
    producer = AcousticProducer(sr=sr, hidden_dim=32)
    imitation_pcm = producer.produce(duration_s)

    # 4. Play imitation on device B, capture through mic
    imitation_result = play_and_capture(
        imitation_pcm,
        input_device=input_device,
        output_device=imitation_device,
        sr=sr,
    )
    imitation_capture = imitation_result["captured_pcm"]

    # 5. Score: how well does the imitation match the stimulus?
    score = similarity.detailed_score(stimulus_capture, imitation_capture)

    # 6. Also score stimulus against itself (reference: perfect match)
    #    by comparing the stimulus capture to the original generated PCM
    self_score = similarity.detailed_score(stimulus_pcm, stimulus_capture)

    return {
        "stimulus": {
            "freq_hz": stimulus_freq_hz,
            "waveform": stimulus_waveform,
            "duration_s": duration_s,
            "device_name": stimulus_result["output_device_name"],
            "capture": {
                "duration_s": stimulus_result["capture_duration_s"],
                "elapsed_s": stimulus_result["capture_elapsed_s"],
            },
        },
        "imitation": {
            "device_name": imitation_result["output_device_name"],
            "capture": {
                "duration_s": imitation_result["capture_duration_s"],
                "elapsed_s": imitation_result["capture_elapsed_s"],
            },
            "freq_hz": float(imitation_result.get("freq_hz", 0)),
        },
        "scores": {
            "imitation_vs_stimulus": {
                "overall": float(score.overall),
                "mel": float(score.mfcc_distance),
                "pitch": float(score.pitch_distance),
                "duration": float(score.duration_distance),
                "loudness": float(score.loudness_distance),
            },
            "stimulus_self": {
                "overall": float(self_score.overall),
                "mel": float(self_score.mfcc_distance),
                "pitch": float(self_score.pitch_distance),
                "duration": float(self_score.duration_distance),
                "loudness": float(self_score.loudness_distance),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 acoustic imitation loop")
    parser.add_argument("--stimulus-freq", type=float, default=220.0)
    parser.add_argument("--stimulus-waveform", type=str, default="sine",
                        choices=["sine", "triangle", "sawtooth", "square"])
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--stimulus-device", type=int, default=3,
                        help="Output device for stimulus (human sound)")
    parser.add_argument("--imitation-device", type=int, default=2,
                        help="Output device for imitation (agent sound)")
    parser.add_argument("--input-device", type=int, default=1,
                        help="Input device (mic)")
    parser.add_argument("--output", type=str, default="/tmp/m9_acoustic_imitation.json")
    args = parser.parse_args()

    print("=" * 72)
    print("M9 Acoustic Imitation Loop — Self-Listening Diagnostic")
    print("=" * 72)
    print(f"  Stimulus: {args.stimulus_freq:.1f} Hz {args.stimulus_waveform}")
    print(f"  Duration: {args.duration:.2f}s")
    print(f"  Stimulus device: {args.stimulus_device}")
    print(f"  Imitation device: {args.imitation_device}")
    print(f"  Input device: {args.input_device}")
    print()

    result = run_imitation_loop(
        stimulus_freq_hz=args.stimulus_freq,
        stimulus_waveform=args.stimulus_waveform,
        stimulus_device=args.stimulus_device,
        imitation_device=args.imitation_device,
        input_device=args.input_device,
        duration_s=args.duration,
    )

    scores = result["scores"]
    ivs = scores["imitation_vs_stimulus"]
    self_s = scores["stimulus_self"]

    print("Stimulus self-score (capture vs original):")
    print(f"  overall={self_s['overall']:.4f} mel={self_s['mel']:.4f} "
          f"pitch={self_s['pitch']:.4f} dur={self_s['duration']:.4f} "
          f"loud={self_s['loudness']:.4f}")
    print()
    print("Imitation vs stimulus score:")
    print(f"  overall={ivs['overall']:.4f} mel={ivs['mel']:.4f} "
          f"pitch={ivs['pitch']:.4f} dur={ivs['duration']:.4f} "
          f"loud={ivs['loudness']:.4f}")
    print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved to: {args.output}")

    if ivs["overall"] >= self_s["overall"] * 0.8:
        print("  STATUS: imitation is within 80% of self-score (good)")
    else:
        print("  STATUS: imitation score is low vs self-score (expected with random weights)")


if __name__ == "__main__":
    main()
