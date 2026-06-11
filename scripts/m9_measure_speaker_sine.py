#!/usr/bin/env python3
"""M9 Speaker Response Measurement — sine tone method.

Plays individual sine tones at key frequencies through the speaker,
captures each with the mic, and measures the relative amplitude.

This gives a cleaner frequency response than a chirp.

Usage:
    python scripts/m9_measure_speaker_sine.py
"""

import sys, json, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.temporal import SR

try:
    import sounddevice as sd
except ImportError:
    sd = None


def play_and_capture(frequency, duration=0.5, sr=SR, playback_device=3, input_device=1):
    """Play a sine tone and capture the response."""
    if sd is None:
        raise RuntimeError("sounddevice is required")

    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    # Use 80% of full scale to avoid clipping
    signal = 0.8 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    # Fade in/out
    fade = int(0.02 * sr)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)

    try:
        captured = sd.playrec(
            signal, samplerate=sr, channels=1, dtype="float32",
            device=(input_device, playback_device), blocking=True,
        )
    except TypeError:
        sd.default.device = (input_device, playback_device)
        captured = sd.playrec(
            signal, samplerate=sr, channels=1, dtype="float32", blocking=True,
        )

    captured = np.asarray(captured, dtype=np.float32).reshape(-1)
    return signal, captured


def measure_tone_response(frequency, signal, captured, sr=SR):
    """Measure the response at a specific frequency using DFT."""
    # Skip the first 50ms (fade-in + delay)
    skip = int(0.05 * sr)
    segment = captured[skip:]

    if len(segment) < 256:
        return 0.0

    # DFT at the target frequency
    t = np.arange(len(segment), dtype=np.float32) / sr
    cos_comp = np.sum(segment * np.cos(2 * np.pi * frequency * t)) * 2 / len(segment)
    sin_comp = np.sum(segment * np.sin(2 * np.pi * frequency * t)) * 2 / len(segment)
    amplitude = np.sqrt(cos_comp**2 + sin_comp**2)

    return float(amplitude)


def main():
    parser = argparse.ArgumentParser(description="Measure speaker response with sine tones")
    parser.add_argument("--playback-device", type=int, default=3)
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="data/m9_speaker_response.json")
    args = parser.parse_args()

    # Frequencies to test — cover the full vowel formant range
    frequencies = [
        100, 150, 200, 250, 300, 400, 500, 600, 700, 800,
        900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
        1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800,
        2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800,
        3900, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000,
    ]

    results = {}

    print(f"Measuring speaker response: device {args.playback_device} -> device {args.input_device}")
    print(f"Tone duration: {args.duration}s, {len(frequencies)} frequencies\n")

    for freq in frequencies:
        signal, captured = play_and_capture(
            freq, duration=args.duration, sr=SR,
            playback_device=args.playback_device,
            input_device=args.input_device,
        )
        response = measure_tone_response(freq, signal, captured, sr=SR)
        results[freq] = response
        print(f"  {freq:5d} Hz: response = {response:.6f}")

    # Normalize to 0 dB at 1kHz (reference)
    ref = results.get(1000, 1.0)
    if ref < 1e-6:
        ref = max(results.values())

    response_db = {}
    for freq, amp in results.items():
        if amp > 1e-8:
            response_db[freq] = 20 * np.log10(amp / ref)
        else:
            response_db[freq] = -100.0

    print("\n=== Normalized Response (dB re: 1kHz) ===")
    for freq in sorted(response_db.keys()):
        bar = '#' * max(0, int((response_db[freq] + 40) / 2))
        print(f"  {freq:5d} Hz: {response_db[freq]:+6.1f} dB {bar}")

    # Save
    output = {
        "frequencies": sorted(results.keys()),
        "response_raw": {str(k): v for k, v in results.items()},
        "response_db": {str(k): v for k, v in response_db.items()},
        "reference_hz": 1000,
        "sample_rate": SR,
        "playback_device": args.playback_device,
        "input_device": args.input_device,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
