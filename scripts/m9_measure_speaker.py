#!/usr/bin/env python3
"""M9 Speaker Response Measurement.

Plays a known test signal through the Mac mini speaker (device 3),
captures it with the mic (device 1), and computes the frequency response.

This lets us measure the actual speaker→mic transfer function
and use it for substrate retraining.

Usage:
    python scripts/m9_measure_speaker.py
    python scripts/m9_measure_speaker.py --playback-device 3 --input-device 1
"""

import sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.signal import chirp, butter, sosfilt, welch, coherence
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.temporal import SR, FRAME_SIZE

try:
    import sounddevice as sd
except ImportError:
    sd = None


def generate_test_signal(duration=2.0, sr=SR):
    """Generate a test signal for speaker response measurement.

    Uses a logarithmic chirp (20Hz-8kHz) that covers the full range
    of interest for vowel formants.
    """
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    # Logarithmic chirp from 20Hz to 8kHz
    f0 = 20.0
    f1 = 8000.0
    # Instantaneous frequency: f(t) = f0 * (f1/f0)**(t/duration)
    phase = 2 * np.pi * f0 * duration / np.log(f1/f0) * ((f1/f0)**(t/duration) - 1)
    signal = 0.5 * np.sin(phase).astype(np.float32)

    # Add fade in/out to avoid clicks
    fade_samples = int(0.05 * sr)
    signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
    signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    return signal


def measure_response(playback_device, input_device, duration=2.0, sr=SR):
    """Play test signal and capture the response."""
    if sd is None:
        raise RuntimeError("sounddevice is required for speaker measurement")

    signal = generate_test_signal(duration=duration, sr=sr)

    print(f"Playing {duration}s chirp (20Hz-8kHz) through device {playback_device}...")
    print(f"Capturing from device {input_device}...")

    # Play and record simultaneously
    try:
        captured = sd.playrec(
            signal,
            samplerate=sr,
            channels=1,
            dtype="float32",
            device=(input_device, playback_device),
            blocking=True,
        )
    except TypeError:
        sd.default.device = (input_device, playback_device)
        captured = sd.playrec(
            signal,
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocking=True,
        )

    captured = np.asarray(captured, dtype=np.float32).reshape(-1)
    print(f"Captured {len(captured)} samples ({len(captured)/sr:.3f}s)")

    return signal, captured


def compute_frequency_response(signal, captured, sr=SR):
    """Compute the frequency response using Welch's method."""
    # Use coherence-based transfer function estimation
    nperseg = 1024

    # Compute power spectral density of input and output
    f_in, psd_in = welch(signal, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
    f_out, psd_out = welch(captured, fs=sr, nperseg=nperseg, noverlap=nperseg//2)

    # Compute transfer function
    f_coh, coh = coherence(signal, captured, fs=sr, nperseg=nperseg, noverlap=nperseg//2)

    # Transfer function estimate: H(f) = PSD_out / PSD_in
    # Only use frequencies where coherence is high
    h_psd = psd_out / np.maximum(psd_in, 1e-10)
    h_db = 10 * np.log10(np.maximum(h_psd, 1e-10))

    # Smooth the response
    from scipy.ndimage import uniform_filter1d
    h_db_smooth = uniform_filter1d(h_db, size=5)

    return f_in, h_db_smooth, coh


def save_results(freqs, response_db, coherence, output_path):
    """Save the measured frequency response."""
    results = {
        "frequencies": freqs.tolist(),
        "response_db": response_db.tolist(),
        "coherence": coherence.tolist(),
        "sample_rate": SR,
    }
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Measure speaker→mic frequency response")
    parser.add_argument("--playback-device", type=int, default=3, help="Output device (Mac mini speakers)")
    parser.add_argument("--input-device", type=int, default=1, help="Input device (mic)")
    parser.add_argument("--duration", type=float, default=2.0, help="Test signal duration (s)")
    parser.add_argument("--output", type=str, default="data/m9_speaker_response.json")
    args = parser.parse_args()

    # Measure
    signal, captured = measure_response(args.playback_device, args.input_device, args.duration)

    # Compute frequency response
    freqs, response_db, coh = compute_frequency_response(signal, captured)

    # Print summary
    print("\n=== Speaker Response Summary ===")
    for f in [100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
        idx = np.argmin(np.abs(freqs - f))
        print(f"  {f:5d} Hz: {response_db[idx]:+.1f} dB (coherence: {coh[idx]:.3f})")

    # Save
    save_results(freqs, response_db, coh, args.output)

    # Also save the raw captured audio for inspection
    from m9_sound.temporal import write_pcm16_wav
    wav_meta = write_pcm16_wav("data/m9_speaker_capture.wav", captured, sr=SR)
    print(f"Captured audio: {wav_meta['path']}")

    # Save the original signal for comparison
    wav_meta = write_pcm16_wav("data/m9_speaker_chirp.wav", signal, sr=SR)
    print(f"Original chirp: {wav_meta['path']}")


if __name__ == "__main__":
    main()
