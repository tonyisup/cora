#!/usr/bin/env python3
"""M10 live consonant imitation demo.

Wait for the user to speak, detect the consonant/vowel structure, synthesize
an imitation that includes consonant-like sounds, and play it back.

Use --dry-run to exercise the full loop without hardware.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound.vowel_transitions import (
    VOWEL_FAMILIES,
    build_vowel_transition_report,
    decode_vowel_transition_audio,
    _synthesize_consonant_sequence,
)
from m9_sound.temporal import FRAME_SIZE, SR, write_pcm16_wav

# Class names for M10
VOWEL_NAMES = ['i', 'E', 'a', 'O', 'u']
CONSONANT_NAMES = ['p', 't', 'k', 's', 'f', 'sh', 'm', 'n']
ALL_NAMES = VOWEL_NAMES + CONSONANT_NAMES

N_MELS = 32

HP_SOS = butter(4, 100, btype="high", fs=SR, output="sos")

# Default consonant-vowel imitation families
CONSONANT_FAMILIES: dict[str, list[str]] = {
    "cv_simple": ["p", "a", "t", "a"],
    "cv_fricative": ["s", "a", "f", "a"],
    "cv_nasal": ["m", "a", "n", "a"],
    "cvc": ["p", "a", "t", "a", "k"],
    "mixed": ["s", "a", "m", "a", "t", "a"],
}

FAMILY_DURATIONS_S: dict[str, float] = {
    "cv_simple": 0.8,
    "cv_fricative": 1.0,
    "cv_nasal": 0.9,
    "cvc": 1.0,
    "mixed": 1.2,
}


def _dry_run_capture_audio(playback_audio: np.ndarray, *, sr: int, **_: object) -> np.ndarray:
    tail = np.zeros(sr // 20, dtype=np.float32)
    return np.concatenate([np.asarray(playback_audio, dtype=np.float32).reshape(-1), tail])


def _detect_speech(audio: np.ndarray, *, threshold: float, min_speech_frames: int) -> tuple[bool, float, int]:
    filtered = np.asarray(sosfilt(HP_SOS, np.asarray(audio, dtype=np.float32).reshape(-1)), dtype=np.float32)
    frame_count = max(1, len(filtered) // FRAME_SIZE)
    speech_frames = 0
    for idx in range(frame_count):
        frame = filtered[idx * FRAME_SIZE : (idx + 1) * FRAME_SIZE]
        if len(frame) < FRAME_SIZE:
            continue
        frame_rms = float(np.sqrt(np.mean(frame**2)))
        if frame_rms >= threshold:
            speech_frames += 1
    rms = float(np.sqrt(np.mean(filtered**2))) if len(filtered) else 0.0
    return speech_frames >= min_speech_frames, rms, speech_frames


def _listen_window_audio(
    *,
    input_device: int,
    listen_window_s: float,
    dry_run: bool,
    listen_family: str,
) -> np.ndarray:
    if dry_run:
        listen_report = build_vowel_transition_report(
            family=listen_family,
            heldout_family=listen_family,
            noise_rms=0.0,
            noise_seed=7,
            device_profile="highpass",
            device_gain=1.0,
            sr=SR,
            frame_size=FRAME_SIZE,
        )
        listen_report = dict(listen_report)
        return np.asarray(cast(list[float], listen_report["rendered_audio"]), dtype=np.float32)

    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError("sounddevice is required for live capture") from exc

    sample_count = max(1, int(round(SR * listen_window_s)))
    recorded = sd.rec(sample_count, samplerate=SR, channels=1, device=input_device, dtype="float32")
    sd.wait()
    return np.asarray(recorded, dtype=np.float32).reshape(-1)


def _synthesize_imitation(
    family: str,
    *,
    duration_scale: float = 1.0,
    noise_seed: int = 7,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Synthesize a consonant-vowel imitation."""
    tokens = CONSONANT_FAMILIES.get(family, CONSONANT_FAMILIES["cv_simple"])
    duration_s = FAMILY_DURATIONS_S.get(family, 0.8) * duration_scale
    return _synthesize_consonant_sequence(
        tokens,
        duration_s=duration_s,
        sr=SR,
        frame_size=FRAME_SIZE,
        noise_seed=noise_seed,
    )


def _load_m10_model(model_path: Path | None = None):
    """Load the M10 trained model if available."""
    if model_path is None:
        model_path = Path("data/m10_v1.json")
    if not model_path.exists():
        return None
    params = json.loads(model_path.read_text())
    return {
        "W1": np.array(params["W1"], dtype=np.float32),
        "b1": np.array(params["b1"], dtype=np.float32),
        "W2": np.array(params["W2"], dtype=np.float32),
        "b2": np.array(params["b2"], dtype=np.float32),
    }


def _decode_and_report(
    audio: np.ndarray,
    *,
    consonant_gate: float,
    silence_threshold: float,
    m10_model: dict | None = None,
) -> dict[str, object]:
    """Decode audio with consonant awareness, using M10 model if available."""
    from m9_sound.temporal import FRAME_SIZE

    # Extract mel features
    audio_filt = np.asarray(sosfilt(HP_SOS, np.asarray(audio, dtype=np.float32).reshape(-1)), dtype=np.float32)
    n_frames = len(audio_filt) // FRAME_SIZE

    if m10_model is not None:
        # Use M10 trained model
        W1 = m10_model["W1"]
        b1 = m10_model["b1"]
        W2 = m10_model["W2"]
        b2 = m10_model["b2"]

        labels = []
        for i in range(n_frames):
            frame = audio_filt[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
            if len(frame) < FRAME_SIZE:
                frame = np.zeros(FRAME_SIZE, dtype=np.float32)
                frame[:len(audio_filt[i * FRAME_SIZE:])] = audio_filt[i * FRAME_SIZE:]
            rms = float(np.sqrt(np.mean(frame**2)))
            if rms < silence_threshold:
                labels.append("_")
            else:
                mel = _mel_feat_for_frame(frame)
                h = np.maximum(mel @ W1 + b1, 0)
                logits = h @ W2 + b2
                pred_idx = int(np.argmax(logits))
                labels.append(ALL_NAMES[pred_idx] if pred_idx < len(ALL_NAMES) else "_")
    else:
        # Fallback: use vowel centroid decoder
        decode = decode_vowel_transition_audio(
            audio,
            consonant_gate=consonant_gate,
            silence_threshold=silence_threshold,
        )
        labels = list(decode["decoded_vowels"])

    consonant_frames = [i for i, l in enumerate(labels) if l in CONSONANT_NAMES]
    vowel_frames = [i for i, l in enumerate(labels) if l in VOWEL_NAMES]
    silence_frames = [i for i, l in enumerate(labels) if l == "_"]
    total = max(len(labels), 1)

    return {
        "labels": labels,
        "consonant_count": len(consonant_frames),
        "consonant_rate": len(consonant_frames) / total,
        "vowel_count": len(vowel_frames),
        "vowel_rate": len(vowel_frames) / total,
        "silence_count": len(silence_frames),
        "silence_rate": len(silence_frames) / total,
        "consonant_frames": consonant_frames[:20],
        "transition_count": sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1]),
    }


def _mel_feat_for_frame(frame: np.ndarray, n_mels: int = N_MELS, n_fft: int = 512) -> np.ndarray:
    """Extract mel features from a single frame."""
    x = np.zeros(n_fft, dtype=np.float32)
    n = min(len(frame), n_fft)
    x[:n] = frame[:n]
    spectrum = np.abs(np.fft.rfft(x))
    mel_fb = _get_mel_fb_demo()
    mel_spec = mel_fb @ spectrum
    mel_spec = np.log(mel_spec + 1e-6).astype(np.float32)
    rms = float(np.sqrt(np.mean(mel_spec**2)))
    if rms > 1e-6:
        mel_spec /= rms
    return mel_spec


def _mel_filterbank_demo(n_mels, n_fft, sr):
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(80), hz_to_mel(7600), n_mels + 2)
    bin_points = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        for k in range(left, center):
            if k < n_bins and center > left:
                fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if k < n_bins and right > center:
                fb[m, k] = (right - k) / (right - center)
    return fb


_MEL_FB_DEMO = None

def _get_mel_fb_demo():
    global _MEL_FB_DEMO
    if _MEL_FB_DEMO is None:
        _MEL_FB_DEMO = _mel_filterbank_demo(N_MELS, 512, SR)
    return _MEL_FB_DEMO


def _parse_family_list(text: str) -> list[str]:
    families = [item.strip() for item in text.split(",") if item.strip()]
    if not families:
        raise ValueError("At least one consonant family is required")
    for family in families:
        if family not in CONSONANT_FAMILIES:
            raise ValueError(f"Unknown consonant family: {family}. Choose from: {', '.join(sorted(CONSONANT_FAMILIES.keys()))}")
    return families


def main() -> None:
    parser = argparse.ArgumentParser(description="M10 live consonant imitation demo")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m10-live-consonant-imitate"))
    parser.add_argument("--families", type=str, default="cv_simple,cv_fricative,cv_nasal", help="Comma-separated consonant families to imitate in rotation")
    parser.add_argument("--listen-family", type=str, default="phase_triad", choices=list(VOWEL_FAMILIES.keys()))
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=3)
    parser.add_argument("--listen-window-s", type=float, default=0.6)
    parser.add_argument("--trigger-threshold", type=float, default=0.003)
    parser.add_argument("--min-speech-frames", type=int, default=2)
    parser.add_argument("--consonant-gate", type=float, default=2.0)
    parser.add_argument("--silence-threshold", type=float, default=0.001)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--pause-s", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--play", action="store_true", default=True, help="Play the imitation audio")
    parser.add_argument("--no-play", action="store_true", help="Skip playing the imitation audio")
    args = parser.parse_args()

    if args.once and args.iterations == 0:
        args.iterations = 1

    families = _parse_family_list(args.families)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    # Load M10 model if available
    m10_model = _load_m10_model()
    if m10_model is not None:
        print("  Model:             m10_v1 (13 classes)")
    else:
        print("  Model:             vowel-centroid fallback (5 classes)")

    print("=" * 78)
    print("M10 Live Consonant Imitation Demo")
    print("=" * 78)
    print("Speak into the microphone. CORA will listen and imitate with consonants.")
    print(f"  Families:          {', '.join(families)}")
    print(f"  Listen family:     {args.listen_family}")
    print(f"  Input device:      {args.input_device}")
    print(f"  Output device:     {args.output_device}")
    print(f"  Consonant gate:    {args.consonant_gate}")
    print(f"  Duration scale:    {args.duration_scale}")
    print(f"  Output dir:        {output_root}")
    print(f"  Iterations:        {'forever' if args.iterations <= 0 else args.iterations}")
    print("  Ctrl+C to stop.\n")

    trigger_count = 0
    passed = 0
    family_index = 0
    try:
        while args.iterations <= 0 or trigger_count < args.iterations:
            listen_audio = _listen_window_audio(
                input_device=args.input_device,
                listen_window_s=args.listen_window_s,
                dry_run=args.dry_run,
                listen_family=args.listen_family,
            )
            heard, listen_rms, speech_frames = _detect_speech(
                listen_audio,
                threshold=args.trigger_threshold,
                min_speech_frames=args.min_speech_frames,
            )
            if not heard:
                print(f"[listen] no trigger (rms={listen_rms:.4f}, speech_frames={speech_frames})")
                if args.pause_s > 0:
                    time.sleep(args.pause_s)
                continue

            trigger_count += 1
            family = families[family_index % len(families)]
            family_index += 1
            print(f"[trigger {trigger_count:04d}] heard speech (rms={listen_rms:.4f}, frames={speech_frames}) -> imitate {family}")

            # Decode what was heard
            heard_report = _decode_and_report(
                listen_audio,
                consonant_gate=args.consonant_gate,
                silence_threshold=args.silence_threshold,
                m10_model=m10_model,
            )
            print(f"  heard: consonant_rate={heard_report['consonant_rate']:.1%} vowel_rate={heard_report['vowel_rate']:.1%}")

            # Synthesize imitation
            imitation_audio, imitation_segs = _synthesize_imitation(
                family,
                duration_scale=args.duration_scale,
                noise_seed=7 + trigger_count,
            )

            # Save artifacts
            iteration_dir = output_root / f"run-{trigger_count:04d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            heard_wav = iteration_dir / "heard.wav"
            imitation_wav = iteration_dir / "imitation.wav"
            write_pcm16_wav(heard_wav, listen_audio, sr=SR)
            write_pcm16_wav(imitation_wav, imitation_audio, sr=SR)

            # Decode the imitation
            imitation_report = _decode_and_report(
                imitation_audio,
                consonant_gate=args.consonant_gate,
                silence_threshold=args.silence_threshold,
            )

            report = {
                "trigger_idx": trigger_count,
                "family": family,
                "heard": heard_report,
                "imitation": imitation_report,
                "imitation_segments": imitation_segs,
                "heard_wav": str(heard_wav),
                "imitation_wav": str(imitation_wav),
            }
            report_path = iteration_dir / "report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            print(f"  imitation: consonant_rate={imitation_report['consonant_rate']:.1%} vowel_rate={imitation_report['vowel_rate']:.1%}")
            print(f"  segments: {len(imitation_segs)}")
            print(f"  saved: {imitation_wav}")

            # Play the imitation
            if args.play and not args.no_play:
                print(f"  Playing imitation...")
                if args.dry_run:
                    subprocess.run(["afplay", str(imitation_wav)], check=True)
                else:
                    try:
                        import sounddevice as sd
                        sd.play(imitation_audio, samplerate=SR, device=args.output_device)
                        sd.wait()
                    except Exception:
                        subprocess.run(["afplay", str(imitation_wav)], check=True)

            if args.pause_s > 0:
                time.sleep(args.pause_s)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print(f"\nLoop summary: {trigger_count} triggers processed")
        latest = output_root / "report.json"
        if latest.exists():
            print(f"Latest report: {latest}")


if __name__ == "__main__":
    main()
