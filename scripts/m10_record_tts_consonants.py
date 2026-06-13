#!/usr/bin/env python3
"""M10 TTS consonant capture — 48kHz version.

Uses macOS `say` to generate consonant-vowel prompts through the speaker,
captures them via the mic at 48kHz, and saves them as M10 training data.

The higher sample rate preserves high-frequency content (sibilants, stops)
that is lost at 16kHz.

Usage:
    ./.venv/bin/python scripts/m10_record_tts_consonants.py --rate 48000
    scripts/m10_record_tts_consonants.py --rate 48000 --voices Samantha Alex
"""

import sys, json, time, uuid, subprocess, tempfile, argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.temporal import SR as SR_16K, write_pcm16_wav

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice required: pip install sounddevice")
    sys.exit(1)

# Consonant prompt definitions
PROMPTS = [
    # Single CV syllables
    {"text": "pa", "label": "p", "category": "stop"},
    {"text": "pi", "label": "p", "category": "stop"},
    {"text": "pu", "label": "p", "category": "stop"},
    {"text": "ta", "label": "t", "category": "stop"},
    {"text": "ti", "label": "t", "category": "stop"},
    {"text": "tu", "label": "t", "category": "stop"},
    {"text": "ka", "label": "k", "category": "stop"},
    {"text": "ki", "label": "k", "category": "stop"},
    {"text": "ku", "label": "k", "category": "stop"},
    {"text": "sa", "label": "s", "category": "fricative"},
    {"text": "si", "label": "s", "category": "fricative"},
    {"text": "su", "label": "s", "category": "fricative"},
    {"text": "fa", "label": "f", "category": "fricative"},
    {"text": "fi", "label": "f", "category": "fricative"},
    {"text": "fu", "label": "f", "category": "fricative"},
    {"text": "sha", "label": "sh", "category": "fricative"},
    {"text": "shi", "label": "sh", "category": "fricative"},
    {"text": "shu", "label": "sh", "category": "fricative"},
    {"text": "ma", "label": "m", "category": "nasal"},
    {"text": "mi", "label": "m", "category": "nasal"},
    {"text": "mu", "label": "m", "category": "nasal"},
    {"text": "na", "label": "n", "category": "nasal"},
    {"text": "ni", "label": "n", "category": "nasal"},
    {"text": "nu", "label": "n", "category": "nasal"},
    # Multi-syllable
    {"text": "pa ta ka", "label": "p", "category": "stop"},
    {"text": "sa fa sha", "label": "s", "category": "fricative"},
    {"text": "ma na pa", "label": "m", "category": "nasal"},
    {"text": "ta ka sa", "label": "t", "category": "stop"},
    {"text": "fa sha ma", "label": "f", "category": "fricative"},
    {"text": "na ma ta", "label": "n", "category": "nasal"},
    {"text": "ka pa na", "label": "k", "category": "stop"},
    {"text": "sha fa sa", "label": "sh", "category": "fricative"},
]

VOICES_48K = ["Samantha", "Alex", "Victoria"]  # Voices that work well at 48kHz


def tts_to_wav_48k(text: str, voice: str, out_wav: Path) -> bool:
    """Generate TTS at 48kHz using macOS `say`."""
    with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as f:
        aiff_path = f.name
    try:
        subprocess.run(["say", "-v", voice, "-o", aiff_path, text], check=True, timeout=15)
        # Convert to 48kHz WAV
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@48000", aiff_path, str(out_wav)],
            check=True, timeout=10
        )
        return True
    except Exception as e:
        print(f"  TTS error: {e}")
        return False
    finally:
        Path(aiff_path).unlink(missing_ok=True)


def resample_to_16k(audio_48k: np.ndarray) -> np.ndarray:
    """Resample 48kHz audio to 16kHz using simple decimation."""
    # 48000 / 16000 = 3, so take every 3rd sample after anti-alias filtering
    from scipy.signal import decimate
    return decimate(audio_48k, 3).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="M10 TTS consonant capture (48kHz)")
    parser.add_argument("--voices", nargs="+", default=VOICES_48K)
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=3)
    parser.add_argument("--speaker", type=str, default="tts48")
    parser.add_argument("--rate", type=int, default=48000, help="Capture sample rate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    SR = args.rate
    HP100 = butter(4, 100, btype='high', fs=SR, output='sos')

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/m10/raw") / args.speaker / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("data/m10/manifests") / "captures.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(PROMPTS) * len(args.voices)
    print("=" * 78)
    print(f"M10 TTS Consonant Capture — {SR//1000}kHz")
    print("=" * 78)
    print(f"  Prompts:     {len(PROMPTS)}")
    print(f"  Voices:      {', '.join(args.voices)} ({len(args.voices)})")
    print(f"  Total:       {total} captures")
    print(f"  Rate:        {SR} Hz")
    print(f"  Input dev:   {args.input_device} ({sd.query_devices(args.input_device)['name']})")
    print(f"  Output dev:  {args.output_device}")
    print(f"  Output dir:  {output_dir}")
    print()

    if args.dry_run:
        for voice in args.voices:
            for p in PROMPTS:
                print(f"  [{voice:10s}] {p['text']:15s} -> {p['label']} ({p['category']})")
        return

    rows = []
    count = 0

    for voice in args.voices:
        print(f"\n=== Voice: {voice} ===")
        for p in PROMPTS:
            count += 1
            text = p["text"]
            label = p["label"]

            print(f"[{count}/{total}] {voice}: '{text}' -> {label}", end="", flush=True)

            # Generate TTS at 48kHz
            tts_wav = output_dir / f"tts48_{voice}_{label}_{count:03d}.wav"
            if not tts_to_wav_48k(text, voice, tts_wav):
                print(" [TTS FAIL]")
                continue

            # Load TTS audio
            import wave
            with wave.open(str(tts_wav), 'rb') as wf:
                audio_48k = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

            # Play at 48kHz and capture at 48kHz
            try:
                captured_48k = sd.playrec(
                    audio_48k, samplerate=SR, channels=1, dtype="float32",
                    device=(args.input_device, args.output_device), blocking=True
                )
            except TypeError:
                sd.default.device = (args.input_device, args.output_device)
                captured_48k = sd.playrec(
                    audio_48k, samplerate=SR, channels=1, dtype="float32", blocking=True
                )

            captured_48k = np.asarray(captured_48k, dtype=np.float32).reshape(-1)
            captured_48k = sosfilt(HP100, captured_48k).astype(np.float32)

            rms_48k = float(np.sqrt(np.mean(captured_48k**2)))
            peak_48k = float(np.max(np.abs(captured_48k)))

            # Save 48kHz capture
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            capture_id = f"{voice[:3].lower()}_{str(uuid.uuid4())[:6]}"
            filename_48k = f"{timestamp}_{label}_{capture_id}_48k.wav"
            wav_path_48k = output_dir / filename_48k

            # Write 48kHz WAV
            with wave.open(str(wav_path_48k), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SR)
                audio_int16 = np.clip(captured_48k * 32767, -32768, 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            # Also save 16kHz downsampled version for compatibility
            captured_16k = resample_to_16k(captured_48k)
            filename_16k = f"{timestamp}_{label}_{capture_id}_16k.wav"
            wav_path_16k = output_dir / filename_16k
            write_pcm16_wav(wav_path_16k, captured_16k, sr=16000)

            # Check high-frequency energy ratio
            spec = np.abs(np.fft.rfft(captured_48k[:min(len(captured_48k), 2048)]))
            freqs = np.linspace(0, SR/2, len(spec))
            low_e = np.mean(spec[freqs < 4000]**2)
            high_e = np.mean(spec[freqs >= 4000]**2)
            total_e = low_e + high_e
            hf_pct = high_e / total_e * 100 if total_e > 0 else 0

            row = {
                "capture_id": capture_id,
                "session_id": session_id,
                "speaker": f"{args.speaker}_{voice}",
                "prompt": text,
                "label": label,
                "category": p["category"],
                "wav_path": str(wav_path_16k.relative_to(Path("."))),
                "wav_path_48k": str(wav_path_48k.relative_to(Path("."))),
                "sample_rate": SR,
                "duration_s": round(len(captured_48k) / SR, 3),
                "rms": rms_48k,
                "peak": peak_48k,
                "hf_pct": float(hf_pct),
                "input_device": args.input_device,
                "output_device": args.output_device,
                "voice": voice,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(row)
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            quality = "GOOD" if rms_48k > 0.02 else "LOW" if rms_48k > 0.008 else "SILENT"
            print(f"  rms={rms_48k:.4f} hf={hf_pct:.0f}% [{quality}]")

    print(f"\nSession complete: {len(rows)} captures saved")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")

    # Summary
    by_label = {}
    for r in rows:
        l = r["label"]
        if l not in by_label:
            by_label[l] = {"count": 0, "rms_list": [], "hf_list": []}
        by_label[l]["count"] += 1
        by_label[l]["rms_list"].append(r["rms"])
        by_label[l]["hf_list"].append(r.get("hf_pct", 0))

    print("\nPer-label summary:")
    for label in sorted(by_label.keys()):
        info = by_label[label]
        avg_rms = float(np.mean(info["rms_list"]))
        avg_hf = float(np.mean(info["hf_list"]))
        print(f"  {label:4s}: {info['count']:3d} captures, avg rms={avg_rms:.4f}, avg hf={avg_hf:.0f}%")


if __name__ == "__main__":
    main()
