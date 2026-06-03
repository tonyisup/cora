#!/usr/bin/env python3
"""M9 live audio promotion gate.

This is the next-step hardware gate for M9. It is intentionally separate from
synthetic M9 validation:
- dry-run validates config and scoring without hardware
- live mode plays a target sound and captures it back from a mic/loopback
- score and report the capture quality for promotion decisions

The synthetic M9 track remains the source of truth for learning behavior;
this script only checks whether that behavior survives real audio I/O.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound import AcousticSimilarity, OscillatorBank, TextAssociationStore

SR = 16000
DEFAULT_TARGETS = [
    ("alpha", 220.0, "sine"),
    ("beta", 247.0, "triangle"),
    ("gamma", 262.0, "sawtooth"),
    ("delta", 294.0, "square"),
]


def parse_targets(spec: str | None) -> list[tuple[str, float, str]]:
    """Parse a comma-separated target spec like 'alpha:220:sine,beta:247:triangle'."""
    if not spec:
        return list(DEFAULT_TARGETS)

    out: list[tuple[str, float, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 3:
            word, freq_s, waveform = parts
        elif len(parts) == 2:
            word, freq_s = parts
            waveform = "sine"
        else:
            raise ValueError(f"Invalid target spec: {item!r}")
        out.append((word.strip(), float(freq_s), waveform.strip()))
    return out


def list_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed.")
        return

    print("Available audio devices:")
    print(f"  {'#':>3}  {'Name':<34}  {'In':>3}  {'Out':>3}")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 or dev["max_output_channels"] > 0:
            print(f"  {i:3d}  {dev['name']:<34}  {dev['max_input_channels']:3d}  {dev['max_output_channels']:3d}")


def make_target_pcm(word: str, freq_hz: float, waveform: str, duration_s: float, sr: int = SR) -> tuple[np.ndarray, dict[str, Any]]:
    store = TextAssociationStore(sr=sr)
    pair = store.add_pair(word, freq_hz=freq_hz, waveform=waveform, duration_s=duration_s)
    sound = store.get_sound_for_word(word)
    if sound is not None and sound.prototype_audio:
        return np.asarray(sound.prototype_audio, dtype=np.float32), {
            "word": word,
            "word_id": pair.word_id,
            "sound_id": pair.sound_id,
            "freq_hz": freq_hz,
            "waveform": waveform,
            "source": "text_association_store",
        }

    osc = OscillatorBank(sr=sr)
    pcm = {
        "sine": osc.sine,
        "triangle": osc.triangle,
        "sawtooth": osc.sawtooth,
        "square": osc.square,
    }[waveform](freq_hz, duration_s, amplitude=0.5)
    return pcm.astype(np.float32), {
        "word": word,
        "word_id": pair.word_id,
        "sound_id": pair.sound_id,
        "freq_hz": freq_hz,
        "waveform": waveform,
        "source": "oscillator",
    }


def capture_during_playback(
    pcm: np.ndarray,
    *,
    input_device: int | None,
    output_device: int | None,
    sr: int = SR,
    lead_in_s: float = 0.25,
    tail_s: float = 0.25,
) -> dict[str, Any]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is required for live capture") from exc

    if pcm.ndim != 1:
        pcm = np.asarray(pcm).reshape(-1)
    pcm = pcm.astype(np.float32)

    if input_device is not None:
        input_info = sd.query_devices(input_device)
        input_sr = int(input_info["default_samplerate"] or sr)
        input_name = str(input_info["name"])
    else:
        input_sr = sr
        input_name = "default"

    output_name = "default"
    if output_device is not None:
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

    capture_finished_at = time.perf_counter()
    if input_sr != sr:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError("scipy is required to resample capture") from exc
        captured = resample_poly(captured, sr, input_sr).astype(np.float32)

    return {
        "captured_pcm": captured.astype(np.float32),
        "input_device_name": input_name,
        "output_device_name": output_name,
        "capture_duration_s": round(total_duration, 4),
        "playback_duration_s": round(playback_duration, 4),
        "capture_elapsed_s": round(capture_finished_at - capture_started_at, 4),
        "playback_started": playback_started_at is not None,
    }


def run_gate(
    *,
    word: str,
    freq_hz: float,
    waveform: str,
    duration_s: float,
    input_device: int | None,
    output_device: int | None,
    threshold: float,
    dry_run: bool,
) -> dict[str, Any]:
    target_pcm, target_meta = make_target_pcm(word, freq_hz, waveform, duration_s)
    similarity = AcousticSimilarity(sr=SR)

    if dry_run:
        captured_pcm = target_pcm.copy()
        capture_meta = {
            "captured_pcm": captured_pcm,
            "input_device_name": "dry-run",
            "output_device_name": "dry-run",
            "capture_duration_s": round(duration_s, 4),
            "playback_duration_s": round(duration_s, 4),
            "capture_elapsed_s": 0.0,
            "playback_started": False,
        }
        score = type(similarity.detailed_score(target_pcm, captured_pcm))(
            overall=1.0,
            mfcc_distance=0.0,
            pitch_distance=0.0,
            duration_distance=0.0,
            loudness_distance=0.0,
            components={"mel": 0.0, "pitch": 0.0, "duration": 0.0, "loudness": 0.0},
        )
    else:
        capture_meta = capture_during_playback(
            target_pcm,
            input_device=input_device,
            output_device=output_device,
            sr=SR,
        )
        captured_pcm = capture_meta["captured_pcm"]
        score = similarity.detailed_score(target_pcm, captured_pcm)
    passed = score.overall >= threshold

    return {
        "target": {
            **target_meta,
            "duration_s": duration_s,
            "sample_count": int(len(target_pcm)),
        },
        "capture": {
            "input_device_name": capture_meta["input_device_name"],
            "output_device_name": capture_meta["output_device_name"],
            "capture_duration_s": capture_meta["capture_duration_s"],
            "playback_duration_s": capture_meta["playback_duration_s"],
            "capture_elapsed_s": capture_meta["capture_elapsed_s"],
            "playback_started": capture_meta["playback_started"],
        },
        "score": {
            "overall": float(score.overall),
            "mfcc_distance": float(score.mfcc_distance),
            "pitch_distance": float(score.pitch_distance),
            "duration_distance": float(score.duration_distance),
            "loudness_distance": float(score.loudness_distance),
        },
        "verdict": {
            "threshold": threshold,
            "passed": passed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 live audio promotion gate")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--word", type=str, default=None, help="Single-target override word")
    parser.add_argument("--target-freq", type=float, default=220.0)
    parser.add_argument("--target-waveform", type=str, default="sine", choices=["sine", "triangle", "sawtooth", "square"])
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated target set like 'alpha:220:sine,beta:247:triangle'")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each target this many times")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output", type=str, default="/tmp/m9_live_audio_gate.json")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.targets:
        target_specs = parse_targets(args.targets)
    elif args.word is not None:
        target_specs = [(args.word, args.target_freq, args.target_waveform)]
    else:
        target_specs = list(DEFAULT_TARGETS)

    repeats = max(1, int(args.repeats))
    rows: list[dict[str, Any]] = []
    for word, freq_hz, waveform in target_specs:
        for repeat_idx in range(repeats):
            row = run_gate(
                word=word,
                freq_hz=freq_hz,
                waveform=waveform,
                duration_s=args.duration,
                input_device=args.input_device,
                output_device=args.output_device,
                threshold=args.threshold,
                dry_run=args.dry_run,
            )
            row["repeat"] = repeat_idx + 1
            rows.append(row)

    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row["target"]["word"]
        by_target.setdefault(key, []).append(row)

    target_summaries = []
    for word, group in by_target.items():
        pass_count = sum(1 for row in group if row["verdict"]["passed"])
        pass_rate = pass_count / len(group)
        scores = [row["score"]["overall"] for row in group]
        target_summaries.append({
            "word": word,
            "freq_hz": group[0]["target"]["freq_hz"],
            "waveform": group[0]["target"]["waveform"],
            "pass_count": pass_count,
            "total": len(group),
            "pass_rate": pass_rate,
            "avg_score": float(np.mean(scores)),
            "worst_score": float(np.min(scores)),
        })

    total_pass_count = sum(summary["pass_count"] for summary in target_summaries)
    total_trials = len(rows)
    pass_rate = total_pass_count / max(total_trials, 1)
    avg_score = float(np.mean([row["score"]["overall"] for row in rows]))
    worst_score = float(np.min([row["score"]["overall"] for row in rows]))
    verdict = pass_rate >= 0.80

    output_payload = {
        "config": {
            "dry_run": args.dry_run,
            "targets": [{"word": w, "freq_hz": f, "waveform": wf} for w, f, wf in target_specs],
            "repeats": repeats,
            "duration": args.duration,
            "input_device": args.input_device,
            "output_device": args.output_device,
            "threshold": args.threshold,
        },
        "results": rows,
        "summary": {
            "pass_count": total_pass_count,
            "total": total_trials,
            "pass_rate": pass_rate,
            "avg_score": avg_score,
            "worst_score": worst_score,
            "verdict": verdict,
            "target_summaries": target_summaries,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output_payload, indent=2))

    print("=" * 72)
    print("M9 Live Audio Promotion Gate")
    print("=" * 72)
    print(f"  mode:      {'dry-run' if args.dry_run else 'live'}")
    print(f"  targets:   {len(target_specs)}")
    print(f"  repeats:   {repeats}")
    print(f"  pass rate: {pass_rate:.2f} ({total_pass_count}/{total_trials})")
    print(f"  avg score: {avg_score:.4f}")
    print(f"  worst:     {worst_score:.4f}")
    print(f"  threshold: {args.threshold:.4f}")
    print(f"  verdict:   {'PASS' if verdict else 'FAIL'}")
    print(f"  saved:     {args.output}")

    if not verdict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
