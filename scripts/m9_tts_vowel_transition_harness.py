#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""M9 TTS vowel-transition speaker harness.

This harness synthesizes short vowel-transition phrases with macOS TTS, plays
those phrases through a selected output device, captures them from the mic, and
scores the recorded audio with the vowel-transition decoder.

It is meant for external-speaker stress testing, not as a promotion baseline.
Use it to compare output devices, playback levels, and phrase sets.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound import decode_vowel_transition_audio, write_pcm16_wav

SR = 16_000
DEFAULT_INPUT_DEVICE = 1
DEFAULT_OUTPUT_DEVICE = 2
DEFAULT_VOICE = "Samantha"
DEFAULT_RATE = "-20%"
DEFAULT_PITCH = "+0Hz"
DEFAULT_LEAD_IN_S = 0.25
DEFAULT_TAIL_S = 0.25
DEFAULT_FRAME_SIZE = 480
DEFAULT_MIN_TRANSITIONS = 2
DEFAULT_MIN_CONTINUITY = 0.88

DEFAULT_PHRASES = [
    "eee eeeh ehh ahh ohh ooo",
    "ooo ooh ohh ahh ehh eee",
    "eee ahh eee ohh ooo",
    "ooo ahh ooo ehh eee",
    "eee ehh ahh ehh eee",
    "ooo ohh ahh ohh ooo",
]


def _read_wav_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if channels != 1:
            raise ValueError(f"Expected mono WAV, got {channels} channels in {path}")
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit WAV, got sample width {sampwidth} in {path}")
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm.reshape(-1)


def _write_tts_audio_to_wav(
    *,
    text: str,
    output_wav: Path,
    voice: str,
    rate: str,
    pitch: str,
) -> np.ndarray:
    """Use macOS `say` to synthesize text, then convert it to 16 kHz WAV."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m9-tts-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        aiff_path = tmpdir_path / "tts.aiff"
        subprocess.run(
            ["say", "-v", voice, "-r", rate.replace("%", "") if rate.endswith("%") else rate, "-o", str(aiff_path), text],
            check=True,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff_path), str(output_wav)],
            check=True,
        )
    return _read_wav_pcm(output_wav)


def _capture_playback_with_playrec(
    pcm: np.ndarray,
    *,
    input_device: int,
    output_device: int,
    lead_in_s: float,
    tail_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - only on real hardware
        raise RuntimeError("sounddevice is required for live capture") from exc

    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    pad_lead = np.zeros(int(round(lead_in_s * SR)), dtype=np.float32)
    pad_tail = np.zeros(int(round(tail_s * SR)), dtype=np.float32)
    playback = np.concatenate([pad_lead, pcm, pad_tail])

    try:
        captured = sd.playrec(
            playback,
            samplerate=SR,
            channels=1,
            dtype="float32",
            device=(input_device, output_device),
            blocking=True,
        )
    except TypeError:
        sd.default.device = (input_device, output_device)
        captured = sd.playrec(
            playback,
            samplerate=SR,
            channels=1,
            dtype="float32",
            blocking=True,
        )
    captured = np.asarray(captured, dtype=np.float32).reshape(-1)
    return captured, {
        "playback_samples": int(len(playback)),
        "capture_samples": int(len(captured)),
        "lead_in_s": float(lead_in_s),
        "tail_s": float(tail_s),
        "input_device": int(input_device),
        "output_device": int(output_device),
    }


def _default_capture_audio_fn(
    pcm: np.ndarray,
    *,
    input_device: int,
    output_device: int,
    lead_in_s: float,
    tail_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _capture_playback_with_playrec(
        pcm,
        input_device=input_device,
        output_device=output_device,
        lead_in_s=lead_in_s,
        tail_s=tail_s,
    )


def _parse_phrases(values: Sequence[str]) -> list[str]:
    phrases = [value.strip() for value in values if value.strip()]
    if not phrases:
        raise ValueError("At least one phrase is required")
    return phrases


def run_tts_vowel_transition_harness(
    *,
    output_dir: str | Path,
    output: str | Path | None = None,
    phrases: Sequence[str] | None = None,
    repeats: int = 1,
    input_device: int = DEFAULT_INPUT_DEVICE,
    output_device: int = DEFAULT_OUTPUT_DEVICE,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    pitch: str = DEFAULT_PITCH,
    lead_in_s: float = DEFAULT_LEAD_IN_S,
    tail_s: float = DEFAULT_TAIL_S,
    threshold_transitions: int = DEFAULT_MIN_TRANSITIONS,
    threshold_continuity: float = DEFAULT_MIN_CONTINUITY,
    frame_size: int = DEFAULT_FRAME_SIZE,
    dry_run: bool = False,
    tts_audio_fn: Callable[..., np.ndarray] | None = None,
    capture_audio_fn: Callable[..., tuple[np.ndarray, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output) if output is not None else output_dir / "report.json"
    phrase_list = _parse_phrases(DEFAULT_PHRASES if phrases is None else phrases)
    repeats = max(1, int(repeats))

    synth_fn = tts_audio_fn or _write_tts_audio_to_wav
    capture_fn = capture_audio_fn or _default_capture_audio_fn

    runs: list[dict[str, Any]] = []
    for repeat_idx in range(repeats):
        for phrase_idx, phrase in enumerate(phrase_list, start=1):
            run_dir = output_dir / f"repeat-{repeat_idx + 1:02d}" / f"phrase-{phrase_idx:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            tts_wav = run_dir / "tts.wav"
            captured_wav = run_dir / "captured.wav"

            target_pcm = synth_fn(
                text=phrase,
                output_wav=tts_wav,
                voice=voice,
                rate=rate,
                pitch=pitch,
            )
            target_pcm = np.asarray(target_pcm, dtype=np.float32).reshape(-1)
            if dry_run:
                captured_pcm = target_pcm.copy()
                capture_meta = {
                    "playback_samples": int(len(target_pcm)),
                    "capture_samples": int(len(target_pcm)),
                    "lead_in_s": float(lead_in_s),
                    "tail_s": float(tail_s),
                    "input_device": int(input_device),
                    "output_device": int(output_device),
                }
            else:
                captured_pcm, capture_meta = capture_fn(
                    pcm=target_pcm,
                    input_device=input_device,
                    output_device=output_device,
                    lead_in_s=lead_in_s,
                    tail_s=tail_s,
                )

            captured_pcm = np.asarray(captured_pcm, dtype=np.float32).reshape(-1)
            captured_meta = write_pcm16_wav(captured_wav, captured_pcm, sr=SR)
            decoded = decode_vowel_transition_audio(captured_pcm, frame_size=frame_size)
            transition_count = int(decoded["decoded_transition_count"])
            continuity = float(decoded["continuity"])
            passed = transition_count >= threshold_transitions and continuity >= threshold_continuity
            runs.append(
                {
                    "repeat": repeat_idx + 1,
                    "phrase_index": phrase_idx,
                    "phrase": phrase,
                    "tts_wav": str(tts_wav),
                    "captured_wav": str(captured_wav),
                    "tts_sample_count": int(len(target_pcm)),
                    "captured_sample_count": int(captured_meta["sample_count"]),
                    "capture": capture_meta,
                    "decoded": {
                        "decoded_transition_count": transition_count,
                        "decoded_unique_vowel_count": int(decoded["decoded_unique_vowel_count"]),
                        "continuity": continuity,
                        "mean_boundary_delta": float(decoded["mean_boundary_delta"]),
                        "max_boundary_delta": float(decoded["max_boundary_delta"]),
                        "decoded_vowels_preview": list(decoded["decoded_vowels"][:20]),
                    },
                    "verdict": {
                        "threshold_transitions": int(threshold_transitions),
                        "threshold_continuity": float(threshold_continuity),
                        "passed": passed,
                    },
                }
            )

    pass_count = sum(1 for run in runs if run["verdict"]["passed"])
    avg_transitions = float(np.mean([run["decoded"]["decoded_transition_count"] for run in runs])) if runs else 0.0
    avg_continuity = float(np.mean([run["decoded"]["continuity"] for run in runs])) if runs else 0.0
    failure_rows = [
        {
            "repeat": run["repeat"],
            "phrase_index": run["phrase_index"],
            "phrase": run["phrase"],
            "decoded_transition_count": run["decoded"]["decoded_transition_count"],
            "continuity": run["decoded"]["continuity"],
        }
        for run in runs
        if not run["verdict"]["passed"]
    ]
    summary = {
        "pass_count": pass_count,
        "total": len(runs),
        "pass_rate": float(pass_count / len(runs)) if runs else 0.0,
        "avg_transition_count": avg_transitions,
        "avg_continuity": avg_continuity,
        "threshold_transitions": int(threshold_transitions),
        "threshold_continuity": float(threshold_continuity),
        "verdict": pass_count == len(runs) if runs else False,
        "failure_rows": failure_rows,
    }
    report = {
        "output_dir": str(output_dir),
        "output": str(output_path),
        "phrases": phrase_list,
        "repeats": repeats,
        "input_device": int(input_device),
        "output_device": int(output_device),
        "voice": voice,
        "rate": rate,
        "pitch": pitch,
        "lead_in_s": float(lead_in_s),
        "tail_s": float(tail_s),
        "dry_run": bool(dry_run),
        "runs": runs,
        "summary": summary,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M9 TTS vowel-transition speaker harness")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m9-tts-vowel-harness"))
    parser.add_argument("--output", type=Path, default=None, help="Optional report JSON path (defaults to output-dir/report.json)")
    parser.add_argument("--phrases", nargs="*", default=DEFAULT_PHRASES, help="Phrases to synthesize and play")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--input-device", type=int, default=DEFAULT_INPUT_DEVICE)
    parser.add_argument("--output-device", type=int, default=DEFAULT_OUTPUT_DEVICE)
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE)
    parser.add_argument("--rate", type=str, default=DEFAULT_RATE)
    parser.add_argument("--pitch", type=str, default=DEFAULT_PITCH)
    parser.add_argument("--lead-in-s", type=float, default=DEFAULT_LEAD_IN_S)
    parser.add_argument("--tail-s", type=float, default=DEFAULT_TAIL_S)
    parser.add_argument("--threshold-transitions", type=int, default=DEFAULT_MIN_TRANSITIONS)
    parser.add_argument("--threshold-continuity", type=float, default=DEFAULT_MIN_CONTINUITY)
    parser.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Skip hardware capture and treat the synthesized phrase as perfectly captured")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_tts_vowel_transition_harness(
        output_dir=args.output_dir,
        output=args.output,
        phrases=args.phrases,
        repeats=args.repeats,
        input_device=args.input_device,
        output_device=args.output_device,
        voice=args.voice,
        rate=args.rate,
        pitch=args.pitch,
        lead_in_s=args.lead_in_s,
        tail_s=args.tail_s,
        threshold_transitions=args.threshold_transitions,
        threshold_continuity=args.threshold_continuity,
        frame_size=args.frame_size,
        dry_run=args.dry_run,
    )

    for run in report["runs"]:
        verdict = "PASS" if run["verdict"]["passed"] else "FAIL"
        decoded = run["decoded"]
        print(
            f"[repeat {run['repeat']:02d} phrase {run['phrase_index']:02d}] {verdict} "
            f"transitions={decoded['decoded_transition_count']} continuity={decoded['continuity']:.3f} "
            f"phrase={run['phrase']}"
        )
    summary = report["summary"]
    print(
        f"Summary: {summary['pass_count']}/{summary['total']} pass, "
        f"avg_transitions={summary['avg_transition_count']:.2f}, "
        f"avg_continuity={summary['avg_continuity']:.3f}"
    )
    print(f"Report: {report['output']}")


if __name__ == "__main__":
    main()
