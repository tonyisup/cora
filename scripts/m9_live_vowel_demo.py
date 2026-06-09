#!/usr/bin/env python3
"""M9 live vowel imitation loop.

This is the main demo for live mic imitation.
It listens for mic speech, classifies the vowel from the latest spectral
substrate + lookup, and synthesizes an imitation in real time.

Use --dry-run to exercise the same pipeline without hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

SR = 16000
FRAME = 480
N_MEL = 26
HIDDEN_DIM = 32

# Load spectral substrate and vowel lookup.
_SUB_CKPT = json.load(open("data/m9_substrate_spectral.ckpt.json"))
W_h = np.array(_SUB_CKPT["W_h"], dtype=np.float32)
W_x = np.array(_SUB_CKPT["W_x"], dtype=np.float32)
b = np.array(_SUB_CKPT["b"], dtype=np.float32)

_LOOKUP_DATA = json.load(open("data/m9_formant_lookup_v5.json"))
vowel_names = _LOOKUP_DATA["vowel_names"]
lookup = _LOOKUP_DATA["lookup"]
centroids = np.array([lookup[v]["centroid"] for v in vowel_names], dtype=np.float32)
tree = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(centroids)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def make_mel_filterbank(n_mels=N_MEL, n_fft=512, sr=SR, fmin=80, fmax=7600):
    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
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


MEL_FB = make_mel_filterbank()
HP_SOS = butter(4, 100, btype="high", fs=SR, output="sos")


def compute_mel(frame, mel_fb=MEL_FB, n_fft=512):
    x = np.zeros(n_fft, dtype=np.float32)
    n = min(len(frame), n_fft)
    x[:n] = frame[:n]
    X = np.abs(np.fft.rfft(x))
    mel_spec = mel_fb @ X
    mel_spec = np.log(mel_spec + 1e-6).astype(np.float32)
    rms = np.sqrt(np.mean(mel_spec**2))
    if rms > 1e-6:
        mel_spec /= rms
    return mel_spec


def synth(vowel, n_frames, phase=0.0) -> tuple[np.ndarray, float]:
    t = lookup[vowel]["template"]
    frames, cp = [], phase
    for _ in range(n_frames):
        tf = np.arange(FRAME, dtype=np.float32) / SR
        fr = np.zeros(FRAME, dtype=np.float32)
        for fk, ak in [("f1", "a1"), ("f2", "a2"), ("f3", "a3")]:
            fr += t[ak] * np.sin(2 * np.pi * t[fk] * tf + cp)
            cp += 2 * np.pi * t[fk] * (FRAME / SR)
        pk = np.max(np.abs(fr))
        if pk > 1.0:
            fr /= pk
        frames.append(fr.astype(np.float32))
        cp %= 2 * np.pi
    return np.concatenate(frames), cp


def synth_interp(template_from, template_to, alpha, n_frames, phase=0.0) -> tuple[np.ndarray, float]:
    """Synthesize frames with interpolated formant parameters."""
    frames, cp = [], phase
    for _ in range(n_frames):
        tf = np.arange(FRAME, dtype=np.float32) / SR
        fr = np.zeros(FRAME, dtype=np.float32)
        for fk, ak in [("f1", "a1"), ("f2", "a2"), ("f3", "a3")]:
            f_interp = template_from[fk] * (1 - alpha) + template_to[fk] * alpha
            a_interp = template_from[ak] * (1 - alpha) + template_to[ak] * alpha
            fr += a_interp * np.sin(2 * np.pi * f_interp * tf + cp)
            cp += 2 * np.pi * f_interp * (FRAME / SR)
        pk = np.max(np.abs(fr))
        if pk > 1.0:
            fr /= pk
        frames.append(fr.astype(np.float32))
        cp %= 2 * np.pi
    return np.concatenate(frames), cp


def make_state(args):
    return {
        "prev": None,
        "phase": 0.0,
        "hold": 0,
        "silence_count": args.feedback_frames,
        "history": [],
        "h": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "transition": 0,
        "transition_from": None,
        "transition_to": None,
    }


def rnn_step(state, mel_features):
    state["h"] = np.tanh(W_h @ state["h"] + W_x @ mel_features + b)


def process_mic_block(mic, state, args) -> tuple[np.ndarray, dict[str, object]]:
    mic = sosfilt(HP_SOS, np.asarray(mic, dtype=np.float32))
    rms = float(np.sqrt(np.mean(mic**2)))
    out_v = None
    confidence = float("inf")
    heard = False
    detected = None

    if rms > args.threshold and state["silence_count"] >= args.feedback_frames:
        heard = True
        block_rms = np.sqrt(np.mean(mic**2))
        mic_scaled = mic / max(block_rms, 1e-6) * 0.3

        state["h"] = np.zeros(HIDDEN_DIM, dtype=np.float32)
        n_speech_frames = 0
        for i in range(len(mic_scaled) // FRAME):
            raw_frame = mic_scaled[i * FRAME : (i + 1) * FRAME].flatten()
            frame_rms = float(np.sqrt(np.mean(raw_frame**2)))
            if frame_rms > args.threshold:
                mel_features = compute_mel(raw_frame)
                rnn_step(state, mel_features)
                n_speech_frames += 1

        if n_speech_frames < 2:
            state["silence_count"] = min(state["silence_count"] + 1, args.feedback_frames * 2)
        else:
            h = state["h"]
            dists, idx = tree.query(h, k=2)
            detected = vowel_names[idx[0]]
            if len(idx) > 1 and dists[0] > 1e-6:
                confidence = float(dists[1] / dists[0])
            else:
                confidence = float("inf")

            state["history"].append(detected)
            if len(state["history"]) > args.smooth_frames:
                state["history"] = state["history"][-args.smooth_frames :]
            counts = Counter(state["history"])
            v = counts.most_common(1)[0][0]

            if args.confidence_threshold > 0 and confidence < args.confidence_threshold:
                if state["prev"] is not None:
                    out_v = state["prev"]
                    state["hold"] = args.hold_frames
            else:
                out_v = v
                state["hold"] = args.hold_frames

            state["silence_count"] = 0
            state["prev"] = v
    elif state["hold"] > 0:
        state["hold"] -= 1
        out_v = state["prev"]
        state["silence_count"] = 0
    else:
        state["silence_count"] = min(state["silence_count"] + 1, args.feedback_frames * 2)
        if state["silence_count"] >= args.feedback_frames:
            state["history"] = []

    audio = np.zeros(len(mic), dtype=np.float32)
    if out_v:
        n_out = max(1, len(mic) // FRAME)
        if out_v != state["prev"] and state["prev"] is not None and args.transition_frames > 0:
            state["transition"] = args.transition_frames
            state["transition_from"] = lookup[state["prev"]]["template"]
            state["transition_to"] = lookup[out_v]["template"]

        if state["transition"] > 0 and state["transition_from"] is not None:
            alpha = 1.0 - (state["transition"] / args.transition_frames)
            audio, state["phase"] = synth_interp(
                state["transition_from"], state["transition_to"], alpha, n_out, state["phase"]
            )
            state["transition"] -= 1
        else:
            audio, state["phase"] = synth(out_v, n_out, state["phase"])

        if len(audio) >= len(mic):
            audio = audio[: len(mic)]
        else:
            padded = np.zeros(len(mic), dtype=np.float32)
            padded[: len(audio)] = audio
            audio = padded

    info = {
        "heard": heard,
        "detected": detected,
        "output_vowel": out_v,
        "confidence": confidence,
        "mic_rms": rms,
    }
    return audio, info


def _dry_run_sequence(args) -> None:
    print("Cora Vowel Imitation (Spectral)")
    print(f"  Input: device {args.input_device} (mic)")
    print(f"  Output: device {args.output_device} (headphones)")
    print(f"  Vowels: {vowel_names}")
    print(f"  Mel bins: {N_MEL}")
    print(f"  Speech gate: RMS>{args.threshold} and feedback gate {args.feedback_frames} frames")
    print(f"  Temporal smoothing: {args.smooth_frames}-frame majority vote")
    print(f"  Vowel transitions: {args.transition_frames}-frame formant interpolation")
    print("\nSpeak clearly. The loop waits for mic speech and imitates it. Ctrl+C to stop.\n")

    # A short synthetic utterance that the same mic→classifier→synth loop can process.
    speech, _ = synth("i", max(2, args.transition_frames + 1), 0.0)
    report_rows = []
    heard_count = 0
    for iteration in range(1, max(1, args.iterations) + 1):
        state = make_state(args)
        audio, info = process_mic_block(speech, state, args)
        heard_count += 1 if info["heard"] else 0
        report_rows.append({
            "iteration": iteration,
            "heard": info["heard"],
            "detected": info["detected"],
            "output_vowel": info["output_vowel"],
            "mic_rms": info["mic_rms"],
            "output_rms": float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0,
        })
        print(
            f"[{iteration:04d}] heard={info['heard']} detected={info['detected']} "
            f"output={info['output_vowel']} rms={info['mic_rms']:.4f}"
        )

    report = {
        "mode": "dry_run",
        "iterations": len(report_rows),
        "heard_iterations": heard_count,
        "rows": report_rows,
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDry-run summary: {heard_count}/{len(report_rows)} iterations heard speech")
    print(f"Report: {output_dir / 'report.json'}")


def _run_live_loop(args):
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - only on live hardware
        raise RuntimeError("sounddevice is required for live capture; install optional audio deps") from exc

    state = make_state(args)
    print("Cora Vowel Imitation (Spectral)")
    print(f"  Input: device {args.input_device} (mic)")
    print(f"  Output: device {args.output_device} (headphones)")
    print(f"  Vowels: {vowel_names}")
    print(f"  Mel bins: {N_MEL}")
    print(f"  Speech gate: RMS>{args.threshold} and feedback gate {args.feedback_frames} frames")
    print(f"  Temporal smoothing: {args.smooth_frames}-frame majority vote")
    print(f"  Vowel transitions: {args.transition_frames}-frame formant interpolation")
    print("\nSpeak clearly. The loop waits for mic speech and imitates it. Ctrl+C to stop.\n")

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"  Stream status: {status}")
        audio, info = process_mic_block(indata[:, 0], state, args)
        outdata[:] = audio.reshape(-1, 1)
        if info["heard"] and info["detected"] is not None:
            confidence = cast(float, info["confidence"])
            conf_str = f" (conf={confidence:.1f})" if confidence < float("inf") else ""
            print(f"  heard {info['detected']}{conf_str} -> output {info['output_vowel']}", end="\r", flush=True)

    try:
        with sd.Stream(
            device=(args.input_device, args.output_device),
            samplerate=SR,
            blocksize=SR // 2,
            channels=(1, 1),
            callback=callback,
            dtype="float32",
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 live vowel imitation loop")
    parser.add_argument("--input-device", type=int, default=1, help="Mic input device")
    parser.add_argument("--output-device", type=int, default=2, help="Headphone output device")
    parser.add_argument("--threshold", type=float, default=0.003, help="RMS threshold for speech")
    parser.add_argument("--feedback-frames", type=int, default=5, help="Frames to ignore input after producing output")
    parser.add_argument("--hold-frames", type=int, default=3, help="Frames to hold vowel")
    parser.add_argument("--smooth-frames", type=int, default=3, help="Majority-vote window for temporal smoothing")
    parser.add_argument("--transition-frames", type=int, default=3, help="Frames to interpolate formants during vowel change")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Min centroid distance ratio for acceptance (0=accept all)")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m9-live-vowel-demo"), help="Where dry-run artifacts are written")
    parser.add_argument("--iterations", type=int, default=2, help="Dry-run iterations (ignored in live mode; 0 means forever when dry-run is off)")
    parser.add_argument("--dry-run", action="store_true", help="Skip hardware and run a synthetic mic-triggered imitation loop")
    args = parser.parse_args()

    if args.dry_run:
        if args.iterations <= 0:
            args.iterations = 2
        _dry_run_sequence(args)
    else:
        _run_live_loop(args)


if __name__ == "__main__":
    main()
