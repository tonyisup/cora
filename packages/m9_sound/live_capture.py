"""Live capture helpers for the M9 temporal smoke path.

This module bridges the hardware-free temporal smoke harness with a real mic/
speaker capture loop. It keeps the render path reusable while allowing tests to
inject a fake capture function.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, cast

import numpy as np

from m9_sound.temporal import (
    DEFAULT_CONTROLLER_CKPT,
    DEFAULT_SUBSTRATE_CKPT,
    FRAME_SIZE,
    SR,
    build_temporal_smoke_artifact,
    write_pcm16_wav,
)
from m9_sound.vowel_transitions import (
    DEFAULT_LOOKUP_CKPT,
    DEFAULT_SUBSTRATE_CKPT as DEFAULT_VOWEL_TRANSITION_SUBSTRATE_CKPT,
    decode_vowel_transition_audio,
    build_vowel_transition_report,
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


def run_live_vowel_transition_capture_report(
    output_dir: str | Path,
    *,
    family: str = "default",
    heldout_family: str = "diphthong",
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
    substrate_ckpt_path: str | Path | None = None,
    lookup_ckpt_path: str | Path | None = None,
    sr: int = SR,
    frame_size: int = 480,
) -> dict[str, object]:
    """Render a vowel-transition family, capture it live, and score phase closure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    render_report = cast(
        dict[str, object],
        build_vowel_transition_report(
            family=family,
            heldout_family=heldout_family,
            noise_rms=input_noise_rms,
            noise_seed=noise_seed,
            device_profile=device_profile,
            device_gain=device_gain,
            substrate_ckpt_path=substrate_ckpt_path or DEFAULT_VOWEL_TRANSITION_SUBSTRATE_CKPT,
            lookup_ckpt_path=lookup_ckpt_path or DEFAULT_LOOKUP_CKPT,
            sr=sr,
            frame_size=frame_size,
        ),
    )
    rendered_audio = np.asarray(cast(list[float], render_report["rendered_audio"]), dtype=np.float32)

    rendered_wav = output_dir / "rendered_vowel_transition.wav"
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

    captured_wav = output_dir / "captured_vowel_transition.wav"
    captured_meta = write_pcm16_wav(captured_wav, live_capture, sr=sr)
    captured_decode = decode_vowel_transition_audio(
        live_capture,
        substrate_ckpt_path=substrate_ckpt_path or DEFAULT_VOWEL_TRANSITION_SUBSTRATE_CKPT,
        lookup_ckpt_path=lookup_ckpt_path or DEFAULT_LOOKUP_CKPT,
        frame_size=frame_size,
    )

    target_transition_count = int(render_report["input_transition_count"])
    captured_transition_count = int(cast(int, captured_decode["decoded_transition_count"]))
    transition_alignment = float(min(1.0, captured_transition_count / max(target_transition_count, 1)))
    continuity = float(cast(float, captured_decode["continuity"]))
    phase_closure_score = float(0.7 * transition_alignment + 0.3 * continuity)
    phase_closure_passed = bool(transition_alignment >= 1.0 and phase_closure_score >= 0.7)

    report: dict[str, object] = dict(render_report)
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
            "captured_decoded_vowels": list(cast(list[str], captured_decode["decoded_vowels"])),
            "captured_decoded_transition_count": captured_transition_count,
            "captured_decoded_unique_vowel_count": int(cast(int, captured_decode["decoded_unique_vowel_count"])),
            "captured_continuity": continuity,
            "captured_mean_boundary_delta": float(cast(float, captured_decode["mean_boundary_delta"])),
            "captured_max_boundary_delta": float(cast(float, captured_decode["max_boundary_delta"])),
            "captured_min_boundary_delta": float(cast(float, captured_decode["min_boundary_delta"])),
            "phase_closure": {
                "target_transition_count": target_transition_count,
                "captured_transition_count": captured_transition_count,
                "transition_alignment": transition_alignment,
                "continuity": continuity,
                "score": phase_closure_score,
                "passed": phase_closure_passed,
            },
        }
    )

    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _dry_run_phase_closure_capture_audio(playback_audio: np.ndarray, *, sr: int, **_: object) -> np.ndarray:
    return np.asarray(playback_audio, dtype=np.float32).reshape(-1)


def run_live_vowel_transition_gate(
    output_dir: str | Path,
    *,
    families: list[str] | None = None,
    repeats: int = 1,
    threshold: float = 0.7,
    dry_run: bool = False,
    capture_audio_fn: CaptureFn | None = None,
    heldout_family: str = "diphthong",
    input_noise_rms: float = 0.0,
    noise_seed: int | None = None,
    device_profile: str = "flat",
    device_gain: float = 1.0,
    input_device: int | None = None,
    output_device: int | None = None,
    lead_in_s: float = 0.25,
    tail_s: float = 0.25,
    substrate_ckpt_path: str | Path | None = None,
    lookup_ckpt_path: str | Path | None = None,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
) -> dict[str, object]:
    """Run a live phase-closure gate over one or more vowel-transition families."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    family_list = list(families or ["default", "phase_triad", "phase_arc"])
    repeats = max(1, int(repeats))
    capture_audio_fn = capture_audio_fn or (_dry_run_phase_closure_capture_audio if dry_run else None)

    rows: list[dict[str, object]] = []
    for family in family_list:
        for repeat_idx in range(repeats):
            row = run_live_vowel_transition_capture_report(
                output_dir=output_dir / family / f"repeat-{repeat_idx + 1}",
                family=family,
                heldout_family=heldout_family,
                input_noise_rms=input_noise_rms,
                noise_seed=noise_seed,
                device_profile=device_profile,
                device_gain=device_gain,
                input_device=input_device,
                output_device=output_device,
                lead_in_s=lead_in_s,
                tail_s=tail_s,
                capture_audio_fn=capture_audio_fn,
                substrate_ckpt_path=substrate_ckpt_path,
                lookup_ckpt_path=lookup_ckpt_path,
                sr=sr,
                frame_size=frame_size,
            )
            phase = dict(cast(dict[str, object], row["phase_closure"]))
            phase["threshold"] = threshold
            phase["passed"] = bool(float(cast(float, phase["score"])) >= threshold)
            row["phase_closure"] = phase
            row["family"] = family
            row["repeat"] = repeat_idx + 1
            row["verdict"] = {"threshold": threshold, "passed": phase["passed"]}
            rows.append(row)

    family_summaries: list[dict[str, object]] = []
    for family in family_list:
        group = [row for row in rows if row["family"] == family]
        if not group:
            continue
        phase_rows = [cast(dict[str, object], row["phase_closure"]) for row in group]
        scores = [float(cast(float, phase["score"])) for phase in phase_rows]
        pass_count = sum(1 for phase in phase_rows if phase["passed"])
        failure_rows = [
            {
                "repeat": row["repeat"],
                "score": float(cast(float, cast(dict[str, object], row["phase_closure"])["score"])),
                "passed": bool(cast(dict[str, object], row["phase_closure"])["passed"]),
                "captured_transition_count": int(cast(int, row["captured_decoded_transition_count"])),
                "captured_continuity": float(cast(float, row["captured_continuity"])),
                "captured_mean_boundary_delta": float(cast(float, row["captured_mean_boundary_delta"])),
                "captured_unique_vowel_count": int(cast(int, row["captured_decoded_unique_vowel_count"])),
            }
            for row in group
            if not bool(cast(dict[str, object], row["phase_closure"])["passed"])
        ]
        family_summaries.append(
            {
                "family": family,
                "pass_count": pass_count,
                "failure_count": len(group) - pass_count,
                "total": len(group),
                "pass_rate": pass_count / len(group),
                "avg_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "std_score": float(np.std(scores)),
                "worst_score": float(np.min(scores)),
                "best_score": float(np.max(scores)),
                "failure_rows": failure_rows,
            }
        )

    phase_rows = [cast(dict[str, object], row["phase_closure"]) for row in rows]
    pass_count = sum(1 for phase in phase_rows if phase["passed"])
    total = len(rows)
    pass_rate = pass_count / max(total, 1)
    scores = [float(cast(float, phase["score"])) for phase in phase_rows]
    avg_score = float(np.mean(scores)) if scores else 0.0
    median_score = float(np.median(scores)) if scores else 0.0
    std_score = float(np.std(scores)) if scores else 0.0
    worst_score = float(np.min(scores)) if scores else 0.0
    best_score = float(np.max(scores)) if scores else 0.0
    failure_rows = [
        {
            "family": row["family"],
            "repeat": row["repeat"],
            "score": float(cast(float, cast(dict[str, object], row["phase_closure"])["score"])),
            "passed": bool(cast(dict[str, object], row["phase_closure"])["passed"]),
            "captured_transition_count": int(cast(int, row["captured_decoded_transition_count"])),
            "captured_continuity": float(cast(float, row["captured_continuity"])),
            "captured_mean_boundary_delta": float(cast(float, row["captured_mean_boundary_delta"])),
            "captured_unique_vowel_count": int(cast(int, row["captured_decoded_unique_vowel_count"])),
        }
        for row in rows
        if not bool(cast(dict[str, object], row["phase_closure"])["passed"])
    ]
    verdict = pass_rate >= 0.80

    report = {
        "config": {
            "dry_run": dry_run,
            "families": family_list,
            "repeats": repeats,
            "threshold": threshold,
            "heldout_family": heldout_family,
            "input_noise_rms": input_noise_rms,
            "noise_seed": noise_seed,
            "device_profile": device_profile,
            "device_gain": device_gain,
            "input_device": input_device,
            "output_device": output_device,
            "lead_in_s": lead_in_s,
            "tail_s": tail_s,
            "sr": sr,
            "frame_size": frame_size,
        },
        "results": rows,
        "summary": {
            "pass_count": pass_count,
            "total": total,
            "pass_rate": pass_rate,
            "avg_score": avg_score,
            "median_score": median_score,
            "std_score": std_score,
            "worst_score": worst_score,
            "best_score": best_score,
            "failure_rows": failure_rows,
            "verdict": verdict,
            "family_summaries": family_summaries,
        },
    }

    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

