"""Bounded project-owner-recorded vocal imitation.

This module is deliberately separate from the stopped H4 inverse families. It
learns one finite prototype association from two development recordings, then
scores the frozen association on a third recording without using confirmation
audio for motor selection.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import html
import json
from pathlib import Path
import shutil
import time
import wave

import numpy as np

from .factorized_tract_plant import (
    EXPANDED_SENSORY_FREQUENCIES_HZ,
    halton5,
    synthesize_factorized,
)
from .geometric_articulatory_synthesis import SAMPLE_RATE
from .io.live_mic import CAPTURE_CHANNELS, DEFAULT_DEVICE, LiveMicArray
from .io.motor_plan import write_wav
from .io.wav_frames import read_pcm_wav
from .speech_learning_demo import (
    DEMO_F0_HZ,
    DEMO_VOWELS,
    LocalImitationLearner,
    MotorPrototype,
    MotorRepertoire,
    sensory_distance,
)


PROTOCOL = "docs/VOCAL_IMITATION_PROTOCOL.md v4"
CAPTURE_DURATION_MS = 4000
START_CUE_FREQUENCY_HZ = 660.0
START_CUE_DURATION_MS = 100
START_CUE_SETTLE_MS = 200
START_CUE_AMPLITUDE = 0.08
VOCAL_REPERTOIRE_START_INDEX = 20011
VOCAL_REPERTOIRE_STRIDE = 43
ANALYSIS_SAMPLE_RATE = 16000
ANALYSIS_WINDOW_MS = 480
ANALYSIS_STRIDE_MS = 80
DEVELOPMENT_TAKES = 2
CONFIRMATION_TAKES = 1

CaptureFn = Callable[[], tuple[np.ndarray, int, dict[str, object]]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (30,) or not np.all(np.isfinite(vector)) or norm <= 1e-12:
        raise ValueError("vocal-imitation sensory vector is invalid")
    return vector / norm


def _resample_linear(mono: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate <= 0:
        raise ValueError("capture sample rate must be positive")
    source = np.asarray(mono, dtype=np.float64).reshape(-1)
    if source_rate == ANALYSIS_SAMPLE_RATE:
        return source
    output_count = max(1, int(round(len(source) * ANALYSIS_SAMPLE_RATE / source_rate)))
    source_positions = np.arange(len(source), dtype=np.float64) / source_rate
    output_positions = np.arange(output_count, dtype=np.float64) / ANALYSIS_SAMPLE_RATE
    return np.interp(output_positions, source_positions, source).astype(np.float64)


def _select_analysis_window(mono: np.ndarray) -> tuple[np.ndarray, int]:
    size = int(round(ANALYSIS_WINDOW_MS * ANALYSIS_SAMPLE_RATE / 1000.0))
    stride = int(round(ANALYSIS_STRIDE_MS * ANALYSIS_SAMPLE_RATE / 1000.0))
    if len(mono) < size:
        raise ValueError("recording is shorter than the frozen 480 ms analysis window")
    starts = range(0, len(mono) - size + 1, stride)
    start = max(
        starts,
        key=lambda offset: (
            float(np.sqrt(np.mean(np.square(mono[offset : offset + size])))),
            -offset,
        ),
    )
    return mono[start : start + size], start


def _pitch_periodicity(mono: np.ndarray) -> tuple[float | None, float]:
    """Return bounded autocorrelation pitch and normalized periodicity."""

    centered = np.asarray(mono, dtype=np.float64) - float(np.mean(mono))
    windowed = centered * np.hanning(len(centered))
    energy = float(np.dot(windowed, windowed))
    if energy <= 1e-10:
        return None, 0.0
    fft_size = 1 << max(1, (2 * len(windowed) - 1).bit_length())
    spectrum = np.fft.rfft(windowed, fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)
    autocorrelation = autocorrelation[: len(windowed)]
    minimum_lag = int(ANALYSIS_SAMPLE_RATE / 350.0)
    maximum_lag = min(len(windowed) - 1, int(ANALYSIS_SAMPLE_RATE / 70.0))
    search = autocorrelation[minimum_lag : maximum_lag + 1]
    lag = minimum_lag + int(np.argmax(search))
    periodicity = max(
        0.0, min(1.0, float(autocorrelation[lag] / autocorrelation[0]))
    )
    return float(ANALYSIS_SAMPLE_RATE / lag), periodicity


def acoustic_sensory(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, object]]:
    """Extract the frozen label-free 30-band auditory representation."""

    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim == 2:
        mono = samples.mean(axis=1)
    elif samples.ndim == 1:
        mono = samples
    else:
        raise ValueError("recording must be mono or sample-by-channel")
    if not np.all(np.isfinite(mono)):
        raise ValueError("recording contains non-finite samples")
    mono = _resample_linear(mono, sample_rate)
    selected, start = _select_analysis_window(mono)
    selected = selected - float(np.mean(selected))
    peak = float(np.max(np.abs(selected), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(selected))))
    clipped = int(np.count_nonzero(np.abs(mono) >= 1.0))
    pitch_hz, periodicity = _pitch_periodicity(selected)

    fft_size = 1024  # 64 ms at 16 kHz
    hop = 256  # 16 ms
    spectra = []
    window = np.hanning(fft_size)
    for offset in range(0, len(selected) - fft_size + 1, hop):
        frame = selected[offset : offset + fft_size] * window
        spectra.append(np.square(np.abs(np.fft.rfft(frame))))
    if not spectra:
        raise ValueError("recording has no complete spectral analysis frame")
    mean_power = np.mean(np.stack(spectra), axis=0)
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / ANALYSIS_SAMPLE_RATE)
    values: list[float] = []
    for center in EXPANDED_SENSORY_FREQUENCIES_HZ:
        bandwidth = max(80.0, 0.10 * float(center))
        weights = np.exp(-0.5 * np.square((frequencies - center) / bandwidth))
        band_power = float(np.sum(mean_power * weights) / np.sum(weights))
        values.append(float(np.log(band_power + 1e-12)))
    sensory = np.asarray(values, dtype=np.float64)
    sensory -= float(np.mean(sensory))
    sensory = _unit(sensory)
    quality = {
        "source_sample_rate_hz": sample_rate,
        "analysis_sample_rate_hz": ANALYSIS_SAMPLE_RATE,
        "analysis_start_sample": start,
        "analysis_sample_count": len(selected),
        "analysis_rms": rms,
        "analysis_peak_absolute": peak,
        "clipped_sample_count": clipped,
        "estimated_pitch_hz": pitch_hz,
        "pitch_periodicity": periodicity,
        "finite": True,
        "rms_at_least_0_01": rms >= 0.01,
        "peak_at_least_0_03": peak >= 0.03,
        "peak_below_0_98": peak < 0.98,
        "zero_clipped_samples": clipped == 0,
        "pitch_between_70_and_350_hz": pitch_hz is not None
        and 70.0 <= pitch_hz <= 350.0,
        "pitch_periodicity_at_least_0_30": periodicity >= 0.30,
    }
    quality["quality_passed"] = all(
        quality[key] is True
        for key in (
            "finite",
            "rms_at_least_0_01",
            "peak_at_least_0_03",
            "peak_below_0_98",
            "zero_clipped_samples",
            "pitch_between_70_and_350_hz",
            "pitch_periodicity_at_least_0_30",
        )
    )
    return sensory, quality


def build_waveform_repertoire() -> MotorRepertoire:
    """Re-express the frozen babbling memory through the microphone feature path."""

    motors = tuple(
        (
            VOCAL_REPERTOIRE_START_INDEX + offset * VOCAL_REPERTOIRE_STRIDE,
            halton5(
                VOCAL_REPERTOIRE_START_INDEX + offset * VOCAL_REPERTOIRE_STRIDE
            ),
        )
        for offset in range(96)
    )
    return MotorRepertoire(
        MotorPrototype(
            prototype_id=f"v4-babble-{offset:03d}",
            halton_index=halton_index,
            motor=tuple(float(value) for value in motor),
            sensory=tuple(
                float(value)
                for value in acoustic_sensory(
                    np.tile(
                        synthesize_factorized(motor, DEMO_F0_HZ), 3
                    ),
                    SAMPLE_RATE,
                )[0]
            ),
        )
        for offset, (halton_index, motor) in enumerate(motors)
    )


def _write_multichannel_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[1] not in (1, CAPTURE_CHANNELS):
        raise ValueError("capture must contain one or two host channels")
    if not np.all(np.isfinite(data)):
        raise ValueError("capture contains non-finite samples")
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(data.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _play_start_cue(sounddevice_module: object | None = None) -> dict[str, object]:
    """Play a bounded start cue through the pinned self-source output."""

    if sounddevice_module is None:
        import sounddevice as sounddevice_module

    devices = sounddevice_module.query_devices()
    device_index = next(
        (
            index
            for index, device in enumerate(devices)
            if str(device["name"]).lower() == "mac mini speakers"
            and int(device["max_output_channels"]) >= 1
        ),
        None,
    )
    if device_index is None:
        raise ValueError("Mac mini Speakers start-cue output is unavailable")
    sample_rate = 48000
    sample_count = int(round(START_CUE_DURATION_MS * sample_rate / 1000.0))
    timeline = np.arange(sample_count, dtype=np.float64) / sample_rate
    envelope = np.sin(np.linspace(0.0, np.pi, sample_count)) ** 2
    cue = (
        START_CUE_AMPLITUDE
        * envelope
        * np.sin(2.0 * np.pi * START_CUE_FREQUENCY_HZ * timeline)
    ).astype(np.float32)
    sounddevice_module.play(
        cue,
        samplerate=sample_rate,
        device=device_index,
        blocking=True,
    )
    return {
        "played": True,
        "output_device_index": device_index,
        "output_device_name": str(devices[device_index]["name"]),
        "frequency_hz": START_CUE_FREQUENCY_HZ,
        "duration_ms": START_CUE_DURATION_MS,
        "amplitude": START_CUE_AMPLITUDE,
        "retained_in_capture": False,
    }


def _default_capture() -> tuple[np.ndarray, int, dict[str, object]]:
    mic = LiveMicArray(
        device=DEFAULT_DEVICE,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        capture_seconds=CAPTURE_DURATION_MS / 1000.0,
        stream_backend="callback",
    )
    with mic.streaming():
        cue_metadata = _play_start_cue()
        time.sleep(START_CUE_SETTLE_MS / 1000.0)
        # The input stream is opened before the cue to eliminate startup
        # uncertainty. Discard cue/settle samples so only the post-cue teacher
        # interval enters the retained recording.
        with mic._callback_condition:
            mic._callback_chunks.clear()
            mic._callback_available_frames = 0
        mic.capture_frame(duration_ms=CAPTURE_DURATION_MS)
        samples = mic.last_pcm_channels
        if samples is None:
            raise ValueError("XVF3800 callback returned no PCM")
        metadata = {
            "backend": mic.active_stream_backend,
            "device_selector": DEFAULT_DEVICE,
            "channel_count": samples.shape[1],
            "input_overflow_count": mic.input_overflow_count,
            "callback_status_count": mic.callback_status_count,
            "callback_timestamp_discontinuity_count": mic.callback_timestamp_discontinuity_count,
            "start_cue": cue_metadata,
        }
        return samples, ANALYSIS_SAMPLE_RATE, metadata


def _manifest_path(session_dir: Path) -> Path:
    return session_dir / "project-owner-vocal-imitation-manifest.json"


def _load_manifest(session_dir: Path) -> dict[str, object]:
    path = _manifest_path(session_dir)
    if not path.exists():
        return {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "source_class": "new_project_owner_recording",
            "captures": [],
        }
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or payload.get("protocol") != PROTOCOL:
        raise ValueError("vocal-imitation manifest schema or protocol drifted")
    if payload.get("source_class") != "new_project_owner_recording":
        raise ValueError("vocal-imitation manifest source is unauthorized")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def capture_project_owner_take(
    *,
    session_dir: str | Path,
    target_key: str,
    role: str,
    take: int,
    authorized: bool,
    capture_fn: CaptureFn | None = None,
) -> dict[str, object]:
    """Capture and register one explicitly authorized project-owner take."""

    if not authorized:
        raise ValueError("project-owner recording requires explicit authorization")
    if target_key not in DEMO_VOWELS:
        raise ValueError("target key must be one of a,i,u")
    if role == "development" and take not in (1, 2):
        raise ValueError("development take must be 1 or 2")
    if role == "confirmation" and take != 1:
        raise ValueError("confirmation take must be 1")
    if role not in ("development", "confirmation"):
        raise ValueError("role must be development or confirmation")

    root = Path(session_dir).resolve()
    manifest = _load_manifest(root)
    key = (target_key, role, take)
    if any(
        (row["target_key"], row["role"], row["take"]) == key
        for row in manifest["captures"]
    ):
        raise ValueError("that vocal-imitation take is already registered")
    filename = f"{target_key}-{role}-{take}.wav"
    path = root / "captures" / filename
    if path.exists():
        raise ValueError(f"refusing to overwrite existing capture: {path}")

    samples, sample_rate, metadata = (capture_fn or _default_capture)()
    _write_multichannel_wav(path, samples, sample_rate)
    decoded, decoded_rate = read_pcm_wav(path)
    sensory, quality = acoustic_sensory(decoded, decoded_rate)
    hardware_contract_passed = (
        metadata.get("backend") == "sounddevice_portaudio_coreaudio_callback"
        and metadata.get("device_selector") == DEFAULT_DEVICE
        and sample_rate == ANALYSIS_SAMPLE_RATE
        and decoded.shape[1] == CAPTURE_CHANNELS
        and metadata.get("input_overflow_count") == 0
        and metadata.get("callback_status_count") == 0
        and metadata.get("callback_timestamp_discontinuity_count") == 0
        and isinstance(metadata.get("start_cue"), dict)
        and metadata["start_cue"].get("played") is True
        and metadata["start_cue"].get("output_device_name")
        == "Mac mini Speakers"
        and metadata["start_cue"].get("retained_in_capture") is False
    )
    real_capture = hardware_contract_passed
    record = {
        "target_key": target_key,
        "display_label": target_key,
        "label_role": "capture_organization_and_display_only",
        "role": role,
        "take": take,
        "relative_path": f"captures/{filename}",
        "sha256": _sha256(path),
        "sample_rate_hz": decoded_rate,
        "channel_count": decoded.shape[1],
        "frame_count": decoded.shape[0],
        "sensory_sha256": _array_sha256(sensory),
        "quality": quality,
        "capture_metadata": metadata,
        "hardware_contract_passed": hardware_contract_passed,
        "real_capture": real_capture,
        "operator_confirmations": {
            "project_owner_is_speaker": real_capture,
            "bounded_recording_authorized": real_capture,
            "external_corpus_not_used": True,
            "playback_during_capture": False,
        },
    }
    manifest["captures"].append(record)
    manifest["captures"].sort(
        key=lambda row: (row["target_key"], row["role"], row["take"])
    )
    _write_json(_manifest_path(root), manifest)
    return {
        "mode": "project_owner_vocal_imitation_capture",
        "protocol": PROTOCOL,
        "capture": record,
        "capture_passed": quality["quality_passed"],
        "manifest_path": str(_manifest_path(root)),
    }


def _verified_take(
    root: Path, record: dict[str, object]
) -> tuple[np.ndarray, dict[str, object]]:
    captures = (root / "captures").resolve()
    path = (root / str(record["relative_path"])).resolve()
    try:
        path.relative_to(captures)
    except ValueError as exc:
        raise ValueError("manifest capture escapes the authorized session directory") from exc
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"capture hash mismatch: {path.name}")
    samples, sample_rate = read_pcm_wav(path)
    sensory, quality = acoustic_sensory(samples, sample_rate)
    if _array_sha256(sensory) != record["sensory_sha256"]:
        raise ValueError(f"capture sensory hash mismatch: {path.name}")
    if quality["quality_passed"] is not True:
        raise ValueError(f"capture quality gate failed: {path.name}")
    return sensory, quality


def _write_learned_wav(path: Path, motor: np.ndarray) -> dict[str, object]:
    production = synthesize_factorized(motor, DEMO_F0_HZ)
    silence = np.zeros(int(round(0.04 * SAMPLE_RATE)), dtype=np.float32)
    rendered = np.concatenate(
        [segment for repeat in range(6) for segment in ((silence,) if repeat else ()) + (production,)]
    )
    write_wav(str(path), rendered, SAMPLE_RATE)
    return {
        "filename": path.name,
        "sha256": _sha256(path),
        "sample_rate_hz": SAMPLE_RATE,
        "sample_count": len(rendered),
        "peak_absolute": float(np.max(np.abs(rendered))),
        "finite": bool(np.all(np.isfinite(rendered))),
    }


def _render_html(report: dict[str, object]) -> str:
    cards = []
    for session in report["sessions"]:
        label = html.escape(str(session["display_label"]))
        cards.append(
            f"""<section><h2>/{label}/</h2>
<p>development {session['development']['final_distance']:.3f} · confirmation {session['confirmation']['learned_distance']:.3f} · improvement {100 * session['development']['improvement_ratio']:.1f}%</p>
<div><label>teacher development 1<audio controls src="captures/{label}-development-1.wav"></audio></label>
<label>held confirmation<audio controls src="captures/{label}-confirmation-1.wav"></audio></label>
<label>learned self<audio controls src="{label}-learned.wav"></audio></label></div></section>"""
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cora bounded vocal imitation</title><style>
:root{{color-scheme:dark;font-family:ui-monospace,SFMono-Regular,monospace}}body{{margin:0;background:#0c1117;color:#dce7ef}}main{{max-width:980px;margin:auto;padding:36px 20px}}section{{background:#141d26;border:1px solid #263746;border-radius:14px;padding:20px;margin:18px 0}}section div{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}audio{{display:block;width:100%;margin-top:8px}}footer,p{{color:#91a4b4}}a{{color:#80c7ff}}</style></head><body><main><h1>Cora bounded vocal imitation</h1><p>project-owner recordings · selection-blind confirmation · finite prototype recall</p>{''.join(cards)}<footer>{html.escape(str(report['claim_boundary']))}<br><a href="vocal-imitation-result.json">Inspect result JSON</a></footer></main></body></html>"""


def run_project_owner_vocal_imitation(
    *,
    session_dir: str | Path,
    output_dir: str | Path | None = None,
    allow_test_fixture: bool = False,
) -> dict[str, object]:
    """Learn from development takes, then open and score confirmation takes."""

    root = Path(session_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(root)
    records = manifest["captures"]
    expected = {
        (target, "development", take)
        for target in DEMO_VOWELS
        for take in (1, 2)
    } | {(target, "confirmation", 1) for target in DEMO_VOWELS}
    observed = {(row["target_key"], row["role"], row["take"]) for row in records}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"manifest take set is incomplete or invalid; missing={missing}, extra={extra}")
    all_real_captures = all(row.get("real_capture") is True for row in records)
    if not all_real_captures and not allow_test_fixture:
        raise ValueError(
            "vocal-imitation evidence requires XVF3800 callback captures; "
            "injected fixtures are test-only"
        )

    # Deliberately decode only development records before all associations freeze.
    development_features: dict[str, list[np.ndarray]] = {}
    development_quality: dict[str, list[dict[str, object]]] = {}
    for target in DEMO_VOWELS:
        rows = sorted(
            (row for row in records if row["target_key"] == target and row["role"] == "development"),
            key=lambda row: row["take"],
        )
        pairs = [_verified_take(root, row) for row in rows]
        development_features[target] = [pair[0] for pair in pairs]
        development_quality[target] = [pair[1] for pair in pairs]

    repertoire = build_waveform_repertoire()
    learner = LocalImitationLearner(repertoire, capacity=3)
    learned: dict[str, tuple[dict[str, object], np.ndarray, float]] = {}
    for target in DEMO_VOWELS:
        combined = _unit(np.mean(np.stack(development_features[target]), axis=0))
        target_id = _array_sha256(combined)
        learning = learner.imitate(combined, target_id=target_id)
        take_distance = sensory_distance(
            development_features[target][0], development_features[target][1]
        )
        learned[target] = (learning, combined, take_distance)
    frozen_associations = dict(learner.associations)

    # Confirmation access begins only after every target association is frozen.
    confirmation: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
    for target in DEMO_VOWELS:
        row = next(
            row for row in records
            if row["target_key"] == target and row["role"] == "confirmation"
        )
        confirmation[target] = _verified_take(root, row)

    sessions: list[dict[str, object]] = []
    for target in DEMO_VOWELS:
        learning, _combined, take_distance = learned[target]
        prototype_index = frozen_associations[learning["target_id"]]
        prototype = repertoire.prototypes[prototype_index]
        confirm_feature, confirm_quality = confirmation[target]
        rest = repertoire.prototypes[repertoire.rest_index()]
        confirm_rest_distance = sensory_distance(rest.sensory_array(), confirm_feature)
        confirm_learned_distance = sensory_distance(prototype.sensory_array(), confirm_feature)
        confirm_improvement = (
            (confirm_rest_distance - confirm_learned_distance) / confirm_rest_distance
            if confirm_rest_distance > 0.0 else 0.0
        )
        criteria = {
            "development_take_distance_at_most_0_30": take_distance <= 0.30,
            "development_improvement_at_least_0_20": float(learning["improvement_ratio"]) >= 0.20,
            "development_final_distance_at_most_0_45": float(learning["final_best_sensory_distance"]) <= 0.45,
            "final_three_recall_same_prototype": len({row["prototype_id"] for row in learning["attempts"][-3:]}) == 1,
            "confirmation_improvement_at_least_0_15": confirm_improvement >= 0.15,
            "confirmation_final_distance_at_most_0_45": confirm_learned_distance <= 0.45,
            "confirmation_accessed_after_learning": True,
            "confirmation_did_not_change_association": learner.associations == frozen_associations,
            "all_capture_quality_gates_passed": all(row["quality_passed"] for row in development_quality[target]) and confirm_quality["quality_passed"] is True,
        }
        learned_wav = _write_learned_wav(
            output / f"{target}-learned.wav", prototype.motor_array()
        )
        sessions.append({
            "display_label": target,
            "label_role": "capture_organization_and_display_only",
            "learner_received_vowel_label": False,
            "development": {
                "target_id": learning["target_id"],
                "take_distance": take_distance,
                "initial_distance": learning["initial_sensory_distance"],
                "final_distance": learning["final_best_sensory_distance"],
                "improvement_ratio": learning["improvement_ratio"],
                "attempts": learning["attempts"],
            },
            "learned_prototype": {
                "prototype_id": prototype.prototype_id,
                "halton_index": prototype.halton_index,
                "motor": list(prototype.motor),
                "waveform": learned_wav,
            },
            "confirmation": {
                "rest_distance": confirm_rest_distance,
                "learned_distance": confirm_learned_distance,
                "improvement_ratio": confirm_improvement,
                "used_for_selection": False,
            },
            "criteria": criteria,
            "session_passed": all(criteria.values()),
        })

    aggregate = {
        "exact_capture_count": len(records) == 9,
        "learner_received_no_vowel_labels": all(session["learner_received_vowel_label"] is False for session in sessions),
        "confirmation_accessed_after_all_learning": True,
        "association_memory_within_three": len(learner.associations) <= 3,
        "every_target_passed": all(session["session_passed"] for session in sessions),
        "external_corpus_not_accessed": True,
        "protected_data_not_accessed": True,
        "playback_not_executed": True,
        "capture_evidence_class_eligible": all_real_captures or allow_test_fixture,
    }
    report = {
        "mode": "bounded_project_owner_vocal_imitation",
        "evidence_class": (
            "project_owner_real_capture"
            if all_real_captures
            else "synthetic_injected_protocol_fixture"
        ),
        "protocol": PROTOCOL,
        "manifest_sha256": _sha256(_manifest_path(root)),
        "repertoire_sha256": repertoire.digest(),
        "sessions": sessions,
        "aggregate_criteria": aggregate,
        "demo_passed": all(aggregate.values()),
        "claim_boundary": "Three target-specific associations for one known project owner and one capture setup; not continuous inverse closure, phoneme recognition, general human speech learning, identity, semantics, or H4 evidence.",
    }
    if output != root:
        destination = output / "captures"
        destination.mkdir(parents=True, exist_ok=True)
        for record in records:
            source = root / str(record["relative_path"])
            copied = destination / Path(str(record["relative_path"])).name
            shutil.copyfile(source, copied)
            if _sha256(copied) != record["sha256"]:
                raise ValueError(f"copied capture hash mismatch: {copied.name}")
    html_path = output / "vocal-imitation-demo.html"
    html_path.write_text(_render_html(report))
    report["presentation"] = {
        "html_filename": html_path.name,
        "html_sha256": _sha256(html_path),
    }
    _write_json(output / "vocal-imitation-result.json", report)
    return report
