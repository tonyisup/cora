"""Product-session V2.2 with a frozen pre-development self calibration.

V2.2 preserves the V2.1 behavioral and scoring contract.  It adds one
machine-generated self-path calibration capture before owner targets, freezes
the resulting causal FIR before development, and supplies a versioned durable
state machine that exercises every held-response prerequisite.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
import re
import threading
import time
import wave
from typing import Any

import numpy as np

from .drained_playback import AudioToolboxOutput
from .drained_playback_v2_1 import (
    PLAYER_VERSION,
    SELF_DEVICE_NAME,
    SELF_DEVICE_UID,
)
from .factorized_tract_plant import synthesize_factorized
from .io.live_mic import CAPTURE_CHANNELS, DEFAULT_DEVICE, LiveMicArray
from .io.motor_plan import synthesize_chirp, write_wav
from .io.wav_frames import read_pcm_wav
from .live_speak_smoke import estimate_playback_latency_ms
from .product_session_live import (
    CAPTURE_DURATION_MS,
    DEVELOPMENT_SCHEDULE,
    HELD_SCHEDULE,
    PRE_ROLL_MS,
    RETENTION_DAYS,
    TARGETS,
    build_prediction_commitment,
    build_self_model,
    classify_transport_integrity,
    evaluate_committed_selection,
    predict_self_capture,
    render_selected_prototype,
    score_held_trial,
    sha256_json,
    sha256_path,
)
from .product_session_v2 import (
    commit_v2_label_blind_selection,
    freeze_v2_development_associations,
)
from .product_session_v2_1 import (
    build_v2_1_read_only_preflight,
    v2_1_playback_callback,
    v2_1_unchanged_behavioral_contract,
)
from .speech_learning_demo import DEMO_F0_HZ
from .vocal_imitation import ANALYSIS_SAMPLE_RATE, build_waveform_repertoire


PROTOCOL_VERSION = "bounded-product-session-v2.2-candidate"
SELF_CALIBRATION_TRIAL_ID = "calibration-01-self-chirp"
SELF_CALIBRATION_ROLE = "self_calibration"
SELF_CALIBRATION_CAPTURE_MS = 1_000
SELF_CALIBRATION_CHIRP_MS = 120
SELF_CALIBRATION_CHIRP_F0_HZ = 250.0
SELF_CALIBRATION_CHIRP_F1_HZ = 3_500.0
SELF_CALIBRATION_PEAK = 0.08
SELF_CALIBRATION_MIN_CORRELATION = 0.10
SELF_CALIBRATION_CAUSAL_WINDOW_MS = 500.0
FIR_SAMPLES = 128
_SESSION_ID = re.compile(r"[a-z0-9][a-z0-9-]{7,79}")


def _array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_multichannel_wav(path: Path, samples: np.ndarray) -> None:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[1] != CAPTURE_CHANNELS:
        raise ValueError("self calibration requires two captured host channels")
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CAPTURE_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(ANALYSIS_SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def build_self_calibration_stimulus() -> np.ndarray:
    """Return the exact bounded 16 kHz self-path calibration chirp."""

    chirp = synthesize_chirp(
        SELF_CALIBRATION_CHIRP_F0_HZ,
        SELF_CALIBRATION_CHIRP_F1_HZ,
        duration_ms=SELF_CALIBRATION_CHIRP_MS,
        sample_rate=ANALYSIS_SAMPLE_RATE,
    )
    return (np.float32(SELF_CALIBRATION_PEAK) * chirp).astype(np.float32)


def analyze_self_calibration(
    known_chirp: np.ndarray,
    captured: np.ndarray,
    *,
    expected_onset_frame: int,
    transport_integrity: Mapping[str, object],
) -> dict[str, object]:
    """Align one preregistered chirp and freeze a bounded causal self model."""

    known = np.asarray(known_chirp, dtype=np.float64).reshape(-1)
    capture = np.asarray(captured, dtype=np.float64)
    if capture.ndim == 2:
        if capture.shape[1] != CAPTURE_CHANNELS:
            raise ValueError("self calibration capture must have two host channels")
        mono = np.mean(capture, axis=1)
    elif capture.ndim == 1:
        mono = capture
    else:
        raise ValueError("self calibration capture must be mono or frame-by-channel")
    if len(known) != SELF_CALIBRATION_CHIRP_MS * ANALYSIS_SAMPLE_RATE // 1_000:
        raise ValueError("self calibration chirp has the wrong frozen duration")
    if expected_onset_frame < 0:
        raise ValueError("expected self-calibration onset must be non-negative")
    if not np.all(np.isfinite(known)) or not np.all(np.isfinite(mono)):
        raise ValueError("self calibration contains non-finite samples")
    if float(np.max(np.abs(known), initial=0.0)) > SELF_CALIBRATION_PEAK + 1e-6:
        raise ValueError("self calibration chirp exceeds the frozen peak")
    if transport_integrity.get("technical_valid") is not True:
        raise ValueError("self calibration transport integrity failed")
    capture_peak = float(np.max(np.abs(mono), initial=0.0))
    if int(np.count_nonzero(np.abs(mono) >= 32767.0 / 32768.0)) != 0:
        raise ValueError("self calibration capture contains clipped samples")
    if capture_peak >= 0.98:
        raise ValueError("self calibration capture peak is too high")

    onset_ms = estimate_playback_latency_ms(
        efference_pcm=known,
        captured_pcm=mono,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        expected_onset_ms=expected_onset_frame * 1000.0 / ANALYSIS_SAMPLE_RATE,
        causal_window_ms=SELF_CALIBRATION_CAUSAL_WINDOW_MS,
        min_peak_correlation=SELF_CALIBRATION_MIN_CORRELATION,
    )
    if onset_ms is None:
        raise ValueError("self calibration chirp was not recovered")
    onset_frame = int(round(onset_ms * ANALYSIS_SAMPLE_RATE / 1000.0))
    required = len(known) + FIR_SAMPLES - 1
    if onset_frame < 0 or onset_frame + required > len(mono):
        raise ValueError("self calibration FIR window is outside the capture")
    observed = mono[onset_frame : onset_frame + required]
    observed_rms = float(np.sqrt(np.mean(np.square(observed))))
    if observed_rms < 0.001:
        raise ValueError("self calibration chirp capture is too quiet")
    model = build_self_model(known, observed)
    impulse = np.asarray(model["impulse"], dtype=np.float64)
    if impulse.shape != (FIR_SAMPLES,) or not np.all(np.isfinite(impulse)):
        raise ValueError("self calibration produced an invalid impulse")
    frozen_model = {
        **model,
        "protocol_version": PROTOCOL_VERSION,
        "calibration_trial_id": SELF_CALIBRATION_TRIAL_ID,
        "calibration_stimulus_sha256": _array_digest(known),
        "calibration_capture_sha256": _array_digest(capture),
        "expected_onset_frame": expected_onset_frame,
        "recovered_onset_frame": onset_frame,
        "recovered_onset_ms": onset_ms,
        "frozen_before_development_capture": True,
        "owner_vocal_audio_used": False,
        "post_response_data_used": False,
    }
    frozen_model["self_model_payload_sha256"] = sha256_json(frozen_model)
    return {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "calibration_passed": True,
        "quality": {
            "finite": True,
            "zero_clipped_samples": True,
            "chirp_recovered": True,
            "recovered_onset_frame": onset_frame,
            "recovered_onset_ms": onset_ms,
            "capture_peak_absolute": capture_peak,
            "observed_chirp_rms": observed_rms,
        },
        "self_model": frozen_model,
    }


def capture_self_calibration_live(
    capture_path: str | Path,
    *,
    stimulus_path: str | Path,
    playback: Callable[..., Mapping[str, object]],
) -> dict[str, object]:
    """Capture the one scheduled self-calibration chirp before development."""

    stimulus, rate = read_pcm_wav(stimulus_path)
    known = np.mean(stimulus, axis=1)
    if rate != ANALYSIS_SAMPLE_RATE or not np.allclose(
        known,
        build_self_calibration_stimulus(),
        atol=1.0 / 32768.0,
    ):
        raise ValueError("self calibration stimulus bytes do not match the contract")
    mic = LiveMicArray(
        device=DEFAULT_DEVICE,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        capture_seconds=SELF_CALIBRATION_CAPTURE_MS / 1000.0,
        stream_backend="callback",
    )
    failure: list[BaseException] = []

    def capture() -> None:
        try:
            mic.capture_frame(duration_ms=SELF_CALIBRATION_CAPTURE_MS)
        except BaseException as exc:
            failure.append(exc)

    with mic.streaming():
        capture_origin = mic.discard_buffered_callback_audio()
        worker = threading.Thread(target=capture, daemon=True)
        worker.start()
        time.sleep(PRE_ROLL_MS / 1000.0)
        playback_onset = mic.callback_stream_frame_position()
        playback_report = dict(playback(str(stimulus_path)))
        playback_end = mic.callback_stream_frame_position()
        worker.join(timeout=SELF_CALIBRATION_CAPTURE_MS / 1000.0 + 2.0)
        if worker.is_alive():
            raise RuntimeError("self calibration capture thread did not finish")
        if failure:
            raise RuntimeError(f"self calibration capture failed: {failure[0]}")
        samples = mic.last_pcm_channels
        if samples is None:
            raise RuntimeError("self calibration callback returned no PCM")
        telemetry = {
            "backend": mic.active_stream_backend,
            "device_selector": DEFAULT_DEVICE,
            "sample_rate_hz": ANALYSIS_SAMPLE_RATE,
            "channel_count": int(samples.shape[1]),
            "sample_count": int(samples.shape[0]),
            "input_overflow_count": mic.input_overflow_count,
            "callback_status_count": mic.callback_status_count,
            "callback_timestamp_discontinuity_count": (
                mic.callback_timestamp_discontinuity_count
            ),
            "capture_peak_absolute": mic.capture_peak_absolute,
            "clipped_sample_count": mic.clipped_sample_count,
            "capture_origin_stream_frame": capture_origin,
            "playback_onset_stream_frame": playback_onset,
            "playback_end_stream_frame": playback_end,
            "capture_end_stream_frame": capture_origin + len(samples),
            "pre_roll_frames": playback_onset - capture_origin,
            "capture_active_through_playback_completion": (
                playback_end <= capture_origin + len(samples)
            ),
            "playback": playback_report,
            "playback_wav_sha256": sha256_path(stimulus_path),
        }
    _write_multichannel_wav(Path(capture_path), samples)
    integrity = classify_transport_integrity(
        telemetry,
        expected_sample_count=SELF_CALIBRATION_CAPTURE_MS
        * ANALYSIS_SAMPLE_RATE
        // 1_000,
    )
    if telemetry["pre_roll_frames"] < PRE_ROLL_MS * ANALYSIS_SAMPLE_RATE // 1_000:
        integrity["technical_valid"] = False
        integrity["failure_reasons"].append("pre_roll_shorter_than_250ms")
    if telemetry["capture_active_through_playback_completion"] is not True:
        integrity["technical_valid"] = False
        integrity["failure_reasons"].append("capture_did_not_span_playback")
    telemetry["transport_integrity"] = integrity
    analysis = analyze_self_calibration(
        known,
        samples,
        expected_onset_frame=playback_onset - capture_origin,
        transport_integrity=integrity,
    )
    return {
        "telemetry": telemetry,
        "analysis": analysis,
        "capture_wav_sha256": sha256_path(capture_path),
    }


def v2_2_planned_capture_schedule() -> tuple[dict[str, str], ...]:
    calibration = ({
        "trial_id": SELF_CALIBRATION_TRIAL_ID,
        "phase": "calibration",
        "label": "machine_chirp",
        "role": SELF_CALIBRATION_ROLE,
    },)
    development = tuple(
        {
            "trial_id": f"development-{index:02d}-{label}",
            "phase": "development",
            "label": label,
            "role": "development_target",
        }
        for index, label in enumerate(DEVELOPMENT_SCHEDULE, 1)
    )
    held = tuple(
        row
        for index, label in enumerate(HELD_SCHEDULE, 1)
        for row in (
            {
                "trial_id": f"held-{index:02d}-{label}",
                "phase": "held",
                "label": label,
                "role": "held_target",
            },
            {
                "trial_id": f"held-{index:02d}-{label}",
                "phase": "held",
                "label": label,
                "role": "held_response",
            },
        )
    )
    return calibration + development + held


class V2_2AppendOnlyAttemptLog:
    """Durable 19-capture V2.2 ledger with explicit completion gates."""

    def __init__(self, path: str | Path, *, session_id: str) -> None:
        if _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("V2.2 session ID must be storage-safe")
        self.path = Path(path)
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        result = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("V2.2 attempt log contains a non-object row")
                result.append(value)
        return result

    def _append(self, value: Mapping[str, object]) -> dict[str, object]:
        row = dict(value)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return row

    def append_open(self, *, trial_id: str, role: str) -> dict[str, object]:
        allowed = {
            SELF_CALIBRATION_ROLE,
            "development_target",
            "held_target",
            "held_response",
        }
        if role not in allowed:
            raise ValueError("unknown V2.2 attempt role")
        rows = self.rows()
        prior = [
            row
            for row in rows
            if row.get("trial_id") == trial_id
            and row.get("role") == role
            and row.get("state") == "opened_before_io"
        ]
        aborted = {
            row.get("attempt_number")
            for row in rows
            if row.get("trial_id") == trial_id
            and row.get("role") == role
            and row.get("state") == "technical_abort"
        }
        if prior and not (len(prior) == 1 and prior[0].get("attempt_number") in aborted):
            raise ValueError("V2.2 planned trial role is already consumed")
        attempt_number = sum(row.get("state") == "opened_before_io" for row in rows) + 1
        return self._append(
            {
                "event_number": len(rows) + 1,
                "attempt_number": attempt_number,
                "session_id": self.session_id,
                "trial_id": trial_id,
                "role": role,
                "state": "opened_before_io",
            }
        )

    def append_technical_abort(
        self,
        *,
        trial_id: str,
        role: str,
        accepted_audio_samples: int,
        partial_wav_created: bool,
        reason: str,
    ) -> dict[str, object]:
        if accepted_audio_samples != 0 or partial_wav_created is not False:
            raise ValueError("technical abort requires zero samples and no partial WAV")
        rows = self.rows()
        opens = [
            row
            for row in rows
            if row.get("trial_id") == trial_id
            and row.get("role") == role
            and row.get("state") == "opened_before_io"
        ]
        aborts = [
            row
            for row in rows
            if row.get("trial_id") == trial_id
            and row.get("role") == role
            and row.get("state") == "technical_abort"
        ]
        if len(opens) != 1 or aborts or not reason.strip():
            raise ValueError("technical abort is unavailable for this V2.2 trial")
        return self._append(
            {
                "event_number": len(rows) + 1,
                "attempt_number": opens[0]["attempt_number"],
                "session_id": self.session_id,
                "trial_id": trial_id,
                "role": role,
                "state": "technical_abort",
                "accepted_audio_samples": 0,
                "partial_wav_created": False,
                "reason": reason,
            }
        )

    @staticmethod
    def _load_json(path: Path) -> Mapping[str, object]:
        if not path.is_file():
            raise ValueError(f"required V2.2 artifact is missing: {path.name}")
        value = json.loads(path.read_text())
        if not isinstance(value, Mapping):
            raise ValueError(f"required V2.2 artifact is invalid: {path.name}")
        return value

    def open_next(self, session_root: str | Path) -> dict[str, object]:
        root = Path(session_root)
        plan = v2_2_planned_capture_schedule()
        rows = self.rows()
        aborted = {
            row.get("attempt_number")
            for row in rows
            if row.get("state") == "technical_abort"
        }
        effective = [
            row
            for row in rows
            if row.get("state") == "opened_before_io"
            and row.get("attempt_number") not in aborted
        ]
        if len(effective) >= len(plan):
            raise ValueError("all nineteen V2.2 captures are consumed")
        for index, row in enumerate(effective):
            expected = plan[index]
            if row.get("trial_id") != expected["trial_id"] or row.get("role") != expected["role"]:
                raise ValueError("V2.2 attempt log diverges from the frozen schedule")
            completion = self._load_json(
                root
                / expected["phase"]
                / expected["trial_id"]
                / f"{expected['role']}-completion.json"
            )
            if completion.get("technical_valid") is not True:
                raise ValueError("technical-invalid V2.2 session cannot continue")
            if completion.get("gate_passed") is not True:
                raise ValueError("failed V2.2 capture gate cannot continue")
            if not isinstance(completion.get("accepted_audio_samples"), int) or int(
                completion["accepted_audio_samples"]
            ) <= 0:
                raise ValueError("completed V2.2 capture lacks accepted audio")

        current = plan[len(effective)]
        if current["phase"] in {"development", "held"}:
            model_path = root / "self-model.json"
            if not model_path.is_file():
                raise ValueError(
                    "development is blocked until the V2.2 self model freezes"
                )
            model = self._load_json(model_path)
            if (
                model.get("protocol_version") != PROTOCOL_VERSION
                or model.get("frozen_before_development_capture") is not True
            ):
                raise ValueError("development is blocked until the V2.2 self model freezes")
        if current["phase"] == "held":
            frozen = self._load_json(root / "frozen-vocal-associations.json")
            key = self._load_json(root / "evaluation-key.json")
            if (
                frozen.get("protocol_version") != PROTOCOL_VERSION
                or frozen.get("frozen_before_held_capture") is not True
                or key.get("frozen_associations_sha256") != sha256_json(frozen)
            ):
                raise ValueError("held capture is blocked until the V2.2 association freeze")
        if current["role"] == "held_response":
            trial_dir = root / "held" / current["trial_id"]
            selection = self._load_json(trial_dir / "selection-commitment.json")
            verdict = self._load_json(trial_dir / "selection-verdict.json")
            prediction = self._load_json(trial_dir / "prediction-commitment.json")
            if selection.get("selection_and_contrastive_passed") is not True:
                raise ValueError("V2.2 response blocked by selection gates")
            if verdict.get("correct_association") is not True:
                raise ValueError("V2.2 response blocked by association verdict")
            if (
                prediction.get("protocol_version") != PROTOCOL_VERSION
                or prediction.get("committed_before_response_capture") is not True
                or prediction.get("self_model_frozen_before_development") is not True
            ):
                raise ValueError("V2.2 response blocked by prediction commitment")
            for name, key_name in (
                ("agent_emitted.wav", "agent_emitted_sha256"),
                ("self_prediction.wav", "self_prediction_sha256"),
            ):
                path = trial_dir / name
                if not path.is_file() or prediction.get(key_name) != sha256_path(path):
                    raise ValueError("V2.2 prediction artifact changed before response")
        return self.append_open(trial_id=current["trial_id"], role=current["role"])


def build_v2_2_completion_record(
    *,
    accepted_audio_samples: int,
    technical_valid: bool,
    gate_passed: bool,
) -> dict[str, object]:
    if accepted_audio_samples <= 0:
        raise ValueError("V2.2 completion requires accepted audio")
    return {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "accepted_audio_samples": accepted_audio_samples,
        "technical_valid": technical_valid,
        "gate_passed": gate_passed,
        "retry_permitted": False,
    }


def freeze_v2_2_development_associations(
    captures: Mapping[str, Sequence[tuple[str, np.ndarray, int]]],
) -> tuple[dict[str, object], dict[str, object]]:
    frozen, key = freeze_v2_development_associations(captures)
    frozen["protocol_version"] = PROTOCOL_VERSION
    frozen["v2_2_behavioral_or_score_threshold_changed"] = False
    key["protocol_version"] = PROTOCOL_VERSION
    key["frozen_associations_sha256"] = sha256_json(frozen)
    return frozen, key


def commit_v2_2_label_blind_selection(
    *,
    target_waveform: np.ndarray,
    sample_rate: int,
    frozen: Mapping[str, object],
) -> dict[str, object]:
    if frozen.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("V2.2 selection requires a V2.2 association freeze")
    commitment = commit_v2_label_blind_selection(
        target_waveform=target_waveform,
        sample_rate=sample_rate,
        frozen=frozen,
    )
    commitment["protocol_version"] = PROTOCOL_VERSION
    return commitment


def build_v2_2_prediction_commitment(
    *,
    agent_emitted_path: str | Path,
    self_prediction_path: str | Path,
    self_model: Mapping[str, object],
) -> dict[str, object]:
    if (
        self_model.get("protocol_version") != PROTOCOL_VERSION
        or self_model.get("frozen_before_development_capture") is not True
    ):
        raise ValueError("V2.2 prediction requires the pre-development self model")
    commitment = build_prediction_commitment(
        agent_emitted_path=agent_emitted_path,
        self_prediction_path=self_prediction_path,
        self_model=self_model,
    )
    commitment.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "self_model_sha256": sha256_json(self_model),
            "self_model_frozen_before_development": True,
            "self_calibration_trial_id": SELF_CALIBRATION_TRIAL_ID,
        }
    )
    return commitment


def build_v2_2_read_only_preflight(
    operating_point: Mapping[str, object],
    *,
    devices: Sequence[AudioToolboxOutput],
    system_volume_percent: int,
    output_muted: bool,
    player_inspection: Mapping[str, object],
) -> dict[str, object]:
    report = build_v2_1_read_only_preflight(
        operating_point,
        devices=devices,
        system_volume_percent=system_volume_percent,
        output_muted=output_muted,
        player_inspection=player_inspection,
    )
    calibration = operating_point.get("self_calibration")
    checks = dict(report["checks"])
    checks.update(
        {
            "self_calibration_is_mapping": isinstance(calibration, Mapping),
            "self_calibration_precedes_development": isinstance(calibration, Mapping)
            and calibration.get("frozen_before_development_capture_required") is True,
            "self_calibration_capture_is_1000ms": isinstance(calibration, Mapping)
            and calibration.get("capture_duration_ms") == SELF_CALIBRATION_CAPTURE_MS,
            "self_calibration_chirp_is_exact": isinstance(calibration, Mapping)
            and calibration.get("chirp_duration_ms") == SELF_CALIBRATION_CHIRP_MS
            and calibration.get("chirp_peak_absolute") == SELF_CALIBRATION_PEAK,
        }
    )
    return {
        **report,
        "protocol_version": PROTOCOL_VERSION,
        "preflight_kind": "v2_2_read_only_route_player_and_self_calibration_contract",
        "checks": checks,
        "preflight_passed": all(checks.values()),
    }


def build_v2_2_session_authorization(
    *,
    session_id: str,
    cohort_id: str,
    ordered_slot_index: int,
    authorization_text: str,
    authorization_date: str,
) -> dict[str, object]:
    if _SESSION_ID.fullmatch(session_id) is None or _SESSION_ID.fullmatch(cohort_id) is None:
        raise ValueError("V2.2 authorization IDs must be storage-safe")
    if not session_id.startswith(cohort_id + "-"):
        raise ValueError("V2.2 session is not namespaced by its cohort")
    if ordered_slot_index not in range(1, 6):
        raise ValueError("V2.2 slot index must be 1 through 5")
    date.fromisoformat(authorization_date)
    if not authorization_text.strip():
        raise ValueError("V2.2 authorization text is required")
    return {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "cohort_id": cohort_id,
        "session_id": session_id,
        "ordered_slot_index": ordered_slot_index,
        "authorization_date": authorization_date,
        "authorization_text": authorization_text,
        "owner_is_sole_speaker": True,
        "owner_may_stop_at_any_time": True,
        "machine_self_calibration_capture_authorized": True,
        "owner_target_recording_authorized": True,
        "playback_authorized": True,
        "one_session_live_io_authorized": True,
        "authorization_applies_to_other_slots": False,
        "retention_days": RETENTION_DAYS,
    }


def v2_2_playback_callback(**kwargs: object) -> Callable[..., Mapping[str, object]]:
    return v2_1_playback_callback(**kwargs)


def v2_2_behavioral_contract() -> dict[str, object]:
    return {
        **v2_1_unchanged_behavioral_contract(),
        "self_calibration_captures": 1,
        "owner_target_and_response_captures": 18,
        "total_microphone_captures_per_session": 19,
        "self_model_frozen_before_development_required": True,
        "behavioral_or_score_threshold_changed_from_v2_1": False,
    }


def _synthetic_prototype_waveform(prototype_index: int) -> np.ndarray:
    repertoire = build_waveform_repertoire()
    prototype = repertoire.prototypes[prototype_index]
    waveform = np.tile(
        synthesize_factorized(prototype.motor_array(), DEMO_F0_HZ),
        13,
    )
    peak = float(np.max(np.abs(waveform), initial=0.0))
    return (waveform * (0.095 / peak)).astype(np.float32)


def run_v2_2_deterministic_dry_run(root: str | Path) -> dict[str, object]:
    """Exercise all 19 state transitions without device access."""

    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    ledger = V2_2AppendOnlyAttemptLog(
        base / "attempts.jsonl",
        session_id="bvips-v2-2-dry-run-session-01",
    )
    known = build_self_calibration_stimulus()
    capture = np.zeros(ANALYSIS_SAMPLE_RATE, dtype=np.float32)
    onset = PRE_ROLL_MS * ANALYSIS_SAMPLE_RATE // 1_000
    capture[onset : onset + len(known)] = known
    opened = ledger.open_next(base)
    assert opened["role"] == SELF_CALIBRATION_ROLE
    analysis = analyze_self_calibration(
        known,
        capture,
        expected_onset_frame=onset,
        transport_integrity={"technical_valid": True},
    )
    _write_json_atomic(base / "self-model.json", analysis["self_model"])
    calibration_dir = base / "calibration" / SELF_CALIBRATION_TRIAL_ID
    _write_json_atomic(
        calibration_dir / f"{SELF_CALIBRATION_ROLE}-completion.json",
        build_v2_2_completion_record(
            accepted_audio_samples=len(capture),
            technical_valid=True,
            gate_passed=True,
        ),
    )

    self_model = analysis["self_model"]
    waveforms = {
        label: predict_self_capture(_synthetic_prototype_waveform(index), self_model)
        for label, index in zip(TARGETS, (16, 70, 78))
    }
    development_captures: dict[str, list[tuple[str, np.ndarray, int]]] = {
        label: [] for label in TARGETS
    }
    for index, label in enumerate(DEVELOPMENT_SCHEDULE, 1):
        trial_id = f"development-{index:02d}-{label}"
        opened = ledger.open_next(base)
        assert opened["trial_id"] == trial_id
        trial_dir = base / "development" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        path = trial_dir / "target_captured.wav"
        write_wav(str(path), waveforms[label], ANALYSIS_SAMPLE_RATE)
        development_captures[label].append((path.name, waveforms[label], ANALYSIS_SAMPLE_RATE))
        _write_json_atomic(
            trial_dir / "development_target-completion.json",
            build_v2_2_completion_record(
                accepted_audio_samples=len(waveforms[label]),
                technical_valid=True,
                gate_passed=True,
            ),
        )

    frozen, key = freeze_v2_2_development_associations(development_captures)
    _write_json_atomic(base / "frozen-vocal-associations.json", frozen)
    _write_json_atomic(base / "evaluation-key.json", key)
    held_passes = []
    for index, label in enumerate(HELD_SCHEDULE, 1):
        trial_id = f"held-{index:02d}-{label}"
        opened = ledger.open_next(base)
        assert opened["role"] == "held_target"
        trial_dir = base / "held" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        target_path = trial_dir / "target_captured.wav"
        write_wav(str(target_path), waveforms[label], ANALYSIS_SAMPLE_RATE)
        _write_json_atomic(
            trial_dir / "held_target-completion.json",
            build_v2_2_completion_record(
                accepted_audio_samples=len(waveforms[label]),
                technical_valid=True,
                gate_passed=True,
            ),
        )
        selection = commit_v2_2_label_blind_selection(
            target_waveform=waveforms[label],
            sample_rate=ANALYSIS_SAMPLE_RATE,
            frozen=frozen,
        )
        verdict = evaluate_committed_selection(selection, key, scheduled_label=label)
        _write_json_atomic(trial_dir / "selection-commitment.json", selection)
        _write_json_atomic(trial_dir / "selection-verdict.json", verdict)
        emitted = render_selected_prototype(frozen, str(selection["selected_opaque_id"]))
        prediction = predict_self_capture(emitted, self_model)
        emitted_path = trial_dir / "agent_emitted.wav"
        prediction_path = trial_dir / "self_prediction.wav"
        write_wav(str(emitted_path), emitted, ANALYSIS_SAMPLE_RATE)
        write_wav(str(prediction_path), prediction, ANALYSIS_SAMPLE_RATE)
        commitment = build_v2_2_prediction_commitment(
            agent_emitted_path=emitted_path,
            self_prediction_path=prediction_path,
            self_model=self_model,
        )
        _write_json_atomic(trial_dir / "prediction-commitment.json", commitment)
        opened = ledger.open_next(base)
        assert opened["role"] == "held_response"
        response_path = trial_dir / "response_captured.wav"
        write_wav(str(response_path), prediction, ANALYSIS_SAMPLE_RATE)
        score = score_held_trial(
            target=waveforms[label],
            response=prediction,
            self_prediction=prediction,
        )
        _write_json_atomic(trial_dir / "report.json", score)
        _write_json_atomic(
            trial_dir / "held_response-completion.json",
            build_v2_2_completion_record(
                accepted_audio_samples=len(prediction),
                technical_valid=True,
                gate_passed=score["all_component_gates_passed"] is True,
            ),
        )
        held_passes.append(
            selection["selection_and_contrastive_passed"] is True
            and verdict["correct_association"] is True
            and score["all_component_gates_passed"] is True
        )
    return {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "device_accessed": False,
        "microphone_opened": False,
        "playback_performed": False,
        "planned_capture_count": len(v2_2_planned_capture_schedule()),
        "opened_capture_count": len(
            [row for row in ledger.rows() if row.get("state") == "opened_before_io"]
        ),
        "self_model_frozen_before_development": True,
        "association_count": frozen["association_count"],
        "three_distinct_prototype_ids": frozen["three_distinct_prototype_ids"],
        "held_trial_count": len(held_passes),
        "all_held_selection_prediction_and_score_gates_passed": all(held_passes),
    }
