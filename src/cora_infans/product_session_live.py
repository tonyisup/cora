"""Fail-closed live runner primitives for bounded product-session protocol V1.

The offline scorer remains in :mod:`cora_infans.product_session`.  This module
owns the state transitions that must happen before that scorer is allowed to
read response audio: cohort and authorization validation, append-only attempt
registration, callback-integrity classification, development-only association
freeze, label-blind held selection, and pre-response prediction commitment.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import threading
import time
import wave

import numpy as np

from .factorized_tract_plant import synthesize_factorized
from .io.live_mic import CAPTURE_CHANNELS, DEFAULT_DEVICE, LiveMicArray
from .io.motor_plan import write_wav
from .live_speak_smoke import play_wav_to_device
from .natural_speech_workspace_binding import FIR_SAMPLES, _estimate_causal_fir
from .operational_acceptance import probe_local_operating_environment
from .product_session import (
    PRODUCT_SCORE_THRESHOLDS,
    SCORING_VERSION,
    evaluate_contrastive_selection,
    score_component_gates,
    score_waveform_pair,
    select_frozen_association,
)
from .speech_learning_demo import DEMO_F0_HZ
from .vocal_imitation import ANALYSIS_SAMPLE_RATE, acoustic_sensory, build_waveform_repertoire


PROTOCOL_VERSION = "bounded-product-session-v1-approved"
TARGETS = ("a", "i", "u")
DEVELOPMENT_SCHEDULE = ("a", "i", "u", "a", "i", "u")
HELD_SCHEDULE = ("a", "i", "u", "a", "i", "u")
CAPTURE_DURATION_MS = 5_000
CAPTURE_SAMPLE_COUNT = CAPTURE_DURATION_MS * ANALYSIS_SAMPLE_RATE // 1_000
PRE_ROLL_MS = 250
START_CUE_HZ = 660.0
START_CUE_MS = 100
START_CUE_SETTLE_MS = 250
START_CUE_AMPLITUDE = 0.08
SELF_OUTPUT_DEVICE_INDEX = 3
SELF_OUTPUT_DEVICE_NAME = "Mac mini Speakers"
SELF_OUTPUT_PEAK_MAX = 0.10
SELF_OUTPUT_VOLUME_DB = 0.0
RETENTION_DAYS = 30
TECHNICAL_COUNTERS = (
    "input_overflow_count",
    "callback_status_count",
    "callback_timestamp_discontinuity_count",
)
SESSION_STATUSES = frozenset(
    {"blocked_pre_io", "technical_invalid", "behavioral_failed", "passed"}
)


def planned_capture_schedule() -> tuple[dict[str, str], ...]:
    """Return the immutable eighteen-capture schedule in execution order."""

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
    return development + held


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def build_cohort_registration(
    *,
    cohort_id: str,
    ordered_session_ids: Sequence[str],
    frozen_digests: Mapping[str, str],
) -> dict[str, object]:
    """Build the immutable five-slot cohort ledger before first live I/O."""

    slots = tuple(ordered_session_ids)
    if len(slots) != 5 or len(set(slots)) != 5:
        raise ValueError("cohort registration requires five distinct ordered slots")
    if not cohort_id or any(not slot.startswith(cohort_id + "-") for slot in slots):
        raise ValueError("every session slot must be namespaced by the cohort identifier")
    required = {
        "protocol",
        "preregistration",
        "threshold_calibration",
        "representation_implementation",
        "live_runner_implementation",
        "private_storage_implementation",
        "operating_point",
    }
    if set(frozen_digests) != required:
        raise ValueError("cohort frozen digests do not match the required set")
    digests = {
        name: _require_sha256(value, name=f"frozen digest {name}")
        for name, value in frozen_digests.items()
    }
    return {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "protocol_version": PROTOCOL_VERSION,
        "ordered_session_slot_ids": list(slots),
        "minimum_fully_passed_slots": 3,
        "passes_must_be_consecutive": False,
        "sixth_or_replacement_session_may_contribute": False,
        "started_slot_consumption_point": "first accepted audio sample",
        "session_statuses": sorted(SESSION_STATUSES),
        "behavioral_repeatability_denominator": "technically valid started sessions",
        "operational_reliability_denominator": "all started session slots",
        "frozen_digests": digests,
        "capture_schedule": {
            "development": list(DEVELOPMENT_SCHEDULE),
            "held": list(HELD_SCHEDULE),
            "capture_duration_ms": CAPTURE_DURATION_MS,
        },
        "registered_before_live_io": True,
    }


def build_session_authorization(
    *,
    session_id: str,
    cohort_registration: Mapping[str, object],
    storage_alias: str,
    authorization_text: str,
    authorization_date: str,
) -> dict[str, object]:
    """Bind one explicit owner authorization to one registered session slot."""

    ordered = cohort_registration.get("ordered_session_slot_ids")
    if not isinstance(ordered, list) or session_id not in ordered:
        raise ValueError("session is not an ordered slot in the registered cohort")
    if not authorization_text.strip():
        raise ValueError("one-session owner authorization text is required")
    try:
        date.fromisoformat(authorization_date)
    except ValueError as exc:
        raise ValueError("authorization_date must be ISO YYYY-MM-DD") from exc
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "cohort_id": cohort_registration["cohort_id"],
        "session_id": session_id,
        "ordered_slot_index": ordered.index(session_id) + 1,
        "authorization_date": authorization_date,
        "authorization_text": authorization_text,
        "owner_is_sole_speaker": True,
        "owner_may_stop_at_any_time": True,
        "targets": [f"/{target}/" for target in TARGETS],
        "recording_authorized": True,
        "playback_authorized": True,
        "device_discovery_authorized": True,
        "one_session_live_io_authorized": True,
        "authorization_applies_to_other_slots": False,
        "storage_alias": storage_alias,
        "raw_audio_outside_git": True,
        "retention_days": RETENTION_DAYS,
        "backup_exclusion_or_inventory_required": True,
        "cryptographic_erasure_required": True,
        "narrow_claim": (
            "bounded label-blind target-specific association and recall for the "
            "authorized owner at the frozen operating point"
        ),
    }


def validate_session_authorization(
    authorization: Mapping[str, object],
    cohort_registration: Mapping[str, object],
    *,
    session_id: str,
) -> None:
    required_true = (
        "owner_is_sole_speaker",
        "owner_may_stop_at_any_time",
        "recording_authorized",
        "playback_authorized",
        "device_discovery_authorized",
        "one_session_live_io_authorized",
        "raw_audio_outside_git",
        "backup_exclusion_or_inventory_required",
        "cryptographic_erasure_required",
    )
    if authorization.get("session_id") != session_id:
        raise ValueError("authorization is bound to a different session")
    if authorization.get("cohort_id") != cohort_registration.get("cohort_id"):
        raise ValueError("authorization is bound to a different cohort")
    if authorization.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("authorization protocol version mismatch")
    if authorization.get("authorization_applies_to_other_slots") is not False:
        raise ValueError("authorization must be limited to one slot")
    if authorization.get("retention_days") != RETENTION_DAYS:
        raise ValueError("authorization retention decision mismatch")
    if any(authorization.get(field) is not True for field in required_true):
        raise ValueError("authorization is missing a required affirmative decision")


def probe_product_session_preflight(
    operating_point: Mapping[str, object],
    *,
    device_probe: Callable[[], Mapping[str, object]] = probe_local_operating_environment,
    volume_probe: Callable[[], int] | None = None,
) -> dict[str, object]:
    """Discover and verify the frozen route without opening an audio stream."""

    if volume_probe is None:
        import subprocess

        def volume_probe() -> int:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    "output volume of (get volume settings)",
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "could not read system volume")
            return int(result.stdout.strip())

    probe = dict(device_probe())
    input_point = operating_point.get("input")
    self_point = operating_point.get("self_source")
    if not isinstance(input_point, Mapping) or not isinstance(self_point, Mapping):
        raise ValueError("operating point is missing input or self-source metadata")
    volume = volume_probe()
    checks = {
        "host_is_macos": probe.get("platform_system") == "Darwin",
        "host_is_mac_mini": probe.get("host_name") == "Mac mini",
        "ffmpeg_audiotoolbox_available": probe.get("ffmpeg_has_audiotoolbox") is True,
        "xvf3800_input_discovered": probe.get("xvf3800_input") is not None,
        "xvf3800_16khz_two_channel_supported": probe.get(
            "xvf3800_16khz_two_channel_supported"
        ) is True,
        "mac_mini_speakers_discovered": SELF_OUTPUT_DEVICE_NAME
        in probe.get("coreaudio_output_device_names", ()),
        "input_selector_is_colon_zero": input_point.get("device_selector") == DEFAULT_DEVICE,
        "input_sample_rate_is_16000": input_point.get("sample_rate_hz") == ANALYSIS_SAMPLE_RATE,
        "input_has_two_host_channels": input_point.get("host_channels") == CAPTURE_CHANNELS,
        "self_output_device_index_is_3": self_point.get("output_device_index")
        == SELF_OUTPUT_DEVICE_INDEX,
        "self_output_name_matches": self_point.get("device") == SELF_OUTPUT_DEVICE_NAME,
        "self_geometry_is_4cm_below": self_point.get("placement")
        == "centered_directly_below_xvf3800"
        and self_point.get("vertical_distance_cm") == 4,
        "system_volume_is_100_percent": volume == 100
        and self_point.get("system_volume_percent") == 100,
    }
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "preflight_kind": "read_only_route_discovery",
        "checks": checks,
        "preflight_passed": all(checks.values()),
        "measured_system_volume_percent": volume,
        "operating_point": dict(operating_point),
        "device_probe": probe,
        "audio_stream_opened": False,
        "playback_performed": False,
        "captured_at": _utc_now(),
    }


class AppendOnlyAttemptLog:
    """Durable JSONL attempt ledger with monotonic sequence enforcement."""

    def __init__(self, path: str | Path, *, session_id: str) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _rows(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("attempt log contains a non-object row")
                rows.append(value)
        return rows

    def append_open(self, *, trial_id: str, role: str) -> dict[str, object]:
        if role not in {"development_target", "held_target", "held_response"}:
            raise ValueError("unknown attempt role")
        rows = self._rows()
        prior_opens = [
            row
            for row in rows
            if row.get("trial_id") == trial_id
            and row.get("role") == role
            and row.get("state") == "opened_before_io"
        ]
        aborted_attempts = {
            row.get("attempt_number")
            for row in rows
            if row.get("trial_id") == trial_id
            and row.get("role") == role
            and row.get("state") == "technical_abort"
        }
        if prior_opens and not (
            len(prior_opens) == 1
            and prior_opens[0].get("attempt_number") in aborted_attempts
        ):
            raise ValueError("planned trial role is already consumed")
        row = {
            "event_number": len(rows) + 1,
            "attempt_number": sum(
                item.get("state") == "opened_before_io" for item in rows
            )
            + 1,
            "session_id": self.session_id,
            "trial_id": trial_id,
            "role": role,
            "state": "opened_before_io",
            "opened_at": _utc_now(),
        }
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return row

    def append_technical_abort(
        self,
        *,
        trial_id: str,
        role: str,
        accepted_audio_samples: int,
        partial_wav_created: bool,
        reason: str,
    ) -> dict[str, object]:
        """Record the one allowed retryable abort only before any audio exists."""

        if accepted_audio_samples != 0 or partial_wav_created is not False:
            raise ValueError("technical abort requires zero samples and no partial WAV")
        if not reason.strip():
            raise ValueError("technical abort requires a machine failure reason")
        rows = self._rows()
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
        if len(opens) != 1 or aborts:
            raise ValueError("technical abort is unavailable for this planned trial")
        row = {
            "event_number": len(rows) + 1,
            "attempt_number": opens[0]["attempt_number"],
            "session_id": self.session_id,
            "trial_id": trial_id,
            "role": role,
            "state": "technical_abort",
            "accepted_audio_samples": 0,
            "partial_wav_created": False,
            "reason": reason,
            "recorded_at": _utc_now(),
        }
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return row

    @staticmethod
    def _trial_dir(root: Path, planned: Mapping[str, str]) -> Path:
        return root / planned["phase"] / planned["trial_id"]

    @staticmethod
    def _load_required_json(path: Path) -> Mapping[str, object]:
        if not path.is_file():
            raise ValueError(f"required state artifact is missing: {path.name}")
        value = json.loads(path.read_text())
        if not isinstance(value, Mapping):
            raise ValueError(f"required state artifact is invalid: {path.name}")
        return value

    def open_next(self, session_root: str | Path) -> dict[str, object]:
        """Open only the next preregistered capture after all prior gates pass."""

        root = Path(session_root)
        plan = planned_capture_schedule()
        rows = self._rows()
        aborted_attempts = {
            row.get("attempt_number")
            for row in rows
            if row.get("state") == "technical_abort"
        }
        effective_rows = [
            row
            for row in rows
            if row.get("state") == "opened_before_io"
            and row.get("attempt_number") not in aborted_attempts
        ]
        if len(effective_rows) >= len(plan):
            raise ValueError("all eighteen planned microphone captures are consumed")
        for index, row in enumerate(effective_rows):
            expected = plan[index]
            if row.get("trial_id") != expected["trial_id"] or row.get("role") != expected["role"]:
                raise ValueError("attempt log diverges from the frozen capture schedule")
            trial_dir = self._trial_dir(root, expected)
            filename = (
                "response_captured.wav"
                if expected["role"] == "held_response"
                else "target_captured.wav"
            )
            if not (trial_dir / filename).is_file():
                raise ValueError("previous opened attempt has no completed capture artifact")
            integrity = self._load_required_json(trial_dir / f"{expected['role']}-integrity.json")
            if integrity.get("technical_valid") is not True:
                raise ValueError("technical-invalid session cannot open another capture")
            quality = self._load_required_json(trial_dir / f"{expected['role']}-quality.json")
            if quality.get("quality_passed") is not True:
                raise ValueError("quality-failed session cannot open another capture")
            if expected["role"] == "held_response":
                score = self._load_required_json(trial_dir / "report.json")
                if score.get("all_component_gates_passed") is not True:
                    raise ValueError("score-failed session cannot open another capture")

        current = plan[len(effective_rows)]
        current_dir = self._trial_dir(root, current)
        current_dir.mkdir(parents=True, exist_ok=True)
        if current["phase"] == "held":
            frozen_path = root / "frozen-vocal-associations.json"
            key_path = root / "evaluation-key.json"
            frozen = self._load_required_json(frozen_path)
            key = self._load_required_json(key_path)
            if frozen.get("frozen_before_held_capture") is not True:
                raise ValueError("held capture is blocked until association freeze")
            if key.get("frozen_associations_sha256") != sha256_json(frozen):
                raise ValueError("evaluation key does not match the frozen associations")
        if current["role"] == "held_response":
            selection = self._load_required_json(current_dir / "selection-commitment.json")
            verdict = self._load_required_json(current_dir / "selection-verdict.json")
            prediction = self._load_required_json(current_dir / "prediction-commitment.json")
            if selection.get("selection_and_contrastive_passed") is not True:
                raise ValueError("held response blocked by selection or contrastive gate")
            if verdict.get("correct_association") is not True:
                raise ValueError("held response blocked by incorrect association")
            if prediction.get("committed_before_response_capture") is not True:
                raise ValueError("held response blocked until prediction commitment")
            for filename in ("agent_emitted.wav", "self_prediction.wav"):
                path = current_dir / filename
                if not path.is_file():
                    raise ValueError(f"held response prerequisite is missing: {filename}")
            if prediction.get("agent_emitted_sha256") != sha256_path(
                current_dir / "agent_emitted.wav"
            ) or prediction.get("self_prediction_sha256") != sha256_path(
                current_dir / "self_prediction.wav"
            ):
                raise ValueError("prediction commitment hashes changed before response capture")
            if (current_dir / "response_captured.wav").exists():
                raise ValueError("response capture already exists")
        return self.append_open(
            trial_id=current["trial_id"],
            role=current["role"],
        )

    def rows(self) -> list[dict[str, object]]:
        return self._rows()


def classify_transport_integrity(
    telemetry: Mapping[str, object], *, expected_sample_count: int = CAPTURE_SAMPLE_COUNT
) -> dict[str, object]:
    """Commit technical validity from machine telemetry before score access."""

    failures: list[str] = []
    if telemetry.get("backend") != "sounddevice_portaudio_coreaudio_callback":
        failures.append("callback_backend_mismatch")
    if telemetry.get("sample_rate_hz") != ANALYSIS_SAMPLE_RATE:
        failures.append("sample_rate_mismatch")
    if telemetry.get("channel_count") != CAPTURE_CHANNELS:
        failures.append("channel_count_mismatch")
    if telemetry.get("sample_count") != expected_sample_count:
        failures.append("truncated_or_oversized_capture")
    for counter in TECHNICAL_COUNTERS:
        value = telemetry.get(counter)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            failures.append(counter)
    return {
        "committed_before_score_access": True,
        "technical_valid": not failures,
        "session_status_if_invalid": None if not failures else "technical_invalid",
        "failure_reasons": failures,
    }


def _array_digest(vector: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(vector, dtype="<f8").tobytes()).hexdigest()


def freeze_development_associations(
    captures: Mapping[str, Sequence[tuple[str, np.ndarray, int]]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Freeze three development-only centroids and fixed-repertoire associations."""

    if set(captures) != set(TARGETS) or any(len(captures[label]) != 2 for label in TARGETS):
        raise ValueError("freeze requires exactly two development captures per target")
    repertoire = build_waveform_repertoire()
    associations: dict[str, object] = {}
    evaluation_key: dict[str, object] = {}
    learned_ids: list[str] = []
    for label in TARGETS:
        vectors = []
        capture_hashes = []
        for path_name, waveform, sample_rate in captures[label]:
            vector, quality = acoustic_sensory(waveform, sample_rate)
            if quality["quality_passed"] is not True:
                raise ValueError(f"development capture failed quality gates: {path_name}")
            vectors.append(vector)
            capture_hashes.append(hashlib.sha256(np.asarray(waveform, dtype="<f4").tobytes()).hexdigest())
        centroid = np.mean(np.stack(vectors), axis=0)
        centroid /= np.linalg.norm(centroid)
        target_id = _array_digest(centroid)
        prototype_index = repertoire.nearest_sensory_index(centroid)
        prototype = repertoire.prototypes[prototype_index]
        prototype_id = prototype.prototype_id
        learned_ids.append(prototype_id)
        opaque_id = "assoc-" + hashlib.sha256((target_id + prototype_id).encode()).hexdigest()[:16]
        associations[opaque_id] = {
            "centroid": centroid.tolist(),
            "centroid_sha256": target_id,
            "prototype_id": prototype_id,
            "prototype_sensory": prototype.sensory_array().tolist(),
            "prototype_sensory_sha256": _array_digest(prototype.sensory_array()),
            "motor": list(prototype.motor),
            "development_capture_sha256": capture_hashes,
            "association_lookup": {
                "method": "unique nearest fixed-repertoire sensory prototype",
                "target_label_provided": False,
                "prototype_index": prototype_index,
            },
        }
        evaluation_key[opaque_id] = {"scheduled_target": label}
    if len(set(learned_ids)) != 3:
        raise ValueError("freeze requires three distinct selected prototype IDs")
    rest = repertoire.prototypes[repertoire.rest_index()]
    frozen = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "representation": "v4-label-blind-30-band-auditory",
        "representation_implementation_sha256": sha256_path(Path(__file__).with_name("vocal_imitation.py")),
        "scoring_version": SCORING_VERSION,
        "score_thresholds": dict(PRODUCT_SCORE_THRESHOLDS),
        "repertoire_sha256": repertoire.digest(),
        "association_count": 3,
        "three_distinct_prototype_ids": True,
        "associations": associations,
        "rest_prototype": {
            "prototype_id": rest.prototype_id,
            "sensory": rest.sensory_array().tolist(),
        },
        "frozen_before_held_capture": True,
    }
    key = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "label_metadata_available_to_selection": False,
        "opaque_association_targets": evaluation_key,
        "frozen_associations_sha256": sha256_json(frozen),
    }
    return frozen, key


def commit_label_blind_selection(
    *,
    target_waveform: np.ndarray,
    sample_rate: int,
    frozen: Mapping[str, object],
) -> dict[str, object]:
    """Select and evaluate controls without access to the scheduled label."""

    target, quality = acoustic_sensory(target_waveform, sample_rate)
    if quality["quality_passed"] is not True:
        raise ValueError("held target failed quality gates")
    records = frozen.get("associations")
    if not isinstance(records, Mapping) or len(records) != 3:
        raise ValueError("frozen association set is invalid")
    centroids = {
        str(opaque_id): record["centroid"]
        for opaque_id, record in records.items()
        if isinstance(record, Mapping)
    }
    selection = select_frozen_association(target, centroids)
    selected_id = str(selection["selected_opaque_id"])
    selected = records[selected_id]
    rest = frozen.get("rest_prototype")
    if not isinstance(selected, Mapping) or not isinstance(rest, Mapping):
        raise ValueError("frozen prototype records are invalid")
    off_targets = {
        str(opaque_id): record["prototype_sensory"]
        for opaque_id, record in records.items()
        if opaque_id != selected_id and isinstance(record, Mapping)
    }
    contrastive = evaluate_contrastive_selection(
        target,
        selected_prototype=selected["prototype_sensory"],
        rest_prototype=rest["sensory"],
        off_target_prototypes=off_targets,
    )
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "selection_input": {
            "target_sensory_sha256": _array_digest(target),
            "candidate_opaque_ids": sorted(centroids),
            "scheduled_label_provided": False,
        },
        "quality": quality,
        "selection": selection,
        "contrastive": contrastive,
        "selected_opaque_id": selected_id,
        "selected_prototype_id": selected["prototype_id"],
        "selection_and_contrastive_passed": (
            selection["selection_passed"] is True
            and contrastive["contrastive_passed"] is True
        ),
        "committed_at": _utc_now(),
    }


def evaluate_committed_selection(
    commitment: Mapping[str, object],
    evaluation_key: Mapping[str, object],
    *,
    scheduled_label: str,
) -> dict[str, object]:
    """Expose the scheduled label only after a selection commitment exists."""

    selected_id = commitment.get("selected_opaque_id")
    key = evaluation_key.get("opaque_association_targets")
    if not isinstance(key, Mapping) or selected_id not in key:
        raise ValueError("selection commitment is absent from the evaluation key")
    row = key[selected_id]
    if not isinstance(row, Mapping):
        raise ValueError("evaluation-key row is invalid")
    correct = row.get("scheduled_target") == scheduled_label
    return {
        "scheduled_label": scheduled_label,
        "committed_opaque_id": selected_id,
        "correct_association": correct,
        "verdict_opened_after_commitment": True,
    }


def render_selected_prototype(
    frozen: Mapping[str, object], selected_opaque_id: str
) -> np.ndarray:
    records = frozen.get("associations")
    if not isinstance(records, Mapping) or selected_opaque_id not in records:
        raise ValueError("selected association is not frozen")
    record = records[selected_opaque_id]
    if not isinstance(record, Mapping):
        raise ValueError("selected frozen association is invalid")
    utterance = np.tile(
        synthesize_factorized(np.asarray(record["motor"], dtype=np.float64), DEMO_F0_HZ),
        13,
    )
    peak = float(np.max(np.abs(utterance), initial=0.0))
    if peak <= 0.0:
        raise ValueError("rendered prototype is silent")
    return (utterance * (0.095 / peak)).astype(np.float32)


def build_self_model(known_chirp: np.ndarray, observed_chirp: np.ndarray) -> dict[str, object]:
    impulse = _estimate_causal_fir(known_chirp, observed_chirp)
    return {
        "schema_version": 1,
        "model": "causal-fir-v1",
        "sample_rate_hz": ANALYSIS_SAMPLE_RATE,
        "fir_samples": FIR_SAMPLES,
        "impulse": impulse.tolist(),
        "impulse_sha256": _array_digest(impulse),
        "known_chirp_sha256": _array_digest(known_chirp),
        "observed_chirp_sha256": _array_digest(observed_chirp),
        "frozen_before_held_response": True,
    }


def predict_self_capture(emitted: np.ndarray, self_model: Mapping[str, object]) -> np.ndarray:
    impulse = np.asarray(self_model.get("impulse"), dtype=np.float64)
    if impulse.shape != (FIR_SAMPLES,) or not np.all(np.isfinite(impulse)):
        raise ValueError("self model has an invalid FIR")
    return np.convolve(np.asarray(emitted, dtype=np.float64), impulse)[: len(emitted)].astype(np.float32)


def build_prediction_commitment(
    *, agent_emitted_path: str | Path, self_prediction_path: str | Path, self_model: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "agent_emitted_sha256": sha256_path(agent_emitted_path),
        "self_prediction_sha256": sha256_path(self_prediction_path),
        "self_model_sha256": sha256_json(self_model),
        "response_capture_exists_at_commitment": False,
        "committed_before_response_capture": True,
        "committed_at": _utc_now(),
    }


def _write_multichannel_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[1] not in (1, CAPTURE_CHANNELS):
        raise ValueError("capture must have one or two channels")
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(data.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _capture_telemetry(mic: LiveMicArray, samples: np.ndarray) -> dict[str, object]:
    return {
        "backend": mic.active_stream_backend,
        "device_selector": DEFAULT_DEVICE,
        "sample_rate_hz": ANALYSIS_SAMPLE_RATE,
        "channel_count": int(samples.shape[1]),
        "sample_count": int(samples.shape[0]),
        "input_overflow_count": mic.input_overflow_count,
        "callback_status_count": mic.callback_status_count,
        "callback_timestamp_discontinuity_count": mic.callback_timestamp_discontinuity_count,
        "capture_peak_absolute": mic.capture_peak_absolute,
        "clipped_sample_count": mic.clipped_sample_count,
    }


def capture_target_live(
    path: str | Path,
    *,
    cue_player: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    """Capture one post-cue owner target with cue and settle samples excluded."""

    mic = LiveMicArray(
        device=DEFAULT_DEVICE,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        capture_seconds=CAPTURE_DURATION_MS / 1000.0,
        stream_backend="callback",
    )
    with mic.streaming():
        cue = dict(cue_player())
        time.sleep(START_CUE_SETTLE_MS / 1000.0)
        capture_origin = mic.discard_buffered_callback_audio()
        mic.capture_frame(duration_ms=CAPTURE_DURATION_MS)
        samples = mic.last_pcm_channels
        if samples is None:
            raise RuntimeError("callback target capture returned no PCM")
        telemetry = _capture_telemetry(mic, samples)
        telemetry["capture_origin_stream_frame"] = capture_origin
        telemetry["capture_end_stream_frame"] = capture_origin + len(samples)
        telemetry["cue"] = cue
        telemetry["cue_and_settle_excluded"] = True
    _write_multichannel_wav(Path(path), samples, ANALYSIS_SAMPLE_RATE)
    telemetry["wav_sha256"] = sha256_path(path)
    telemetry["transport_integrity"] = classify_transport_integrity(telemetry)
    return telemetry


def capture_response_live(
    path: str | Path,
    *,
    emitted_path: str | Path,
    playback: Callable[..., Mapping[str, object]] = play_wav_to_device,
) -> dict[str, object]:
    """Capture a full-span response with playback anchored in stream frames."""

    mic = LiveMicArray(
        device=DEFAULT_DEVICE,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        capture_seconds=CAPTURE_DURATION_MS / 1000.0,
        stream_backend="callback",
    )
    failure: list[BaseException] = []

    def capture() -> None:
        try:
            mic.capture_frame(duration_ms=CAPTURE_DURATION_MS)
        except BaseException as exc:  # propagated after thread joins
            failure.append(exc)

    with mic.streaming():
        capture_origin = mic.discard_buffered_callback_audio()
        worker = threading.Thread(target=capture, daemon=True)
        worker.start()
        time.sleep(PRE_ROLL_MS / 1000.0)
        playback_onset = mic.callback_stream_frame_position()
        playback_result = dict(
            playback(
                str(emitted_path),
                device_index=SELF_OUTPUT_DEVICE_INDEX,
                sample_rate=ANALYSIS_SAMPLE_RATE,
                volume_db=SELF_OUTPUT_VOLUME_DB,
            )
        )
        playback_end = mic.callback_stream_frame_position()
        worker.join(timeout=CAPTURE_DURATION_MS / 1000.0 + 2.0)
        if worker.is_alive():
            raise RuntimeError("response capture thread did not finish")
        if failure:
            raise RuntimeError(f"response capture failed: {failure[0]}") from failure[0]
        samples = mic.last_pcm_channels
        if samples is None:
            raise RuntimeError("callback response capture returned no PCM")
        capture_end = capture_origin + len(samples)
        telemetry = _capture_telemetry(mic, samples)
        telemetry.update(
            {
                "capture_origin_stream_frame": capture_origin,
                "playback_onset_stream_frame": playback_onset,
                "playback_end_stream_frame": playback_end,
                "capture_end_stream_frame": capture_end,
                "pre_roll_frames": playback_onset - capture_origin,
                "capture_active_through_playback_completion": playback_end <= capture_end,
                "playback": playback_result,
                "playback_wav_sha256": sha256_path(emitted_path),
            }
        )
    _write_multichannel_wav(Path(path), samples, ANALYSIS_SAMPLE_RATE)
    telemetry["wav_sha256"] = sha256_path(path)
    integrity = classify_transport_integrity(telemetry)
    expected_pre_roll = PRE_ROLL_MS * ANALYSIS_SAMPLE_RATE // 1_000
    if telemetry["pre_roll_frames"] < expected_pre_roll:
        integrity["technical_valid"] = False
        integrity["failure_reasons"].append("pre_roll_shorter_than_250ms")
    if telemetry["capture_active_through_playback_completion"] is not True:
        integrity["technical_valid"] = False
        integrity["failure_reasons"].append("capture_did_not_span_playback")
    if integrity["technical_valid"] is False:
        integrity["session_status_if_invalid"] = "technical_invalid"
    telemetry["transport_integrity"] = integrity
    return telemetry


def score_held_trial(
    *,
    target: np.ndarray,
    response: np.ndarray,
    self_prediction: np.ndarray,
) -> dict[str, object]:
    imitation = score_waveform_pair(target, response)
    self_fidelity = score_waveform_pair(self_prediction, response)
    gates = {
        "imitation_fidelity": score_component_gates(imitation),
        "self_model_fidelity": score_component_gates(self_fidelity),
    }
    return {
        "scoring_version": SCORING_VERSION,
        "scores": {
            "imitation_fidelity": imitation,
            "self_model_fidelity": self_fidelity,
        },
        "component_gates": gates,
        "all_component_gates_passed": all(
            passed for family in gates.values() for passed in family.values()
        ),
    }


def retention_deadline(completed_date: date) -> str:
    return (completed_date + timedelta(days=RETENTION_DAYS)).isoformat()


def build_session_manifest(
    session_root: str | Path,
    *,
    required_relative_paths: Sequence[str],
) -> dict[str, object]:
    """Hash the completed private evidence tree without circular self-hashing."""

    root = Path(session_root).resolve()
    required = tuple(required_relative_paths)
    if len(required) != len(set(required)) or not required:
        raise ValueError("manifest requires a non-empty unique artifact list")
    artifacts: dict[str, str] = {}
    for relative in required:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("manifest artifact escapes session root") from exc
        if candidate.name in {"session-manifest.json", "session-attestation.json"}:
            raise ValueError("manifest and attestation cannot self-hash")
        if not candidate.is_file():
            raise ValueError(f"required session artifact is missing: {relative}")
        artifacts[relative] = sha256_path(candidate)
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_sha256": artifacts,
        "artifact_count": len(artifacts),
        "all_required_artifacts_present": True,
        "v4_evidence_mutated": False,
        "completed_at": _utc_now(),
    }


def build_session_attestation(
    manifest_path: str | Path,
    *,
    anchor_type: str,
    anchor_value: str,
) -> dict[str, object]:
    """Externally anchor a completed manifest by Git commit or owner signature."""

    if anchor_type == "git_commit":
        if len(anchor_value) not in (40, 64) or any(
            character not in "0123456789abcdef" for character in anchor_value
        ):
            raise ValueError("git commit anchor must be a lowercase object ID")
    elif anchor_type == "owner_signature":
        if len(anchor_value.strip()) < 16:
            raise ValueError("owner signature anchor is missing")
    else:
        raise ValueError("unsupported manifest anchor type")
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "manifest_sha256": sha256_path(manifest_path),
        "anchor_type": anchor_type,
        "anchor_value": anchor_value,
        "externally_anchored": True,
        "attested_at": _utc_now(),
    }


def classify_completed_session(
    *,
    transport_integrity_records: Sequence[Mapping[str, object]],
    behavioral_gates: Mapping[str, bool],
    manifest_attestation: Mapping[str, object] | None,
) -> dict[str, object]:
    """Apply technical classification before behavioral results are consulted."""

    technical_failures = [
        list(record.get("failure_reasons", ()))
        for record in transport_integrity_records
        if record.get("technical_valid") is not True
    ]
    if technical_failures:
        return {
            "status": "technical_invalid",
            "technical_classification_committed_before_score_access": True,
            "technical_failure_reasons": technical_failures,
            "behavioral_gates_consulted": False,
        }
    all_behavioral = bool(behavioral_gates) and all(
        value is True for value in behavioral_gates.values()
    )
    anchored = (
        manifest_attestation is not None
        and manifest_attestation.get("externally_anchored") is True
    )
    status = "passed" if all_behavioral and anchored else "behavioral_failed"
    return {
        "status": status,
        "technical_classification_committed_before_score_access": True,
        "technical_failure_reasons": [],
        "behavioral_gates_consulted": True,
        "behavioral_gates": dict(behavioral_gates),
        "all_behavioral_gates_passed": all_behavioral,
        "manifest_externally_anchored": anchored,
    }


def aggregate_cohort_status(slot_statuses: Sequence[str | None]) -> dict[str, object]:
    if len(slot_statuses) != 5:
        raise ValueError("cohort aggregation requires exactly five slots")
    if any(status is not None and status not in SESSION_STATUSES for status in slot_statuses):
        raise ValueError("cohort contains an unknown session status")
    passed = sum(status == "passed" for status in slot_statuses)
    started = sum(status in {"technical_invalid", "behavioral_failed", "passed"} for status in slot_statuses)
    technically_valid = sum(status in {"behavioral_failed", "passed"} for status in slot_statuses)
    resolved = all(status is not None and status != "blocked_pre_io" for status in slot_statuses)
    return {
        "slot_statuses": list(slot_statuses),
        "resolved": resolved,
        "passed_slots": passed,
        "product_level_passed": resolved and passed >= 3,
        "behavioral_repeatability": None if technically_valid == 0 else passed / technically_valid,
        "operational_reliability": None if started == 0 else technically_valid / started,
        "passes_must_be_consecutive": False,
    }
