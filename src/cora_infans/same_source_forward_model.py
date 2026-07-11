"""Bounded same-source acoustic forward model for H3 development."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io.wav_frames import read_pcm_wav
from .natural_speech_workspace_spectral_v3 import BAND_EDGES_HZ
from .natural_speech_workspace_template_v2 import SAMPLE_RATE, SPEECH_MS
from .real_workspace_binding import BoundedEfferenceWorkspace


@dataclass(frozen=True)
class SpentCommand:
    key: str
    panel: str
    panel_index: int
    command_path: Path
    command_sha256: str
    calibration_capture_path: Path
    calibration_capture_sha256: str
    calibration_alignment: int
    self_probe_capture_path: Path
    self_probe_capture_sha256: str
    self_probe_alignment: int
    world_only_capture_path: Path
    world_only_capture_sha256: str
    overlap_capture_path: Path | None
    overlap_capture_sha256: str | None
    overlap_alignment: int | None


class SameSourceForwardModel:
    """Running-mean log-spectral path signature with fixed 13-scalar state."""

    def __init__(self) -> None:
        self._path_signature = np.zeros(12, dtype=np.float64)
        self._count = 0

    @classmethod
    def from_frozen(
        cls, path_signature: np.ndarray, *, calibration_count: int
    ) -> SameSourceForwardModel:
        signature = _feature_vector(path_signature)
        if calibration_count <= 0:
            raise ValueError("frozen calibration count must be positive")
        model = cls()
        model._path_signature = signature.copy()
        model._count = calibration_count
        return model

    @property
    def count(self) -> int:
        return self._count

    @property
    def state_scalars(self) -> int:
        return 13

    @property
    def path_signature(self) -> np.ndarray:
        return self._path_signature.copy()

    def update(self, command_centered: np.ndarray, captured_centered: np.ndarray) -> None:
        command = _feature_vector(command_centered)
        captured = _feature_vector(captured_centered)
        observation = captured - command
        self._count += 1
        self._path_signature += (observation - self._path_signature) / self._count

    def predict(self, command_centered: np.ndarray) -> np.ndarray:
        if self._count == 0:
            raise ValueError("same-source forward model has no calibration")
        predicted = _feature_vector(command_centered) + self._path_signature
        predicted -= float(np.mean(predicted))
        return _normalize(predicted)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.shape != (12,) or not np.all(np.isfinite(value)):
        raise ValueError("same-source feature must contain 12 finite scalars")
    return value


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = _feature_vector(vector)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("same-source feature has insufficient shape energy")
    return value / norm


def centered_log_bands(pcm: np.ndarray) -> np.ndarray:
    samples = np.asarray(pcm, dtype=np.float64)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    if samples.ndim != 1 or len(samples) != SAMPLE_RATE * SPEECH_MS // 1000:
        raise ValueError("same-source analysis requires one 600 ms mono window")
    centered = samples - float(np.mean(samples))
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequencies = np.fft.rfftfreq(len(centered), 1.0 / SAMPLE_RATE)
    vector = np.array(
        [
            np.log(
                float(
                    np.sum(power[(frequencies >= low) & (frequencies < high)])
                )
                + 1e-12
            )
            for low, high in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])
        ]
    )
    vector -= float(np.mean(vector))
    if float(np.linalg.norm(vector)) <= 1e-12:
        raise ValueError("same-source spectral window has insufficient energy")
    return vector


def forward_distance(
    model: SameSourceForwardModel,
    *,
    command_centered: np.ndarray,
    current_centered: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(model.predict(command_centered) - _normalize(current_centered))
    )


def raw_efference_distance(
    *, command_centered: np.ndarray, current_centered: np.ndarray
) -> float:
    return float(
        np.linalg.norm(_normalize(command_centered) - _normalize(current_centered))
    )


def _artifact_path(root: Path, trial_id: str, kind: str) -> Path:
    return root / f"{trial_id}_{kind}.wav"


def _artifact_by_trial(result: dict[str, object]) -> dict[str, dict[str, object]]:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("spent result is missing artifacts")
    return {str(item["trial_id"]): item for item in artifacts}


def _validate_result_modes(
    template_v2: dict[str, object], spectral_v3: dict[str, object]
) -> None:
    if template_v2.get("mode") != "natural_speech_workspace_template_v2_real_failure_audit":
        raise ValueError("unexpected template-v2 result mode")
    if spectral_v3.get("mode") != "natural_speech_workspace_spectral_v3_real_result":
        raise ValueError("unexpected spectral-v3 result mode")


def _alignment(artifact: dict[str, object]) -> int:
    value = artifact.get("alignment_samples")
    if value is None:
        value = artifact.get("reconstructed_alignment_samples")
    if not isinstance(value, int):
        raise ValueError("spent artifact is missing alignment")
    return value


def _make_command(
    *,
    key: str,
    panel: str,
    panel_index: int,
    prefix: str,
    artifacts: dict[str, dict[str, object]],
    artifact_root: Path,
    has_overlap: bool,
) -> SpentCommand:
    ids = {
        condition: f"{prefix}_{condition}_{panel_index}"
        for condition in ("self_calibration", "self_probe", "world_only", "overlap")
    }
    calibration = artifacts[ids["self_calibration"]]
    probe = artifacts[ids["self_probe"]]
    world = artifacts[ids["world_only"]]
    overlap = artifacts.get(ids["overlap"]) if has_overlap else None
    command_path = _artifact_path(artifact_root, ids["self_calibration"], "self")
    calibration_path = _artifact_path(
        artifact_root, ids["self_calibration"], "capture"
    )
    probe_path = _artifact_path(artifact_root, ids["self_probe"], "capture")
    world_path = _artifact_path(artifact_root, ids["world_only"], "capture")
    overlap_path = (
        _artifact_path(artifact_root, ids["overlap"], "capture")
        if overlap is not None
        else None
    )
    return SpentCommand(
        key=key,
        panel=panel,
        panel_index=panel_index,
        command_path=command_path,
        command_sha256=str(calibration["self_sha256"]),
        calibration_capture_path=calibration_path,
        calibration_capture_sha256=str(calibration["captured_sha256"]),
        calibration_alignment=_alignment(calibration),
        self_probe_capture_path=probe_path,
        self_probe_capture_sha256=str(probe["captured_sha256"]),
        self_probe_alignment=_alignment(probe),
        world_only_capture_path=world_path,
        world_only_capture_sha256=str(world["captured_sha256"]),
        overlap_capture_path=overlap_path,
        overlap_capture_sha256=(
            str(overlap["captured_sha256"]) if overlap is not None else None
        ),
        overlap_alignment=_alignment(overlap) if overlap is not None else None,
    )


def load_spent_commands(
    *,
    template_v2_result: str | Path,
    template_v2_artifacts: str | Path,
    spectral_v3_result: str | Path,
    spectral_v3_artifacts: str | Path,
) -> tuple[SpentCommand, ...]:
    v2 = json.loads(Path(template_v2_result).read_text())
    v3 = json.loads(Path(spectral_v3_result).read_text())
    _validate_result_modes(v2, v3)
    v2_artifacts = _artifact_by_trial(v2)
    v3_artifacts = _artifact_by_trial(v3)
    v2_root = Path(template_v2_artifacts)
    v3_root = Path(spectral_v3_artifacts)
    commands = []
    for split in ("development", "held_out"):
        for index in range(3):
            commands.append(
                _make_command(
                    key=f"v2_{split}_{index}",
                    panel=f"v2_{split}",
                    panel_index=index,
                    prefix=split,
                    artifacts=v2_artifacts,
                    artifact_root=v2_root,
                    has_overlap=split == "held_out",
                )
            )
    for index in range(3):
        commands.append(
            _make_command(
                key=f"v3_{index}",
                panel="v3",
                panel_index=index,
                prefix="held_out",
                artifacts=v3_artifacts,
                artifact_root=v3_root,
                has_overlap=True,
            )
        )
    if len(commands) != 9:
        raise AssertionError("same-source development requires nine commands")
    for command in commands:
        pairs = (
            (command.command_path, command.command_sha256),
            (
                command.calibration_capture_path,
                command.calibration_capture_sha256,
            ),
            (command.self_probe_capture_path, command.self_probe_capture_sha256),
            (
                command.world_only_capture_path,
                command.world_only_capture_sha256,
            ),
        )
        for path, expected in pairs:
            if _sha256(path) != expected:
                raise ValueError(f"spent artifact hash mismatch: {path}")
        if command.overlap_capture_path is not None:
            if _sha256(command.overlap_capture_path) != command.overlap_capture_sha256:
                raise ValueError(
                    f"spent artifact hash mismatch: {command.overlap_capture_path}"
                )
    return tuple(commands)


def _speech_from_command(path: Path) -> np.ndarray:
    samples, rate = read_pcm_wav(path)
    if rate != SAMPLE_RATE:
        raise ValueError("same-source command sample rate mismatch")
    start = SAMPLE_RATE * 220 // 1000
    end = start + SAMPLE_RATE * SPEECH_MS // 1000
    return np.mean(samples[start:end], axis=1)


def _speech_from_capture(path: Path, alignment: int) -> np.ndarray:
    samples, rate = read_pcm_wav(path)
    if rate != SAMPLE_RATE or samples.shape[1] != 2:
        raise ValueError("same-source capture geometry mismatch")
    start = alignment + SAMPLE_RATE * 220 // 1000
    end = start + SAMPLE_RATE * SPEECH_MS // 1000
    if alignment < 0 or end > len(samples):
        raise ValueError("same-source capture does not contain speech window")
    return np.mean(samples[start:end], axis=1)


def _classification(actual: list[bool], predicted: list[bool]) -> dict[str, object]:
    tp = sum(want and got for want, got in zip(actual, predicted))
    fp = sum((not want) and got for want, got in zip(actual, predicted))
    tn = sum((not want) and (not got) for want, got in zip(actual, predicted))
    fn = sum(want and (not got) for want, got in zip(actual, predicted))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "world_recall": tp / (tp + fn) if tp + fn else 0.0,
        "self_suppression_specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _strategy(
    rows: list[dict[str, object]],
    *,
    capacity: int,
    strategy: str,
    threshold: float,
) -> dict[str, object]:
    actual, predicted = [], []
    for panel in ("v2_development", "v2_held_out", "v3"):
        workspace = BoundedEfferenceWorkspace(capacity)
        for index in range(3):
            workspace.push(f"{panel}_self_{index}")
        for row in (item for item in rows if item["panel"] == panel):
            bound = workspace.contains(str(row["efference_id"]) if row["efference_id"] else None)
            if strategy == "flat":
                usable = True
            elif strategy == "fixed_self_window":
                usable = not bound
            elif strategy == "scheduled_identity_plus_same_source_forward_model":
                usable = not bound or float(row["world_evidence"]) >= threshold
            else:
                raise ValueError(f"unknown same-source strategy: {strategy}")
            actual.append(bool(row["expected_world_usable"]))
            predicted.append(usable)
    return {
        "strategy": strategy,
        "capacity": capacity,
        "bounded_binding_slots": capacity,
        "metrics": _classification(actual, predicted),
    }


def evaluate_spent_same_source_forward_model(
    *,
    template_v2_result: str | Path,
    template_v2_artifacts: str | Path,
    spectral_v3_result: str | Path,
    spectral_v3_artifacts: str | Path,
) -> dict[str, object]:
    commands = load_spent_commands(
        template_v2_result=template_v2_result,
        template_v2_artifacts=template_v2_artifacts,
        spectral_v3_result=spectral_v3_result,
        spectral_v3_artifacts=spectral_v3_artifacts,
    )
    command_features = {
        command.key: centered_log_bands(_speech_from_command(command.command_path))
        for command in commands
    }
    fold_reports = []
    self_distances, raw_distances, overlap_distances = [], [], []
    rows: list[dict[str, object]] = []
    for held_out in commands:
        model = SameSourceForwardModel()
        training_keys = []
        for training in commands:
            if training.key == held_out.key:
                continue
            captured = centered_log_bands(
                _speech_from_capture(
                    training.calibration_capture_path,
                    training.calibration_alignment,
                )
            )
            model.update(command_features[training.key], captured)
            training_keys.append(training.key)
        probe = centered_log_bands(
            _speech_from_capture(
                held_out.self_probe_capture_path, held_out.self_probe_alignment
            )
        )
        predicted_distance = forward_distance(
            model,
            command_centered=command_features[held_out.key],
            current_centered=probe,
        )
        raw_distance = raw_efference_distance(
            command_centered=command_features[held_out.key], current_centered=probe
        )
        self_distances.append(predicted_distance)
        raw_distances.append(raw_distance)
        rows.extend(
            (
                {
                    "trial_id": f"{held_out.key}_self_probe",
                    "panel": held_out.panel,
                    "efference_id": f"{held_out.panel}_self_{held_out.panel_index}",
                    "condition": "self_probe",
                    "expected_world_usable": False,
                    "world_evidence": predicted_distance,
                },
                {
                    "trial_id": f"{held_out.key}_world_only",
                    "panel": held_out.panel,
                    "efference_id": None,
                    "condition": "world_only",
                    "expected_world_usable": True,
                    "world_evidence": 2.0,
                },
            )
        )
        overlap_distance = None
        if held_out.overlap_capture_path is not None:
            assert held_out.overlap_alignment is not None
            overlap = centered_log_bands(
                _speech_from_capture(
                    held_out.overlap_capture_path, held_out.overlap_alignment
                )
            )
            overlap_distance = forward_distance(
                model,
                command_centered=command_features[held_out.key],
                current_centered=overlap,
            )
            overlap_distances.append(overlap_distance)
            rows.append(
                {
                    "trial_id": f"{held_out.key}_overlap",
                    "panel": held_out.panel,
                    "efference_id": f"{held_out.panel}_self_{held_out.panel_index}",
                    "condition": "overlap",
                    "expected_world_usable": True,
                    "world_evidence": overlap_distance,
                }
            )
        fold_reports.append(
            {
                "held_out_command": held_out.key,
                "training_commands": training_keys,
                "training_count": model.count,
                "held_out_calibration_consumed": False,
                "self_probe_forward_distance": predicted_distance,
                "self_probe_raw_efference_distance": raw_distance,
                "forward_improves_raw": predicted_distance < raw_distance,
                "overlap_forward_distance": overlap_distance,
                "model_state_scalars": model.state_scalars,
                "path_signature": model.path_signature.tolist(),
            }
        )
    maximum_self = max(self_distances)
    minimum_overlap = min(overlap_distances)
    clean_gap = minimum_overlap > maximum_self
    threshold = (
        maximum_self + 0.10 * (minimum_overlap - maximum_self)
        if clean_gap
        else None
    )
    score_threshold = threshold if threshold is not None else math.inf
    strategies = (
        "flat",
        "fixed_self_window",
        "scheduled_identity_plus_same_source_forward_model",
    )
    reports = {
        str(capacity): {
            strategy: _strategy(
                rows,
                capacity=capacity,
                strategy=strategy,
                threshold=score_threshold,
            )
            for strategy in strategies
        }
        for capacity in (1, 2, 3, 4)
    }
    selected = reports["3"]
    scheduled = selected[strategies[2]]["metrics"]
    flat = selected["flat"]["metrics"]
    fixed = selected["fixed_self_window"]["metrics"]
    criteria = {
        "clean_self_overlap_distance_gap": clean_gap,
        "all_nine_self_probe_forward_distances_below_raw_efference_distances": all(
            forward < raw for forward, raw in zip(self_distances, raw_distances)
        ),
        "selected_capacity_scheduled_f1_at_least_0_90": scheduled["f1"] >= 0.90,
        "selected_capacity_scheduled_minus_flat_f1_at_least_0_20": scheduled[
            "f1"
        ]
        - flat["f1"]
        >= 0.20 - 1e-12,
        "selected_capacity_world_recall_at_least_0_90": scheduled[
            "world_recall"
        ]
        >= 0.90,
        "selected_capacity_self_suppression_specificity_at_least_0_90": scheduled[
            "self_suppression_specificity"
        ]
        >= 0.90,
        "fixed_window_world_recall_below_scheduled": fixed["world_recall"]
        < scheduled["world_recall"],
        "capacity_one_f1_below_capacity_three": reports["1"][strategies[2]][
            "metrics"
        ]["f1"]
        < scheduled["f1"],
        "capacity_three_f1_equals_capacity_four": math.isclose(
            reports["4"][strategies[2]]["metrics"]["f1"], scheduled["f1"]
        ),
        "all_nine_leave_one_command_out_folds_present": len(fold_reports) == 9,
        "all_artifact_hashes_verified": True,
        "maximum_shared_model_scalars_at_most_13": all(
            fold["model_state_scalars"] <= 13 for fold in fold_reports
        ),
        "confirmation_audio_not_accessed": True,
    }
    passed = all(criteria.values())
    return {
        "mode": "h3_same_source_forward_model_spent_development",
        "evidence_scope": "already_spent_h3_captures_only",
        "representation": {
            "bands": 12,
            "band_edges_hz": BAND_EDGES_HZ.tolist(),
            "shared_path_scalars": 12,
            "shared_path_count_scalars": 1,
            "per_utterance_sensory_templates": 0,
            "workspace_slots": 3,
            "runtime_labels_consumed": False,
        },
        "fold_count": len(fold_reports),
        "self_probe_distances": self_distances,
        "raw_efference_self_probe_distances": raw_distances,
        "overlap_distances": overlap_distances,
        "maximum_self_probe_distance": maximum_self,
        "minimum_overlap_distance": minimum_overlap,
        "clean_gap": minimum_overlap - maximum_self,
        "selected_threshold": threshold,
        "threshold_rule": "max_self + 0.10 * (min_overlap - max_self)",
        "folds": fold_reports,
        "rows": rows,
        "reports": reports,
        "criteria": criteria,
        "all_development_gates_passed": passed,
        "fresh_live_preregistration_allowed": passed,
        "fresh_live_audio_accessed": False,
        "promotion_allowed": False,
        "claim_boundary": (
            "A pass supports only fresh-test readiness for one bounded same-source "
            "forward model at the existing geometry. It does not extend H3 without "
            "a separately preregistered fresh live result."
        ),
    }
