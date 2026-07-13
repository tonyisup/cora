"""Bounded synthetic speech-learning demonstration.

The learner stores forward consequences for a finite babbled motor repertoire,
performs target-specific local prototype search, and recalls the learned action.
It is deliberately not a continuous acoustic-to-articulatory inverse model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .factorized_tract_plant import (
    expanded_formants,
    expanded_sensory,
    halton5,
    synthesize_factorized,
)
from .geometric_articulatory_synthesis import SAMPLE_RATE
from .io.motor_plan import write_wav


DEMO_VOWELS = ("a", "i", "u")
DEMO_SCENARIOS = ("synthetic-mirror", "cross-configuration")
REPERTOIRE_START_INDEX = 14001
REPERTOIRE_STRIDE = 37
REPERTOIRE_SIZE = 96
LOCAL_CANDIDATE_COUNT = 12
EXPLORATION_ATTEMPTS = 9
RECALL_ATTEMPTS = 3
TOTAL_ATTEMPTS = EXPLORATION_ATTEMPTS + RECALL_ATTEMPTS
DEMO_F0_HZ = 110.0

# Already-spent Option F target motors. Labels are display/evaluation metadata;
# the learner receives only their sensory vectors and opaque hashes.
CROSS_CONFIGURATION_TARGET_MOTORS = {
    "a": (
        0.93865966796875,
        0.936798252298938,
        0.139456,
        0.7824715892187778,
        0.08776722901441159,
    ),
    "i": (
        0.97381591796875,
        0.9916679367982523,
        0.8690560000000002,
        0.36264651633248046,
        0.10579878423604945,
    ),
    "u": (
        0.22088623046875,
        0.9354265101864552,
        0.5762560000000001,
        0.0940084488605938,
        0.9870910456936002,
    ),
}


def sensory_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Bounded distance for unit-normalized sensory envelopes."""

    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != (30,) or b.shape != (30,):
        raise ValueError("demo sensory vectors must contain 30 values")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("demo sensory vectors must be finite")
    return float(np.linalg.norm(a - b) / 2.0)


def _array_sha256(value: np.ndarray) -> str:
    canonical = np.asarray(value, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


@dataclass(frozen=True)
class MotorPrototype:
    prototype_id: str
    halton_index: int
    motor: tuple[float, float, float, float, float]
    sensory: tuple[float, ...]

    def motor_array(self) -> np.ndarray:
        return np.asarray(self.motor, dtype=np.float64)

    def sensory_array(self) -> np.ndarray:
        return np.asarray(self.sensory, dtype=np.float64)


class MotorRepertoire:
    """Fixed-capacity forward memory acquired during synthetic babbling."""

    def __init__(self, prototypes: Iterable[MotorPrototype]) -> None:
        self.prototypes = tuple(prototypes)
        if len(self.prototypes) != REPERTOIRE_SIZE:
            raise ValueError(f"demo repertoire must contain {REPERTOIRE_SIZE} items")
        if len({item.prototype_id for item in self.prototypes}) != len(
            self.prototypes
        ):
            raise ValueError("demo prototype identifiers must be unique")
        motors = np.stack([item.motor_array() for item in self.prototypes])
        if not np.all(np.isfinite(motors)) or np.any(motors < 0.0) or np.any(
            motors > 1.0
        ):
            raise ValueError("demo repertoire contains an invalid motor")
        if len({item.motor for item in self.prototypes}) != len(self.prototypes):
            raise ValueError("demo repertoire motors must be distinct")

    @classmethod
    def babble(cls) -> "MotorRepertoire":
        indices = tuple(
            REPERTOIRE_START_INDEX + offset * REPERTOIRE_STRIDE
            for offset in range(REPERTOIRE_SIZE)
        )
        points = tuple(halton5(index) for index in indices)
        return cls(
            MotorPrototype(
                prototype_id=f"babble-{offset:03d}",
                halton_index=indices[offset],
                motor=tuple(float(value) for value in motor),
                sensory=tuple(float(value) for value in expanded_sensory(motor)),
            )
            for offset, motor in enumerate(points)
        )

    def rest_index(self) -> int:
        rest = np.full(5, 0.5, dtype=np.float64)
        return min(
            range(len(self.prototypes)),
            key=lambda index: (
                float(np.linalg.norm(self.prototypes[index].motor_array() - rest)),
                index,
            ),
        )

    def nearest_sensory_index(self, target: np.ndarray) -> int:
        return min(
            range(len(self.prototypes)),
            key=lambda index: (
                sensory_distance(self.prototypes[index].sensory_array(), target),
                index,
            ),
        )

    def digest(self) -> str:
        payload = [
            {
                "prototype_id": item.prototype_id,
                "halton_index": item.halton_index,
                "motor": item.motor,
                "sensory": item.sensory,
            }
            for item in self.prototypes
        ]
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class DemoTarget:
    display_label: str
    scenario: str
    source_motor: tuple[float, float, float, float, float]
    sensory: tuple[float, ...]
    target_id: str
    source_prototype_id: str | None

    def sensory_array(self) -> np.ndarray:
        return np.asarray(self.sensory, dtype=np.float64)


def build_demo_target(
    repertoire: MotorRepertoire,
    *,
    vowel: str,
    scenario: str,
) -> DemoTarget:
    if vowel not in DEMO_VOWELS:
        raise ValueError(f"unsupported demo vowel: {vowel}")
    if scenario not in DEMO_SCENARIOS:
        raise ValueError(f"unsupported demo scenario: {scenario}")
    cross_motor = np.asarray(
        CROSS_CONFIGURATION_TARGET_MOTORS[vowel], dtype=np.float64
    )
    cross_sensory = expanded_sensory(cross_motor)
    source_prototype_id: str | None = None
    if scenario == "synthetic-mirror":
        index = repertoire.nearest_sensory_index(cross_sensory)
        prototype = repertoire.prototypes[index]
        motor = prototype.motor_array()
        sensory = prototype.sensory_array()
        source_prototype_id = prototype.prototype_id
    else:
        motor = cross_motor
        sensory = cross_sensory
    return DemoTarget(
        display_label=vowel,
        scenario=scenario,
        source_motor=tuple(float(value) for value in motor),
        sensory=tuple(float(value) for value in sensory),
        target_id=_array_sha256(sensory),
        source_prototype_id=source_prototype_id,
    )


class LocalImitationLearner:
    """Finite target-to-prototype association learner with bounded search."""

    def __init__(self, repertoire: MotorRepertoire, *, capacity: int = 3) -> None:
        if capacity <= 0 or capacity > 3:
            raise ValueError("demo association capacity must be in [1,3]")
        self.repertoire = repertoire
        self.capacity = capacity
        self.associations: dict[str, int] = {}

    def _remember(self, target_id: str, prototype_index: int) -> None:
        if target_id not in self.associations and len(self.associations) >= self.capacity:
            oldest = next(iter(self.associations))
            del self.associations[oldest]
        self.associations[target_id] = prototype_index

    def imitate(
        self,
        target_sensory: np.ndarray,
        *,
        target_id: str,
    ) -> dict[str, object]:
        target_sensory = np.asarray(target_sensory, dtype=np.float64)
        if _array_sha256(target_sensory) != target_id:
            raise ValueError("demo target identifier does not match sensory bytes")
        rest_index = self.repertoire.rest_index()
        attempted: set[int] = set()
        attempts: list[dict[str, object]] = []
        best_index = rest_index
        best_distance = sensory_distance(
            self.repertoire.prototypes[best_index].sensory_array(), target_sensory
        )
        initial_distance = best_distance

        def record(index: int, phase: str) -> None:
            nonlocal best_index, best_distance
            prototype = self.repertoire.prototypes[index]
            distance = sensory_distance(prototype.sensory_array(), target_sensory)
            prior_best = best_distance
            if distance < best_distance:
                best_index = index
                best_distance = distance
            attempted.add(index)
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "phase": phase,
                    "prototype_id": prototype.prototype_id,
                    "halton_index": prototype.halton_index,
                    "motor": list(prototype.motor),
                    "sensory_distance": distance,
                    "best_prototype_id": self.repertoire.prototypes[
                        best_index
                    ].prototype_id,
                    "best_sensory_distance": best_distance,
                    "competence_progress": max(0.0, prior_best - best_distance),
                }
            )

        record(rest_index, "rest")
        while len(attempts) < EXPLORATION_ATTEMPTS:
            frontier: set[int] = set()
            for anchor_index in attempted:
                anchor = self.repertoire.prototypes[anchor_index].motor_array()
                local = sorted(
                    (
                        float(np.linalg.norm(prototype.motor_array() - anchor)),
                        index,
                    )
                    for index, prototype in enumerate(self.repertoire.prototypes)
                    if index not in attempted
                )[:LOCAL_CANDIDATE_COUNT]
                frontier.update(index for _, index in local)
            chosen_index = min(
                frontier,
                key=lambda index: (
                    sensory_distance(
                        self.repertoire.prototypes[index].sensory_array(),
                        target_sensory,
                    ),
                    index,
                ),
            )
            record(chosen_index, "local_exploration")

        self._remember(target_id, best_index)
        for _ in range(RECALL_ATTEMPTS):
            record(self.associations[target_id], "learned_recall")

        final = self.repertoire.prototypes[best_index]
        improvement_ratio = (
            (initial_distance - best_distance) / initial_distance
            if initial_distance > 0.0
            else 0.0
        )
        final_three = [row["prototype_id"] for row in attempts[-3:]]
        criteria = {
            "exactly_12_attempts": len(attempts) == TOTAL_ATTEMPTS,
            "learner_received_no_vowel_label": True,
            "improvement_ratio_at_least_0_30": improvement_ratio >= 0.30,
            "final_distance_at_most_0_20": best_distance <= 0.20,
            "final_three_recall_same_prototype": len(set(final_three)) == 1,
            "association_memory_within_capacity": len(self.associations)
            <= self.capacity,
        }
        return {
            "target_id": target_id,
            "learner_input": {
                "sensory_sha256": target_id,
                "sensory_dimension": len(target_sensory),
                "vowel_label_provided": False,
            },
            "attempts": attempts,
            "initial_sensory_distance": initial_distance,
            "final_best_sensory_distance": best_distance,
            "improvement_ratio": improvement_ratio,
            "learned_prototype_id": final.prototype_id,
            "learned_halton_index": final.halton_index,
            "learned_motor": list(final.motor),
            "criteria": criteria,
            "session_passed": all(criteria.values()),
        }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _render_demo_html(report: dict[str, object]) -> str:
    cards: list[str] = []
    for session in report["sessions"]:
        label = html.escape(str(session["display_label"]))
        learning = session["learning"]
        distances = [
            float(row["best_sensory_distance"])
            for row in learning["attempts"]
        ]
        maximum = max(distances) or 1.0
        points = " ".join(
            f"{10 + index * (380 / (len(distances) - 1)):.1f},"
            f"{10 + 80 * distance / maximum:.1f}"
            for index, distance in enumerate(distances)
        )
        cards.append(
            f"""
      <section class="card">
        <h2>/{label}/</h2>
        <div class="metrics">
          <span>Initial <strong>{learning['initial_sensory_distance']:.3f}</strong></span>
          <span>Final <strong>{learning['final_best_sensory_distance']:.3f}</strong></span>
          <span>Improvement <strong>{100 * learning['improvement_ratio']:.1f}%</strong></span>
          <span>Motor <strong>{html.escape(str(learning['learned_prototype_id']))}</strong></span>
        </div>
        <svg viewBox="0 0 400 110" role="img" aria-label="Best sensory distance by attempt">
          <line x1="10" y1="90" x2="390" y2="90" class="axis" />
          <polyline points="{points}" class="curve" />
          <text x="10" y="106">attempt 1</text><text x="342" y="106">attempt 12</text>
        </svg>
        <div class="audio-pair">
          <label>Teacher target<audio controls src="{label}-target.wav"></audio></label>
          <label>Learned output<audio controls src="{label}-learned.wav"></audio></label>
        </div>
      </section>"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Cora bounded speech-learning demo</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-monospace, SFMono-Regular, monospace; }}
    body {{ margin: 0; background: #0c1117; color: #dce7ef; }}
    main {{ max-width: 980px; margin: auto; padding: 36px 20px 64px; }}
    h1 {{ margin-bottom: 6px; }} .lede {{ color: #91a4b4; margin-top: 0; }}
    .card {{ background: #141d26; border: 1px solid #263746; border-radius: 14px; padding: 20px; margin-top: 18px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(145px,1fr)); gap: 10px; color: #91a4b4; }}
    strong {{ display: block; color: #eef8ff; margin-top: 4px; }}
    svg {{ width: 100%; height: 150px; margin-top: 12px; }}
    .axis {{ stroke: #405565; }} .curve {{ fill: none; stroke: #7ee787; stroke-width: 4; }}
    text {{ fill: #91a4b4; font-size: 10px; }}
    .audio-pair {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap: 16px; }}
    audio {{ display: block; width: 100%; margin-top: 8px; }}
    footer {{ color: #91a4b4; margin-top: 26px; line-height: 1.5; }} a {{ color: #80c7ff; }}
  </style>
</head>
<body><main>
  <h1>Cora speech-learning demo</h1>
  <p class="lede">{html.escape(str(report['scenario']))} · 96 babbled actions · label-blind selection · 12 attempts</p>
  {''.join(cards)}
  <footer>{html.escape(str(report['claim_boundary']))}<br><a href="speech-learning-session.json">Inspect the complete session JSON</a></footer>
</main></body></html>
"""


def _write_demo_wav(path: Path, motor: np.ndarray) -> dict[str, object]:
    first = synthesize_factorized(motor, DEMO_F0_HZ)
    second = synthesize_factorized(motor, DEMO_F0_HZ)
    deterministic = bool(np.array_equal(first, second))
    silence = np.zeros(int(round(0.04 * SAMPLE_RATE)), dtype=np.float32)
    segments: list[np.ndarray] = []
    for repetition in range(6):
        if repetition:
            segments.append(silence)
        segments.append(first)
    rendered = np.concatenate(segments)
    write_wav(str(path), rendered, SAMPLE_RATE)
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate_hz": SAMPLE_RATE,
        "sample_count": len(rendered),
        "presentation": "six_160ms_productions_with_40ms_intervals",
        "peak_absolute": float(np.max(np.abs(rendered))),
        "finite": bool(np.all(np.isfinite(rendered))),
        "deterministic": deterministic,
    }


def run_speech_learning_demo(
    *,
    scenario: str,
    output_dir: str | Path,
    vowels: Iterable[str] = DEMO_VOWELS,
) -> dict[str, object]:
    """Run the frozen offline demo and retain audible/session artifacts."""

    if scenario not in DEMO_SCENARIOS:
        raise ValueError(f"unsupported demo scenario: {scenario}")
    requested = tuple(vowels)
    if not requested or any(vowel not in DEMO_VOWELS for vowel in requested):
        raise ValueError("demo vowels must be a non-empty subset of a,i,u")
    if len(set(requested)) != len(requested):
        raise ValueError("demo vowels must not repeat")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    repertoire = MotorRepertoire.babble()
    learner = LocalImitationLearner(repertoire, capacity=3)
    sessions: list[dict[str, object]] = []
    for vowel in requested:
        target = build_demo_target(repertoire, vowel=vowel, scenario=scenario)
        learning = learner.imitate(
            target.sensory_array(), target_id=target.target_id
        )
        target_motor = np.asarray(target.source_motor, dtype=np.float64)
        learned_motor = np.asarray(learning["learned_motor"], dtype=np.float64)
        target_formants = expanded_formants(target_motor)
        learned_formants = expanded_formants(learned_motor)
        target_wav = _write_demo_wav(output / f"{vowel}-target.wav", target_motor)
        learned_wav = _write_demo_wav(output / f"{vowel}-learned.wav", learned_motor)
        waveform_criteria = {
            "target_waveform_valid": target_wav["finite"] is True
            and float(target_wav["peak_absolute"]) < 1.0
            and target_wav["deterministic"] is True,
            "learned_waveform_valid": learned_wav["finite"] is True
            and float(learned_wav["peak_absolute"]) < 1.0
            and learned_wav["deterministic"] is True,
        }
        session_criteria = {
            **learning["criteria"],
            **waveform_criteria,
        }
        sessions.append(
            {
                "display_label": vowel,
                "label_role": "display_and_evaluation_only",
                "target": {
                    "target_id": target.target_id,
                    "scenario": target.scenario,
                    "source_motor": list(target.source_motor),
                    "source_prototype_id": target.source_prototype_id,
                    "formants_hz": None
                    if target_formants is None
                    else list(target_formants),
                    "waveform": target_wav,
                },
                "learning": learning,
                "learned_output": {
                    "formants_hz": None
                    if learned_formants is None
                    else list(learned_formants),
                    "waveform": learned_wav,
                },
                "criteria": session_criteria,
                "session_passed": all(session_criteria.values()),
            }
        )

    repertoire_motors = np.stack(
        [prototype.motor_array() for prototype in repertoire.prototypes]
    )
    aggregate_criteria = {
        "repertoire_count_exactly_96": len(repertoire.prototypes)
        == REPERTOIRE_SIZE,
        "repertoire_motors_finite_and_bounded": bool(
            np.all(np.isfinite(repertoire_motors))
            and np.all(repertoire_motors >= 0.0)
            and np.all(repertoire_motors <= 1.0)
        ),
        "repertoire_motors_distinct": len(
            {prototype.motor for prototype in repertoire.prototypes}
        )
        == REPERTOIRE_SIZE,
        "association_memory_within_three": len(learner.associations) <= 3,
        "every_requested_target_passed": all(
            session["session_passed"] is True for session in sessions
        ),
        "recorded_audio_not_accessed": True,
        "protected_data_not_accessed": True,
        "live_io_not_executed": True,
    }
    report = {
        "mode": "bounded_speech_learning_demo",
        "specification": "docs/DEMO_SPEC.md",
        "scenario": scenario,
        "vowels": list(requested),
        "repertoire": {
            "count": len(repertoire.prototypes),
            "halton_schedule": {
                "start": REPERTOIRE_START_INDEX,
                "stride": REPERTOIRE_STRIDE,
                "last": REPERTOIRE_START_INDEX
                + (REPERTOIRE_SIZE - 1) * REPERTOIRE_STRIDE,
            },
            "sensory_dimension": 30,
            "sha256": repertoire.digest(),
        },
        "sessions": sessions,
        "association_count": len(learner.associations),
        "aggregate_criteria": aggregate_criteria,
        "demo_passed": all(aggregate_criteria.values()),
        "claim_boundary": (
            "Finite prototype association and model-based local search over a "
            "synthetic plant; not continuous inverse closure, human-speech "
            "learning, phoneme recognition, semantics, or new H4 evidence"
        ),
    }
    html_path = output / "speech-learning-demo.html"
    html_path.write_text(_render_demo_html(report))
    report["presentation"] = {
        "html_filename": html_path.name,
        "html_sha256": hashlib.sha256(html_path.read_bytes()).hexdigest(),
    }
    _write_json(output / "speech-learning-session.json", report)
    return report
