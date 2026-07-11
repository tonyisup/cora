"""Design-stage factorized five-coordinate tract plant for H4 Option E."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np

from .geometric_articulatory_synthesis import (
    FORMANT_FREQUENCIES_HZ,
    NEUTRAL_AREAS_CM2,
    SENSORY_FREQUENCIES_HZ,
    TractGeometry,
)
from .io.motor_plan import VOWEL_FORMANTS_HZ


VOWELS = ("a", "e", "i", "o", "u")
HALTON_BASES = (2, 3, 5, 7, 11)
JAW_WEIGHTS = np.array(
    [0.25, 0.45, 0.7, 0.9, 1.0, 1.0, 1.0, 0.9, 0.6, 0.3],
    dtype=np.float64,
)
LIP_WEIGHTS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.6, 1.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class FactorizedPlantConfig:
    jaw_gain: float
    constriction_depth: float
    constriction_width: float
    lip_gain: float
    protrusion_narrowing: float
    maximum_length_cm: float


def _motor5(value: np.ndarray) -> np.ndarray:
    motor = np.asarray(value, dtype=np.float64).reshape(-1)
    if motor.shape != (5,) or not np.all(np.isfinite(motor)):
        raise ValueError("factorized motor must contain five finite coordinates")
    if np.any(motor < 0.0) or np.any(motor > 1.0):
        raise ValueError("factorized motor violates [0,1] bounds")
    return motor


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    value = index
    while value:
        result += factor * (value % base)
        value //= base
        factor /= base
    return result


def halton5(index: int) -> np.ndarray:
    if index <= 0:
        raise ValueError("Halton index must be positive")
    return np.array(
        [_radical_inverse(index, base) for base in HALTON_BASES],
        dtype=np.float64,
    )


def halton5_points(start: int, stop: int) -> np.ndarray:
    if stop < start:
        raise ValueError("Halton stop must not precede start")
    return np.stack([halton5(index) for index in range(start, stop + 1)])


def design_family() -> tuple[FactorizedPlantConfig, ...]:
    values = itertools.product(
        (0.3, 0.6),
        (2.2, 3.0),
        (0.09, 0.13),
        (1.2, 2.0),
        (0.3, 0.8),
        (18.5, 20.0),
    )
    return tuple(FactorizedPlantConfig(*row) for row in values)


def factorized_area_function(
    motor: np.ndarray, config: FactorizedPlantConfig
) -> TractGeometry:
    j, c, p, lip_aperture, lip_protrusion = _motor5(motor)
    centers = (np.arange(10, dtype=np.float64) + 0.5) / 10.0
    constriction_center = 0.12 + 0.68 * p
    constriction = np.exp(
        -0.5 * ((centers - constriction_center) / config.constriction_width) ** 2
    )
    log_areas = (
        np.log(NEUTRAL_AREAS_CM2)
        + config.jaw_gain * (j - 0.5) * JAW_WEIGHTS
        - config.constriction_depth * c * constriction
        + config.lip_gain * (lip_aperture - 0.5) * LIP_WEIGHTS
        - config.protrusion_narrowing * lip_protrusion * LIP_WEIGHTS
    )
    areas = np.clip(np.exp(log_areas), 0.04, 9.0)
    total_length = 15.5 + (config.maximum_length_cm - 15.5) * lip_protrusion
    return TractGeometry(
        areas_cm2=areas,
        lengths_cm=np.full(10, total_length / 10.0),
        total_length_cm=float(total_length),
    )


def factorized_transfer(
    motor: np.ndarray, frequencies_hz: np.ndarray, config: FactorizedPlantConfig
) -> np.ndarray:
    geometry = factorized_area_function(motor, config)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    if np.any(frequencies < 0.0) or not np.all(np.isfinite(frequencies)):
        raise ValueError("transfer frequencies must be finite and nonnegative")
    rho = 1.14
    speed = 35_000.0
    angular_wave = 2.0 * np.pi * frequencies / speed
    gamma = 0.002 + 0.00001 * frequencies + 1j * angular_wave
    a = np.ones_like(gamma, dtype=np.complex128)
    b = np.zeros_like(gamma, dtype=np.complex128)
    c = np.zeros_like(gamma, dtype=np.complex128)
    d = np.ones_like(gamma, dtype=np.complex128)
    for area, length in zip(geometry.areas_cm2, geometry.lengths_cm):
        impedance = rho * speed / area
        cosh = np.cosh(gamma * length)
        sinh = np.sinh(gamma * length)
        a, b, c, d = (
            a * cosh + b * (sinh / impedance),
            a * (impedance * sinh) + b * cosh,
            c * cosh + d * (sinh / impedance),
            c * (impedance * sinh) + d * cosh,
        )
    lip_area = float(geometry.areas_cm2[-1])
    lip_radius = math.sqrt(lip_area / math.pi)
    ka = angular_wave * lip_radius
    lip_impedance = (rho * speed / lip_area) * (
        0.25 * ka**2 + 1j * 0.61 * ka
    )
    denominator = c * lip_impedance + d
    return 1.0 / np.where(np.abs(denominator) > 1e-18, denominator, 1e-18)


def factorized_formants(
    motor: np.ndarray, config: FactorizedPlantConfig
) -> tuple[float, float, float] | None:
    magnitude = np.abs(factorized_transfer(motor, FORMANT_FREQUENCIES_HZ, config))
    candidates = [
        index
        for index in range(1, len(magnitude) - 1)
        if magnitude[index] > magnitude[index - 1]
        and magnitude[index] >= magnitude[index + 1]
    ]
    selected: list[float] = []
    for index in candidates:
        frequency = float(FORMANT_FREQUENCIES_HZ[index])
        if not selected or frequency - selected[-1] >= 150.0:
            selected.append(frequency)
        if len(selected) == 3:
            return selected[0], selected[1], selected[2]
    return None


def factorized_sensory(motor: np.ndarray, config: FactorizedPlantConfig) -> np.ndarray:
    magnitude = np.abs(factorized_transfer(motor, SENSORY_FREQUENCIES_HZ, config))
    vector = np.log(magnitude + 1e-12)
    vector -= float(np.mean(vector))
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("factorized plant envelope has insufficient structure")
    return vector / norm


def relative_formant_rmse(vowel: str, formants: tuple[float, float, float]) -> float:
    reference = np.asarray(VOWEL_FORMANTS_HZ[vowel], dtype=np.float64)
    measured = np.asarray(formants, dtype=np.float64)
    return float(np.sqrt(np.mean(((measured - reference) / reference) ** 2)))


def nearest_vowel(formants: tuple[float, float, float]) -> str:
    return min(VOWELS, key=lambda vowel: (relative_formant_rmse(vowel, formants), vowel))


def _neighbors(motor: np.ndarray, delta: float) -> tuple[np.ndarray, ...]:
    values = [motor.copy()]
    for axis in range(5):
        for direction in (-1.0, 1.0):
            candidate = motor.copy()
            candidate[axis] = np.clip(candidate[axis] + direction * delta, 0.0, 1.0)
            values.append(candidate)
    return tuple(values)


def select_targets(
    points: np.ndarray, config: FactorizedPlantConfig
) -> tuple[list[dict[str, object]], list[tuple[float, float, float] | None]]:
    formants = [factorized_formants(point, config) for point in points]
    used: set[int] = set()
    targets: list[dict[str, object]] = []
    for vowel in VOWELS:
        candidates = [
            (relative_formant_rmse(vowel, value), index)
            for index, value in enumerate(formants)
            if value is not None and index not in used
        ]
        cost, index = min(candidates, key=lambda row: (row[0], row[1]))
        used.add(index)
        value = formants[index]
        assert value is not None
        valid_neighbors = 0
        neighbors = _neighbors(points[index], 0.04)
        for neighbor in neighbors:
            neighbor_formants = factorized_formants(neighbor, config)
            if (
                neighbor_formants is not None
                and nearest_vowel(neighbor_formants) == vowel
                and relative_formant_rmse(vowel, neighbor_formants) <= 0.30
            ):
                valid_neighbors += 1
        targets.append(
            {
                "vowel": vowel,
                "design_offset": index,
                "motor": points[index].tolist(),
                "formants_hz": list(value),
                "relative_formant_rmse": cost,
                "nearest_vowel": nearest_vowel(value),
                "design_valid": nearest_vowel(value) == vowel and cost <= 0.25,
                "valid_neighbor_fraction": valid_neighbors / len(neighbors),
            }
        )
    return targets, formants


def evaluate_design_family() -> dict[str, object]:
    points = halton5_points(4097, 5120)
    summaries: list[dict[str, object]] = []
    for ordinal, config in enumerate(design_family()):
        targets, formants = select_targets(points, config)
        rmses = [float(target["relative_formant_rmse"]) for target in targets]
        summary = {
            "ordinal": ordinal,
            "config": asdict(config),
            "points_with_three_formants": sum(value is not None for value in formants),
            "valid_vowel_count": sum(bool(target["design_valid"]) for target in targets),
            "minimum_valid_neighbor_fraction": min(
                float(target["valid_neighbor_fraction"]) for target in targets
            ),
            "maximum_target_relative_formant_rmse": max(rmses),
            "mean_target_relative_formant_rmse": float(np.mean(rmses)),
            "targets": targets,
        }
        summaries.append(summary)

    def key(summary: dict[str, object]) -> tuple[float, ...]:
        return (
            -float(summary["valid_vowel_count"]),
            -float(summary["minimum_valid_neighbor_fraction"]),
            float(summary["maximum_target_relative_formant_rmse"]),
            float(summary["mean_target_relative_formant_rmse"]),
            float(summary["ordinal"]),
        )

    selected = min(summaries, key=key)
    return {
        "mode": "h4_option_e_factorized_plant_design_only",
        "evidence_status": "development_not_confirmation",
        "design_indices_inclusive": [4097, 5120],
        "confirmation_indices_accessed": False,
        "candidate_count": len(summaries),
        "selected_ordinal": selected["ordinal"],
        "selected_config": selected["config"],
        "selected_key": list(key(selected)),
        "selected_targets": selected["targets"],
        "candidates": summaries,
        "recorded_audio_accessed": False,
        "protected_data_accessed": False,
        "promotion_allowed": False,
    }
