"""Design-stage factorized five-coordinate tract plant for H4 Option E."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np

from .geometric_articulatory_synthesis import (
    DURATION_MS,
    F0_CONDITIONS_HZ,
    FORMANT_FREQUENCIES_HZ,
    NEUTRAL_AREAS_CM2,
    SAMPLE_RATE,
    SENSORY_FREQUENCIES_HZ,
    TractGeometry,
)
from .io.motor_plan import VOWEL_FORMANTS_HZ, _source_pulse_train


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


FROZEN_OPTION_E_CONFIG = FactorizedPlantConfig(
    jaw_gain=0.3,
    constriction_depth=3.0,
    constriction_width=0.09,
    lip_gain=2.0,
    protrusion_narrowing=0.8,
    maximum_length_cm=20.0,
)
EXPANDED_FORMANT_FREQUENCIES_HZ = np.arange(80.0, 5000.0 + 5.0, 5.0)
EXPANDED_SENSORY_FREQUENCIES_HZ = np.geomspace(150.0, 5000.0, 30)


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


def expanded_formants(
    motor: np.ndarray, config: FactorizedPlantConfig = FROZEN_OPTION_E_CONFIG
) -> tuple[float, float, float] | None:
    magnitude = np.abs(
        factorized_transfer(motor, EXPANDED_FORMANT_FREQUENCIES_HZ, config)
    )
    candidates = [
        index
        for index in range(1, len(magnitude) - 1)
        if magnitude[index] > magnitude[index - 1]
        and magnitude[index] >= magnitude[index + 1]
    ]
    selected: list[float] = []
    for index in candidates:
        frequency = float(EXPANDED_FORMANT_FREQUENCIES_HZ[index])
        if not selected or frequency - selected[-1] >= 150.0:
            selected.append(frequency)
        if len(selected) == 3:
            return selected[0], selected[1], selected[2]
    return None


def expanded_sensory(
    motor: np.ndarray, config: FactorizedPlantConfig = FROZEN_OPTION_E_CONFIG
) -> np.ndarray:
    magnitude = np.abs(
        factorized_transfer(motor, EXPANDED_SENSORY_FREQUENCIES_HZ, config)
    )
    vector = np.log(magnitude + 1e-12)
    vector -= float(np.mean(vector))
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("expanded plant envelope has insufficient structure")
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


def synthesize_factorized(
    motor: np.ndarray,
    f0_hz: float,
    config: FactorizedPlantConfig = FROZEN_OPTION_E_CONFIG,
) -> np.ndarray:
    m = _motor5(motor)
    if f0_hz not in F0_CONDITIONS_HZ:
        raise ValueError("F0 is outside the frozen synthetic conditions")
    samples = int(round(DURATION_MS * SAMPLE_RATE / 1000.0))
    source = _source_pulse_train(samples, SAMPLE_RATE, f0_hz)
    frequencies = np.fft.rfftfreq(samples, 1.0 / SAMPLE_RATE)
    response = factorized_transfer(m, frequencies, config)
    waveform = np.fft.irfft(np.fft.rfft(source) * response, n=samples)
    ramp = min(SAMPLE_RATE // 100, samples // 2)
    envelope = np.ones(samples, dtype=np.float64)
    envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
    envelope[-ramp:] = np.linspace(1.0, 0.0, ramp)
    waveform *= envelope
    peak = float(np.max(np.abs(waveform))) or 1.0
    return (waveform * (0.95 / peak)).astype(np.float32)


def _confirmation_targets(
    points: np.ndarray,
    formants: list[tuple[float, float, float] | None],
) -> list[dict[str, object]]:
    used: set[int] = set()
    targets: list[dict[str, object]] = []
    for vowel in VOWELS:
        candidates = [
            (relative_formant_rmse(vowel, value), index)
            for index, value in enumerate(formants)
            if value is not None and index not in used
        ]
        cost, offset = min(candidates, key=lambda row: (row[0], row[1]))
        used.add(offset)
        value = formants[offset]
        assert value is not None
        motor = points[offset]
        neighbor_rows: list[dict[str, object]] = []
        for neighbor in _neighbors(motor, 0.03):
            neighbor_formants = factorized_formants(neighbor, FROZEN_OPTION_E_CONFIG)
            valid = (
                neighbor_formants is not None
                and nearest_vowel(neighbor_formants) == vowel
                and relative_formant_rmse(vowel, neighbor_formants) <= 0.25
            )
            neighbor_rows.append(
                {
                    "motor": neighbor.tolist(),
                    "formants_hz": None
                    if neighbor_formants is None
                    else list(neighbor_formants),
                    "valid": valid,
                }
            )
        targets.append(
            {
                "vowel": vowel,
                "halton_index": 8193 + offset,
                "motor": motor.tolist(),
                "formants_hz": list(value),
                "relative_formant_rmse": cost,
                "nearest_vowel": nearest_vowel(value),
                "valid_neighbor_fraction": sum(
                    bool(row["valid"]) for row in neighbor_rows
                )
                / len(neighbor_rows),
                "neighbors": neighbor_rows,
            }
        )
    return targets


def evaluate_option_e_confirmation() -> dict[str, object]:
    points = halton5_points(8193, 9216)
    formants: list[tuple[float, float, float] | None] = []
    geometry_valid = True
    transfer_valid = True
    for point in points:
        geometry = factorized_area_function(point, FROZEN_OPTION_E_CONFIG)
        geometry_valid &= bool(
            np.all(np.isfinite(geometry.areas_cm2))
            and np.all(geometry.areas_cm2 >= 0.04)
            and np.all(geometry.areas_cm2 <= 9.0)
            and 15.5 <= geometry.total_length_cm <= 20.0
        )
        transfer = factorized_transfer(
            point, SENSORY_FREQUENCIES_HZ, FROZEN_OPTION_E_CONFIG
        )
        transfer_valid &= bool(np.all(np.isfinite(transfer)))
        formants.append(factorized_formants(point, FROZEN_OPTION_E_CONFIG))

    targets = _confirmation_targets(points, formants)
    waveform_rows: list[dict[str, object]] = []
    for target in targets:
        motor = np.asarray(target["motor"], dtype=np.float64)
        for f0_hz in F0_CONDITIONS_HZ:
            first = synthesize_factorized(motor, f0_hz)
            second = synthesize_factorized(motor, f0_hz)
            waveform_rows.append(
                {
                    "vowel": target["vowel"],
                    "f0_hz": f0_hz,
                    "finite": bool(np.all(np.isfinite(first))),
                    "peak": float(np.max(np.abs(first))),
                    "deterministic": bool(np.array_equal(first, second)),
                }
            )

    target_formants = np.asarray(
        [target["formants_hz"] for target in targets], dtype=np.float64
    )
    references = np.asarray(
        [VOWEL_FORMANTS_HZ[target["vowel"]] for target in targets],
        dtype=np.float64,
    )
    formant_mae = np.mean(np.abs(target_formants - references), axis=0)
    pairwise_rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(targets, 2):
        left_sensory = factorized_sensory(
            np.asarray(left["motor"]), FROZEN_OPTION_E_CONFIG
        )
        right_sensory = factorized_sensory(
            np.asarray(right["motor"]), FROZEN_OPTION_E_CONFIG
        )
        pairwise_rows.append(
            {
                "pair": f"{left['vowel']}{right['vowel']}",
                "half_euclidean_sensory_distance": float(
                    np.linalg.norm(left_sensory - right_sensory) / 2.0
                ),
            }
        )

    p0_criteria = {
        "all_confirmation_geometry_valid": bool(geometry_valid),
        "all_confirmation_transfers_finite": bool(transfer_valid),
        "all_confirmation_points_have_three_formants": all(
            value is not None for value in formants
        ),
        "all_target_f0_waveforms_finite_and_below_one": all(
            bool(row["finite"]) and float(row["peak"]) < 1.0
            for row in waveform_rows
        ),
        "all_target_f0_waveforms_deterministic": all(
            bool(row["deterministic"]) for row in waveform_rows
        ),
    }
    p2_criteria = {
        "each_target_nearest_identity_matches": all(
            target["nearest_vowel"] == target["vowel"] for target in targets
        ),
        "maximum_target_relative_formant_rmse_at_most_0_15": max(
            float(target["relative_formant_rmse"]) for target in targets
        )
        <= 0.15,
        "mean_target_f1_mae_at_most_75_hz": float(formant_mae[0]) <= 75.0,
        "mean_target_f2_mae_at_most_150_hz": float(formant_mae[1]) <= 150.0,
        "mean_target_f3_mae_at_most_150_hz": float(formant_mae[2]) <= 150.0,
        "minimum_pairwise_sensory_distance_at_least_0_25": min(
            float(row["half_euclidean_sensory_distance"])
            for row in pairwise_rows
        )
        >= 0.25,
        "minimum_valid_neighborhood_fraction_at_least_0_80": min(
            float(target["valid_neighbor_fraction"]) for target in targets
        )
        >= 0.8,
    }
    p0_passed = all(p0_criteria.values())
    p2_passed = all(p2_criteria.values())
    return {
        "mode": "h4_option_e_factorized_plant_confirmation",
        "evidence_scope": "synthetic_plant_only",
        "confirmation_indices_inclusive": [8193, 9216],
        "confirmation_count": len(points),
        "config": asdict(FROZEN_OPTION_E_CONFIG),
        "p0_numerical_physical": {
            "points_with_three_formants": sum(value is not None for value in formants),
            "criteria": p0_criteria,
            "gate_passed": p0_passed,
        },
        "p2_expressivity": {
            "targets": targets,
            "formant_mae_hz": {
                "f1": float(formant_mae[0]),
                "f2": float(formant_mae[1]),
                "f3": float(formant_mae[2]),
            },
            "pairwise_sensory_distances": pairwise_rows,
            "waveforms": waveform_rows,
            "criteria": p2_criteria,
            "gate_passed": p2_passed,
        },
        "all_plant_gates_passed": p0_passed and p2_passed,
        "forward_learning_executed": False,
        "inverse_search_executed": False,
        "recorded_audio_accessed": False,
        "protected_data_accessed": False,
        "promotion_allowed": False,
    }


def _option_f_targets(
    points: np.ndarray,
    formants: list[tuple[float, float, float] | None],
) -> list[dict[str, object]]:
    used: set[int] = set()
    targets: list[dict[str, object]] = []
    for vowel in VOWELS:
        candidates = [
            (relative_formant_rmse(vowel, value), index)
            for index, value in enumerate(formants)
            if value is not None and index not in used
        ]
        cost, offset = min(candidates, key=lambda row: (row[0], row[1]))
        used.add(offset)
        value = formants[offset]
        assert value is not None
        motor = points[offset]
        neighbor_rows: list[dict[str, object]] = []
        for neighbor in _neighbors(motor, 0.03):
            neighbor_formants = expanded_formants(neighbor)
            valid = (
                neighbor_formants is not None
                and nearest_vowel(neighbor_formants) == vowel
                and relative_formant_rmse(vowel, neighbor_formants) <= 0.25
            )
            neighbor_rows.append(
                {
                    "motor": neighbor.tolist(),
                    "formants_hz": None
                    if neighbor_formants is None
                    else list(neighbor_formants),
                    "valid": valid,
                }
            )
        targets.append(
            {
                "vowel": vowel,
                "halton_index": 12289 + offset,
                "motor": motor.tolist(),
                "formants_hz": list(value),
                "relative_formant_rmse": cost,
                "nearest_vowel": nearest_vowel(value),
                "valid_neighbor_fraction": sum(
                    bool(row["valid"]) for row in neighbor_rows
                )
                / len(neighbor_rows),
                "neighbors": neighbor_rows,
            }
        )
    return targets


def evaluate_option_f_confirmation() -> dict[str, object]:
    points = halton5_points(12289, 13312)
    formants: list[tuple[float, float, float] | None] = []
    geometry_valid = True
    transfer_valid = True
    for point in points:
        geometry = factorized_area_function(point, FROZEN_OPTION_E_CONFIG)
        geometry_valid &= bool(
            np.all(np.isfinite(geometry.areas_cm2))
            and np.all(geometry.areas_cm2 >= 0.04)
            and np.all(geometry.areas_cm2 <= 9.0)
            and 15.5 <= geometry.total_length_cm <= 20.0
        )
        transfer_valid &= bool(
            np.all(
                np.isfinite(
                    factorized_transfer(
                        point,
                        EXPANDED_SENSORY_FREQUENCIES_HZ,
                        FROZEN_OPTION_E_CONFIG,
                    )
                )
            )
        )
        formants.append(expanded_formants(point))

    targets = _option_f_targets(points, formants)
    waveform_rows: list[dict[str, object]] = []
    for target in targets:
        motor = np.asarray(target["motor"], dtype=np.float64)
        for f0_hz in F0_CONDITIONS_HZ:
            first = synthesize_factorized(motor, f0_hz)
            second = synthesize_factorized(motor, f0_hz)
            waveform_rows.append(
                {
                    "vowel": target["vowel"],
                    "f0_hz": f0_hz,
                    "finite": bool(np.all(np.isfinite(first))),
                    "peak": float(np.max(np.abs(first))),
                    "deterministic": bool(np.array_equal(first, second)),
                }
            )

    measured = np.asarray([target["formants_hz"] for target in targets])
    references = np.asarray(
        [VOWEL_FORMANTS_HZ[target["vowel"]] for target in targets]
    )
    formant_mae = np.mean(np.abs(measured - references), axis=0)
    pairwise_rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(targets, 2):
        distance = float(
            np.linalg.norm(
                expanded_sensory(np.asarray(left["motor"]))
                - expanded_sensory(np.asarray(right["motor"]))
            )
            / 2.0
        )
        pairwise_rows.append(
            {
                "pair": f"{left['vowel']}{right['vowel']}",
                "half_euclidean_expanded_sensory_distance": distance,
            }
        )

    p0_criteria = {
        "all_confirmation_geometry_valid": bool(geometry_valid),
        "all_expanded_transfers_finite": bool(transfer_valid),
        "all_confirmation_points_have_three_formants_through_5khz": all(
            value is not None for value in formants
        ),
        "all_target_f0_waveforms_finite_and_below_one": all(
            bool(row["finite"]) and float(row["peak"]) < 1.0
            for row in waveform_rows
        ),
        "all_target_f0_waveforms_deterministic": all(
            bool(row["deterministic"]) for row in waveform_rows
        ),
    }
    p2_criteria = {
        "each_target_nearest_identity_matches": all(
            target["nearest_vowel"] == target["vowel"] for target in targets
        ),
        "maximum_target_relative_formant_rmse_at_most_0_15": max(
            float(target["relative_formant_rmse"]) for target in targets
        )
        <= 0.15,
        "mean_target_f1_mae_at_most_75_hz": float(formant_mae[0]) <= 75.0,
        "mean_target_f2_mae_at_most_150_hz": float(formant_mae[1]) <= 150.0,
        "mean_target_f3_mae_at_most_150_hz": float(formant_mae[2]) <= 150.0,
        "minimum_pairwise_expanded_sensory_distance_at_least_0_25": min(
            float(row["half_euclidean_expanded_sensory_distance"])
            for row in pairwise_rows
        )
        >= 0.25,
        "minimum_valid_neighborhood_fraction_at_least_0_80": min(
            float(target["valid_neighbor_fraction"]) for target in targets
        )
        >= 0.8,
    }
    p0_passed = all(p0_criteria.values())
    p2_passed = all(p2_criteria.values())
    return {
        "mode": "h4_option_f_expanded_band_plant_confirmation",
        "evidence_scope": "synthetic_plant_only",
        "confirmation_indices_inclusive": [12289, 13312],
        "confirmation_count": len(points),
        "expanded_formant_band_hz": [80.0, 5000.0],
        "expanded_sensory_dimension": 30,
        "p0": {
            "points_with_three_formants": sum(value is not None for value in formants),
            "criteria": p0_criteria,
            "gate_passed": p0_passed,
        },
        "p2": {
            "targets": targets,
            "formant_mae_hz": {
                "f1": float(formant_mae[0]),
                "f2": float(formant_mae[1]),
                "f3": float(formant_mae[2]),
            },
            "pairwise_expanded_sensory_distances": pairwise_rows,
            "waveforms": waveform_rows,
            "criteria": p2_criteria,
            "gate_passed": p2_passed,
        },
        "all_plant_gates_passed": p0_passed and p2_passed,
        "forward_learning_executed": False,
        "inverse_search_executed": False,
        "recorded_audio_accessed": False,
        "protected_data_accessed": False,
        "promotion_allowed": False,
    }
