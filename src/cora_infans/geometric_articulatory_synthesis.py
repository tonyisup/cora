"""Synthetic-only H4 Option D geometric articulatory candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .io.motor_plan import VOWEL_FORMANTS_HZ, _source_pulse_train


SAMPLE_RATE = 16_000
DURATION_MS = 160
SECTION_COUNT = 10
NEUTRAL_AREAS_CM2 = np.array(
    [1.0, 1.35, 1.9, 2.7, 3.45, 3.9, 3.7, 3.15, 2.55, 2.0],
    dtype=np.float64,
)
LIP_WEIGHTS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0],
    dtype=np.float64,
)
SENSORY_FREQUENCIES_HZ = np.geomspace(150.0, 3500.0, 24)
FORMANT_FREQUENCIES_HZ = np.arange(80.0, 3500.0 + 5.0, 5.0)
F0_CONDITIONS_HZ = (90.0, 110.0, 140.0, 180.0)
REST_STATE = np.array([0.65, 0.5, 0.2], dtype=np.float64)
VOWELS = ("a", "e", "i", "o", "u")


@dataclass(frozen=True)
class TractGeometry:
    areas_cm2: np.ndarray
    lengths_cm: np.ndarray
    total_length_cm: float


@dataclass(frozen=True)
class PlantOutput:
    motor: np.ndarray
    f0_hz: float
    geometry: TractGeometry
    sensory: np.ndarray
    formants_hz: tuple[float, float, float] | None
    waveform: np.ndarray


@dataclass(frozen=True)
class InverseResult:
    initial_motor: np.ndarray
    final_motor: np.ndarray
    initial_loss: float
    final_loss: float
    forward_prediction_count: int
    accepted_update_count: int


def _motor(value: np.ndarray) -> np.ndarray:
    motor = np.asarray(value, dtype=np.float64).reshape(-1)
    if motor.shape != (3,) or not np.all(np.isfinite(motor)):
        raise ValueError("motor state must contain three finite coordinates")
    if np.any(motor < 0.0) or np.any(motor > 1.0):
        raise ValueError("motor state violates [0,1] bounds")
    return motor


def area_function(motor: np.ndarray) -> TractGeometry:
    m = _motor(motor)
    centers = (np.arange(SECTION_COUNT, dtype=np.float64) + 0.5) / SECTION_COUNT
    constriction_center = 0.20 + 0.60 * m[1]
    constriction = np.exp(
        -0.5 * ((centers - constriction_center) / 0.12) ** 2
    )
    log_areas = (
        np.log(NEUTRAL_AREAS_CM2)
        + 0.35 * (m[0] - 0.5)
        - 2.2 * (1.0 - m[0]) * constriction
        - m[2] * LIP_WEIGHTS
    )
    areas = np.clip(np.exp(log_areas), 0.05, 8.0)
    total_length = 16.0 + 2.0 * m[2]
    lengths = np.full(SECTION_COUNT, total_length / SECTION_COUNT)
    return TractGeometry(
        areas_cm2=areas,
        lengths_cm=lengths,
        total_length_cm=float(total_length),
    )


def transfer_function(
    motor: np.ndarray, frequencies_hz: np.ndarray
) -> np.ndarray:
    geometry = area_function(motor)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    if np.any(frequencies < 0.0) or not np.all(np.isfinite(frequencies)):
        raise ValueError("transfer frequencies must be finite and nonnegative")
    rho = 1.14
    speed = 35_000.0
    angular_wave = 2.0 * np.pi * frequencies / speed
    attenuation = 0.002 + 0.00001 * frequencies
    gamma = attenuation + 1j * angular_wave
    a = np.ones_like(gamma, dtype=np.complex128)
    b = np.zeros_like(gamma, dtype=np.complex128)
    c = np.zeros_like(gamma, dtype=np.complex128)
    d = np.ones_like(gamma, dtype=np.complex128)
    for area, length in zip(geometry.areas_cm2, geometry.lengths_cm):
        impedance = rho * speed / area
        cosh = np.cosh(gamma * length)
        sinh = np.sinh(gamma * length)
        next_a = a * cosh + b * (sinh / impedance)
        next_b = a * (impedance * sinh) + b * cosh
        next_c = c * cosh + d * (sinh / impedance)
        next_d = c * (impedance * sinh) + d * cosh
        a, b, c, d = next_a, next_b, next_c, next_d
    lip_area = float(geometry.areas_cm2[-1])
    lip_radius = math.sqrt(lip_area / math.pi)
    ka = angular_wave * lip_radius
    lip_impedance = (rho * speed / lip_area) * (
        0.25 * ka**2 + 1j * 0.61 * ka
    )
    denominator = c * lip_impedance + d
    return 1.0 / np.where(np.abs(denominator) > 1e-18, denominator, 1e-18)


def sensory_vector(motor: np.ndarray) -> np.ndarray:
    magnitude = np.abs(transfer_function(motor, SENSORY_FREQUENCIES_HZ))
    vector = np.log(magnitude + 1e-12)
    vector -= float(np.mean(vector))
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("plant sensory envelope has insufficient structure")
    return vector / norm


def analytic_formants(motor: np.ndarray) -> tuple[float, float, float] | None:
    magnitude = np.abs(transfer_function(motor, FORMANT_FREQUENCIES_HZ))
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


def synthesize_tract(
    motor: np.ndarray, *, f0_hz: float, duration_ms: int = DURATION_MS
) -> np.ndarray:
    m = _motor(motor)
    if f0_hz not in F0_CONDITIONS_HZ:
        raise ValueError("F0 is outside the frozen synthetic conditions")
    samples = int(round(duration_ms * SAMPLE_RATE / 1000.0))
    source = _source_pulse_train(samples, SAMPLE_RATE, f0_hz)
    frequencies = np.fft.rfftfreq(samples, 1.0 / SAMPLE_RATE)
    response = transfer_function(m, frequencies)
    waveform = np.fft.irfft(np.fft.rfft(source) * response, n=samples)
    ramp = min(SAMPLE_RATE // 100, samples // 2)
    envelope = np.ones(samples, dtype=np.float64)
    if ramp:
        envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
        envelope[-ramp:] = np.linspace(1.0, 0.0, ramp)
    waveform *= envelope
    peak = float(np.max(np.abs(waveform))) or 1.0
    return (waveform * (0.95 / peak)).astype(np.float32)


def run_plant(motor: np.ndarray, *, f0_hz: float) -> PlantOutput:
    m = _motor(motor)
    return PlantOutput(
        motor=m.copy(),
        f0_hz=f0_hz,
        geometry=area_function(m),
        sensory=sensory_vector(m),
        formants_hz=analytic_formants(m),
        waveform=synthesize_tract(m, f0_hz=f0_hz),
    )


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    value = index
    while value:
        result += factor * (value % base)
        value //= base
        factor /= base
    return result


def halton_motor(index: int) -> np.ndarray:
    if index <= 0:
        raise ValueError("Halton index must be positive")
    return np.array(
        [_radical_inverse(index, base) for base in (2, 3, 5)],
        dtype=np.float64,
    )


def halton_motors(start: int, stop: int) -> np.ndarray:
    if stop < start:
        raise ValueError("Halton stop must not precede start")
    return np.stack([halton_motor(index) for index in range(start, stop + 1)])


def bounded_transition(previous: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _motor(previous)
    destination = _motor(target)
    hops = max(1, int(math.ceil(float(np.max(np.abs(destination - source))) / 0.25)))
    return np.stack(
        [source + (destination - source) * (step / hops) for step in range(1, hops + 1)]
    )


class LocalAffineForwardMap:
    """Fixed-center, winner-local affine map learned from babbling."""

    def __init__(self, centers: np.ndarray) -> None:
        value = np.asarray(centers, dtype=np.float64)
        if value.shape != (64, 3):
            raise ValueError("forward map requires exactly 64 three-scalar centers")
        if np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError("prototype center violates motor bounds")
        self.centers = value.copy()
        self._xtx = np.zeros((64, 4, 4), dtype=np.float64)
        self._xty = np.zeros((64, 4, 24), dtype=np.float64)
        self.counts = np.zeros(64, dtype=np.int64)
        self.coefficients: np.ndarray | None = None

    @property
    def training_state_scalars(self) -> int:
        return int(self.centers.size + self._xtx.size + self._xty.size + self.counts.size)

    @property
    def retained_state_scalars(self) -> int:
        return int(self.centers.size + 64 * 4 * 24 + self.counts.size)

    def winner_index(self, motor: np.ndarray) -> int:
        m = _motor(motor)
        distances = np.sum((self.centers - m) ** 2, axis=1)
        return int(np.argmin(distances))

    def update(self, motor: np.ndarray, sensory: np.ndarray) -> None:
        if self.coefficients is not None:
            raise ValueError("cannot update a finalized forward map")
        m = _motor(motor)
        target = np.asarray(sensory, dtype=np.float64).reshape(-1)
        if target.shape != (24,) or not np.all(np.isfinite(target)):
            raise ValueError("forward target must contain 24 finite values")
        winner = self.winner_index(m)
        design = np.concatenate(([1.0], m - self.centers[winner]))
        self._xtx[winner] += np.outer(design, design)
        self._xty[winner] += np.outer(design, target)
        self.counts[winner] += 1

    def finalize(self) -> None:
        if np.any(self.counts == 0):
            raise ValueError("every forward prototype must receive babbling support")
        ridge = np.diag([0.000001, 0.0001, 0.0001, 0.0001])
        coefficients = np.empty((64, 4, 24), dtype=np.float64)
        for index in range(64):
            coefficients[index] = np.linalg.solve(
                self._xtx[index] + ridge, self._xty[index]
            )
        self.coefficients = coefficients

    def _normalized(self, vector: np.ndarray) -> np.ndarray:
        # Predictions can be views into the retained coefficient table.  Work on
        # a copy so normalization remains a read-only operation on the model.
        value = np.asarray(vector, dtype=np.float64).reshape(24).copy()
        value -= float(np.mean(value))
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            raise ValueError("forward prediction has insufficient structure")
        return value / norm

    def predict(self, motor: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("forward map must be finalized before prediction")
        m = _motor(motor)
        winner = self.winner_index(m)
        design = np.concatenate(([1.0], m - self.centers[winner]))
        return self._normalized(design @ self.coefficients[winner])

    def intercept_prediction(self, index: int) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("forward map must be finalized before prediction")
        if not 0 <= index < 64:
            raise ValueError("prototype index out of range")
        return self._normalized(self.coefficients[index, 0])


def fit_babbling_forward_map() -> tuple[LocalAffineForwardMap, dict[str, object]]:
    training = halton_motors(1, 1024)
    model = LocalAffineForwardMap(training[:64])
    previous = REST_STATE.copy()
    maximum_hop = 0.0
    interpolation_hops = 0
    for ordinal, motor in enumerate(training):
        transition = bounded_transition(previous, motor)
        if len(transition):
            maximum_hop = max(
                maximum_hop,
                float(
                    np.max(
                        np.abs(
                            np.diff(
                                np.vstack((previous[None, :], transition)), axis=0
                            )
                        )
                    )
                ),
            )
        interpolation_hops += len(transition)
        model.update(motor, sensory_vector(motor))
        previous = motor
    model.finalize()
    return model, {
        "training_count": len(training),
        "f0_assignment_counts": {
            str(f0): sum(
                F0_CONDITIONS_HZ[index % len(F0_CONDITIONS_HZ)] == f0
                for index in range(len(training))
            )
            for f0 in F0_CONDITIONS_HZ
        },
        "interpolation_hop_count": interpolation_hops,
        "maximum_coordinate_delta_per_hop": maximum_hop,
        "minimum_prototype_support": int(np.min(model.counts)),
        "maximum_prototype_support": int(np.max(model.counts)),
        "training_state_scalars": model.training_state_scalars,
        "retained_state_scalars": model.retained_state_scalars,
    }


def _peak_indices(vector: np.ndarray) -> tuple[int, int, int]:
    value = np.asarray(vector, dtype=np.float64).reshape(24)
    local = [
        index
        for index in range(1, 23)
        if value[index] > value[index - 1] and value[index] >= value[index + 1]
    ]
    ranked = sorted(local, key=lambda index: (-value[index], index))
    for index in sorted(range(24), key=lambda item: (-value[item], item)):
        if index not in ranked:
            ranked.append(index)
        if len(ranked) >= 3:
            break
    return tuple(sorted(ranked[:3]))  # type: ignore[return-value]


def sensory_distance(target: np.ndarray, predicted: np.ndarray, *, f0_hz: float) -> float:
    wanted = np.asarray(target, dtype=np.float64).reshape(24)
    actual = np.asarray(predicted, dtype=np.float64).reshape(24)
    envelope = float(np.linalg.norm(wanted - actual) / 2.0)
    target_peaks = _peak_indices(wanted)
    predicted_peaks = _peak_indices(actual)
    resonance = float(
        np.mean(np.abs(np.array(target_peaks) - np.array(predicted_peaks))) / 23.0
    )
    harmonics = np.arange(1, int(math.ceil(3500.0 / f0_hz)) + 1) * f0_hz
    nearest = np.min(
        np.abs(SENSORY_FREQUENCIES_HZ[:, None] - harmonics[None, :]), axis=1
    )
    weights = 0.5 + 0.5 * np.exp(
        -(nearest**2) / (2.0 * (0.35 * f0_hz) ** 2)
    )
    harmonic = float(np.sqrt(np.sum(weights * (wanted - actual) ** 2) / np.sum(weights)))
    return envelope + 0.1 * resonance + 0.05 * harmonic


def composite_loss(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    motor: np.ndarray,
    previous_motor: np.ndarray,
    f0_hz: float,
) -> float:
    m = _motor(motor)
    previous = _motor(previous_motor)
    return (
        sensory_distance(target, predicted, f0_hz=f0_hz)
        + 0.0005 * float(np.sum((m - REST_STATE) ** 2))
        + 0.001 * float(np.sum((m - previous) ** 2))
    )


def invert_sensory(
    target: np.ndarray,
    model: LocalAffineForwardMap,
    *,
    f0_hz: float,
    previous_motor: np.ndarray = REST_STATE,
) -> InverseResult:
    previous = _motor(previous_motor)
    initialization = []
    for index, center in enumerate(model.centers):
        predicted = model.intercept_prediction(index)
        loss = composite_loss(
            target,
            predicted,
            motor=center,
            previous_motor=previous,
            f0_hz=f0_hz,
        )
        effort = float(np.sum((center - REST_STATE) ** 2))
        initialization.append((loss, effort, index, center))
    _stored_loss, _effort, _index, initial = min(
        initialization, key=lambda item: (item[0], item[1], item[2])
    )
    current = initial.copy()
    current_loss = composite_loss(
        target,
        model.predict(current),
        motor=current,
        previous_motor=previous,
        f0_hz=f0_hz,
    )
    initial_loss = current_loss
    predictions = 1
    updates = 0
    steps = np.array([0.16, 0.16, 0.16], dtype=np.float64)
    for _round in range(3):
        for coordinate in range(3):
            for direction in (-1.0, 1.0):
                candidate = current.copy()
                candidate[coordinate] += direction * steps[coordinate]
                if np.any(candidate < 0.0) or np.any(candidate > 1.0):
                    continue
                predicted = model.predict(candidate)
                predictions += 1
                loss = composite_loss(
                    target,
                    predicted,
                    motor=candidate,
                    previous_motor=previous,
                    f0_hz=f0_hz,
                )
                if loss < current_loss:
                    current = candidate
                    current_loss = loss
                    updates += 1
        steps /= 2.0
    return InverseResult(
        initial_motor=initial.copy(),
        final_motor=current,
        initial_loss=initial_loss,
        final_loss=current_loss,
        forward_prediction_count=predictions,
        accepted_update_count=updates,
    )


def nearest_vowel(formants_hz: tuple[float, float, float]) -> str:
    return min(
        VOWELS,
        key=lambda vowel: sum(
            ((actual - expected) / expected) ** 2
            for actual, expected in zip(formants_hz, VOWEL_FORMANTS_HZ[vowel])
        ),
    )


def select_vowel_targets() -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    values = np.arange(0.05, 1.0, 0.10)
    candidates = []
    for m1 in values:
        for m2 in values:
            for m3 in values:
                motor = np.array([m1, m2, m3], dtype=np.float64)
                formants = analytic_formants(motor)
                if formants is not None:
                    candidates.append((motor, formants))
    selected: dict[str, np.ndarray] = {}
    reports = []
    used: set[tuple[float, float, float]] = set()
    for vowel in VOWELS:
        reference = VOWEL_FORMANTS_HZ[vowel]
        eligible = []
        for motor, formants in candidates:
            key = tuple(float(value) for value in motor)
            if key in used:
                continue
            cost = sum(
                ((actual - expected) / expected) ** 2
                for actual, expected in zip(formants, reference)
            )
            eligible.append((cost, key, motor, formants))
        cost, key, motor, formants = min(eligible, key=lambda item: (item[0], item[1]))
        used.add(key)
        selected[vowel] = motor.copy()
        relative_rmse = math.sqrt(cost / 3.0)
        reports.append(
            {
                "vowel": vowel,
                "motor": motor.tolist(),
                "plant_formants_hz": list(formants),
                "reference_formants_hz": list(reference),
                "relative_formant_rmse": relative_rmse,
                "nearest_vowel": nearest_vowel(formants),
                "calibration_valid": nearest_vowel(formants) == vowel
                and relative_rmse <= 0.25,
            }
        )
    return selected, reports


def identifiable_motor_error(target: np.ndarray, inferred: np.ndarray) -> dict[str, object]:
    truth = _motor(target)
    estimate = _motor(inferred)
    delta = 0.005
    jacobian = np.empty((24, 3), dtype=np.float64)
    for coordinate in range(3):
        lower = truth.copy()
        upper = truth.copy()
        lower[coordinate] = max(0.0, lower[coordinate] - delta)
        upper[coordinate] = min(1.0, upper[coordinate] + delta)
        denominator = upper[coordinate] - lower[coordinate]
        jacobian[:, coordinate] = (
            sensory_vector(upper) - sensory_vector(lower)
        ) / denominator
    _u, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    threshold = 0.01 * singular[0]
    rank = int(np.sum(singular > threshold))
    projector = right[:rank].T @ right[:rank]
    error = float(np.linalg.norm(projector @ (estimate - truth)))
    return {
        "projected_motor_error": error,
        "jacobian_rank": rank,
        "singular_values": singular.tolist(),
        "rank_threshold": float(threshold),
    }


def evaluate_forward_generalization(
    model: LocalAffineForwardMap,
) -> dict[str, object]:
    held_out = halton_motors(2049, 2304)
    neutral = sensory_vector(REST_STATE)
    distances = []
    baseline_distances = {"neutral_tract": [], "nearest_prototype": [], "persistence": []}
    octants: dict[str, list[float]] = {f"{index:03b}": [] for index in range(8)}
    previous = neutral
    for motor in held_out:
        target = sensory_vector(motor)
        predicted = model.predict(motor)
        winner = model.winner_index(motor)
        distance = float(np.linalg.norm(target - predicted))
        distances.append(distance)
        baseline_distances["neutral_tract"].append(float(np.linalg.norm(target - neutral)))
        baseline_distances["nearest_prototype"].append(
            float(np.linalg.norm(target - model.intercept_prediction(winner)))
        )
        baseline_distances["persistence"].append(float(np.linalg.norm(target - previous)))
        octant = "".join("1" if value >= 0.5 else "0" for value in motor)
        octants[octant].append(distance)
        previous = target
    wins = {
        name: sum(
            distance < baseline
            for distance, baseline in zip(distances, baseline_values)
        )
        / len(distances)
        for name, baseline_values in baseline_distances.items()
    }
    criteria = {
        "strict_win_fraction_vs_neutral_at_least_0_75": wins["neutral_tract"] >= 0.75,
        "strict_win_fraction_vs_nearest_prototype_at_least_0_75": wins[
            "nearest_prototype"
        ]
        >= 0.75,
        "strict_win_fraction_vs_persistence_at_least_0_75": wins["persistence"]
        >= 0.75,
        "median_unit_vector_distance_at_most_0_20": float(np.median(distances)) <= 0.20,
        "all_eight_motor_octants_reported": all(octants.values()),
        "training_state_scalars_at_most_7424": model.training_state_scalars <= 7424,
        "retained_state_scalars_at_most_6400": model.retained_state_scalars <= 6400,
        "all_prototypes_supported": bool(np.all(model.counts > 0)),
    }
    return {
        "held_out_count": len(held_out),
        "unit_vector_distances": distances,
        "median_unit_vector_distance": float(np.median(distances)),
        "p95_unit_vector_distance": float(np.percentile(distances, 95)),
        "strict_win_fraction": wins,
        "baseline_median_distances": {
            name: float(np.median(values))
            for name, values in baseline_distances.items()
        },
        "octant_mean_distance": {
            key: float(np.mean(values)) for key, values in octants.items()
        },
        "criteria": criteria,
        "gate_passed": all(criteria.values()),
    }


def evaluate_plant_validity(
    vowel_reports: list[dict[str, object]],
    vowel_targets: dict[str, np.ndarray],
) -> dict[str, object]:
    points = np.vstack(
        (
            halton_motors(1, 1024),
            halton_motors(2049, 2304),
            np.array(
                [
                    [m1, m2, m3]
                    for m1 in np.arange(0.05, 1.0, 0.10)
                    for m2 in np.arange(0.05, 1.0, 0.10)
                    for m3 in np.arange(0.05, 1.0, 0.10)
                ]
            ),
            np.stack(list(vowel_targets.values())),
        )
    )
    geometry_valid = True
    transfer_valid = True
    waveform_valid = True
    formants_present = True
    maximum_peak = 0.0
    for index, motor in enumerate(points):
        geometry = area_function(motor)
        geometry_valid &= bool(
            np.all(np.isfinite(geometry.areas_cm2))
            and np.all(geometry.areas_cm2 >= 0.05)
            and np.all(geometry.areas_cm2 <= 8.0)
            and 16.0 <= geometry.total_length_cm <= 18.0
        )
        response = transfer_function(motor, SENSORY_FREQUENCIES_HZ)
        transfer_valid &= bool(np.all(np.isfinite(response)))
        formants_present &= analytic_formants(motor) is not None
        f0 = F0_CONDITIONS_HZ[index % len(F0_CONDITIONS_HZ)]
        waveform = synthesize_tract(motor, f0_hz=f0)
        maximum_peak = max(maximum_peak, float(np.max(np.abs(waveform))))
        waveform_valid &= bool(
            np.all(np.isfinite(waveform)) and float(np.max(np.abs(waveform))) < 1.0
        )
    repeated_a = synthesize_tract(REST_STATE, f0_hz=110.0)
    repeated_b = synthesize_tract(REST_STATE, f0_hz=110.0)
    criteria = {
        "all_areas_and_lengths_finite_and_within_bounds": geometry_valid,
        "all_transfer_samples_finite": transfer_valid,
        "all_waveforms_finite_and_below_one": waveform_valid,
        "deterministic_repeat_synthesis": bool(np.array_equal(repeated_a, repeated_b)),
        "three_formants_present_for_every_point": formants_present,
        "each_selected_vowel_calibration_valid": all(
            bool(report["calibration_valid"]) for report in vowel_reports
        ),
    }
    return {
        "point_count": len(points),
        "maximum_waveform_peak": maximum_peak,
        "vowel_calibration": vowel_reports,
        "criteria": criteria,
        "gate_passed": all(criteria.values()),
    }


def evaluate_inverse_closure(
    model: LocalAffineForwardMap,
    vowel_targets: dict[str, np.ndarray],
) -> dict[str, object]:
    rows = []
    errors = [[], [], []]
    relative_errors = [[], [], []]
    confusion = {vowel: {guess: 0 for guess in VOWELS} for vowel in VOWELS}
    correct = 0
    combined_agreement = 0
    for vowel in VOWELS:
        target_motor = vowel_targets[vowel]
        target_sensory = sensory_vector(target_motor)
        target_formants = analytic_formants(target_motor)
        assert target_formants is not None
        for f0 in F0_CONDITIONS_HZ:
            result = invert_sensory(target_sensory, model, f0_hz=f0)
            final_sensory = sensory_vector(result.final_motor)
            final_formants = analytic_formants(result.final_motor)
            if final_formants is None:
                final_formants = (math.nan, math.nan, math.nan)
                identity = "a"
            else:
                identity = nearest_vowel(final_formants)
            is_correct = identity == vowel
            correct += int(is_correct)
            confusion[vowel][identity] += 1
            formant_errors = [
                abs(actual - expected)
                for actual, expected in zip(final_formants, target_formants)
            ]
            for index, error in enumerate(formant_errors):
                errors[index].append(error)
                relative_errors[index].append(error / target_formants[index])
            acoustic = float(np.linalg.norm(target_sensory - final_sensory) / 2.0)
            motor_agreement = identifiable_motor_error(target_motor, result.final_motor)
            agreed = acoustic <= 0.10 and motor_agreement["projected_motor_error"] <= 0.12
            combined_agreement += int(agreed)
            final_geometry = area_function(result.final_motor)
            rows.append(
                {
                    "expected_vowel": vowel,
                    "f0_hz": f0,
                    "target_motor": target_motor.tolist(),
                    "initial_motor": result.initial_motor.tolist(),
                    "final_motor": result.final_motor.tolist(),
                    "target_formants_hz": list(target_formants),
                    "final_formants_hz": list(final_formants),
                    "formant_errors_hz": formant_errors,
                    "predicted_vowel": identity,
                    "correct_identity": is_correct,
                    "acoustic_cycle_distance": acoustic,
                    **motor_agreement,
                    "combined_agreement": agreed,
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                    "accepted_update_count": result.accepted_update_count,
                    "forward_prediction_count": result.forward_prediction_count,
                    "geometry_within_bounds": bool(
                        np.all(final_geometry.areas_cm2 >= 0.05)
                        and np.all(final_geometry.areas_cm2 <= 8.0)
                        and 16.0 <= final_geometry.total_length_cm <= 18.0
                    ),
                }
            )
    trial_count = len(rows)
    formant_mae = {
        f"f{index + 1}": float(np.mean(values))
        for index, values in enumerate(errors)
    }
    relative = {
        f"f{index + 1}": {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
        for index, values in enumerate(relative_errors)
    }
    identity_accuracy = correct / trial_count
    agreement_fraction = combined_agreement / trial_count
    criteria = {
        "identity_preservation_at_least_0_95": identity_accuracy >= 0.95,
        "acoustic_and_equivalence_agreement_at_least_0_90": agreement_fraction >= 0.90,
        "analytic_f1_mae_at_most_30_hz": formant_mae["f1"] <= 30.0,
        "analytic_f2_mae_at_most_30_hz": formant_mae["f2"] <= 30.0,
        "analytic_f3_mae_at_most_30_hz": formant_mae["f3"] <= 30.0,
        "all_plans_and_geometry_within_bounds": all(
            bool(row["geometry_within_bounds"]) for row in rows
        ),
        "maximum_forward_predictions_at_most_23": max(
            int(row["forward_prediction_count"]) for row in rows
        )
        <= 23,
        "all_final_losses_no_worse_than_initial": all(
            float(row["final_loss"]) <= float(row["initial_loss"]) for row in rows
        ),
        "labels_absent_from_learning_initialization_and_search": True,
    }
    return {
        "trial_count": trial_count,
        "identity_accuracy": identity_accuracy,
        "combined_agreement_fraction": agreement_fraction,
        "formant_mae_hz": formant_mae,
        "relative_formant_error": relative,
        "confusion_matrix": confusion,
        "maximum_forward_predictions": max(
            int(row["forward_prediction_count"]) for row in rows
        ),
        "criteria": criteria,
        "gate_passed": all(criteria.values()),
        "rows": rows,
    }


def evaluate_option_d_synthetic() -> dict[str, object]:
    vowel_targets, vowel_reports = select_vowel_targets()
    plant = evaluate_plant_validity(vowel_reports, vowel_targets)
    model, babbling = fit_babbling_forward_map()
    forward = evaluate_forward_generalization(model)
    inverse = evaluate_inverse_closure(model, vowel_targets)
    criteria = {
        "gate_0_plant_validity_passed": plant["gate_passed"],
        "gate_1_forward_generalization_passed": forward["gate_passed"],
        "gate_2_inverse_closure_passed": inverse["gate_passed"],
        "synthetic_only": True,
        "recorded_audio_accessed": False,
        "protected_data_accessed": False,
    }
    passed = all(criteria.values())
    return {
        "mode": "h4_option_d_geometric_articulatory_synthesis_synthetic",
        "evidence_scope": "synthetic_only",
        "plant": plant,
        "babbling": babbling,
        "forward_generalization": forward,
        "inverse_closure": inverse,
        "criteria": criteria,
        "all_synthetic_gates_passed": passed,
        "spent_recorded_development_preregistration_allowed": passed,
        "recorded_audio_accessed": False,
        "protected_data_accessed": False,
        "promotion_allowed": False,
        "claim_boundary": (
            "A pass establishes synthetic-only closure for this simplified geometric "
            "plant and bounded local learner. It does not promote H4, establish "
            "anatomical or infant fidelity, or authorize recorded speech."
        ),
    }
