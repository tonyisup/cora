from __future__ import annotations

import json
import sys
from typing import Any

from .io.live_listener import LiveListener
from .io.live_mic import DEFAULT_DEVICE, LiveMicArray, LiveMicError
from .io.offline_eval import CorpusLoader, OfflineEvalHarness
from .simulation import (
    SimulationParams,
    _builtin_sequence_family,
    build_builtin_sequence_report,
    build_live_frame_report,
    build_sequence_report,
    build_sequence_report_with_scheduling,
    build_simulation_report,
)
from .io.synthetic_stream import build_synthetic_corpus_frames
from .io.motor_plan import (  # noqa: E402
    ANALYSIS_BY_SYNTHESIS_TABLE,
    VOWEL_FORMANTS_HZ,
    MotorPlanner,
    analyze_vowel,
    synthesize_vowel,
    write_wav,
)
from .io.self_suppression import (
    SelfSuppressionParams,
    run_forward_model_learning_trial,
    run_forward_model_promotion_gate,
    run_forward_model_robustness_sweep,
    run_self_suppression_trial,
)


OutputMode = str
SequenceArgs = tuple[str, bool]


def describe() -> dict[str, object]:
    return {
        "project": "cora-infans",
        "implemented_slice": "saliency_workspace_live_attention",
        "implemented_subsystems": [
            "synthetic_mic_array_front_end",
            "live_xvf3800_front_end",
            "acoustic_saliency_gate",
            "predictive_hierarchy",
            "stateful_event_workspace",
            "theta_gamma_scheduler",
            "xvf3800_hardware_tuner",
            "commanded_active_attention",
            "offline_corpus_evaluation",
            "motor_plan_synthesis",
            "theta_gamma_motor_subwindow",
            "simulation_only_signal_self_suppression",
            "simulation_only_forward_model_learning",
            "simulation_only_forward_model_robustness_sweep",
            "simulation_only_forward_model_promotion_gate",
        ],

        "planned_subsystems": [
            "non_speech_source_localization",
            "training_and_evaluation_harnesses",
            "general_user_authored_sequences",
        ],
        "verdicts": {
            "source_localization": "speech_stimulus_recoverable_non_speech_negative",
            "beam_steering": "verified_commanded_top_down_actuator",
        },
    }


def _parse_channel_energies(raw: str) -> list[float]:
    try:
        return [float(part) for part in raw.split(",") if part]
    except ValueError as exc:
        raise ValueError("invalid channel energies") from exc



def _parse_mic_geometry(raw: str) -> tuple[str, ...]:
    geometry = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not geometry:
        raise ValueError("invalid mic geometry")
    return geometry


def parse_sequence_cli_args(args: list[str]) -> tuple[str, bool]:
    family = "sweep"
    with_scheduling = True  # now the default report path
    i = 1
    while i < len(args):
        option = args[i]
        if option == "--family":
            if i + 1 >= len(args):
                raise ValueError("missing value for --family")
            family = args[i + 1]
            i += 2
        elif option == "--with-scheduling":
            with_scheduling = True
            i += 1
        else:
            raise ValueError(f"unknown option: {option}")
    return family, with_scheduling



def parse_simulation_cli_args(args: list[str]) -> tuple[SimulationParams, OutputMode]:
    params: dict[str, Any] = {
        "timestamp_ms": 125,
        "dominant_frequency_hz": 1200.0,
        "channel_energies": [0.2, 0.9, 0.4, 0.1],
        "initial_prediction_hz": 1000.0,
        "mic_channel_geometry": (
            "channel_0",
            "channel_1",
            "channel_2",
            "channel_3",
        ),
        "saliency_threshold": 0.7,
        "output": "json",
    }
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--timestamp-ms":
            try:
                params["timestamp_ms"] = int(value)
            except ValueError as exc:
                raise ValueError("invalid timestamp") from exc
        elif option == "--dominant-frequency-hz":
            try:
                params["dominant_frequency_hz"] = float(value)
            except ValueError as exc:
                raise ValueError("invalid dominant frequency") from exc
        elif option == "--channel-energies":
            params["channel_energies"] = _parse_channel_energies(value)
        elif option == "--initial-prediction-hz":
            try:
                params["initial_prediction_hz"] = float(value)
            except ValueError as exc:
                raise ValueError("invalid initial prediction") from exc
        elif option == "--mic-geometry":
            params["mic_channel_geometry"] = _parse_mic_geometry(value)
        elif option == "--saliency-threshold":
            try:
                params["saliency_threshold"] = float(value)
            except ValueError as exc:
                raise ValueError("invalid saliency threshold") from exc
        elif option == "--output":
            if value not in {"json", "pretty"}:
                raise ValueError("invalid output mode")
            params["output"] = value
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2

    simulation_params = SimulationParams(
        timestamp_ms=params["timestamp_ms"],
        dominant_frequency_hz=params["dominant_frequency_hz"],
        channel_energies=params["channel_energies"],
        initial_prediction_hz=params["initial_prediction_hz"],
        mic_channel_geometry=params["mic_channel_geometry"],
        saliency_threshold=params["saliency_threshold"],
    )
    return simulation_params, params["output"]


def simulate_frame(args: list[str] | None = None) -> dict[str, object]:
    simulation_params, _output_mode = parse_simulation_cli_args(
        args or ["--simulate-frame"]
    )
    return build_simulation_report(simulation_params)



def simulate_sequence(args: list[str] | None = None) -> dict[str, object]:
    family, with_scheduling = parse_sequence_cli_args(args or ["--simulate-sequence"])
    # Scheduling is part of the default report path; the flag can still request
    # the bare flat-workspace view when explicitly set False via code path.
    report = build_sequence_report(
        _builtin_sequence_family(family), with_scheduling=with_scheduling
    )
    report["family"] = family
    return report


def capture_live(args: list[str] | None = None) -> dict[str, object]:
    parsed = args or ["--capture-live"]
    device = DEFAULT_DEVICE
    duration_ms: int | None = None
    i = 1
    while i < len(parsed):
        option = parsed[i]
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {option}")
        value = parsed[i + 1]
        if option == "--device":
            device = value
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2

    array = LiveMicArray(device=device)
    return build_live_frame_report(array, duration_ms=duration_ms)


def listen_live(args: list[str] | None = None) -> dict[str, object]:
    parsed = args or ["--listen-live"]
    device = DEFAULT_DEVICE
    duration_ms = 2000
    frame_ms = 250
    with_stimulus = False
    with_scheduling = False
    with_attention = False
    with_motor_imitation = False
    i = 1
    while i < len(parsed):
        option = parsed[i]
        if option == "--with-stimulus":
            with_stimulus = True
            i += 1
            continue
        if option == "--with-scheduling":
            with_scheduling = True
            i += 1
            continue
        if option == "--with-attention":
            with_attention = True
            i += 1
            continue
        if option == "--with-motor-imitation":
            with_motor_imitation = True
            i += 1
            continue
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {option}")
        value = parsed[i + 1]
        if option == "--device":
            device = value
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--frame-ms":
            try:
                frame_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid frame ms") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2

    listener = LiveListener(LiveMicArray(device=device))
    return listener.listen(
        duration_ms=duration_ms,
        frame_ms=frame_ms,
        with_stimulus=with_stimulus,
        with_scheduling=with_scheduling,
        with_attention=with_attention,
        with_motor_imitation=with_motor_imitation,
    )


def eval_corpus(args: list[str] | None = None) -> dict[str, object]:
    """Run the offline evaluation harness over a labeled corpus.

    Usage: --eval-corpus --root DIR --label-csv FILE [--saliency-threshold F]
           [--frame-ms N]

    The harness scores a deterministic synthetic frame stream derived from each
    clip's labeled intervals; no hardware or real WAV decoding is required.
    """
    parsed = args or ["--eval-corpus"]
    root = "."
    label_csv: str | None = None
    saliency_threshold = 0.7
    frame_ms = 250
    i = 1
    while i < len(parsed):
        option = parsed[i]
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {option}")
        value = parsed[i + 1]
        if option == "--root":
            root = value
        elif option == "--label-csv":
            label_csv = value
        elif option == "--saliency-threshold":
            try:
                saliency_threshold = float(value)
            except ValueError as exc:
                raise ValueError("invalid saliency threshold") from exc
        elif option == "--frame-ms":
            try:
                frame_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid frame ms") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2

    if label_csv is None:
        raise ValueError("--label-csv is required")

    loader = CorpusLoader(root=root)
    clips = loader.load_labels(label_csv)
    harness = OfflineEvalHarness(
        saliency_threshold=saliency_threshold, frame_ms=frame_ms
    )
    report = harness.evaluate_corpus(clips, build_synthetic_corpus_frames)
    report["label_csv"] = label_csv
    report["clip_loaded"] = len(clips)
    return report


def synthesize_vowel_cli(args: list[str] | None = None) -> dict[str, object]:
    """Option C CLI: synthesize a vowel (motor-plan production) to a WAV file.

    Usage: --synthesize-vowel a --out /tmp/a.wav [--duration-ms N]
           [--sample-rate N] [--f0-hz F] [--analyze]

    If ``--analyze`` is given, the written WAV is read back and analyzed
    (perception) so the analysis-by-synthesis loop is closed in one command.
    """
    parsed = args or ["--synthesize-vowel", "a"]
    vowel: str | None = None
    out_path: str | None = None
    duration_ms = 250
    sample_rate = 16000
    f0_hz = 110.0
    analyze = False
    i = 0
    while i < len(parsed):
        option = parsed[i]
        if option == "--analyze":
            analyze = True
            i += 1
            continue
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {option}")
        value = parsed[i + 1]
        if option == "--synthesize-vowel":
            vowel = value
        elif option == "--out":
            out_path = value
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--sample-rate":
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise ValueError("invalid sample rate") from exc
        elif option == "--f0-hz":
            try:
                f0_hz = float(value)
            except ValueError as exc:
                raise ValueError("invalid f0") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2

    if vowel is None:
        raise ValueError("--synthesize-vowel is required")
    if vowel not in VOWEL_FORMANTS_HZ:
        raise ValueError(f"unknown vowel: {vowel!r}")
    if out_path is None:
        raise ValueError("--out WAV path is required")

    wav = synthesize_vowel(
        vowel,  # type: ignore[arg-type]
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        f0_hz=f0_hz,
    )
    write_wav(out_path, wav, sample_rate)

    report: dict[str, object] = {
        "vowel": vowel,
        "out": out_path,
        "samples": len(wav),
        "sample_rate": sample_rate,
        "formant_target_hz": list(VOWEL_FORMANTS_HZ[vowel]),  # type: ignore[index]
        "motor_plan": ANALYSIS_BY_SYNTHESIS_TABLE[vowel],  # type: ignore[index]
    }
    if analyze:
        import wave

        with wave.open(out_path, "rb") as w:
            frames = w.readframes(w.getnframes())
            import numpy as np

            pcm = (
                np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
            )
        feat = analyze_vowel(pcm, sample_rate=sample_rate)
        report["perceived_formants_hz"] = list(feat.formant_hz)
        report["perceived_nearest_vowel"] = feat.nearest_vowel
        planner = MotorPlanner()
        plan = planner.plan_from_analysis(feat)
        report["motor_plan_from_perception"] = {
            "vowel": plan.vowel,
            "f1_hz": plan.f1_hz,
            "f2_hz": plan.f2_hz,
            "f3_hz": plan.f3_hz,
        }
    return report


def simulate_self_suppression_cli(args: list[str]) -> dict[str, object]:
    vowel = "a"
    world_vowel: str | None = None
    sample_rate = 16000
    duration_ms = 300
    world_gain = 0.0
    svsr_pass_db = 20.0
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown vowel: {value!r}")
            vowel = value
        elif option == "--world-vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown world vowel: {value!r}")
            world_vowel = value
        elif option == "--sample-rate":
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise ValueError("invalid sample rate") from exc
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--world-gain":
            try:
                world_gain = float(value)
            except ValueError as exc:
                raise ValueError("invalid world gain") from exc
        elif option == "--svsr-pass-db":
            try:
                svsr_pass_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid SVSR threshold") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    params = SelfSuppressionParams(
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        world_gain=world_gain,
        svsr_pass_db=svsr_pass_db,
    )
    return run_self_suppression_trial(
        vowel=vowel,  # type: ignore[arg-type]
        world_vowel=world_vowel,  # type: ignore[arg-type]
        params=params,
    )



def learn_self_forward_model_cli(args: list[str]) -> dict[str, object]:
    vowel = "a"
    world_vowel: str | None = None
    sample_rate = 16000
    duration_ms = 300
    observed_delay_ms = 18.0
    observed_attenuation = 0.55
    observed_echo_decay = 0.22
    world_gain = 0.0
    svsr_pass_db = 20.0
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown vowel: {value!r}")
            vowel = value
        elif option == "--world-vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown world vowel: {value!r}")
            world_vowel = value
        elif option == "--sample-rate":
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise ValueError("invalid sample rate") from exc
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--observed-delay-ms":
            try:
                observed_delay_ms = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed delay") from exc
        elif option == "--observed-attenuation":
            try:
                observed_attenuation = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed attenuation") from exc
        elif option == "--observed-echo-decay":
            try:
                observed_echo_decay = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed echo decay") from exc
        elif option == "--world-gain":
            try:
                world_gain = float(value)
            except ValueError as exc:
                raise ValueError("invalid world gain") from exc
        elif option == "--svsr-pass-db":
            try:
                svsr_pass_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid SVSR threshold") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return run_forward_model_learning_trial(
        vowel=vowel,  # type: ignore[arg-type]
        world_vowel=world_vowel,  # type: ignore[arg-type]
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        observed_delay_ms=observed_delay_ms,
        observed_attenuation=observed_attenuation,
        observed_echo_decay=observed_echo_decay,
        world_gain=world_gain,
        svsr_pass_db=svsr_pass_db,
    )



def _parse_float_tuple(raw: str, *, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in raw.split(",") if part)
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc
    if not values:
        raise ValueError(f"invalid {label}")
    return values


def sweep_self_forward_model_cli(args: list[str]) -> dict[str, object]:
    vowel = "a"
    world_vowel: str | None = None
    sample_rate = 16000
    duration_ms = 300
    observed_delay_ms = 18.0
    observed_attenuation = 0.55
    observed_echo_decay = 0.22
    delay_errors_ms = (-20.0, -10.0, 0.0, 10.0, 20.0)
    attenuation_scales = (0.75, 1.0, 1.25)
    echo_decay_scales = (0.5, 1.0, 1.5)
    world_gains = (0.0,)
    svsr_pass_db = 20.0
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown vowel: {value!r}")
            vowel = value
        elif option == "--world-vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown world vowel: {value!r}")
            world_vowel = value
        elif option == "--sample-rate":
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise ValueError("invalid sample rate") from exc
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--observed-delay-ms":
            try:
                observed_delay_ms = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed delay") from exc
        elif option == "--observed-attenuation":
            try:
                observed_attenuation = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed attenuation") from exc
        elif option == "--observed-echo-decay":
            try:
                observed_echo_decay = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed echo decay") from exc
        elif option == "--delay-errors-ms":
            delay_errors_ms = _parse_float_tuple(value, label="delay errors")
        elif option == "--attenuation-scales":
            attenuation_scales = _parse_float_tuple(value, label="attenuation scales")
        elif option == "--echo-decay-scales":
            echo_decay_scales = _parse_float_tuple(value, label="echo decay scales")
        elif option == "--world-gain":
            world_gains = (0.0,) + _parse_float_tuple(value, label="world gain")
        elif option == "--world-gains":
            world_gains = _parse_float_tuple(value, label="world gains")
        elif option == "--svsr-pass-db":
            try:
                svsr_pass_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid SVSR threshold") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return run_forward_model_robustness_sweep(
        vowel=vowel,  # type: ignore[arg-type]
        world_vowel=world_vowel,  # type: ignore[arg-type]
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        observed_delay_ms=observed_delay_ms,
        observed_attenuation=observed_attenuation,
        observed_echo_decay=observed_echo_decay,
        delay_errors_ms=delay_errors_ms,
        attenuation_scales=attenuation_scales,
        echo_decay_scales=echo_decay_scales,
        world_gains=world_gains,
        svsr_pass_db=svsr_pass_db,
    )



def promote_self_forward_model_cli(args: list[str]) -> dict[str, object]:
    vowel = "a"
    world_vowel: str | None = "o"
    sample_rate = 16000
    duration_ms = 300
    observed_delay_ms = 18.0
    observed_attenuation = 0.55
    observed_echo_decay = 0.22
    world_gain = 0.25
    min_learning_improvement_db = 3.0
    min_exact_svsr_db = 20.0
    min_delay_degradation_db = 10.0
    min_world_correlation = 0.2
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown vowel: {value!r}")
            vowel = value
        elif option == "--world-vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown world vowel: {value!r}")
            world_vowel = value
        elif option == "--sample-rate":
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise ValueError("invalid sample rate") from exc
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--observed-delay-ms":
            try:
                observed_delay_ms = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed delay") from exc
        elif option == "--observed-attenuation":
            try:
                observed_attenuation = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed attenuation") from exc
        elif option == "--observed-echo-decay":
            try:
                observed_echo_decay = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed echo decay") from exc
        elif option == "--world-gain":
            try:
                world_gain = float(value)
            except ValueError as exc:
                raise ValueError("invalid world gain") from exc
        elif option == "--min-learning-improvement-db":
            try:
                min_learning_improvement_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid learning threshold") from exc
        elif option == "--min-exact-svsr-db":
            try:
                min_exact_svsr_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid exact SVSR threshold") from exc
        elif option == "--min-delay-degradation-db":
            try:
                min_delay_degradation_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid delay degradation threshold") from exc
        elif option == "--min-world-correlation":
            try:
                min_world_correlation = float(value)
            except ValueError as exc:
                raise ValueError("invalid world correlation threshold") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return run_forward_model_promotion_gate(
        vowel=vowel,  # type: ignore[arg-type]
        world_vowel=world_vowel,  # type: ignore[arg-type]
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        observed_delay_ms=observed_delay_ms,
        observed_attenuation=observed_attenuation,
        observed_echo_decay=observed_echo_decay,
        world_gain=world_gain,
        min_learning_improvement_db=min_learning_improvement_db,
        min_exact_svsr_db=min_exact_svsr_db,
        min_delay_degradation_db=min_delay_degradation_db,
        min_world_correlation=min_world_correlation,
    )



def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args == ["--describe"]:
        print(json.dumps(describe()))
        return 0
    if args and args[0] == "--simulate-frame":
        try:
            simulation_params, output_mode = parse_simulation_cli_args(args)
            payload = build_simulation_report(simulation_params)
            indent = 2 if output_mode == "pretty" else None
            print(json.dumps(payload, indent=indent))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args == ["--simulate-sequence"] or (args and args[0] == "--simulate-sequence"):
        try:
            print(json.dumps(simulate_sequence(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--capture-live":
        try:
            print(json.dumps(capture_live(args)))
            return 0
        except (ValueError, LiveMicError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--listen-live":
        try:
            print(json.dumps(listen_live(args)))
            return 0
        except (ValueError, LiveMicError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--eval-corpus":
        try:
            print(json.dumps(eval_corpus(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--synthesize-vowel":
        try:
            print(json.dumps(synthesize_vowel_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--simulate-self-suppression":
        try:
            print(json.dumps(simulate_self_suppression_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--learn-self-forward-model":
        try:
            print(json.dumps(learn_self_forward_model_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--sweep-self-forward-model":
        try:
            print(json.dumps(sweep_self_forward_model_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--promote-self-forward-model":
        try:
            print(json.dumps(promote_self_forward_model_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] in ("--snapshot-firmware", "--tune-raw-linear"):
        from .io.hardware_tuner import HardwareTuner, HardwareTunerError

        try:
            tuner = HardwareTuner()
            if args[0] == "--tune-raw-linear":
                result = {"applied": tuner.apply_raw_linear(), "snapshot": tuner.snapshot()}
            else:
                result = {"snapshot": tuner.snapshot()}
            print(json.dumps(result))
            return 0
        except (HardwareTunerError, Exception) as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 2

    print(
        "usage: python -m cora_infans --describe | --simulate-frame | --simulate-sequence "
        "[--timestamp-ms N --dominant-frequency-hz F --channel-energies a,b,c,d --initial-prediction-hz F --mic-geometry l0,l1,l2,l3 --saliency-threshold F --output json|pretty] "
        "[--family sweep|zigzag|gap|multi_gap|delayed_recovery] [--with-scheduling] "
        "[--capture-live [--device :0] [--duration-ms N]] "
        "[--listen-live [--duration-ms N] [--frame-ms N] [--with-stimulus] [--with-attention] [--with-scheduling] [--with-motor-imitation]] "
        "[--eval-corpus --root DIR --label-csv FILE [--saliency-threshold F] [--frame-ms N]] "
        "[--synthesize-vowel a|e|i|o|u --out PATH.wav [--duration-ms N] [--sample-rate N] [--f0-hz F] [--analyze]] "
        "[--simulate-self-suppression [--vowel a|e|i|o|u] [--world-vowel a|e|i|o|u] [--world-gain F] [--duration-ms N] [--sample-rate N]] "
        "[--learn-self-forward-model [--vowel a|e|i|o|u] [--observed-delay-ms F] [--observed-attenuation F] [--observed-echo-decay F] [--world-vowel a|e|i|o|u] [--world-gain F]] "
        "[--sweep-self-forward-model [--vowel a|e|i|o|u] [--delay-errors-ms csv] [--attenuation-scales csv] [--echo-decay-scales csv] [--world-vowel a|e|i|o|u] [--world-gain F]] "
        "[--promote-self-forward-model [--vowel a|e|i|o|u] [--world-vowel a|e|i|o|u] [--world-gain F] [--min-world-correlation F]]",
        file=sys.stderr,
    )
    return 2
