from __future__ import annotations

import json
import sys
from typing import Any

from .io.live_listener import LiveListener
from .io.live_mic import DEFAULT_DEVICE, LiveMicArray, LiveMicError
from .simulation import (
    SimulationParams,
    build_builtin_sequence_report,
    build_live_frame_report,
    build_sequence_report,
    build_sequence_report_with_scheduling,
    build_simulation_report,
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
        ],
        "planned_subsystems": [
            "analysis_by_synthesis",
            "automatic_source_localization",
            "training_and_evaluation_harnesses",
            "general_user_authored_sequences",
        ],
        "verdicts": {
            "source_localization": "not_proven_non_speech_doa_geometry_negative",
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
    with_scheduling = False
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
    if with_scheduling:
        return build_sequence_report_with_scheduling(family)
    return build_builtin_sequence_report(family)


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
        "[--listen-live [--duration-ms N] [--frame-ms N] [--with-stimulus] [--with-attention] [--with-scheduling]]",
        file=sys.stderr,
    )
    return 2
