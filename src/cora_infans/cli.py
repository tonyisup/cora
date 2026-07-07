from __future__ import annotations

import json
import sys
from typing import Any

from .simulation import (
    SimulationParams,
    build_builtin_sequence_report,
    build_sequence_report,
    build_simulation_report,
)


OutputMode = str
SequenceFamily = str


def describe() -> dict[str, object]:
    return {
        "project": "cora-infans",
        "implemented_slice": "saliency_gate_to_workspace",
        "planned_subsystems": [
            "mic_array_front_end",
            "predictive_hierarchy",
            "stateful_event_workspace",
            "analysis_by_synthesis",
        ],
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


def parse_sequence_cli_args(args: list[str]) -> SequenceFamily:
    family = "sweep"
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--family":
            family = value
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return family



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
    family = parse_sequence_cli_args(args or ["--simulate-sequence"])
    return build_builtin_sequence_report(family)


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

    print(
        "usage: python -m cora_infans --describe | --simulate-frame | --simulate-sequence "
        "[--timestamp-ms N --dominant-frequency-hz F --channel-energies a,b,c,d --initial-prediction-hz F --mic-geometry l0,l1,l2,l3 --saliency-threshold F --output json|pretty] "
        "[--family sweep|zigzag|gap|multi_gap|delayed_recovery]",
        file=sys.stderr,
    )
    return 2
