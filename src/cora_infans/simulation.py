from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import CoraInfansConfig
from .io.mic_array import SyntheticMicArray
from .pipeline import CoraInfansPipeline
from .predictive import PredictiveCoder
from .scenario import builtin_scenario_definition, lower_scenario_definition
from .predictive import FocusTrajectoryCoder, PredictiveCoder, PredictiveHierarchy


@dataclass(frozen=True)
class SimulationParams:
    timestamp_ms: int = 125
    dominant_frequency_hz: float = 1200.0
    channel_energies: list[float] = field(default_factory=lambda: [0.2, 0.9, 0.4, 0.1])
    initial_prediction_hz: float = 1000.0
    mic_channel_geometry: tuple[str, ...] = (
        "channel_0",
        "channel_1",
        "channel_2",
        "channel_3",
    )
    saliency_threshold: float = 0.7



def _build_config(params: SimulationParams) -> CoraInfansConfig:
    return CoraInfansConfig(
        mic_channel_geometry=params.mic_channel_geometry,
        saliency_threshold=params.saliency_threshold,
        workspace_window_ms=250,
    )



def _build_pipeline(config: CoraInfansConfig, params: SimulationParams) -> CoraInfansPipeline:
    predictive_hierarchy = PredictiveHierarchy(
        frequency_coder=PredictiveCoder(
            initial_prediction_hz=params.initial_prediction_hz,
            learning_rate=0.5,
        ),
        focus_coder=FocusTrajectoryCoder(
            channels=config.mic_channel_geometry,
        ),
    )
    return CoraInfansPipeline(config=config, predictive_hierarchy=predictive_hierarchy)



def _capture_frame(params: SimulationParams, mic_array: SyntheticMicArray):
    return mic_array.capture_frame(
        timestamp_ms=params.timestamp_ms,
        channel_energies=params.channel_energies,
        dominant_frequency_hz=params.dominant_frequency_hz,
    )



def _builtin_sequence_family(family: str) -> list[SimulationParams]:
    scenario = builtin_scenario_definition(family)
    return lower_scenario_definition(scenario)



def build_simulation_report(params: SimulationParams) -> dict[str, object]:
    config = _build_config(params)
    mic_array = SyntheticMicArray(channel_count=config.mic_channel_count)
    pipeline = _build_pipeline(config, params)

    frame = _capture_frame(params, mic_array)
    event = pipeline.process_frame(frame)
    workspace_snapshot = [asdict(item) for item in pipeline.workspace.snapshot()]
    return {
        "status": "salient" if event is not None else "non_salient",
        "focus_path": pipeline.workspace.focus_path(),
        "resolved_config": asdict(config),
        "frame": asdict(frame),
        "event": asdict(event) if event is not None else None,
        "workspace_snapshot": workspace_snapshot,
    }



def build_sequence_report(sequence: list[SimulationParams]) -> dict[str, object]:
    if not sequence:
        return {
            "frame_count": 0,
            "salient_frame_count": 0,
            "non_salient_frame_count": 0,
            "dropout_count": 0,
            "recovery_count": 0,
            "emitted_event_count": 0,
            "focus_path": [],
            "focus_transitions": [],
            "workspace_snapshot": [],
        }

    config = _build_config(sequence[0])
    mic_array = SyntheticMicArray(channel_count=config.mic_channel_count)
    pipeline = _build_pipeline(config, sequence[0])

    frames: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    frame_is_salient: list[bool] = []
    for params in sequence:
        frame = _capture_frame(params, mic_array)
        event = pipeline.process_frame(frame)
        is_salient = event is not None
        frame_is_salient.append(is_salient)
        frames.append(asdict(frame))
        if is_salient:
            events.append(asdict(event))

    focus_path = pipeline.workspace.focus_path()
    salient_frame_count = len(events)
    non_salient_frame_count = len(sequence) - salient_frame_count
    dropout_count = sum(1 for is_salient in frame_is_salient if not is_salient)
    recovery_count = sum(
        1
        for previous_is_salient, current_is_salient in zip(frame_is_salient, frame_is_salient[1:])
        if (not previous_is_salient) and current_is_salient
    )
    return {
        "frame_count": len(sequence),
        "salient_frame_count": salient_frame_count,
        "non_salient_frame_count": non_salient_frame_count,
        "dropout_count": dropout_count,
        "recovery_count": recovery_count,
        "emitted_event_count": len(events),
        "focus_path": focus_path,
        "focus_transitions": [
            f"{left}->{right}"
            for left, right in zip(focus_path, focus_path[1:])
        ],
        "frames": frames,
        "events": events,
        "workspace_snapshot": [asdict(item) for item in pipeline.workspace.snapshot()],
    }



def build_builtin_sequence_report(family: str = "sweep") -> dict[str, object]:
    report = build_sequence_report(_builtin_sequence_family(family))
    report["family"] = family
    return report
