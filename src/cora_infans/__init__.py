from .config import CoraInfansConfig
from .pipeline import AudioFrame, CoraInfansPipeline, WorkspaceEvent
from .predictive import PredictionUpdate, PredictiveCoder
from .saliency import compute_saliency_reward
from .scenario import (
    ChannelLabel,
    EnvelopeKind,
    ForegroundKind,
    ScenarioDefinition,
    ScenarioSegment,
    builtin_scenario_definition,
    load_scenario_definition_yaml,
    lower_scenario_definition,
)
from .simulation import (
    SimulationParams,
    build_builtin_sequence_report,
    build_sequence_report,
    build_simulation_report,
)
from .spatial import resolve_focus_label
from .workspace import EventWorkspace

__all__ = [
    "AudioFrame",
    "CoraInfansConfig",
    "CoraInfansPipeline",
    "ChannelLabel",
    "EnvelopeKind",
    "EventWorkspace",
    "ForegroundKind",
    "PredictiveCoder",
    "PredictionUpdate",
    "ScenarioDefinition",
    "ScenarioSegment",
    "SimulationParams",
    "WorkspaceEvent",
    "build_builtin_sequence_report",
    "builtin_scenario_definition",
    "build_sequence_report",
    "build_simulation_report",
    "compute_saliency_reward",
    "load_scenario_definition_yaml",
    "lower_scenario_definition",
    "resolve_focus_label",
]
__version__ = "0.1.0"
