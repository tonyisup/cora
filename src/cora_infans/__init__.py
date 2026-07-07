from .config import CoraInfansConfig
from .pipeline import AudioFrame, CoraInfansPipeline, WorkspaceEvent
from .predictive import PredictionUpdate, PredictiveCoder
from .saliency import compute_saliency_reward
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
    "EventWorkspace",
    "PredictiveCoder",
    "PredictionUpdate",
    "SimulationParams",
    "WorkspaceEvent",
    "build_builtin_sequence_report",
    "build_sequence_report",
    "build_simulation_report",
    "compute_saliency_reward",
    "resolve_focus_label",
]
__version__ = "0.1.0"
