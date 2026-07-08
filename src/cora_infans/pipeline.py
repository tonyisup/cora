from __future__ import annotations

from dataclasses import dataclass

from .config import CoraInfansConfig
from .predictive import PredictiveCoder, PredictiveHierarchy
from .saliency import compute_saliency_reward
from .spatial import resolve_focus_label
from .workspace import EventWorkspace


@dataclass(frozen=True)
class AudioFrame:
    timestamp_ms: int
    channel_energies: list[float]
    dominant_frequency_hz: float


@dataclass(frozen=True)
class WorkspaceEvent:
    kind: str
    saliency_score: float
    workspace_state: dict[str, int | float | str]


class CoraInfansPipeline:
    def __init__(
        self,
        saliency_threshold: float | None = None,
        config: CoraInfansConfig | None = None,
        predictive_coder: PredictiveCoder | None = None,
        predictive_hierarchy: PredictiveHierarchy | None = None,
    ) -> None:
        self.config = config or CoraInfansConfig()
        self.saliency_threshold = (
            saliency_threshold
            if saliency_threshold is not None
            else self.config.saliency_threshold
        )
        # The hierarchy subsumes the single frequency coder when provided; a
        # bare predictive_coder is still accepted for backward compatibility.
        self.predictive_coder = predictive_coder
        self.predictive_hierarchy = predictive_hierarchy
        self.workspace = EventWorkspace(window_ms=self.config.workspace_window_ms)

    def process_frame(self, frame: AudioFrame) -> WorkspaceEvent | None:
        saliency_score = max(frame.channel_energies, default=0.0)
        if saliency_score < self.saliency_threshold:
            return None

        focus_channel = frame.channel_energies.index(saliency_score)
        focus_label = resolve_focus_label(focus_channel, self.config.mic_channel_geometry)
        workspace_state: dict[str, int | float | str] = {
            "focus_channel": focus_channel,
            "focus_label": focus_label,
            "dominant_frequency_hz": frame.dominant_frequency_hz,
            "timestamp_ms": frame.timestamp_ms,
            "saliency_reward": compute_saliency_reward(frame.channel_energies),
        }
        if self.predictive_hierarchy is not None:
            freq = self.predictive_hierarchy.update_frequency(frame.dominant_frequency_hz)
            focus = self.predictive_hierarchy.update_focus(focus_label)
            workspace_state["predictive_residual_hz"] = freq.residual_hz
            workspace_state["updated_prediction_hz"] = freq.updated_prediction_hz
            workspace_state["higher_predictive_previous_label"] = (
                focus.previous_prediction_label
                if focus.previous_prediction_label is not None
                else ""
            )
            workspace_state["higher_predictive_residual_miss"] = focus.residual_miss
        elif self.predictive_coder is not None:
            prediction = self.predictive_coder.update(frame.dominant_frequency_hz)
            workspace_state["predictive_residual_hz"] = prediction.residual_hz
            workspace_state["updated_prediction_hz"] = prediction.updated_prediction_hz

        event = WorkspaceEvent(
            kind="salient_audio",
            saliency_score=saliency_score,
            workspace_state=workspace_state,
        )
        self.workspace.push(event)
        return event
