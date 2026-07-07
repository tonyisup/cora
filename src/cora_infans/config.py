from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoraInfansConfig:
    mic_channel_count: int = 4
    mic_channel_geometry: tuple[str, ...] = (
        "channel_0",
        "channel_1",
        "channel_2",
        "channel_3",
    )
    saliency_threshold: float = 0.7
    workspace_window_ms: int = 250

    def __post_init__(self) -> None:
        if len(self.mic_channel_geometry) != self.mic_channel_count:
            raise ValueError(
                "mic channel geometry length must match mic channel count"
            )
