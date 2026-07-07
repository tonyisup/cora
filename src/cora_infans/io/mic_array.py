from __future__ import annotations

from cora_infans.pipeline import AudioFrame


class SyntheticMicArray:
    def __init__(self, channel_count: int = 4) -> None:
        self.channel_count = channel_count

    def capture_frame(
        self,
        *,
        timestamp_ms: int,
        channel_energies: list[float],
        dominant_frequency_hz: float,
    ) -> AudioFrame:
        if len(channel_energies) != self.channel_count:
            raise ValueError(
                f"expected {self.channel_count} channels, got {len(channel_energies)}"
            )

        return AudioFrame(
            timestamp_ms=timestamp_ms,
            channel_energies=channel_energies,
            dominant_frequency_hz=dominant_frequency_hz,
        )
