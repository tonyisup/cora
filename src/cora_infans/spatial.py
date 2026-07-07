from __future__ import annotations


def resolve_focus_label(
    focus_channel: int,
    mic_channel_geometry: tuple[str, ...],
) -> str:
    return mic_channel_geometry[focus_channel]
