from cora_infans.pipeline import WorkspaceEvent
from cora_infans.workspace import EventWorkspace



def test_workspace_keeps_events_within_configured_time_window() -> None:
    workspace = EventWorkspace(window_ms=250)

    workspace.push(
        WorkspaceEvent(
            kind="salient_audio",
            saliency_score=0.9,
            workspace_state={
                "timestamp_ms": 100,
                "focus_channel": 1,
                "focus_label": "front_right",
            },
        )
    )
    workspace.push(
        WorkspaceEvent(
            kind="salient_audio",
            saliency_score=0.8,
            workspace_state={
                "timestamp_ms": 300,
                "focus_channel": 2,
                "focus_label": "rear_right",
            },
        )
    )
    workspace.push(
        WorkspaceEvent(
            kind="salient_audio",
            saliency_score=0.7,
            workspace_state={
                "timestamp_ms": 401,
                "focus_channel": 0,
                "focus_label": "front_left",
            },
        )
    )

    events = workspace.snapshot()

    assert len(events) == 2
    assert [event.workspace_state["timestamp_ms"] for event in events] == [300, 401]
    assert workspace.focus_path() == ["rear_right", "front_left"]
