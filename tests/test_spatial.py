from cora_infans.spatial import resolve_focus_label



def test_resolve_focus_label_uses_configured_geometry_order() -> None:
    focus_label = resolve_focus_label(2, ("front_left", "front_right", "rear_right", "rear_left"))

    assert focus_label == "rear_right"
