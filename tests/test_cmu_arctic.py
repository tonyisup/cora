import pytest

from cora_infans.cmu_arctic import parse_label_index, parse_phone_labels


def test_parse_label_index_returns_sorted_unique_names() -> None:
    html = (
        '<a href="arctic_b0002.lab">x</a>'
        '<a href="arctic_a0001.lab">x</a>'
        '<a href="arctic_b0002.lab">x</a>'
    )

    assert parse_label_index(html) == ["arctic_a0001.lab", "arctic_b0002.lab"]


def test_parse_phone_labels_converts_cumulative_endpoints() -> None:
    segments = parse_phone_labels(
        "#\n0.040000 125 pau\n0.200000 125 ao\n0.300000 125 th\n"
    )

    assert segments[1].phone == "ao"
    assert segments[1].start_seconds == pytest.approx(0.04)
    assert segments[1].end_seconds == pytest.approx(0.2)
    assert segments[1].duration_seconds == pytest.approx(0.16)


def test_parse_phone_labels_rejects_nonincreasing_time() -> None:
    with pytest.raises(ValueError, match="endpoints must increase"):
        parse_phone_labels("0.1 125 aa\n0.1 125 eh\n")
