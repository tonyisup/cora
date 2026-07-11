import pytest

from cora_infans.cmu_arctic import parse_label_index, parse_phone_labels, select_tokens


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


def test_select_tokens_is_lexical_balanced_and_duration_bounded(tmp_path) -> None:
    phones = ("aa", "eh", "iy", "ao", "uw")
    for file_index in (2, 1):
        end = 0.0
        rows = ["#"]
        for phone in phones:
            end += 0.12 if file_index == 1 else 0.13
            rows.append(f"{end:.3f} 125 {phone}")
        (tmp_path / f"arctic_a{file_index:04d}.lab").write_text("\n".join(rows))

    selected = select_tokens(
        tmp_path, per_vowel=1, minimum_duration_seconds=0.12
    )

    assert len(selected) == 5
    assert {token["vowel"] for token in selected} == set("aeiou")
    assert {token["label_name"] for token in selected} == {"arctic_a0001.lab"}
    assert all(float(token["duration_seconds"]) >= 0.12 for token in selected)
