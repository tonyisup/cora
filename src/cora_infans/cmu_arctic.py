"""CMU Arctic alignment parsing and development-only acquisition helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


TARGET_PHONE_TO_VOWEL = {
    "aa": "a",
    "eh": "e",
    "iy": "i",
    "ao": "o",
    "uw": "u",
}
LABEL_NAME = re.compile(r'href="(?P<name>arctic_[ab]\d{4}\.lab)"')


@dataclass(frozen=True)
class PhoneSegment:
    phone: str
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def parse_label_index(html: str) -> list[str]:
    names = sorted({match.group("name") for match in LABEL_NAME.finditer(html)})
    if not names:
        raise ValueError("CMU Arctic label index contains no label files")
    return names


def parse_phone_labels(text: str) -> list[PhoneSegment]:
    segments: list[PhoneSegment] = []
    start = 0.0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 3:
            raise ValueError(f"unexpected CMU Arctic label row: {line}")
        end = float(fields[0])
        phone = fields[2].lower()
        if end <= start:
            raise ValueError("CMU Arctic label endpoints must increase")
        segments.append(PhoneSegment(phone, start, end))
        start = end
    if not segments:
        raise ValueError("CMU Arctic label file contains no segments")
    return segments


def sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_tokens(
    label_dir: str | Path,
    *,
    per_vowel: int,
    minimum_duration_seconds: float,
    excluded_label_names: set[str] | None = None,
) -> list[dict[str, object]]:
    if per_vowel <= 0 or minimum_duration_seconds <= 0.0:
        raise ValueError("token selection limits must be positive")
    counts = {vowel: 0 for vowel in TARGET_PHONE_TO_VOWEL.values()}
    selected: list[dict[str, object]] = []
    excluded = excluded_label_names or set()
    for label_path in sorted(Path(label_dir).glob("arctic_[ab][0-9][0-9][0-9][0-9].lab")):
        if label_path.name in excluded:
            continue
        for segment_index, segment in enumerate(
            parse_phone_labels(label_path.read_text())
        ):
            vowel = TARGET_PHONE_TO_VOWEL.get(segment.phone)
            if (
                vowel is None
                or counts[vowel] >= per_vowel
                or segment.duration_seconds < minimum_duration_seconds
            ):
                continue
            selected.append(
                {
                    "label_name": label_path.name,
                    "wav_name": f"{label_path.stem}.wav",
                    "segment_index": segment_index,
                    "phone": segment.phone,
                    "vowel": vowel,
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "duration_seconds": segment.duration_seconds,
                    "selection_ordinal_for_vowel": counts[vowel],
                }
            )
            counts[vowel] += 1
        if all(count == per_vowel for count in counts.values()):
            break
    if any(count != per_vowel for count in counts.values()):
        raise ValueError(f"insufficient qualifying CMU Arctic tokens: {counts}")
    return selected
