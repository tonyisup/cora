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
