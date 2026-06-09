#!/usr/bin/env python3
"""Compatibility shim for the current M9 live vowel demo.

Keep this path working, but route it to the latest demo implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m9_live_vowel_demo import main


if __name__ == "__main__":
    main()
