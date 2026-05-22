import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))

from thalamus.attributor import SourceVerdict, ThalamicAttributor


class AlwaysSelf:
    def is_self(self, pcm):
        return True, 0.95


def test_m4_plumbing_promotes_self_when_detector_accepts():
    attributor = ThalamicAttributor(
        m4_detector=AlwaysSelf(),
        m4_self_threshold=0.1,
        sr=16000,
        m4_min_buffer_ms=1,
    )

    frame = np.ones(16000 // 30, dtype=np.float32)

    res = attributor.attribute(
        match_score=0.8,
        observed_onset_samples=800,
        expected_onset_samples=800,
        rms_db=-30.0,
        is_playing=True,
        is_speech=True,
        frame_audio=frame,
    )

    assert res.verdict == SourceVerdict.SELF
    assert res.metadata is not None
    assert res.metadata.get("m4_attempted") is True
    assert res.metadata.get("m4_confidence") == 0.95
