"""Tests for Step 1: Adaptive Motor-Imitation Loop (live loop closure).

Wires the perception->production bridge into the live listener so a high-
saliency speech frame (firmware VAD fired -> finite processed source-DoA)
is passed through analyze_vowel() and the MotorPlanner re-derives a motor
target. The loop is *logged* as an internal vocalization trial (no live
playback to the speaker, to avoid acoustic feedback contamination).

These run with a fake mic array + bus-free tuner, so no XVF3800 or USB is
touched. The trigger for "speech frame" is exactly a finite processed source
azimuth (AUDIO_MGR_SELECTED_AZIMUTHS[0]), matching Addendum 6.
"""

import math

import numpy as np

from cora_infans.io.live_listener import LiveListener, _NullTuner
from cora_infans.pipeline import AudioFrame
from cora_infans.io.motor_plan import synthesize_vowel


class _FakeArray:
    """Yields scripted frames; exposes a real PCM waveform per frame via
    ``last_pcm`` (mirroring the real LiveMicArray contract added for Step 1)."""

    channel_count = 2

    def __init__(self, items):
        self._items = items
        self._i = 0
        self.last_pcm: np.ndarray | None = None

    def capture_frame(self, duration_ms=None, dominant_frequency_hz=None):
        item = self._items[self._i % len(self._items)]
        self._i += 1
        self.last_pcm = item["pcm"]
        return item["frame"]


class _SpeechTuner(_NullTuner):
    """Bus-free tuner that reports a finite processed source-DoA on demand.

    The listener calls read_selected_azimuths() once per salient frame; we
    return a finite (speech) azimuth when the current frame is flagged speech.
    """

    def __init__(self, speech_flags):
        super().__init__()
        self._flags = speech_flags
        self._j = 0

    def read_selected_azimuths(self):
        flag = self._flags[self._j % len(self._flags)]
        self._j += 1
        if flag:
            return [math.radians(185.0), math.radians(163.0)]
        return [math.nan, math.nan]


def _frame(ch0, ch1, vowel="a", dom=300.0):
    pcm = synthesize_vowel(vowel, duration_ms=240, sample_rate=16000)
    return {
        "pcm": pcm,
        "frame": AudioFrame(
            timestamp_ms=0, channel_energies=[ch0, ch1], dominant_frequency_hz=dom
        ),
    }


def _make_listener(items, speech_flags):
    array = _FakeArray(items)
    tuner = _SpeechTuner(speech_flags)
    return LiveListener(array=array, tuner=tuner, saliency_threshold=0.02)


def test_motor_imitation_logs_trial_on_speech_frame() -> None:
    # Frame 0 non-salient (no tuner call); frame 1 salient + speech -> one trial.
    items = [_frame(0.01, 0.01), _frame(0.20, 0.25)]
    listener = _make_listener(items, speech_flags=[True])
    report = listener.listen(
        duration_ms=500, frame_ms=250, with_attention=True, with_motor_imitation=True
    )
    mi = report["motor_imitation"]
    assert mi is not None
    assert mi["enabled"] is True
    assert mi["trial_count"] >= 1
    trial = mi["trials"][0]
    assert "perceived_formants_hz" in trial
    assert "motor_plan_vowel" in trial
    assert trial["source_azimuth_deg"] is not None
    assert "resynth_samples" in trial and trial["resynth_samples"] > 0


def test_motor_imitation_silent_on_nonspeech() -> None:
    items = [_frame(0.20, 0.25), _frame(0.18, 0.22)]
    listener = _make_listener(items, speech_flags=[False, False])
    report = listener.listen(
        duration_ms=500, frame_ms=250, with_attention=True, with_motor_imitation=True
    )
    mi = report["motor_imitation"]
    assert mi["trial_count"] == 0
    assert mi["trials"] == []


def test_motor_imitation_disabled_by_default() -> None:
    items = [_frame(0.20, 0.25)]
    listener = _make_listener(items, speech_flags=[True])
    report = listener.listen(
        duration_ms=250, frame_ms=250, with_attention=True, with_motor_imitation=False
    )
    assert report["motor_imitation"] is None


def test_motor_imitation_requires_attention_enabled() -> None:
    items = [_frame(0.20, 0.25)]
    listener = _make_listener(items, speech_flags=[True])
    report = listener.listen(
        duration_ms=250, frame_ms=250, with_attention=False, with_motor_imitation=True
    )
    assert report["motor_imitation"] is None
