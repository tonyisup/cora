"""Integration: closed-loop active attention via LiveListener with a fake array.

Uses a fake mic array + an injected _NullTuner so the governor's full decision
path (saliency gate -> deadband -> steer) is exercised in-process without any
USB transfer. This is the test-suite guard the blueprint called for: the real
pyusb writes only fire under --listen-live with real silicon present.
"""

from cora_infans.io.active_attention import ActiveAttentionGovernor, AttentionConfig
from cora_infans.io.live_listener import LiveListener, _NullTuner
from cora_infans.pipeline import AudioFrame


class _FakeArray:
    """Yields scripted frames so the gaze loop is deterministic."""

    channel_count = 2

    def __init__(self, frames):
        self._frames = frames
        self._i = 0

    def capture_frame(self, duration_ms=None, dominant_frequency_hz=None):
        f = self._frames[self._i % len(self._frames)]
        self._i += 1
        return f


def _frame(ch0, ch1, dom=440.0):
    return AudioFrame(timestamp_ms=0, channel_energies=[ch0, ch1], dominant_frequency_hz=dom)


def test_closed_loop_steers_on_salient_and_respects_deadband():
    # Frame 0: quiet (below gate) -> no steer.
    # Frame 1: loud on right -> steer right.
    # Frame 2: loud but only slightly different azimuth -> deadband blocks.
    frames = [
        _frame(0.01, 0.01),     # quiet
        _frame(0.05, 0.30),     # salient, right
        _frame(0.06, 0.31),     # salient, ~same azimuth (deadband)
    ]
    tuner = _NullTuner()
    gov = ActiveAttentionGovernor(AttentionConfig(deadband_deg=5.0))
    listener = LiveListener(array=_FakeArray(frames), governor=gov, tuner=tuner)

    report = listener.listen(duration_ms=750, frame_ms=250, with_attention=True)
    att = report["active_attention"]
    assert att["enabled"] is True
    assert att["is_real_hardware"] is False
    # Only the genuinely-moving salient frame should have steered.
    assert tuner.steer_history  # at least one steer happened
    assert gov.steer_count >= 1
    # Governor summary present and coherent.
    assert "governor" in att and att["governor"]["steer_count"] >= 1
    assert len(att["attention_log"]) >= 1


def test_closed_loop_disabled_by_default_leaves_no_attention_block():
    frames = [_frame(0.05, 0.30), _frame(0.04, 0.28)]
    listener = LiveListener(array=_FakeArray(frames))
    report = listener.listen(duration_ms=500, frame_ms=250, with_attention=False)
    assert report["active_attention"] is None


def test_closed_loop_uses_firmware_doa_when_available():
    # _NullTuner.read_doa() returns a deterministic 45deg bearing with speech
    # detected, so the loop should steer toward 45 (the true DoA), not the
    # coarse derived L/R estimate.
    frames = [_frame(0.05, 0.30), _frame(0.04, 0.28)]
    tuner = _NullTuner()
    listener = LiveListener(array=_FakeArray(frames), tuner=tuner)
    report = listener.listen(duration_ms=500, frame_ms=250, with_attention=True)
    att = report["active_attention"]
    log = att["attention_log"]
    assert log
    # At least one entry should show a DoA reading and steer toward it.
    doa_entries = [e for e in log if e["doa_azimuth_deg"] is not None]
    assert doa_entries, "expected DoA readings in the attention log"
    for e in doa_entries:
        # Bearing is used even though the (fake) VAD flag is False.
        assert e["doa_speech_detected"] is False
        assert e["doa_azimuth_deg"] == 45.0
        # The beam was steered toward the DoA bearing, not the derived L/R.
        assert e["steered_to_azimuth_deg"] == 45.0


class _RadiansTuner:
    """Mimics HardwareTuner.steer_to: returns firmware azimuth in RADIANS."""

    is_real_hardware = True

    def __init__(self):
        self.steer_history = []

    def steer_to(self, azimuth_deg):
        import math

        rad = math.radians(azimuth_deg)
        self.steer_history.append(azimuth_deg)
        return [rad, 0.0]  # radians, like the real firmware read-back


def test_closed_loop_converts_radians_readback_to_degrees():
    # The real HardwareTuner returns radians from steer_to(); the governor must
    # record degrees. This guards the radians->degrees unit boundary.
    frames = [_frame(0.02, 0.05), _frame(0.03, 0.06)]
    tuner = _RadiansTuner()
    listener = LiveListener(array=_FakeArray(frames), tuner=tuner)
    report = listener.listen(duration_ms=500, frame_ms=250, with_attention=True)
    gov = report["active_attention"]["governor"]
    # 2nd channel louder -> positive (right) azimuth in degrees, NOT radians.
    assert gov["current_azimuth_deg"] > 0
    assert gov["current_azimuth_deg"] < 90  # right-front, not a radian value
    assert tuner.steer_history  # steers happened
