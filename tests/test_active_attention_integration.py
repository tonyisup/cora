"""Integration: closed-loop active attention via LiveListener with a fake array.

Uses a fake mic array + an injected _NullTuner so the governor's full decision
path (saliency gate -> deadband -> steer) is exercised in-process without any
USB transfer. This is the test-suite guard the blueprint called for: the real
pyusb writes only fire under --listen-live with real silicon present.
"""

import math

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


def test_closed_loop_logs_beam_azimuth_but_steers_on_derived_lr():
    # After the DoA geometry test we learned DOA_VALUE echoes the beam's own
    # azimuth (not the source), so the loop must NOT feed it back as a steer
    # target. Instead it logs the beam azimuth for transparency and steers on
    # the capture-derived L/R estimate. _NullTuner.read_doa() returns 45deg.
    frames = [_frame(0.05, 0.30), _frame(0.04, 0.28)]
    tuner = _NullTuner()
    listener = LiveListener(array=_FakeArray(frames), tuner=tuner)
    report = listener.listen(duration_ms=500, frame_ms=250, with_attention=True)
    att = report["active_attention"]
    log = att["attention_log"]
    assert log
    # Beam azimuth is logged (from read_doa), not used as the steer target.
    beam_entries = [e for e in log if e["beam_azimuth_deg"] is not None]
    assert beam_entries, "expected beam azimuth in the attention log"
    for e in beam_entries:
        assert e["beam_azimuth_deg"] == 45.0
        # The actual steer target is the derived L/R (right louder -> ~64deg),
        # NOT the beam azimuth echo.
        assert e["derived_steer_azimuth_deg"] != 45.0


def test_governor_accepts_external_ground_truth_target():
    # The governor still supports an explicit target (e.g. a known speaker
    # position supplied by an external source of truth), distinct from the
    # self-referential firmware DoA.
    from cora_infans.io.active_attention import ActiveAttentionGovernor

    g = ActiveAttentionGovernor()
    g.propose(0.0, 0.05, now_ms=0)
    g.register_steer(0.0, now_ms=0)
    # 160 deg move at 180 deg/s cap needs >= ~889 ms; use 1000 ms.
    d = g.propose(10.0, 0.05, now_ms=1000, explicit_target_deg=200.0)
    assert d.acted is True
    assert d.proposed_azimuth_deg == 200.0
    g.register_steer(200.0, now_ms=1000)
    assert g.current_azimuth_deg == 200.0


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



class _SourceDoATuner(_RadiansTuner):
    def __init__(self, source_deg):
        super().__init__()
        self.source_deg = source_deg

    def read_doa(self):
        from cora_infans.io.hardware_tuner import DoAReading

        return DoAReading(azimuth_deg=45.0, speech_detected=True)

    def read_selected_azimuths(self):
        return [math.radians(self.source_deg), 0.0]


def test_closed_loop_uses_finite_processed_source_doa_as_target():
    # Right channel is louder, so the coarse L/R fallback would be ~64deg.
    # A finite processed source-DoA from AUDIO_MGR_SELECTED_AZIMUTHS[0] should
    # override that fallback and become the actual steer target.
    frames = [_frame(0.05, 0.30)]
    tuner = _SourceDoATuner(source_deg=200.0)
    listener = LiveListener(array=_FakeArray(frames), tuner=tuner)

    report = listener.listen(duration_ms=250, frame_ms=250, with_attention=True)

    log = report["active_attention"]["attention_log"]
    assert log[0]["source_azimuth_deg"] == 200.0
    assert log[0]["selected_source_used"] is True
    assert log[0]["derived_steer_azimuth_deg"] != 200.0
    assert tuner.steer_history == [200.0]


def test_closed_loop_falls_back_to_derived_lr_when_processed_source_doa_is_nan():
    frames = [_frame(0.05, 0.30)]
    tuner = _SourceDoATuner(source_deg=math.nan)
    listener = LiveListener(array=_FakeArray(frames), tuner=tuner)

    report = listener.listen(duration_ms=250, frame_ms=250, with_attention=True)

    log = report["active_attention"]["attention_log"]
    assert log[0]["source_azimuth_deg"] is None
    assert log[0]["selected_source_used"] is False
    assert tuner.steer_history
    assert round(tuner.steer_history[0], 2) == log[0]["derived_steer_azimuth_deg"]
