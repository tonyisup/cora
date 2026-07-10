"""Tests for Step 3: integrate vocalizations into the theta-gamma scheduler.

Wires the Step-1 motor-plan trial (an internal vocalization intention) into a
reserved gamma sub-window of the theta cycle, and models corollary discharge:
an external salient-audio event sharing that gamma bin in the same cycle is
flagged self_suppressed (the agent does not re-hear its own vocalization).

All tests are deterministic and hardware-free.
"""

from __future__ import annotations

import numpy as np

from cora_infans.theta_gamma import ThetaGammaScheduler, motor_bin_of
from cora_infans.pipeline import WorkspaceEvent, AudioFrame


def _audio(
    ts: int,
    label: str,
    *,
    expected_self_audio: bool = False,
    efference_id: str | None = None,
) -> WorkspaceEvent:
    state = {
        "timestamp_ms": ts,
        "focus_label": label,
        "saliency_reward": 0.1,
    }
    if expected_self_audio:
        state["expected_self_audio"] = True
    if efference_id is not None:
        state["efference_id"] = efference_id
    return WorkspaceEvent(
        kind="salient_audio",
        saliency_score=0.5,
        workspace_state=state,
    )


def _motor(ts: int, efference_id: str | None = None) -> WorkspaceEvent:
    state = {"timestamp_ms": ts, "perceived_vowel": "a"}
    if efference_id is not None:
        state["efference_id"] = efference_id
    return WorkspaceEvent(
        kind="motor_intention",
        saliency_score=0.3,
        workspace_state=state,
    )


def test_motor_subwindow_is_last_gamma_bin() -> None:
    sched = ThetaGammaScheduler(theta_ms=200, gamma_ms=25)
    # 200/25 = 8 bins, indexed 0..7; motor bin is the last one.
    assert sched.motor_subwindow_bin() == 7
    assert motor_bin_of(200, 25) == 7
    assert motor_bin_of(200, 40) == 4


def test_motor_timestamp_lands_in_reserved_bin() -> None:
    sched = ThetaGammaScheduler(theta_ms=200, gamma_ms=25)
    ts = sched.reserve_motor_timestamp(now_ms=130)  # cycle 0, mid-cycle
    assert (ts % 200) // 25 == sched.motor_subwindow_bin()
    # Mid-bin placement.
    assert ts == 7 * 25 + 12


def test_motor_intention_phase_locked_into_subwindow() -> None:
    sched = ThetaGammaScheduler(theta_ms=200, gamma_ms=25)
    # External audio early in cycle 0.
    sched.push(_audio(10, "L_mid"))
    # Motor intention placed in the reserved sub-window of cycle 0.
    motor_ts = sched.reserve_motor_timestamp(10)
    sched.push(_motor(motor_ts))
    summary = sched.summary()
    assert summary["motor_intentions"] == 1
    assert summary["motor_subwindow_bin"] == 7
    # Force consolidation by opening cycle 1.
    sched.push(_audio(210, "L_mid"))
    cyc = sched.consolidated_cycles()[0]
    bins = {(int(ev.workspace_state["timestamp_ms"]) % 200) // 25 for ev in cyc.events}
    assert sched.motor_subwindow_bin() in bins


def test_corollary_discharge_suppresses_matching_verified_self_audio() -> None:
    sched = ThetaGammaScheduler(theta_ms=200, gamma_ms=25)
    # External audio arriving in the SAME gamma bin as the motor sub-window
    # (bin 7 spans 175..200 ms). Place audio at 185 ms.
    sched.push(
        _audio(185, "L_mid", expected_self_audio=True, efference_id="trial-1")
    )
    motor_ts = sched.reserve_motor_timestamp(185)  # -> bin 7 mid (187)
    sched.push(_motor(motor_ts, efference_id="trial-1"))
    # The external audio in bin 7 must be flagged self_suppressed.
    summary = sched.summary()
    assert summary["self_suppressed_events"] == 1
    # Inspect the flagged event directly.
    sched.push(_audio(385, "L_mid"))  # consolidate cycle 0
    cyc = sched.consolidated_cycles()[0]
    suppressed = [ev for ev in cyc.events if ev.workspace_state.get("self_suppressed") is True]
    assert len(suppressed) == 1
    assert suppressed[0].kind == "salient_audio"


def test_phase_coincident_world_audio_is_not_suppressed() -> None:
    sched = ThetaGammaScheduler(theta_ms=200, gamma_ms=25)
    sched.push(_audio(185, "world"))
    sched.push(_motor(sched.reserve_motor_timestamp(185), efference_id="trial-1"))

    assert sched.summary()["self_suppressed_events"] == 0


def test_no_suppression_when_bins_differ() -> None:
    sched = ThetaGammaScheduler(theta_ms=200, gamma_ms=25)
    # Audio in bin 0 (0..25 ms), motor in bin 7 -> different bins, no suppression.
    sched.push(_audio(5, "L_mid"))
    sched.push(_motor(sched.reserve_motor_timestamp(5)))
    assert sched.summary()["self_suppressed_events"] == 0
    assert sched.summary()["motor_intentions"] == 1


def test_listener_phase_locks_motor_intention_under_scheduling() -> None:
    from cora_infans.io.live_listener import LiveListener, _NullTuner
    from cora_infans.io.motor_plan import synthesize_vowel

    class _FakeArray:
        channel_count = 2
        last_pcm = None

        def __init__(self, items):
            self._items = items
            self._idx = 0

        def capture_frame(self, duration_ms=250):
            item = self._items[min(self._idx, len(self._items) - 1)]
            self._idx += 1
            wav = item["wav"]
            frame = AudioFrame(
                timestamp_ms=item["ts"],
                channel_energies=item["energy"],
                dominant_frequency_hz=200.0,
            )
            self.last_pcm = wav
            return frame

    class _FakeTuner(_NullTuner):
        def read_selected_azimuths(self):
            return [1.0, 0.0]  # finite speech source azimuth

    # Real vowel waveform so the Step-1 perception bridge can run.
    speech_wav = synthesize_vowel("a", duration_ms=300, sample_rate=16000)
    items = [
        {"ts": 0, "energy": [0.01, 0.01], "wav": None},
        {"ts": 250, "energy": [0.20, 0.25], "wav": speech_wav},
    ]

    listener = LiveListener(array=_FakeArray(items), tuner=_FakeTuner())
    report = listener.listen(
        duration_ms=500,
        frame_ms=250,
        with_attention=True,
        with_motor_imitation=True,
        with_scheduling=True,
    )
    tg = report["theta_gamma"]
    assert tg is not None
    # The motor intention was phase-locked into the scheduler.
    assert tg["motor_intentions"] >= 1
    assert tg["motor_subwindow_bin"] == 7
    # And the matching vocalization trial is in the motor-imitation log.
    mi = report["motor_imitation"]
    assert mi["enabled"] is True
    assert mi["trial_count"] >= 1
