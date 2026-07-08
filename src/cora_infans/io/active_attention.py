"""Active top-down auditory oculomotor gaze (superior-colliculus analogue).

This module is the "Active Attention Governor" -- the closed-loop brain between
the software saliency map and the XVF3800 hardware beam. It decides WHEN and
WHERE to point the acoustic laser, subject to three honest physical constraints:

1. Saliency gate  -- never orient toward a frame below the verified 0.02 RMS
   baseline; quiet-room noise does not earn a saccade.
2. Spatial deadband (Delta-theta) -- only re-steer if the target azimuth moved
   more than `deadband_deg` (default 5 deg). Prevents the beam from jittering
   on minor echo / reverberation.
3. Velocity slew limit -- the beam cannot sweep faster than `max_slew_deg_per_s`
   (default 180 deg/s, a biologically-plausible upper bound). Prevents
   un-biological, bus-spamming tracking.

The governor is PURE (no USB / hardware imports). It is driven by the
LiveListener, which supplies the HardwareTuner only when real silicon is present
under `--listen-live`; synthetic/test runs use a no-op tuner so the test suite
never touches the bus.

Honesty note on spatial resolution: the XVF3800 fuses its 4 mics into 2 USB
channels, so the saliency map can only resolve LEFT vs RIGHT (a front
hemisphere, +/-90 deg) from the captured stream. The hardware beam itself can
point anywhere in 0..359 deg, but DERIVING a target finer than L/R from 2 fused
channels would be inventing localization we do not have. The governor therefore
exposes `derive_target_azimuth()` returning a coarse L/R estimate and lets
callers (or future true-DoA firmware reads) override the target directly.
"""
from __future__ import annotations

from dataclasses import dataclass

# Verified live saliency baseline from the capture tuning work: real RMS
# energies sit ~0.01-0.03, so 0.02 is the honest "is this a real signal?" gate.
DEFAULT_SALIENCY_GATE = 0.02
DEFAULT_DEADBAND_DEG = 5.0
# Biologically-plausible upper bound on saccadic sweep speed (deg/sec).
DEFAULT_MAX_SLEW_DEG_PER_S = 180.0


@dataclass
class SteerDecision:
    """Transparent record of one governor evaluation."""

    acted: bool
    reason: str
    proposed_azimuth_deg: float
    current_azimuth_deg: float
    gaze_error_deg: float
    elapsed_since_last_steer_ms: float | None


@dataclass
class AttentionConfig:
    saliency_gate: float = DEFAULT_SALIENCY_GATE
    deadband_deg: float = DEFAULT_DEADBAND_DEG
    max_slew_deg_per_s: float = DEFAULT_MAX_SLEW_DEG_PER_S
    # Front-hemisphere mapping bounds for the 2-channel L/R estimate.
    front_hemisphere_deg: float = 90.0


def derive_target_azimuth(
    channel_energies: list[float], front_hemisphere_deg: float = 90.0
) -> float:
    """Coarse L/R target azimuth from 2 fused channels (honest resolution).

    Returns a front-hemisphere angle in [-front_hemisphere_deg, +front_hemisphere_deg]:
    negative = left, positive = right. With 2 channels we can only resolve which
    side is louder, not a full 360 deg bearing -- so we map the normalized
    energy imbalance linearly across the front hemisphere. Monaural / silent
    input returns 0.0 (straight ahead).
    """
    if len(channel_energies) < 2:
        # Single channel: no lateral information; look straight ahead.
        return 0.0
    left, right = channel_energies[0], channel_energies[1]
    total = left + right
    if total <= 0.0:
        return 0.0
    # Imbalance in [-1, 1]; positive => right louder.
    imbalance = (right - left) / total
    return imbalance * front_hemisphere_deg


class ActiveAttentionGovernor:
    """Decides when/where to steer the acoustic beam based on the saliency map.

    Pure logic -- no hardware, no time source beyond the `now_ms` passed in.
    The owner (LiveListener) is responsible for actually issuing the steer via
    a HardwareTuner and for calling `register_steer` after the bus write.
    """

    def __init__(
        self,
        config: AttentionConfig | None = None,
        *,
        initial_azimuth_deg: float = 0.0,
    ) -> None:
        self.config = config or AttentionConfig()
        self.current_azimuth_deg = float(initial_azimuth_deg)
        self.target_azimuth_deg = float(initial_azimuth_deg)
        self._last_steer_ms: float | None = None
        self.gaze_error_deg: float = 0.0
        self.steer_count: int = 0
        self.reward_sum: float = 0.0
        self.reward_count: int = 0
        self.history: list[SteerDecision] = []

    # -- target proposal ----------------------------------------------------
    def propose(
        self,
        azimuth_deg: float,
        saliency_score: float,
        now_ms: float,
        explicit_target_deg: float | None = None,
    ) -> SteerDecision:
        """Evaluate a candidate target; return whether to act (and why).

        `azimuth_deg` is the target used for the gate/deadband/slew math. When
        the caller has an external higher-fidelity bearing (for example a known
        speaker position from an experiment), pass it as `explicit_target_deg`;
        the governor will STEER toward that explicit bearing while still
        allowing `azimuth_deg` to be the fallback estimate. Do NOT feed the
        XVF3800 `DOA_VALUE` register back here on firmware 2.0.6: the geometry
        test showed it echoes the beam's own azimuth, not the acoustic source.
        """
        target = explicit_target_deg if explicit_target_deg is not None else azimuth_deg
        elapsed = (
            (now_ms - self._last_steer_ms)
            if self._last_steer_ms is not None
            else None
        )

        # 1. Saliency gate.
        if saliency_score < self.config.saliency_gate:
            decision = SteerDecision(
                acted=False,
                reason="below_saliency_gate",
                proposed_azimuth_deg=target,
                current_azimuth_deg=self.current_azimuth_deg,
                gaze_error_deg=self.gaze_error_deg,
                elapsed_since_last_steer_ms=elapsed,
            )
            self.history.append(decision)
            return decision

        # 2. Spatial deadband.
        delta = _angular_delta(target, self.current_azimuth_deg)
        if delta < self.config.deadband_deg:
            decision = SteerDecision(
                acted=False,
                reason="within_deadband",
                proposed_azimuth_deg=target,
                current_azimuth_deg=self.current_azimuth_deg,
                gaze_error_deg=delta,
                elapsed_since_last_steer_ms=elapsed,
            )
            self.history.append(decision)
            return decision

        # 3. Velocity slew limit (deg/sec). No prior steer => always allowed.
        if elapsed is not None and elapsed > 0:
            slew = delta / (elapsed / 1000.0)
            if slew > self.config.max_slew_deg_per_s:
                decision = SteerDecision(
                    acted=False,
                    reason="slew_limited",
                    proposed_azimuth_deg=target,
                    current_azimuth_deg=self.current_azimuth_deg,
                    gaze_error_deg=delta,
                    elapsed_since_last_steer_ms=elapsed,
                )
                self.history.append(decision)
                return decision

        # Passed all gates: this is a real, justified saccade.
        self.target_azimuth_deg = target
        self.gaze_error_deg = delta
        decision = SteerDecision(
            acted=True,
            reason="steer",
            proposed_azimuth_deg=target,
            current_azimuth_deg=self.current_azimuth_deg,
            gaze_error_deg=delta,
            elapsed_since_last_steer_ms=elapsed,
        )
        self.history.append(decision)
        return decision

    def register_steer(self, actual_azimuth_deg: float, now_ms: float) -> None:
        """Owner calls this AFTER the bus write succeeds, to advance state."""
        self.current_azimuth_deg = float(actual_azimuth_deg)
        self._last_steer_ms = now_ms
        self.steer_count += 1

    def reward(self, saliency_score: float) -> None:
        """Accumulate saliency reward while the gaze is on a salient target."""
        self.reward_sum += saliency_score
        self.reward_count += 1

    # -- reporting ----------------------------------------------------------
    def summary(self) -> dict[str, object]:
        return {
            "current_azimuth_deg": round(self.current_azimuth_deg, 2),
            "target_azimuth_deg": round(self.target_azimuth_deg, 2),
            "gaze_error_deg": round(self.gaze_error_deg, 2),
            "steer_count": self.steer_count,
            "reward_sum": round(self.reward_sum, 4),
            "reward_count": self.reward_count,
            "history": [
                {
                    "acted": d.acted,
                    "reason": d.reason,
                    "proposed": round(d.proposed_azimuth_deg, 2),
                    "current": round(d.current_azimuth_deg, 2),
                    "error": round(d.gaze_error_deg, 2),
                }
                for d in self.history
            ],
        }


def _angular_delta(a: float, b: float) -> float:
    """Smallest absolute angular difference (handles wraparound at 360)."""
    diff = abs((a % 360.0) - (b % 360.0))
    return min(diff, 360.0 - diff)
