"""Theta-gamma workspace scheduling (deterministic, simulation-only layer).

This models the VISION.md "Theta-Gamma Multiplexing" claim as a faithful,
test-backed scheduling layer over the event stream -- NOT as invented
neurophysics:

- A theta cycle is a slow temporal window (default 200 ms ~= 5 Hz, inside the
  cited 4-8 Hz band). It defines a volatile workspace window.
- Each theta cycle is subdivided into fast gamma sub-windows (default 25 ms
  ~= 40 Hz, inside the cited 30-80 Hz band). Events arriving in the same cycle
  are "phase-locked bound" into that cycle's assembly.
- At a theta-cycle boundary the active cycle is consolidated (frozen, read-only)
  and a fresh volatile cycle opens. Only the current cycle is mutable.
- Cross-talk is structurally impossible: a cycle's assembly is built only from
  events whose timestamp falls inside that cycle.
- A small capacity bound keeps only the most recent N cycles resident, modeling
  the working-memory capacity limit the vision invokes theta-gamma to solve.

The scheduler consumes `WorkspaceEvent`s (the exact objects the pipeline
emits) and never replaces the flat `EventWorkspace`; it is an interpretation
layer above it. The pipeline and existing reports are unchanged unless a caller
explicitly requests theta-gamma scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import WorkspaceEvent

DEFAULT_THETA_MS = 200
DEFAULT_GAMMA_MS = 25
DEFAULT_CAPACITY_CYCLES = 4

# Reserved gamma sub-window for self-generated motor intentions (vocalization
# trials). It is the LAST gamma bin of each theta cycle, so external salient
# events (which arrive at their own capture timestamps) and internal motor
# intentions (placed here deliberately) are cleanly separable within a cycle.
MOTOR_SUBWINDOW_BIN = -1  # resolved to (theta_ms // gamma_ms) - 1 at runtime


def motor_bin_of(theta_ms: int, gamma_ms: int) -> int:
    """Index of the reserved motor-assembly gamma sub-window in a theta cycle."""
    if gamma_ms <= 0:
        raise ValueError("gamma_ms must be positive")
    return max(0, (theta_ms // gamma_ms) - 1)


@dataclass
class ThetaCycle:
    index: int
    events: list = field(default_factory=list)

    @property
    def gamma_bins_occupied(self) -> int:
        bins: set[int] = set()
        for ev in self.events:
            ts = int(ev.workspace_state["timestamp_ms"])
            bins.add((ts % DEFAULT_THETA_MS) // DEFAULT_GAMMA_MS)
        return len(bins)

    @property
    def bound_focus_labels(self) -> list[str]:
        seen: list[str] = []
        for ev in self.events:
            label = str(ev.workspace_state.get("focus_label", ""))
            if label and label not in seen:
                seen.append(label)
        return seen

    @property
    def event_count(self) -> int:
        return len(self.events)

    def dominant_label(self) -> str | None:
        counts: dict[str, int] = {}
        order: list[str] = []
        for ev in self.events:
            label = str(ev.workspace_state.get("focus_label", ""))
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
            if label not in order:
                order.append(label)
        if not counts:
            return None
        best = max(order, key=lambda l: (counts[l], -order.index(l)))
        return best


class ThetaGammaScheduler:
    """Nested theta/gamma temporal-binding scheduler over workspace events."""

    def __init__(
        self,
        theta_ms: int = DEFAULT_THETA_MS,
        gamma_ms: int = DEFAULT_GAMMA_MS,
        capacity_cycles: int = DEFAULT_CAPACITY_CYCLES,
    ) -> None:
        if theta_ms <= 0 or gamma_ms <= 0:
            raise ValueError("theta_ms and gamma_ms must be positive")
        if gamma_ms > theta_ms:
            raise ValueError("gamma_ms must not exceed theta_ms")
        self.theta_ms = theta_ms
        self.gamma_ms = gamma_ms
        self.capacity_cycles = capacity_cycles
        self._consolidated: list[ThetaCycle] = []
        self._active: ThetaCycle | None = None

    def _cycle_of(self, timestamp_ms: int) -> int:
        return int(timestamp_ms) // self.theta_ms

    def motor_subwindow_bin(self) -> int:
        """Reserved gamma-bin index for motor intentions in any theta cycle."""
        return motor_bin_of(self.theta_ms, self.gamma_ms)

    def reserve_motor_timestamp(self, now_ms: int) -> int:
        """Return a timestamp placed inside the motor sub-window of the cycle
        containing ``now_ms`` (used to phase-lock a vocalization trial)."""
        cycle_idx = self._cycle_of(now_ms)
        bin_idx = self.motor_subwindow_bin()
        return cycle_idx * self.theta_ms + bin_idx * self.gamma_ms + (self.gamma_ms // 2)

    def _gamma_bin(self, event: "WorkspaceEvent") -> int:
        ts = int(event.workspace_state["timestamp_ms"])
        return (ts % self.theta_ms) // self.gamma_ms

    def push(self, event: "WorkspaceEvent") -> None:
        ts = int(event.workspace_state["timestamp_ms"])
        cycle_idx = self._cycle_of(ts)
        if self._active is None or cycle_idx != self._active.index:
            self._consolidate_active()
            self._active = ThetaCycle(index=cycle_idx)
        self._active.events.append(event)
        # Corollary discharge / sensory suppression: a self-generated motor
        # intention in a gamma bin suppresses any concurrent external
        # salient-audio event in the SAME bin of the SAME cycle (the agent does
        # not re-process its own vocalization). Deterministic, no invented
        # neurophysics -- it is a bookkeeping flag on the event state.
        self._apply_corollary_discharge(event)

    def _apply_corollary_discharge(self, event: "WorkspaceEvent") -> None:
        if self._active is None:
            return
        new_bin = self._gamma_bin(event)
        if event.kind == "motor_intention":
            # Suppress external audio already in this bin.
            for ev in self._active.events:
                if (
                    ev is not event
                    and ev.kind == "salient_audio"
                    and self._gamma_bin(ev) == new_bin
                ):
                    ev.workspace_state["self_suppressed"] = True
        elif event.kind == "salient_audio":
            # If a motor intention already occupies this bin, suppress self.
            for ev in self._active.events:
                if (
                    ev is not event
                    and ev.kind == "motor_intention"
                    and self._gamma_bin(ev) == new_bin
                ):
                    event.workspace_state["self_suppressed"] = True

    def _consolidate_active(self) -> None:
        if self._active is None:
            return
        self._consolidated.append(self._active)
        overflow = len(self._consolidated) - self.capacity_cycles
        if overflow > 0:
            del self._consolidated[:overflow]
        self._active = None

    @property
    def current_cycle_index(self) -> int | None:
        return self._active.index if self._active is not None else None

    def active_assembly(self) -> list[str]:
        return self._active.bound_focus_labels if self._active is not None else []

    def consolidated_cycles(self) -> list[ThetaCycle]:
        return list(self._consolidated)

    def total_event_count(self) -> int:
        total = sum(c.event_count for c in self._consolidated)
        if self._active is not None:
            total += self._active.event_count
        return total

    def summary(self) -> dict[str, object]:
        """Structured, analysis-ready view of the scheduling state."""
        cycles_view = [
            {
                "cycle": c.index,
                "event_count": c.event_count,
                "gamma_bins_occupied": c.gamma_bins_occupied,
                "bound_focus_labels": c.bound_focus_labels,
                "dominant_label": c.dominant_label(),
            }
            for c in self._consolidated
        ]
        if self._active is not None:
            cycles_view.append(
                {
                    "cycle": self._active.index,
                    "event_count": self._active.event_count,
                    "gamma_bins_occupied": self._active.gamma_bins_occupied,
                    "bound_focus_labels": self._active.bound_focus_labels,
                    "dominant_label": self._active.dominant_label(),
                    "active": True,
                }
            )

        # Binding stability: fraction of consecutive consolidated cycles whose
        # dominant label repeats the previous one (a deterministic attractor
        # stability proxy, not a biological claim).
        dominants = [c.dominant_label() for c in self._consolidated if c.dominant_label() is not None]
        stable = 0
        for prev, cur in zip(dominants, dominants[1:]):
            if prev == cur:
                stable += 1
        stable_fraction = (stable / (len(dominants) - 1)) if len(dominants) > 1 else 0.0

        motor_intentions = sum(
            1
            for c in self._consolidated + ([self._active] if self._active else [])
            for ev in c.events
            if ev.kind == "motor_intention"
        )
        suppressed = sum(
            1
            for c in self._consolidated + ([self._active] if self._active else [])
            for ev in c.events
            if ev.workspace_state.get("self_suppressed") is True
        )

        return {
            "theta_ms": self.theta_ms,
            "gamma_ms": self.gamma_ms,
            "capacity_cycles": self.capacity_cycles,
            "motor_subwindow_bin": self.motor_subwindow_bin(),
            "resident_cycle_count": len(self._consolidated) + (1 if self._active else 0),
            "active_cycle": self.current_cycle_index,
            "active_assembly": self.active_assembly(),
            "binding_stability_fraction": stable_fraction,
            "motor_intentions": motor_intentions,
            "self_suppressed_events": suppressed,
            "cycles": cycles_view,
        }
