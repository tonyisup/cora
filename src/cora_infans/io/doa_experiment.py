"""Direction-of-Arrival geometry test for the XVF3800.

This is a research measurement, not a product feature: it drives a single
physical earcup of the External Headphones (Core Audio device 2) with a tone,
samples the firmware's native DOA_VALUE while the tone plays, and reports the
mean bearing. By running it for the LEFT channel, RIGHT channel, and BOTH, we
can verify (a) whether device-2 channel 0 maps to left or right (we do NOT
assume it), and (b) whether the reported bearing tracks the physical speaker
angle (~+/-45 deg from front in the user's current setup).

Everything is real: real USB DoA reads, real audiotoolbox playback, real mic
capture is not even required (DoA comes from the firmware's own 4-mic DSP). All
firmware writes are volatile; the beam is left in whatever state the reads
leave it (reads do not mutate state).

Usage:
    python -m cora_infans.io.doa_experiment --side left
    python -m cora_infans.io.doa_experiment --side right
    python -m cora_infans.io.doa_experiment --side both
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time

from .hardware_tuner import DoAReading, HardwareTuner, HardwareTunerError
from .live_listener import ParenteseStimulus


def run_geometry_test(
    side: str,
    *,
    duration_s: float = 2.0,
    sample_hz: float = 20.0,
    sweeps: int = 4,
    sweep_ms: int = 400,
) -> dict[str, object]:
    """Play a one-sided tone and sample the firmware DoA while it plays.

    `side` is "left" (-1 pan), "right" (+1 pan), or "both" (0 pan).
    Returns mean/std/min/max/n bearing plus the raw samples for audit.
    """
    if side not in ("left", "right", "both"):
        raise ValueError(f"side must be left|right|both, got {side!r}")
    pan = {"left": -1.0, "right": 1.0, "both": 0.0}[side]

    try:
        tuner = HardwareTuner()
    except HardwareTunerError as exc:
        return {"error": f"no XVF3800: {exc}"}

    stim = ParenteseStimulus()
    path = tempfile.mktemp(suffix=".wav")
    stim.synthesize_wav(path, sweeps=sweeps, sweep_ms=sweep_ms, pan=pan, channels=2)
    proc = stim.play(path)  # non-blocking, routes to device 2 earcups

    samples: list[float] = []
    speech_hits = 0
    interval = 1.0 / sample_hz
    deadline = time.monotonic() + duration_s
    try:
        while time.monotonic() < deadline:
            try:
                doa: DoAReading = tuner.read_doa()
                if doa.speech_detected:
                    speech_hits += 1
                # Record the bearing whenever it is finite; if NaN (no signal on
                # fixed beams) skip so we don't poison the mean with garbage.
                az = doa.azimuth_deg
                if az == az and 0.0 <= az < 360.0:
                    samples.append(az)
            except Exception:
                pass
            time.sleep(interval)
    finally:
        try:
            proc.wait(timeout=max(1, int(duration_s) + 2))
        except Exception:
            proc.kill()
        import os

        try:
            os.unlink(path)
        except OSError:
            pass

    if not samples:
        return {
            "side": side,
            "pan": pan,
            "samples": 0,
            "speech_hits": speech_hits,
            "mean_deg": None,
            "note": "no finite DoA readings captured",
        }

    return {
        "side": side,
        "pan": pan,
        "samples": len(samples),
        "speech_hits": speech_hits,
        "mean_deg": round(statistics.mean(samples), 2),
        "std_deg": round(statistics.pstdev(samples), 2),
        "min_deg": round(min(samples), 2),
        "max_deg": round(max(samples), 2),
        "raw": [round(s, 2) for s in samples],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="XVF3800 DoA geometry test")
    ap.add_argument("--side", choices=["left", "right", "both"], default="both")
    ap.add_argument("--duration-s", type=float, default=2.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    args = ap.parse_args()
    result = run_geometry_test(
        args.side, duration_s=args.duration_s, sample_hz=args.sample_hz
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
