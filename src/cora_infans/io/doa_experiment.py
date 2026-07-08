"""Direction-of-Arrival geometry test for the XVF3800.

This is a research measurement, not a product feature: it drives a single
physical earcup of the External Headphones (Core Audio device 2) with a tone,
samples the firmware's DOA_VALUE register while the tone plays, and reports the
mean readback. By running it for the LEFT channel, RIGHT channel, and BOTH, we
test whether the register tracks the physical speaker angle (~+/-45 deg from
front in the user's current setup) or merely reflects firmware beam state.

Everything is real: real USB reads and real audiotoolbox playback. Current
verdict: tone/non-speech localization is negative (`DOA_VALUE` echoes beam
azimuth and processed source-DoA stays NAN), but macOS TTS speech can fire the
firmware VAD and make `AUDIO_MGR_SELECTED_AZIMUTHS[0]` return a source bearing.
Reads do not mutate firmware state.

Usage:
    python -m cora_infans.io.doa_experiment --stimulus tone --side left
    python -m cora_infans.io.doa_experiment --stimulus speech --side right
    python -m cora_infans.io.doa_experiment --stimulus speech --side both
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import tempfile
import time

from .hardware_tuner import DoAReading, HardwareTuner, HardwareTunerError
from .live_listener import DEFAULT_STIMULUS_SAMPLE_RATE, ParenteseStimulus


_SIDE_TO_PAN = {"left": -1.0, "right": 1.0, "both": 0.0}
_SIDE_TO_FFMPEG_PAN = {
    "left": "pan=stereo|c0=c0|c1=0*c0",
    "right": "pan=stereo|c0=0*c0|c1=c0",
    "both": "pan=stereo|c0=c0|c1=c0",
}


class SpeechStimulus:
    """macOS TTS speech stimulus, panned to one physical earcup.

    This is intentionally simple and system-native: `/usr/bin/say` creates a
    speech-like waveform that has a better chance of tripping the XVF3800's
    firmware VAD than a pure tone; `ffmpeg` then converts/pans it into a stereo
    WAV for the External Headphones output path used by `ParenteseStimulus`.
    """

    def __init__(
        self,
        *,
        text: str = "Hello baby. Look here. Hello baby. Look here.",
        sample_rate: int = DEFAULT_STIMULUS_SAMPLE_RATE,
        output: ParenteseStimulus | None = None,
    ) -> None:
        self.text = text
        self.sample_rate = sample_rate
        self._output = output or ParenteseStimulus(sample_rate=sample_rate)

    def synthesize_wav(self, path: str, *, side: str = "both") -> None:
        if side not in _SIDE_TO_FFMPEG_PAN:
            raise ValueError(f"side must be left|right|both, got {side!r}")
        tmp_aiff = tempfile.mktemp(suffix=".aiff")
        try:
            subprocess.run(
                [
                    "/usr/bin/say",
                    "-o",
                    tmp_aiff,
                    self.text,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    tmp_aiff,
                    "-af",
                    _SIDE_TO_FFMPEG_PAN[side],
                    "-ar",
                    str(self.sample_rate),
                    "-ac",
                    "2",
                    path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        finally:
            try:
                os.unlink(tmp_aiff)
            except OSError:
                pass

    def play(self, path: str):
        return self._output.play(path)


def run_geometry_test(
    side: str,
    *,
    stimulus: str = "tone",
    duration_s: float = 2.0,
    sample_hz: float = 20.0,
    sweeps: int = 4,
    sweep_ms: int = 400,
) -> dict[str, object]:
    """Play a one-sided tone and sample DOA_VALUE while it plays.

    `side` is "left" (-1 pan), "right" (+1 pan), or "both" (0 pan).
    Returns mean/std/min/max/n readback plus the raw samples for audit.
    """
    if side not in _SIDE_TO_PAN:
        raise ValueError(f"side must be left|right|both, got {side!r}")
    if stimulus not in ("tone", "speech"):
        raise ValueError(f"stimulus must be tone|speech, got {stimulus!r}")
    pan = _SIDE_TO_PAN[side]

    try:
        tuner = HardwareTuner()
    except HardwareTunerError as exc:
        return {"error": f"no XVF3800: {exc}"}

    path = tempfile.mktemp(suffix=".wav")
    if stimulus == "speech":
        stim = SpeechStimulus()
        stim.synthesize_wav(path, side=side)
    else:
        stim = ParenteseStimulus()
        stim.synthesize_wav(path, sweeps=sweeps, sweep_ms=sweep_ms, pan=pan, channels=2)
    proc = stim.play(path)  # non-blocking, routes to device 2 earcups

    samples: list[float] = []
    selected_source_samples: list[float] = []
    selected_beam_samples: list[float] = []
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
                try:
                    source_rad, beam_rad = tuner.read_selected_azimuths()
                    source_deg = math.degrees(float(source_rad)) % 360
                    beam_deg = math.degrees(float(beam_rad)) % 360
                    if math.isfinite(source_deg):
                        selected_source_samples.append(source_deg)
                    if math.isfinite(beam_deg):
                        selected_beam_samples.append(beam_deg)
                except Exception:
                    pass
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
            "stimulus": stimulus,
            "pan": pan,
            "samples": 0,
            "selected_source_samples": len(selected_source_samples),
            "selected_source_mean_deg": _mean_deg(selected_source_samples),
            "selected_beam_samples": len(selected_beam_samples),
            "selected_beam_mean_deg": _mean_deg(selected_beam_samples),
            "speech_hits": speech_hits,
            "mean_deg": None,
            "note": "no finite DoA readings captured",
        }

    return {
        "side": side,
        "stimulus": stimulus,
        "pan": pan,
        "samples": len(samples),
        "selected_source_samples": len(selected_source_samples),
        "selected_source_mean_deg": _mean_deg(selected_source_samples),
        "selected_beam_samples": len(selected_beam_samples),
        "selected_beam_mean_deg": _mean_deg(selected_beam_samples),
        "speech_hits": speech_hits,
        "mean_deg": round(statistics.mean(samples), 2),
        "std_deg": round(statistics.pstdev(samples), 2),
        "min_deg": round(min(samples), 2),
        "max_deg": round(max(samples), 2),
        "raw": [round(s, 2) for s in samples],
        "selected_source_raw": [round(s, 2) for s in selected_source_samples],
        "selected_beam_raw": [round(s, 2) for s in selected_beam_samples],
    }


def _mean_deg(samples: list[float]) -> float | None:
    return round(statistics.mean(samples), 2) if samples else None


def main() -> int:
    ap = argparse.ArgumentParser(description="XVF3800 DoA geometry test")
    ap.add_argument("--side", choices=["left", "right", "both"], default="both")
    ap.add_argument("--stimulus", choices=["tone", "speech"], default="tone")
    ap.add_argument("--duration-s", type=float, default=2.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    args = ap.parse_args()
    result = run_geometry_test(
        args.side,
        stimulus=args.stimulus,
        duration_s=args.duration_s,
        sample_hz=args.sample_hz,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
