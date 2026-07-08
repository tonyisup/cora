"""Continuous live listener for the XVF3800 with an optional parentese stimulus.

This is the "A" slice: a real-time listening loop that captures frames from the
board, runs each through the deterministic pipeline, and accumulates focus-path
tracking, dropout, and recovery over a time window -- the same sequence-report
shape the built-in families produce, but fed by genuine hardware audio.

Stimulus (parentese):
- A `ParenteseStimulus` generates a simple infant-directed-speech-like contour
  (slow rising/falling pitch sweep) and plays it to the External Headphones
  output device (audiotoolbox index 2) so the mic has a controlled acoustic
  target. The listener does NOT claim the "infant" perceives anything; it only
  provides a repeatable stimulus so the saliency gate has something to detect.

Honesty boundary (per VISION.md / README.md):
- Every captured frame is tagged `stimulus_active` so downstream analysis can
  separate stimulated from quiet-room input. No frame is ever fabricated.
- The listener fails loudly if it cannot open the mic; it never substitutes
  synthetic audio for a missed capture.
"""

from __future__ import annotations

import math
import subprocess
import tempfile

import numpy as np

from ..config import CoraInfansConfig
from ..pipeline import AudioFrame, CoraInfansPipeline, WorkspaceEvent
from .live_mic import (
    DEFAULT_DEVICE,
    DEFAULT_LIVE_SALIENCY_THRESHOLD,
    CAPTURE_CHANNELS,
    LiveMicArray,
    LiveMicError,
    _require_ffmpeg,
)
from ..predictive import FocusTrajectoryCoder, PredictiveCoder, PredictiveHierarchy
from .active_attention import (
    ActiveAttentionGovernor,
    AttentionConfig,
    derive_target_azimuth,
)
from .hardware_tuner import HardwareTuner, HardwareTunerError, DoAReading
from .motor_plan import MotorPlanner, analyze_vowel, synthesize_vowel

# External Headphones output device index (audiotoolbox), per host enumeration.
STIMULUS_OUTPUT_DEVICE_INDEX = 2
DEFAULT_STIMULUS_SAMPLE_RATE = 48000


class _NullTuner:
    """Bus-free stand-in for HardwareTuner in synthetic / test / no-silicon runs.

    Records every steer it is asked to perform so the governor's decisions are
    still observable in reports, but never issues a USB control transfer. This
    is how the closed loop stays fully exercised under `pytest` without touching
    the XVF3800.
    """

    def __init__(self) -> None:
        self.steer_history: list[float] = []
        self.released = False

    def steer_to(self, azimuth_deg: float) -> list[float]:
        # Validate the same range the real HardwareTuner enforces, so the
        # closed loop is exercised honestly under tests too.
        if not 0 <= azimuth_deg < 360:
            raise ValueError(f"_NullTuner: azimuth out of range: {azimuth_deg}")
        self.steer_history.append(azimuth_deg)
        return [azimuth_deg, 0.0]

    def release_beam(self) -> list[float]:
        self.released = True
        return [0]

    def read_doa(self) -> "DoAReading":
        # Bus-free stand-in: report a deterministic beam-azimuth readback. This
        # value is logged as state only; it must not be fed back as a source
        # target (the real 2.0.6 firmware echoes the beam azimuth here).
        return DoAReading(azimuth_deg=45.0, speech_detected=False)

    def read_selected_azimuths(self) -> list[float]:
        # Bus-free stand-in for AUDIO_MGR_SELECTED_AZIMUTHS: no processed
        # source-DoA unless a fake/real tuner explicitly supplies one.
        return [math.nan, math.nan]

    @property
    def is_real_hardware(self) -> bool:
        return False


class ParenteseStimulus:
    """Synthesizes and plays a simple parentese-like pitch contour.

    Parentese is characterized by a higher, exaggerated, and slowly modulated
    pitch relative to adult speech. We model only that acoustic envelope: a
    few rising/falling pitch sweeps on a pure tone, at a gentle tempo. This is
    a research stimulus, not a speech-synthesis claim.
    """

    def __init__(
        self,
        output_device_index: int = STIMULUS_OUTPUT_DEVICE_INDEX,
        sample_rate: int = DEFAULT_STIMULUS_SAMPLE_RATE,
    ) -> None:
        self.output_device_index = output_device_index
        self.sample_rate = sample_rate

    def synthesize_wav(
        self,
        path: str,
        *,
        sweeps: int = 3,
        sweep_ms: int = 400,
        pan: float = 0.0,
        channels: int = 2,
    ) -> None:
        """Write a parentese-like stimulus (pitch sweeps) to `path`.

        `pan` routes the tone: -1 = left channel only, 0 = both, +1 = right
        only. `channels` is the output channel count (default 2 = stereo, so a
        single-earcup can be driven for Direction-of-Arrival geometry tests).
        The default `pan=0, channels=2` preserves the original both-speakers
        behavior; callers wanting a one-sided source pass pan=-1/+1.
        """
        sr = self.sample_rate
        total = int(sweeps * sweep_ms / 1000.0 * sr)
        t = np.arange(total) / sr
        # Each sweep rises then falls; consecutive sweeps step the center pitch
        # up and down to mimic the exaggerated prosody of parentese.
        freq = np.zeros(total, dtype=np.float64)
        seg = int(sweep_ms / 1000.0 * sr)
        base = 300.0
        for i in range(sweeps):
            s = i * seg
            e = s + seg
            phase = np.linspace(0.0, np.pi, seg)
            center = base * (1.0 + 0.6 * np.sin(i * np.pi / max(sweeps - 1, 1)))
            amp = (np.sin(phase) * 0.5 + 0.5) * 250.0  # 0..250 Hz swing
            freq[s:e] = center + amp
        phase = np.cumsum(2 * np.pi * freq / sr)
        tone = 0.95 * np.sin(phase)
        # Soft amplitude envelope so each sweep has a gentle attack/decay.
        env = np.ones(total)
        ramp = int(0.05 * sr)
        env[:ramp] = np.linspace(0.0, 1.0, ramp)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp)
        mono = (tone * env * 32767.0).astype("<i2")

        # Build the requested channel layout from the mono signal.
        left_gain = 1.0 if pan <= 0 else 0.0
        right_gain = 1.0 if pan >= 0 else 0.0
        frames: list[np.ndarray] = []
        for ch in range(channels):
            gain = left_gain if ch == 0 else right_gain
            frames.append((mono.astype(np.float32) * gain).astype("<i2"))
        interleaved = np.column_stack(frames) if channels > 1 else frames[0]

        import wave

        with wave.open(path, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(interleaved.tobytes())

    def play(self, path: str) -> subprocess.Popen[bytes]:
        """Play `path` to the stimulus output device (non-blocking).

        Returns the Popen handle so callers can overlap capture with playback.
        Raises LiveMicError if ffmpeg cannot start the playback process.
        """
        ffmpeg = _require_ffmpeg()
        sink = "/tmp/_stim_sink.wav"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", path,
            "-af", "volume=12dB",
            "-f", "audiotoolbox", "-audio_device_index", str(self.output_device_index),
            sink,
        ]
        try:
            return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise LiveMicError(f"stimulus playback failed to start: {exc}") from exc


class LiveListener:
    """Captures a time window of live frames and accumulates tracking metrics."""

    def __init__(
        self,
        array: LiveMicArray | None = None,
        *,
        saliency_threshold: float = DEFAULT_LIVE_SALIENCY_THRESHOLD,
        initial_prediction_hz: float = 1000.0,
        workspace_window_ms: int = 250,
        stimulus: ParenteseStimulus | None = None,
        governor: ActiveAttentionGovernor | None = None,
        tuner: HardwareTuner | _NullTuner | None = None,
        enable_motor_imitation: bool = False,
    ) -> None:
        self.array = array or LiveMicArray(device=DEFAULT_DEVICE)
        self.saliency_threshold = saliency_threshold
        self.initial_prediction_hz = initial_prediction_hz
        self.workspace_window_ms = workspace_window_ms
        self.stimulus = stimulus or ParenteseStimulus()
        self._pipeline: CoraInfansPipeline | None = None
        # The ActiveAttentionGovernor mediates between software saliency state
        # and hardware execution. If no tuner is supplied we resolve one lazily
        # at listen() time: real silicon only when actually live, else a
        # bus-free _NullTuner so tests never touch the USB bus.
        self.governor = governor or ActiveAttentionGovernor()
        self._tuner = tuner  # None => resolve in listen()
        # Step 1: perception->production bridge. Off by default; enabled per-run
        # via listen(with_motor_imitation=True). No live playback happens here
        # (to avoid acoustic feedback); matches are logged as internal trials.
        self._enable_motor_imitation = enable_motor_imitation
        self._planner = MotorPlanner()

    def _resolve_tuner(self) -> HardwareTuner | _NullTuner:
        if self._tuner is not None:
            return self._tuner
        try:
            self._tuner = HardwareTuner()
        except HardwareTunerError:
            # No XVF3800 on the bus (e.g. CI, synthetic runs). Use the no-op.
            self._tuner = _NullTuner()
        return self._tuner

    def _motor_imitation_trial(
        self,
        pcm: "np.ndarray | None",
        source_azimuth_deg: float,
        now_ms: int,
        frame_idx: int,
    ) -> "dict[str, object] | None":
        """Run one perception->production trial on a captured speech frame.

        Returns a trial record, or None if there is no analyzable waveform.
        """
        if pcm is None or len(pcm) < 1600:  # need >= 100 ms @ 16 kHz
            return None
        try:
            feat = analyze_vowel(pcm, sample_rate=16000)
        except ValueError:
            return None
        # The perceived vowel selects the table-anchored motor target; the
        # planner re-synthesizes that target (the "internal vocalization").
        plan = self._planner.plan_from_analysis(feat)
        resynth = self._planner.plan_wav(
            plan.vowel, duration_ms=200, sample_rate=16000
        )
        return {
            "frame": frame_idx,
            "now_ms": now_ms,
            "source_azimuth_deg": round(source_azimuth_deg, 2),
            "perceived_formants_hz": [round(f, 2) for f in feat.formant_hz],
            "perceived_nearest_vowel": feat.nearest_vowel,
            "motor_plan_vowel": plan.vowel,
            "motor_plan_f1_hz": plan.f1_hz,
            "resynth_samples": len(resynth),
        }

    def _build_motor_imitation_report(
        self, log: list[dict[str, object]], enabled: bool
    ) -> "dict[str, object] | None":
        if not enabled:
            return None
        return {
            "enabled": True,
            "trial_count": len(log),
            "trials": log,
        }

    def _build_pipeline(self) -> CoraInfansPipeline:
        ch = self.array.channel_count
        geometry = tuple(f"ch{i}" for i in range(ch))
        config = CoraInfansConfig(
            mic_channel_count=ch,
            mic_channel_geometry=geometry,
            saliency_threshold=self.saliency_threshold,
            workspace_window_ms=self.workspace_window_ms,
        )
        hierarchy = PredictiveHierarchy(
            frequency_coder=PredictiveCoder(
                initial_prediction_hz=self.initial_prediction_hz, learning_rate=0.5
            ),
            focus_coder=FocusTrajectoryCoder(channels=geometry),
        )
        return CoraInfansPipeline(config=config, predictive_hierarchy=hierarchy)

    def listen(
        self,
        *,
        duration_ms: int = 2000,
        frame_ms: int = 250,
        with_stimulus: bool = False,
        with_scheduling: bool = True,
        with_attention: bool = False,
        with_motor_imitation: bool = False,
        theta_ms: int = 200,
        gamma_ms: int = 25,
        capacity_cycles: int = 4,
    ) -> dict[str, object]:
        """Capture `duration_ms` of live audio in `frame_ms` steps.

        Returns an accumulated report (salient/non-salient counts, dropout,
        recovery, focus_path, focus_transitions) plus a per-frame log tagged
        with `stimulus_active`. Theta-gamma scheduling is ON by default;
        salient events are fed into a theta-gamma scheduler and its summary is
        attached under `theta_gamma`. Pass `with_scheduling=False` for the bare
        flat-workspace view.

        When `with_attention` is set, each salient frame drives the
        ActiveAttentionGovernor, which may issue a hardware steer_to() toward
        the saliency-derived target azimuth (subject to the saliency gate,
        spatial deadband, and velocity slew limit). The tuner is real silicon
        only when the XVF3800 is actually present; otherwise a bus-free no-op
        tuner keeps the loop fully exercised under tests.
        """
        from ..theta_gamma import ThetaGammaScheduler

        pipeline = self._build_pipeline()
        self._pipeline = pipeline
        scheduler = (
            ThetaGammaScheduler(
                theta_ms=theta_ms, gamma_ms=gamma_ms, capacity_cycles=capacity_cycles
            )
            if with_scheduling
            else None
        )
        tuner = self._resolve_tuner() if with_attention else None

        stim_proc = None
        stim_path = None
        if with_stimulus:
            stim_path = tempfile.mktemp(suffix=".wav")
            self.stimulus.synthesize_wav(stim_path)
            stim_proc = self.stimulus.play(stim_path)

        frames: list[dict[str, object]] = []
        events: list[dict[str, object]] = []
        frame_is_salient: list[bool] = []
        attention_log: list[dict[str, object]] = []
        motor_imitation_log: list[dict[str, object]] = []
        target_frames = max(1, duration_ms // frame_ms)
        for i in range(target_frames):
            now_ms = i * frame_ms
            frame = self.array.capture_frame(duration_ms=frame_ms)
            event = pipeline.process_frame(frame)
            is_salient = event is not None
            frame_is_salient.append(is_salient)
            frame_view = asdict_public(frame)
            frame_view["stimulus_active"] = with_stimulus
            frames.append(frame_view)
            if is_salient and event is not None:
                events.append(asdict_public(event))
                if scheduler is not None:
                    scheduler.push(event)
                # --- closed-loop active attention -------------------------
                if tuner is not None:
                    saliency_score = float(event.saliency_score)
                    # derive_target_azimuth returns a SIGNED front-hemisphere
                    # angle [-90, +90] (left negative, right positive). The
                    # hardware beam expects [0, 360) with 0=front,90=right,
                    # 270=left, so fold the signed estimate into that range.
                    target = derive_target_azimuth(frame.channel_energies)
                    steer_azimuth = (target + 360.0) % 360.0
                    # HONEST NOTE (verified by DoA geometry tests): on firmware
                    # 2.0.6, DOA_VALUE echoes the beam's OWN steered azimuth,
                    # not the acoustic source bearing. Log it as beam state.
                    # The real processed source-DoA lives at
                    # AUDIO_MGR_SELECTED_AZIMUTHS[0]: finite only when firmware
                    # VAD/classification fires, NAN for tone/non-speech. Use it
                    # as an explicit target when finite; otherwise fall back to
                    # the coarse capture-derived L/R estimate.
                    beam_azimuth = None
                    doa_speech_detected = None
                    if hasattr(tuner, "read_doa"):
                        try:
                            doa = tuner.read_doa()
                            beam_azimuth = doa.azimuth_deg
                            doa_speech_detected = doa.speech_detected
                        except Exception:
                            beam_azimuth = None
                            doa_speech_detected = None
                    source_azimuth = None
                    selected_beam_azimuth = None
                    if hasattr(tuner, "read_selected_azimuths"):
                        try:
                            selected = tuner.read_selected_azimuths()
                            if len(selected) >= 1:
                                source_candidate = math.degrees(float(selected[0])) % 360.0
                                if math.isfinite(source_candidate):
                                    source_azimuth = source_candidate
                            if len(selected) >= 2:
                                beam_candidate = math.degrees(float(selected[1])) % 360.0
                                if math.isfinite(beam_candidate):
                                    selected_beam_azimuth = beam_candidate
                        except Exception:
                            source_azimuth = None
                            selected_beam_azimuth = None
                    decision = self.governor.propose(
                        steer_azimuth,
                        saliency_score,
                        now_ms,
                        explicit_target_deg=source_azimuth,
                    )
                    if decision.acted:
                        # HardwareTuner.steer_to returns the firmware's
                        # azimuth read-back in RADIANS; the governor's world is
                        # degrees, so convert before recording execution.
                        actual = tuner.steer_to(decision.proposed_azimuth_deg)
                        self.governor.register_steer(
                            actual[0] * 180.0 / math.pi, now_ms
                        )
                    if saliency_score >= self.governor.config.saliency_gate:
                        self.governor.reward(saliency_score)
                    attention_log.append(
                        {
                            "frame": i,
                            "now_ms": now_ms,
                            "target_azimuth_deg": round(target, 2),
                            "derived_steer_azimuth_deg": round(steer_azimuth, 2),
                            "beam_azimuth_deg": (
                                round(beam_azimuth, 2) if beam_azimuth is not None else None
                            ),
                            "source_azimuth_deg": (
                                round(source_azimuth, 2) if source_azimuth is not None else None
                            ),
                            "selected_beam_azimuth_deg": (
                                round(selected_beam_azimuth, 2)
                                if selected_beam_azimuth is not None
                                else None
                            ),
                            "selected_source_used": source_azimuth is not None,
                            "doa_speech_detected": doa_speech_detected,
                            "saliency_score": round(saliency_score, 4),
                            "acted": decision.acted,
                            "reason": decision.reason,
                        }
                    )

                    # --- Step 1: adaptive motor-imitation loop ------------
                    # When a salient frame is classified as speech by the
                    # firmware VAD (finite processed source-DoA), run it through
                    # the perception->production bridge: analyze the captured
                    # waveform, map the perceived vowel to its motor target via
                    # the MotorPlanner, and re-synthesize the motor plan. This
                    # is logged as an internal vocalization trial (echolalia /
                    # babbling model). No live playback occurs here, to avoid
                    # acoustic feedback contamination of the capture.
                    if with_motor_imitation and source_azimuth is not None:
                        pcm = getattr(self.array, "last_pcm", None)
                        trial = self._motor_imitation_trial(
                            pcm, source_azimuth, now_ms, i
                        )
                        if trial is not None:
                            motor_imitation_log.append(trial)
                            # --- Step 3: phase-lock the motor intention into
                            # the reserved theta-gamma motor sub-window. The
                            # scheduler applies a SCHEDULING-LEVEL corollary
                            # discharge: an external salient event landing in the
                            # same gamma bin of the same cycle is flagged
                            # self_suppressed (bookkeeping flag, not signal-level
                            # spectral subtraction -- see theta_gamma.py docstring).
                            if scheduler is not None:
                                motor_ts = scheduler.reserve_motor_timestamp(now_ms)
                                motor_event = WorkspaceEvent(
                                    kind="motor_intention",
                                    saliency_score=float(
                                        trial.get("resynth_samples", 0)
                                    )
                                    / 1000.0,
                                    workspace_state={
                                        "timestamp_ms": motor_ts,
                                        "perceived_vowel": str(trial.get("perceived_vowel", "")),
                                        "motor_target_vowel": str(trial.get("motor_target_vowel", "")),
                                        "source_azimuth_deg": round(float(source_azimuth), 2),
                                        "frame": i,
                                    },
                                )
                                scheduler.push(motor_event)

        if stim_proc is not None:
            try:
                stim_proc.wait(timeout=max(1, int(duration_ms / 1000) + 2))
            except Exception:
                stim_proc.kill()
            # Unlink only after playback has finished reading the file.
            if stim_path is not None:
                import os

                try:
                    os.unlink(stim_path)
                except OSError:
                    pass

        salient = len(events)
        non_salient = len(frames) - salient
        dropout = sum(1 for s in frame_is_salient if not s)
        recovery = sum(
            1
            for prev, cur in zip(frame_is_salient, frame_is_salient[1:])
            if (not prev) and cur
        )
        focus_path = pipeline.workspace.focus_path()
        return {
            "source": "live_xvf3800",
            "with_stimulus": with_stimulus,
            "frame_count": len(frames),
            "salient_frame_count": salient,
            "non_salient_frame_count": non_salient,
            "dropout_count": dropout,
            "recovery_count": recovery,
            "emitted_event_count": salient,
            "focus_path": focus_path,
            "focus_transitions": [
                f"{a}->{b}" for a, b in zip(focus_path, focus_path[1:])
            ],
            "frames": frames,
            "events": events,
            "workspace_snapshot": [
                asdict_public(ev) for ev in pipeline.workspace.snapshot()
            ],
            "theta_gamma": scheduler.summary() if scheduler is not None else None,
            "active_attention": (
                {
                    "enabled": with_attention,
                    "is_real_hardware": getattr(tuner, "is_real_hardware", False)
                    if tuner is not None
                    else False,
                    "governor": self.governor.summary(),
                    "attention_log": attention_log,
                }
                if with_attention
                else None
            ),
            "motor_imitation": self._build_motor_imitation_report(
                motor_imitation_log, with_motor_imitation and with_attention
            ),
        }


def asdict_public(obj: object) -> dict[str, object]:
    """Public dict view for AudioFrame / WorkspaceEvent dataclasses."""
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return {k: v for k, v in dataclasses.asdict(obj).items()}  # type: ignore[arg-type]
    return dict(obj)  # type: ignore[arg-type]
