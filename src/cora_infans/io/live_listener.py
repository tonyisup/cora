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

import hashlib
import inspect
import math
import subprocess
import tempfile
import time
from pathlib import Path

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
from ..runtime_metrics import RuntimeLatencyTracker
from ..resource_metrics import ProcessResourceTracker
from .active_attention import (
    ActiveAttentionGovernor,
    AttentionConfig,
    derive_target_azimuth,
)
from .hardware_tuner import HardwareTuner, HardwareTunerError, DoAReading
from .motor_plan import (
    MotorPlanner,
    VowelAnalysis,
    analyze_vowel,
    synthesize_vowel,
    write_wav,
)
from ..live_speak_smoke import (
    _capture_during_playback,
    _read_wav_mono_float32,
    capture_to_wav,
    play_wav_to_device,
)
from .self_suppression import (  # Phase 2: real-data forward-model fitting / subtraction
    compute_svsr_db,
    fit_forward_model,
    predict_self_pcm,
    rms,
    spectral_subtract,
)

# External Headphones output device index (audiotoolbox), per host enumeration.
# This is the WORLD / test-output sink (stimulus playback, noise sim, etc.).
STIMULUS_OUTPUT_DEVICE_INDEX = 2
# Mac mini Speakers output device index (audiotoolbox), per host enumeration.
# This is the IMITATION efference sink: Cora's own re-synthesized vocalization
# plays here, kept separate from the world/test device so the corollary feedback
# path is audibly distinct and the user hears the imitation on the main speakers.
IMITATION_OUTPUT_DEVICE_INDEX = 3
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
        volume_db: float = 12.0,
    ) -> None:
        self.output_device_index = output_device_index
        self.sample_rate = sample_rate
        self.volume_db = float(volume_db)

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
            "-af", f"volume={self.volume_db:g}dB",
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
        output_device_index: int = STIMULUS_OUTPUT_DEVICE_INDEX,
        imitation_device_index: int = IMITATION_OUTPUT_DEVICE_INDEX,
    ) -> None:
        self.array = array or LiveMicArray(device=DEFAULT_DEVICE)
        self.saliency_threshold = saliency_threshold
        self.initial_prediction_hz = initial_prediction_hz
        self.workspace_window_ms = workspace_window_ms
        # World/test stimulus sink (External Headphones by default).
        self.stimulus = stimulus or ParenteseStimulus(output_device_index=output_device_index)
        self._pipeline: CoraInfansPipeline | None = None
        # Step 1: perception->production bridge. Off by default; enabled per-run
        # via listen(with_motor_imitation=True). No live playback happens here
        # (to avoid acoustic feedback); matches are logged as internal trials.
        self._enable_motor_imitation = enable_motor_imitation
        self._planner = MotorPlanner()
        # ActiveAttentionGovernor mediates between software saliency state and
        # hardware execution; resolved to a no-op tuner under tests.
        self.governor = governor or ActiveAttentionGovernor()
        # Imitation efference sink (Mac mini Speakers by default). Kept separate
        # from output_device_index so the world/test stimulus and Cora's own
        # played imitation can use different physical outputs.
        self.imitation_device_index = imitation_device_index
        self._tuner = tuner  # None => resolve in listen()

    def play_efference_if_authorized(
        self,
        *,
        wav_path: str,
        gate_fn,
        volume_db: float = 12.0,
        playback_fn=None,
        bounded_playback_authorized: bool = False,
    ) -> dict[str, object]:
        """Play an imitation efference to the speaker only if the readiness gate passes.

        Bounded + gated: zero playback occurs unless the gate verdict is
        ``promote_simulation_only``. This is the same authorization pattern as the
        SPEAK smoke; it never performs looped/free playback. ``playback_fn`` defaults
        to the real ffmpeg path and may be injected for tests.
        """
        gate = gate_fn()
        if gate.get("verdict") != "promote_simulation_only":
            return {"played": False, "reason": "gate_not_promoted", "gate": gate}
        if not bounded_playback_authorized:
            return {
                "played": False,
                "reason": "bounded_live_io_not_authorized",
                "gate": gate,
            }
        player = playback_fn or play_wav_to_device
        result = player(
            wav_path,
            device_index=self.imitation_device_index,
            sample_rate=48000,
            volume_db=volume_db,
        )
        return {"played": True, "reason": "authorized", "playback": result}

    def capture_corollary_window(
        self,
        *,
        out_dir: str,
        capture_fn=None,
        latency_ms: float | None = None,
        capture_ms: int = 400,
        input_device: str = DEFAULT_DEVICE,
        sample_rate: int = 48000,
    ) -> dict[str, object]:
        """Capture the corollary-feedback window after an imitation playback.

        The played efference re-enters the mic; this captures that window so it
        can be tagged and (later) subtracted. Phase 1 only captures + records the
        measured latency anchor; no signal subtraction happens here.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        captured_wav = str(out / "corollary_capture.wav")
        fn = capture_fn or capture_to_wav
        result = fn(captured_wav, device=input_device, sample_rate=sample_rate, duration_ms=capture_ms)
        return {
            "captured": True,
            "captured_wav": captured_wav,
            "latency_aligned_ms": latency_ms,
            "capture": result,
        }

    def corollary_frame_tags(
        self,
        *,
        latency_ms: float | None,
        capture_ms: int = 400,
        playback_offset_ms: float = 0.0,
        efference_duration_ms: int = 300,
    ) -> list[dict[str, object]]:
        """Tag the captured corollary window against the measured latency anchor.

        Reuses the same frame-tag schema as the SPEAK smoke (READY_TO_SPEAK /
        SPEAKING / POST_SPEAK_SUPPRESSION / RECOVERING) so a downstream consumer
        can tell which captured frames are Cora's own played audio (self) versus
        any continuing human speech (world). Anchored at the known latency.
        """
        from ..live_speak_smoke import _frame_tags

        tags = _frame_tags(
            efference_id="corollary",
            duration_ms=min(capture_ms, efference_duration_ms),
            max_capture_ms=capture_ms,
            latency_aligned_ms=latency_ms,
            playback_offset_ms=playback_offset_ms,
        )
        for tag in tags:
            tag["source"] = "corollary_feedback"
        return tags

    @staticmethod
    def _mono_downmix(pcm: np.ndarray) -> np.ndarray:
        """Downmix raw WAV PCM to mono float32 in [-1, 1].

        Accepts int16 or float input; any
        integer samples are scaled by 1/32768 so the captured ``observed`` shares
        the float scale of the efference/predicted signals used for self-
        subtraction. This scaling is load-bearing: the forward-model fit and SVSR
        are meaningless if observed is int16-scale while predicted is float-scale.
        """
        arr = np.asarray(pcm)
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 32768.0
        else:
            arr = arr.astype(np.float32)
        if arr.ndim == 2:
            return (arr[:, 0] + arr[:, 1]) / 2.0
        return arr

    def _resynth_efference_pcm(self, plan, *, sample_rate: int) -> np.ndarray:
        """Re-synthesize the motor plan at an arbitrary rate for subtraction.

        Phase 1 analyzed + played via a 16 kHz WAV; the corollary capture is at
        the 48 kHz hardware rate, so the efference used for self-prediction must
        match that rate to align samples with the captured mixture.
        """
        return self._planner.plan_wav(
            plan.vowel, duration_ms=200, sample_rate=sample_rate
        )

    def _fit_self_forward_model(
        self,
        *,
        efference_pcm: np.ndarray,
        observed_pcm: np.ndarray,
        sample_rate: int,
        latency_anchor_ms: float | None,
    ) -> dict[str, object]:
        """Fit the toy forward model to a real efference->observation pair.

        Phase 2 (measurement-only in Task 1): grid-search delay around the
        measured anchor, plus attenuation/echo grids, to find params that
        minimize self-prediction residual. Returns the learned params + SVSR.
        No signal is subtracted or routed anywhere by this method.
        """
        grid_center = 0.0 if latency_anchor_ms is None else float(latency_anchor_ms)
        delay_grid_ms = tuple(round(grid_center + d, 3) for d in (-2.0, 0.0, 2.0, 4.0))
        return fit_forward_model(
            efference_pcm,
            observed_pcm,
            sample_rate=sample_rate,
            delay_grid_ms=delay_grid_ms,
            attenuation_grid=(0.3, 0.45, 0.55, 0.65, 0.8),
            echo_decay_grid=(0.05, 0.12, 0.22, 0.35),
            svsr_pass_db=20.0,
        )

    @staticmethod
    def _real_self_subtraction_promotion_gate(
        *,
        svsr_db: float,
        world_residual_correlation: float | None,
        svsr_pass_db: float = 20.0,
        min_world_corr: float = 0.05,
        stable_calibration: bool = False,
        synchronized_world_reference: bool = False,
        held_out_trial_count: int = 0,
    ) -> dict[str, object]:
        """Real-data go/no-go for promoting tag_only -> self_subtract_live.

        SEPARATE from ``run_forward_model_promotion_gate`` (which is
        simulation-only and forces ``live_playback_allowed=False``). This gate
        judges whether the *actual captured* self-subtraction preserved the
        world signal while suppressing self: require SVSR above threshold AND a
        non-negative world-residual correlation (world energy survived).

        The three session/capture-level criteria are evidence-backed, not
        constants -- callers must feed the real values:
          - ``stable_calibration``: a synchronized loopback latency anchor was
            loaded for the session (consumer passes it from the calibration
            artifact's ``synchronized_capture`` flag).
          - ``synchronized_world_reference``: the corollary window was genuinely
            captured around playback with a verified overlap and a stream
            byte-count position source (set by the motor-imitation loop from the
            capture I/O timing, never defaulted True).
          - ``held_out_trial_count``: the operator committed to this many
            independent held-out trials (``>=3`` required).
        """
        criteria = {
            "svsr_above_threshold": {
                "passed": float(svsr_db) >= svsr_pass_db,
                "value": float(svsr_db),
                "threshold": svsr_pass_db,
            },
            "world_signal_preserved": {
                "passed": world_residual_correlation is not None
                and float(world_residual_correlation) >= min_world_corr,
                "value": world_residual_correlation,
                "threshold": min_world_corr,
            },
            "stable_calibration": {
                "passed": stable_calibration,
                "value": stable_calibration,
                "threshold": True,
            },
            "synchronized_world_reference": {
                "passed": synchronized_world_reference,
                "value": synchronized_world_reference,
                "threshold": True,
            },
            "held_out_trials": {
                "passed": held_out_trial_count >= 3,
                "value": held_out_trial_count,
                "threshold": 3,
            },
        }
        passed = all(bool(item["passed"]) for item in criteria.values())
        return {
            "verdict": "promote" if passed else "hold",
            "criteria": criteria,
            "policy_if_promoted": "self_subtract_live",
            "policy_if_held": "tag_only",
        }

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
        *,
        pcm_sample_rate: int = 16000,
        play: bool = False,
        gate_fn=None,
        playback_fn=None,
        capture_fn=None,
        out_dir: str | None = None,
        volume_db: float = 12.0,
        latency_anchor_ms: float | None = None,
        do_self_subtraction: bool = False,
        bounded_playback_authorized: bool = False,
        capture_preroll_ms: int = 100,
        stable_calibration: bool = False,
        held_out_trial_count: int = 0,
    ) -> "dict[str, object] | None":
        """Run one perception->production trial on a captured speech frame.

        Returns a trial record, or None if there is no analyzable waveform.
        When ``play`` is set AND the readiness gate passes, the re-synthesized
        motor plan (the imitation efference) is written to a WAV and played to
        the speaker through the gated helper. This closes the loop from heard
        sound -> internal motor plan -> played imitation, bounded and gated.

        When ``capture_fn`` is supplied (real-io), the corollary-feedback window
        spans playback; the real efference (at the capture rate) and
        the captured mixture drive a forward-model fit. With ``do_self_subtraction``
        the fit is applied to subtract Cora's own audio (Phase 2); otherwise it is
        recorded as measurement-only evidence (Task 1).
        """
        if pcm is None or len(pcm) < max(1, pcm_sample_rate // 10):
            return None
        try:
            feat = analyze_vowel(pcm, sample_rate=pcm_sample_rate)
        except ValueError:
            return None
        # The perceived vowel selects the table-anchored motor target; the
        # planner re-synthesizes that target (the "internal vocalization").
        plan = self._planner.plan_from_analysis(feat)
        resynth = self._planner.plan_wav(
            plan.vowel, duration_ms=200, sample_rate=48000
        )
        trial: dict[str, object] = {
            "frame": frame_idx,
            "now_ms": now_ms,
            "timestamp_ms": now_ms,
            "source_pcm_sample_rate": pcm_sample_rate,
            "source_azimuth_deg": round(source_azimuth_deg, 2),
            "perceived_formants_hz": [round(f, 2) for f in feat.formant_hz],
            "perceived_nearest_vowel": feat.nearest_vowel,
            "motor_plan_vowel": plan.vowel,
            "motor_plan_f1_hz": plan.f1_hz,
            "motor_plan_formants_hz": [plan.f1_hz, plan.f2_hz, plan.f3_hz],
            "resynth_samples": len(resynth),
        }
        efference_wav: str | None = None
        emitted_pcm = np.asarray(resynth, dtype=np.float32)
        played = False
        cap: dict[str, object] | None = None
        if play and out_dir is not None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            efference_wav = str(out / f"imit_{frame_idx}.wav")
            digital_gain = 10.0 ** (volume_db / 20.0)
            write_wav(
                efference_wav,
                np.clip(emitted_pcm * digital_gain, -1.0, 1.0),
                sample_rate=48000,
            )
            emitted_pcm, emitted_sample_rate = _read_wav_mono_float32(efference_wav)
            if emitted_sample_rate != 48000:
                raise ValueError(
                    f"efference sample rate mismatch: expected 48000, got {emitted_sample_rate}"
                )
            trial["efference_wav"] = efference_wav
            efference_duration_ms = int(round(len(emitted_pcm) / 48.0))
            if playback_fn is not None:
                gate = gate_fn() if gate_fn is not None else {}
                if gate.get("verdict") != "promote_simulation_only":
                    result = {"played": False, "reason": "gate_not_promoted", "gate": gate}
                elif not bounded_playback_authorized:
                    result = {
                        "played": False,
                        "reason": "bounded_live_io_not_authorized",
                        "gate": gate,
                    }
                elif capture_fn is not None:
                    captured_wav = str(out / f"corollary_capture_f{frame_idx}.wav")

                    def _capture(
                        path: str,
                        *,
                        sample_rate: int,
                        duration_ms: int,
                        ready_event,
                        cancel_event,
                        capture_timeline,
                    ):
                        kwargs: dict[str, object] = {
                            "device": DEFAULT_DEVICE,
                            "sample_rate": sample_rate,
                            "duration_ms": duration_ms,
                            "ready_event": ready_event,
                        }
                        parameters = list(inspect.signature(capture_fn).parameters.values())
                        if any(
                            parameter.name == "capture_timeline"
                            or parameter.kind is inspect.Parameter.VAR_KEYWORD
                            for parameter in parameters
                        ):
                            kwargs["capture_timeline"] = capture_timeline
                        if any(
                            parameter.name == "cancel_event"
                            or parameter.kind is inspect.Parameter.VAR_KEYWORD
                            for parameter in parameters
                        ):
                            kwargs["cancel_event"] = cancel_event
                        return capture_fn(path, **kwargs)

                    def _playback(path: str):
                        return playback_fn(
                            path,
                            device_index=self.imitation_device_index,
                            sample_rate=48000,
                            volume_db=0.0,
                        )

                    capture_duration_ms = max(
                        400, capture_preroll_ms + efference_duration_ms + 150
                    )
                    io_result = _capture_during_playback(
                        capture_path=captured_wav,
                        playback_path=efference_wav,
                        sample_rate=48000,
                        capture_duration_ms=capture_duration_ms,
                        capture_fn=_capture,
                        playback_fn=_playback,
                        capture_preroll_ms=capture_preroll_ms,
                    )
                    playback_offset = io_result["playback_offset_from_capture_ms"]
                    if not isinstance(playback_offset, (int, float)):
                        raise TypeError("playback offset must be numeric")
                    result = {
                        "played": True,
                        "reason": "authorized",
                        "playback": io_result["playback"],
                        "io_timing": {
                            "capture_start_ms": io_result["capture_start_ms"],
                            "capture_end_ms": io_result["capture_end_ms"],
                            "playback_onset_ms": io_result["playback_onset_ms"],
                            "playback_end_ms": io_result["playback_end_ms"],
                            "playback_offset_samples": io_result[
                                "playback_offset_samples"
                            ],
                            "playback_offset_from_capture_ms": io_result[
                                "playback_offset_from_capture_ms"
                            ],
                            "overlap_verified": io_result["overlap_verified"],
                            "stream_position_verified": io_result[
                                "stream_position_verified"
                            ],
                            "timestamp_source": io_result["timestamp_source"],
                        },
                    }
                    cap = {
                        "path": captured_wav,
                        "captured_wav": captured_wav,
                        "capture": io_result["capture"],
                        "frame_tags": self.corollary_frame_tags(
                            latency_ms=latency_anchor_ms,
                            capture_ms=capture_duration_ms,
                            playback_offset_ms=float(playback_offset),
                            efference_duration_ms=efference_duration_ms,
                        ),
                        "io_timing": result["io_timing"],
                    }
                else:
                    result = self.play_efference_if_authorized(
                        wav_path=efference_wav,
                        gate_fn=lambda: gate,
                        volume_db=0.0,
                        playback_fn=playback_fn,
                        bounded_playback_authorized=True,
                    )
                played = bool(result.get("played", False))
                trial["playback"] = result
            # Replace the imitation TARGET with the sound that actually left the
            # speaker: re-analyze the played WAV (not the internal plan) and
            # record the prediction error vs the perceived human vowel.
            try:
                played_feat = analyze_vowel(emitted_pcm, sample_rate=48000)
                trial["played_formants_hz"] = [round(f, 2) for f in played_feat.formant_hz]
                trial["played_nearest_vowel"] = played_feat.nearest_vowel
                trial["imitation_prediction_error_hz"] = [
                    round(abs(p - w), 2)
                    for p, w in zip(played_feat.formant_hz, feat.formant_hz)
                ]
            except Exception:
                # Analysis of the played artifact is best-effort; never fatal.
                trial["played_formants_hz"] = None
            # Phase 2: use the corollary window captured around playback to fit the
            # forward model on the real efference + observed mixture. Task 1 is
            # MEASUREMENT-ONLY: record the fit; subtraction is added in Task 2.
            if cap is not None:
                trial["corollary_capture"] = cap
                try:
                    observed, observed_sample_rate = _read_wav_mono_float32(
                        str(cap["captured_wav"])
                    )
                    if observed_sample_rate != 48000:
                        raise ValueError(
                            f"capture sample rate mismatch: expected 48000, got {observed_sample_rate}"
                        )
                    timing = cap.get("io_timing")
                    if not isinstance(timing, dict):
                        raise ValueError("capture timing metadata is missing")
                    playback_offset = timing.get("playback_offset_from_capture_ms")
                    if not isinstance(playback_offset, (int, float)):
                        raise ValueError("capture playback offset is missing")
                    playback_offset_samples = timing.get("playback_offset_samples")
                    if not isinstance(playback_offset_samples, int):
                        raise ValueError("capture playback frame offset is missing")
                    onset_samples = max(0, playback_offset_samples)
                    observed = observed[onset_samples:]
                    if observed.size == 0:
                        raise ValueError("capture contains no samples after playback onset")
                    trial["capture_alignment"] = {
                        "playback_offset_ms": round(float(playback_offset), 3),
                        "discarded_preroll_samples": onset_samples,
                    }
                    eff_48k = emitted_pcm
                    fit = self._fit_self_forward_model(
                        efference_pcm=eff_48k,
                        observed_pcm=observed,
                        sample_rate=48000,
                        latency_anchor_ms=latency_anchor_ms,
                    )
                    learned = fit.get("learned_params") or {}
                    trial["forward_model_fit"] = {
                        "learned_params": learned,
                        "svsr_db": fit.get("svsr_db"),
                        "residual_rms": fit.get("residual_rms"),
                        "observed_rms": fit.get("observed_rms"),
                        "verdict": fit.get("verdict"),
                    }
                    # Task 2: apply the fitted forward model as a real
                    # self-subtraction, producing the world_residual (the human's
                    # continuing speech, isolated from Cora's own played audio).
                    # Honesty: tag_only still holds at the policy level until the
                    # real-data promotion gate (Task 3) passes; here we only
                    # COMPUTE + record the residual as evidence.
                    if do_self_subtraction and learned:
                        predicted = predict_self_pcm(
                            eff_48k,
                            sample_rate=48000,
                            delay_ms=float(learned.get("delay_ms", 0.0)),
                            attenuation=float(learned.get("attenuation", 0.65)),
                            echo_decay=float(learned.get("echo_decay", 0.18)),
                        )
                        residual = spectral_subtract(observed, predicted)
                        svsr = compute_svsr_db(observed, residual)
                        # World proxy: the perceived human frame (resampled to
                        # 48k) is an imperfect proxy for the true world signal;
                        # correlation here is a sanity check only. Align on the
                        # shared length (no zero-pad bias) before correlating.
                        world_proxy = self._mono_downmix(np.asarray(pcm))
                        if pcm_sample_rate != 48000:
                            target_n = max(
                                1, int(round(len(world_proxy) * 48000 / pcm_sample_rate))
                            )
                            world_proxy = np.interp(
                                np.linspace(0.0, 1.0, target_n, endpoint=False),
                                np.linspace(0.0, 1.0, len(world_proxy), endpoint=False),
                                world_proxy,
                            ).astype(np.float32)
                        m = min(len(world_proxy), len(residual))
                        world_proxy_m = world_proxy[:m]
                        residual_m = residual[:m]
                        from .self_suppression import _corr

                        corr = _corr(world_proxy_m, residual_m)
                        trial["world_residual"] = {
                            "svsr_db": round(float(svsr), 3),
                            "observed_rms": round(float(rms(observed)), 9),
                            "residual_rms": round(float(rms(residual)), 9),
                            "predicted_rms": round(float(rms(predicted)), 9),
                            "world_residual_correlation": (
                                round(float(corr), 4) if corr is not None else None
                            ),
                            "applied": True,
                        }
                        # Real-data promotion gate (Task 3): decide whether this
                        # subtraction is good enough to flip the honesty policy
                        # from tag_only to self_subtract_live. Record the gate so
                        # downstream consumers see the evidence + verdict. The
                        # three session/scapture-level criteria are fed the real
                        # values the loop actually has, not constants:
                        #  - stable_calibration: a synchronized latency anchor was
                        #    loaded for this session (else promotion is barred);
                        #  - synchronized_world_reference: the corollary window was
                        #    genuinely captured around playback with a verified
                        #    overlap and a stream-byte-count position source;
                        #  - held_out_trial_count: the operator committed to this
                        #    many independent held-out trials (>=3 required).
                        cap_timing = cap.get("io_timing") if isinstance(cap, dict) else None
                        synchronized_world_reference = bool(
                            isinstance(cap_timing, dict)
                            and cap_timing.get("overlap_verified")
                            and cap_timing.get("stream_position_verified")
                        )
                        gate = LiveListener._real_self_subtraction_promotion_gate(
                            svsr_db=float(svsr),
                            world_residual_correlation=trial["world_residual"][
                                "world_residual_correlation"
                            ],
                            stable_calibration=stable_calibration,
                            synchronized_world_reference=synchronized_world_reference,
                            held_out_trial_count=held_out_trial_count,
                        )
                        trial["self_subtraction_gate"] = gate
                        trial["contamination_policy"] = (
                            gate["policy_if_promoted"]
                            if gate["verdict"] == "promote"
                            else gate["policy_if_held"]
                        )
                except Exception as exc:  # noqa: BLE001
                    # Playback remains a completed physical fact, but processing
                    # failures must remain auditable evidence rather than vanish.
                    trial["processing_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "stage": "forward_model_fit_or_subtraction",
                    }
                    trial["forward_model_fit"] = trial.get("forward_model_fit")
                    trial["world_residual"] = None
        trial["played"] = played
        return trial

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
        stimulus_wav_path: str | None = None,
        with_scheduling: bool = True,
        with_attention: bool = False,
        with_motor_imitation: bool = False,
        theta_ms: int = 200,
        gamma_ms: int = 25,
        capacity_cycles: int = 4,
        motor_imitation_gate_fn=None,
        motor_imitation_playback_fn=None,
        motor_imitation_capture_fn=None,
        motor_imitation_out_dir: str | None = None,
        motor_imitation_volume_db: float = 12.0,
        latency_anchor_ms: float | None = None,
        do_self_subtraction: bool = False,
        bounded_playback_authorized: bool = False,
        motor_imitation_capture_preroll_ms: int = 100,
        self_subtraction_stable_calibration: bool = False,
        self_subtraction_held_out_trials: int = 0,
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
        temporary_stimulus = False
        stimulus_wav_sha256 = None
        if with_stimulus:
            if stimulus_wav_path is None:
                stim_path = tempfile.mktemp(suffix=".wav")
                temporary_stimulus = True
                self.stimulus.synthesize_wav(stim_path)
            else:
                source_path = Path(stimulus_wav_path).resolve()
                if not source_path.is_file():
                    raise ValueError(f"stimulus WAV does not exist: {source_path}")
                stim_path = str(source_path)
                stimulus_wav_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            stim_proc = self.stimulus.play(stim_path)

        frames: list[dict[str, object]] = []
        events: list[dict[str, object]] = []
        frame_is_salient: list[bool] = []
        attention_log: list[dict[str, object]] = []
        motor_imitation_log: list[dict[str, object]] = []
        latency_tracker = RuntimeLatencyTracker(deadline_ms=frame_ms)
        resource_tracker = ProcessResourceTracker()
        target_frames = max(1, duration_ms // frame_ms)
        for i in range(target_frames):
            now_ms = i * frame_ms
            capture_started_ns = time.perf_counter_ns()
            frame = self.array.capture_frame(duration_ms=frame_ms)
            capture_ended_ns = time.perf_counter_ns()
            event = pipeline.process_frame(frame)
            decision_ended_ns = time.perf_counter_ns()
            device_timing_fn = getattr(
                self.array, "device_frame_latency_at_decision", None
            )
            device_timing = (
                device_timing_fn() if callable(device_timing_fn) else None
            )
            latency_tracker.record(
                capture_ms=(capture_ended_ns - capture_started_ns) / 1_000_000.0,
                pipeline_ms=(decision_ended_ns - capture_ended_ns) / 1_000_000.0,
                total_ms=(decision_ended_ns - capture_started_ns) / 1_000_000.0,
                adc_onset_to_decision_ms=(
                    device_timing["adc_frame_onset_to_workspace_decision_ms"]
                    if device_timing is not None
                    else None
                ),
                adc_end_to_decision_ms=(
                    device_timing["adc_frame_end_to_workspace_decision_ms"]
                    if device_timing is not None
                    else None
                ),
                workload_label="salient" if event is not None else "quiet",
            )
            is_salient = event is not None
            frame_is_salient.append(is_salient)
            frame_view = asdict_public(frame)
            frame_view["stimulus_active"] = with_stimulus
            frame_view["capture_peak_absolute"] = float(
                getattr(self.array, "last_frame_peak_absolute", 0.0)
            )
            frame_view["clipped_sample_count"] = int(
                getattr(self.array, "last_frame_clipped_sample_count", 0)
            )
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
                            pcm,
                            source_azimuth,
                            frame.timestamp_ms,
                            i,
                            pcm_sample_rate=int(
                                getattr(self.array, "last_pcm_sample_rate", 16000)
                            ),
                            play=motor_imitation_playback_fn is not None,
                            gate_fn=motor_imitation_gate_fn,
                            playback_fn=motor_imitation_playback_fn,
                            capture_fn=motor_imitation_capture_fn,
                            out_dir=motor_imitation_out_dir,
                            volume_db=motor_imitation_volume_db,
                            latency_anchor_ms=latency_anchor_ms,
                            do_self_subtraction=do_self_subtraction,
                            bounded_playback_authorized=bounded_playback_authorized,
                            capture_preroll_ms=motor_imitation_capture_preroll_ms,
                            stable_calibration=self_subtraction_stable_calibration,
                            held_out_trial_count=self_subtraction_held_out_trials,
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
                                motor_ts = scheduler.reserve_motor_timestamp(frame.timestamp_ms)
                                motor_event = WorkspaceEvent(
                                    kind="motor_intention",
                                    saliency_score=float(
                                        trial.get("resynth_samples", 0)
                                    )
                                    / 1000.0,
                                    workspace_state={
                                        "timestamp_ms": motor_ts,
                                        "perceived_vowel": str(
                                            trial.get("perceived_nearest_vowel", "")
                                        ),
                                        "motor_target_vowel": str(
                                            trial.get("motor_plan_vowel", "")
                                        ),
                                        "motor_target_formants_hz": trial.get(
                                            "motor_plan_formants_hz", []
                                        ),
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
            if temporary_stimulus and stim_path is not None:
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
            "stimulus": (
                {
                    "output_device_index": self.stimulus.output_device_index,
                    "sample_rate_hz": self.stimulus.sample_rate,
                    "volume_db": self.stimulus.volume_db,
                    "kind": (
                        "external_wav" if stimulus_wav_path is not None else "synthetic_contour"
                    ),
                    "source_wav_path": (
                        str(Path(stimulus_wav_path).resolve())
                        if stimulus_wav_path is not None
                        else None
                    ),
                    "source_wav_sha256": stimulus_wav_sha256,
                }
                if with_stimulus
                else None
            ),
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
            "runtime_performance": latency_tracker.summary(
                capture_backend=(
                    getattr(self.array, "active_stream_backend", None)
                    or type(self.array).__name__
                ),
                real_capture=bool(getattr(self.array, "is_real_capture", False)),
                backend_reported_input_latency_ms=(
                    getattr(self.array, "stream_metadata", {}) or {}
                ).get("reported_input_latency_ms"),
                input_overflow_count=int(
                    getattr(self.array, "input_overflow_count", 0)
                ),
            ),
            "runtime_resources": resource_tracker.summary(),
            "capture_stream_fallback_reason": getattr(
                self.array, "stream_fallback_reason", None
            ),
            "capture_stream_metadata": getattr(self.array, "stream_metadata", None),
            "capture_callback_status_count": int(
                getattr(self.array, "callback_status_count", 0)
            ),
            "capture_callback_timestamp_discontinuity_count": int(
                getattr(self.array, "callback_timestamp_discontinuity_count", 0)
            ),
            "capture_quality": {
                "peak_absolute": float(
                    getattr(self.array, "capture_peak_absolute", 0.0)
                ),
                "captured_sample_count": int(
                    getattr(self.array, "captured_sample_count", 0)
                ),
                "clipped_sample_count": int(
                    getattr(self.array, "clipped_sample_count", 0)
                ),
                "clipped_sample_fraction": (
                    int(getattr(self.array, "clipped_sample_count", 0))
                    / int(getattr(self.array, "captured_sample_count", 0))
                    if int(getattr(self.array, "captured_sample_count", 0)) > 0
                    else 0.0
                ),
            },
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
