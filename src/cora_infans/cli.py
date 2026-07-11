from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from .io.live_listener import LiveListener
from .benchmark import benchmark_event_pipeline
from .io.live_mic import DEFAULT_DEVICE, LiveMicArray, LiveMicError
from .io.ids_corpus import build_ids_manifest
from .io.recorded_frames import (
    build_corollary_capture_manifest,
    build_recorded_npy_frames,
)
from .io.offline_eval import (
    CorpusLoader,
    OfflineEvalHarness,
    compare_saliency_baselines,
)
from .io.wav_frames import build_wav_corpus_frames
from .simulation import (
    SimulationParams,
    _builtin_sequence_family,
    build_builtin_sequence_report,
    build_live_frame_report,
    build_sequence_report,
    build_sequence_report_with_scheduling,
    build_simulation_report,
)
from .predictive_learning import (
    evaluate_online_predictor_safety,
    evaluate_predictive_corpus,
)
from .register_saliency import (
    SpeakerAdaptiveRegisterModel,
    VocalActivityGate,
    build_register_example,
    evaluate_leave_one_fieldsite_out,
    evaluate_register_pipeline,
    select_register_thresholds,
)
from .workspace_ablation import run_workspace_ablation
from .io.motor_plan import (  # noqa: E402
    ANALYSIS_BY_SYNTHESIS_TABLE,
    VOWEL_FORMANTS_HZ,
    MotorPlanner,
    analyze_vowel,
    evaluate_vowel_round_trip,
    synthesize_vowel,
    write_wav,
)
from .io.self_suppression import (
    SelfSuppressionParams,
    run_forward_model_learning_trial,
    run_forward_model_promotion_gate,
    run_forward_model_robustness_sweep,
    run_self_suppression_trial,
)
from .speaking_simulator import (
    run_speaking_policy_promotion_gate,
    run_speaking_readiness_gate,
)
from .structured_prediction import (
    CALIBRATED_V2_DEVELOPMENT_SEEDS,
    CALIBRATED_V2_HELD_OUT_SEEDS,
    CALIBRATED_V2_STD_FLOOR,
    DROPOUT_V4_DEVELOPMENT_SEEDS,
    DROPOUT_V4_HELD_OUT_SEEDS,
    DROPOUT_V5_DEVELOPMENT_SEEDS,
    DROPOUT_V5_HELD_OUT_SEEDS,
    ROBUST_V3_DEVELOPMENT_SEEDS,
    ROBUST_V3_HELD_OUT_SEEDS,
    build_robust_v3_trial_conditions,
    build_dropout_v4_trial_conditions,
    build_dropout_v5_trial_conditions,
    run_structured_prediction_protocol,
)
from .natural_prediction_audit import audit_natural_prediction_grid
from .natural_event_prediction import (
    NaturalEventConfig,
    audit_natural_event_grid,
    evaluate_natural_event_ablations,
    evaluate_natural_event_cross_fieldsite,
)
from .real_workspace_binding import evaluate_real_workspace_binding_design
from .recorded_vowel_audit import evaluate_recorded_vowels
from .structured_workspace_binding import (
    WorkspaceBindingConfig,
    build_workspace_binding_protocol,
    run_fake_workspace_binding_protocol,
    run_workspace_binding_acquisition,
)
from .live_speak_smoke import (
    StimulusSpec,
    capture_to_wav,
    play_wav_to_device,
    run_live_speak_smoke_trial,
    run_real_io_speak_trial,
    run_real_io_speak_trials,
)


OutputMode = str
SequenceArgs = tuple[str, bool]


def describe() -> dict[str, object]:
    return {
        "project": "cora-infans",
        "implemented_slice": "evidence_first_auditory_learning",
        "implemented_subsystems": [
            "synthetic_mic_array_front_end",
            "live_xvf3800_front_end",
            "native_coreaudio_continuous_front_end",
            "acoustic_saliency_gate",
            "predictive_hierarchy",
            "stateful_event_workspace",
            "theta_gamma_scheduler",
            "xvf3800_hardware_tuner",
            "commanded_active_attention",
            "offline_corpus_evaluation",
            "real_wav_corpus_evaluation",
            "recorded_npy_corpus_evaluation",
            "dev_held_out_saliency_baselines",
            "prior_art_capture_replay",
            "motor_plan_synthesis",
            "clean_synthetic_vowel_round_trip_gate",
            "theta_gamma_motor_subwindow",
            "simulation_only_signal_self_suppression",
            "simulation_only_forward_model_learning",
            "simulation_only_forward_model_robustness_sweep",
            "simulation_only_forward_model_promotion_gate",
            "simulation_only_speaking_policy",
            "simulation_only_speaking_policy_promotion_gate",
            "simulation_only_speaking_readiness_gate",
            "bounded_live_speak_smoke",
            "event_pipeline_microbenchmark",
            "live_capture_to_workspace_latency",
            "ids_corpus_manifest_builder",
            "external_ids_real_audio_evaluation",
            "participant_disjoint_parentese_confirmation",
            "cross_fieldsite_parentese_falsification",
            "speaker_and_mode_adaptive_register_gate",
            "long_window_vocal_activity_gate",
            "joint_register_pipeline_falsification",
            "expanded_register_refit_falsification",
            "safe_online_predictor_regret_gate",
            "structured_prediction_fake_harness",
            "structured_prediction_real_runner",
            "synthetic_workspace_ablation",
            "phase_template_top_down_prediction",
        ],

        "planned_subsystems": [
            "non_speech_source_localization",
            "environmental_pass_through_policy",
            "cold_start_parentese_detection",
            "cross_cultural_parentese_robustness",
            "structured_prediction_real_capture_evidence",
            "cross_protocol_predictive_generalization",
            "device_timestamped_latency",
            "energy_benchmark",
            "general_user_authored_sequences",
        ],
        "verdicts": {
            "source_localization": "speech_stimulus_recoverable_non_speech_negative",
            "beam_steering": "verified_commanded_top_down_actuator",
        },
    }


def _parse_channel_energies(raw: str) -> list[float]:
    try:
        return [float(part) for part in raw.split(",") if part]
    except ValueError as exc:
        raise ValueError("invalid channel energies") from exc



def _parse_mic_geometry(raw: str) -> tuple[str, ...]:
    geometry = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not geometry:
        raise ValueError("invalid mic geometry")
    return geometry


def parse_sequence_cli_args(args: list[str]) -> tuple[str, bool]:
    family = "sweep"
    with_scheduling = True  # now the default report path
    i = 1
    while i < len(args):
        option = args[i]
        if option == "--family":
            if i + 1 >= len(args):
                raise ValueError("missing value for --family")
            family = args[i + 1]
            i += 2
        elif option == "--with-scheduling":
            with_scheduling = True
            i += 1
        else:
            raise ValueError(f"unknown option: {option}")
    return family, with_scheduling



def parse_simulation_cli_args(args: list[str]) -> tuple[SimulationParams, OutputMode]:
    params: dict[str, Any] = {
        "timestamp_ms": 125,
        "dominant_frequency_hz": 1200.0,
        "channel_energies": [0.2, 0.9, 0.4, 0.1],
        "initial_prediction_hz": 1000.0,
        "mic_channel_geometry": (
            "channel_0",
            "channel_1",
            "channel_2",
            "channel_3",
        ),
        "saliency_threshold": 0.7,
        "output": "json",
    }
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--timestamp-ms":
            try:
                params["timestamp_ms"] = int(value)
            except ValueError as exc:
                raise ValueError("invalid timestamp") from exc
        elif option == "--dominant-frequency-hz":
            try:
                params["dominant_frequency_hz"] = float(value)
            except ValueError as exc:
                raise ValueError("invalid dominant frequency") from exc
        elif option == "--channel-energies":
            params["channel_energies"] = _parse_channel_energies(value)
        elif option == "--initial-prediction-hz":
            try:
                params["initial_prediction_hz"] = float(value)
            except ValueError as exc:
                raise ValueError("invalid initial prediction") from exc
        elif option == "--mic-geometry":
            params["mic_channel_geometry"] = _parse_mic_geometry(value)
        elif option == "--saliency-threshold":
            try:
                params["saliency_threshold"] = float(value)
            except ValueError as exc:
                raise ValueError("invalid saliency threshold") from exc
        elif option == "--output":
            if value not in {"json", "pretty"}:
                raise ValueError("invalid output mode")
            params["output"] = value
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2

    simulation_params = SimulationParams(
        timestamp_ms=params["timestamp_ms"],
        dominant_frequency_hz=params["dominant_frequency_hz"],
        channel_energies=params["channel_energies"],
        initial_prediction_hz=params["initial_prediction_hz"],
        mic_channel_geometry=params["mic_channel_geometry"],
        saliency_threshold=params["saliency_threshold"],
    )
    return simulation_params, params["output"]


def simulate_frame(args: list[str] | None = None) -> dict[str, object]:
    simulation_params, _output_mode = parse_simulation_cli_args(
        args or ["--simulate-frame"]
    )
    return build_simulation_report(simulation_params)



def simulate_sequence(args: list[str] | None = None) -> dict[str, object]:
    family, with_scheduling = parse_sequence_cli_args(args or ["--simulate-sequence"])
    # Scheduling is part of the default report path; the flag can still request
    # the bare flat-workspace view when explicitly set False via code path.
    report = build_sequence_report(
        _builtin_sequence_family(family), with_scheduling=with_scheduling
    )
    report["family"] = family
    return report


def capture_live(args: list[str] | None = None) -> dict[str, object]:
    parsed = args or ["--capture-live"]
    device = DEFAULT_DEVICE
    duration_ms: int | None = None
    i = 1
    while i < len(parsed):
        option = parsed[i]
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {option}")
        value = parsed[i + 1]
        if option == "--device":
            device = value
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2

    array = LiveMicArray(device=device)
    return build_live_frame_report(array, duration_ms=duration_ms)


def listen_live(args: list[str] | None = None) -> dict[str, object]:
    parsed = args or ["--listen-live"]
    device = DEFAULT_DEVICE
    duration_ms = 2000
    frame_ms = 250
    capture_backend = "auto"
    with_stimulus = False
    with_scheduling = False
    with_attention = False
    with_motor_imitation = False
    use_real_io = False
    output_device_index = 2
    imitation_device_index = 3
    with_self_subtraction = False
    volume_db = 12.0
    from .io.live_mic import DEFAULT_LIVE_SALIENCY_THRESHOLD

    saliency_threshold = DEFAULT_LIVE_SALIENCY_THRESHOLD
    latency_anchor_ms: float | None = None
    latency_anchor_path: str | None = None
    stable_calibration: bool = False
    # Held-out trial count the operator commits to before any self_subtract_live
    # promotion is even considered (>=3 required by the real-data gate). Defaults
    # to 0 so the gate cannot promote unless the operator explicitly commits.
    held_out_trials: int = 0
    i = 1
    while i < len(parsed):
        option = parsed[i]
        if option == "--with-stimulus":
            with_stimulus = True
            i += 1
            continue
        if option == "--with-scheduling":
            with_scheduling = True
            i += 1
            continue
        if option == "--with-attention":
            with_attention = True
            i += 1
            continue
        if option == "--with-motor-imitation":
            with_motor_imitation = True
            i += 1
            continue
        if option == "--real-io":
            use_real_io = True
            i += 1
            continue
        if option == "--with-self-subtraction":
            with_self_subtraction = True
            i += 1
            continue
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {option}")
        value = parsed[i + 1]
        if option == "--device":
            device = value
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--frame-ms":
            try:
                frame_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid frame ms") from exc
        elif option == "--capture-backend":
            if value not in ("auto", "native", "ffmpeg"):
                raise ValueError("invalid capture backend")
            capture_backend = value
        elif option == "--output-device-index":
            try:
                output_device_index = int(value)
            except ValueError as exc:
                raise ValueError("invalid output device index") from exc
        elif option == "--imitation-device-index":
            try:
                imitation_device_index = int(value)
            except ValueError as exc:
                raise ValueError("invalid imitation device index") from exc
        elif option == "--volume-db":
            try:
                volume_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid volume db") from exc
        elif option == "--saliency-threshold":
            try:
                saliency_threshold = float(value)
            except ValueError as exc:
                raise ValueError("invalid saliency threshold") from exc
        elif option == "--latency-anchor":
            latency_anchor_path = value
        elif option == "--held-out-trials":
            try:
                held_out_trials = int(value)
            except ValueError as exc:
                raise ValueError("invalid held-out-trials count") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2

    # Load a measured latency anchor (the calibration.json we can now write) so
    # the corollary-feedback window is tagged against the real loopback delay.
    if latency_anchor_path is not None:
        import json

        with open(latency_anchor_path, encoding="utf-8") as fh:
            calibration = json.load(fh)
        if calibration.get("synchronized_capture") is not True:
            raise ValueError(
                "latency anchor is missing synchronized-capture provenance; recalibrate"
            )
        latency_anchor_ms = float(calibration["latency_ms"])
        stable_calibration = bool(calibration.get("synchronized_capture") is True)

    # Real-io wiring: inject the actual ffmpeg playback/capture fns and consult
    # the speaking readiness gate. Without --real-io, motor imitation stays
    # analysis-only (no playback fns are supplied -> trial does not play).
    gate_fn = None
    playback_fn = None
    capture_fn = None
    if use_real_io and with_motor_imitation:
        from .live_speak_smoke import (
            StimulusSpec,
            capture_to_wav,
            play_wav_to_device,
            run_real_io_speak_trials,
            run_speaking_readiness_gate,
        )

        gate_fn = run_speaking_readiness_gate
        playback_fn = lambda path, **kw: play_wav_to_device(  # type: ignore[misc]
            path, device_index=imitation_device_index, sample_rate=48000, volume_db=volume_db
        )
        capture_fn = lambda path, **kw: capture_to_wav(  # type: ignore[misc]
            path,
            device=device,
            sample_rate=48000,
            duration_ms=kw.get("duration_ms", 400),
            ready_event=kw.get("ready_event"),
            cancel_event=kw.get("cancel_event"),
            capture_timeline=kw.get("capture_timeline"),
        )

        # A latency measurement is specific to the current device/room/session.
        # When no legacy anchor was explicitly supplied, measure it now and keep
        # it in memory rather than requiring a separately generated JSON file.
        if latency_anchor_path is None:
            import tempfile

            calibration = run_real_io_speak_trials(
                out_dir=tempfile.mkdtemp(prefix="cora_calibration_"),
                playback_fn=lambda path: play_wav_to_device(
                    path,
                    device_index=output_device_index,
                    sample_rate=48000,
                    volume_db=24.0,
                ),
                capture_fn=lambda path, **kw: capture_to_wav(
                    path,
                    device=device,
                    sample_rate=kw.get("sample_rate", 48000),
                    duration_ms=kw.get("duration_ms", 1500),
                    ready_event=kw.get("ready_event"),
                    cancel_event=kw.get("cancel_event"),
                    capture_timeline=kw.get("capture_timeline"),
                ),
                trials=6,
                stimulus=StimulusSpec(kind="chirp", f0_hz=200.0, f1_hz=4000.0),
                # Include CoreAudio device-open latency. Repeatability across the
                # six trials, rather than a preselected 4 ms window, rejects noise.
                causal_window_ms=500.0,
                volume_db=24.0,
                max_capture_ms=3000,
                bounded_live_io_authorized=True,
            )
            if (
                calibration.get("latency_stable") is not True
                or calibration.get("synchronized_capture") is not True
                or calibration.get("latency_aligned_ms") is None
            ):
                failures = [
                    str(trial.get("error") or trial.get("verdict"))
                    for trial in calibration.get("per_trial", [])
                    if isinstance(trial, dict) and trial.get("verdict") != "real_io_complete"
                ]
                detail = f" First failure: {failures[0]}" if failures else ""
                raise RuntimeError(
                    "session latency calibration failed: synchronized stable latency was not measured."
                    f" Successful trials: {calibration.get('trials_ok', 0)}/6;"
                    f" stats: {calibration.get('latency_stats')}.{detail}"
                )
            latency_anchor_ms = float(calibration["latency_aligned_ms"])
            stable_calibration = True
    elif with_self_subtraction:
        # Self-subtraction requires real capture to fit+subtract the loopback;
        # it is meaningless without --real-io.
        raise ValueError("--with-self-subtraction requires --real-io")

    out_dir = None
    if use_real_io:
        import tempfile

        out_dir = tempfile.mkdtemp(prefix="cora_imit_")

    array = (
        LiveMicArray(device=device)
        if capture_backend == "auto"
        else LiveMicArray(device=device, stream_backend=capture_backend)
    )
    listener = LiveListener(
        array,
        output_device_index=output_device_index,
        imitation_device_index=imitation_device_index,
        saliency_threshold=saliency_threshold,
    )
    stream_context = (
        array.streaming() if callable(getattr(array, "streaming", None)) else nullcontext()
    )
    with stream_context:
        return listener.listen(
            duration_ms=duration_ms,
            frame_ms=frame_ms,
            with_stimulus=with_stimulus,
            with_scheduling=with_scheduling,
            with_attention=with_attention,
            with_motor_imitation=with_motor_imitation,
            motor_imitation_gate_fn=gate_fn,
            motor_imitation_playback_fn=playback_fn,
            motor_imitation_capture_fn=capture_fn,
            motor_imitation_out_dir=out_dir,
            motor_imitation_volume_db=volume_db,
            latency_anchor_ms=latency_anchor_ms,
            do_self_subtraction=with_self_subtraction,
            bounded_playback_authorized=use_real_io,
            self_subtraction_stable_calibration=stable_calibration,
            self_subtraction_held_out_trials=held_out_trials,
        )


def eval_corpus(args: list[str] | None = None) -> dict[str, object]:
    """Run the offline evaluation harness over a labeled corpus.

    Usage: --eval-corpus --root DIR --label-csv FILE [--saliency-threshold F]
           [--frame-ms N] [--score-mode peak_rms|spatial_contrast]
           [--input-mode wav|recorded_npy|synthetic_fixture] [--compare-baselines]
           [--evaluate-prediction FEATURE]
           [--evaluate-online-safety FEATURE[,FEATURE]]
           [--positive-label LABEL[,LABEL]]

    Real uncompressed WAV decoding is the default and is the only WAV mode
    reported as real evidence. The synthetic fixture mode exists solely for
    plumbing tests.
    """
    parsed = args or ["--eval-corpus"]
    root = "."
    label_csv: str | None = None
    saliency_threshold = 0.02
    frame_ms = 250
    score_mode = "peak_rms"
    input_mode = "wav"
    compare_baselines = False
    predictive_feature: str | None = None
    safety_features: tuple[str, ...] | None = None
    positive_labels: set[str] | None = None
    i = 1
    while i < len(parsed):
        option = parsed[i]
        if option == "--compare-baselines":
            compare_baselines = True
            i += 1
            continue
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {option}")
        value = parsed[i + 1]
        if option == "--root":
            root = value
        elif option == "--label-csv":
            label_csv = value
        elif option == "--saliency-threshold":
            try:
                saliency_threshold = float(value)
            except ValueError as exc:
                raise ValueError("invalid saliency threshold") from exc
        elif option == "--frame-ms":
            try:
                frame_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid frame ms") from exc
        elif option == "--score-mode":
            if value not in ("peak_rms", "spatial_contrast"):
                raise ValueError("invalid score mode")
            score_mode = value
        elif option == "--input-mode":
            if value not in ("wav", "recorded_npy", "synthetic_fixture"):
                raise ValueError("invalid input mode")
            input_mode = value
        elif option == "--evaluate-prediction":
            predictive_feature = value
        elif option == "--evaluate-online-safety":
            parsed_features = tuple(
                feature.strip() for feature in value.split(",") if feature.strip()
            )
            if not parsed_features:
                raise ValueError("online safety feature list must not be empty")
            safety_features = parsed_features
        elif option == "--positive-label":
            parsed_labels = {label.strip() for label in value.split(",") if label.strip()}
            if not parsed_labels:
                raise ValueError("positive label must not be empty")
            positive_labels = parsed_labels
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2

    if label_csv is None:
        raise ValueError("--label-csv is required")

    loader = CorpusLoader(root=root)
    clips = loader.load_labels(label_csv)
    harness = OfflineEvalHarness(
        saliency_threshold=saliency_threshold,
        frame_ms=frame_ms,
        score_mode=score_mode,  # type: ignore[arg-type]
        positive_labels=positive_labels,
    )
    if input_mode == "wav":
        frame_provider = build_wav_corpus_frames
    elif input_mode == "recorded_npy":
        frame_provider = build_recorded_npy_frames
    else:
        from .io.synthetic_stream import build_synthetic_corpus_frames

        frame_provider = build_synthetic_corpus_frames
    selected_evaluations = sum(
        (compare_baselines, predictive_feature is not None, safety_features is not None)
    )
    if selected_evaluations > 1:
        raise ValueError(
            "baseline, predictive, and online-safety evaluations are mutually exclusive"
        )
    if predictive_feature is not None:
        if input_mode not in ("wav", "recorded_npy"):
            raise ValueError("predictive corpus evidence requires real audio input")
        report = evaluate_predictive_corpus(
            clips,
            frame_provider,
            frame_ms=frame_ms,
            feature=predictive_feature,
        )
    elif safety_features is not None:
        if input_mode not in ("wav", "recorded_npy"):
            raise ValueError("online safety evidence requires real audio input")
        report = evaluate_online_predictor_safety(
            clips,
            frame_provider,
            frame_ms=frame_ms,
            features=safety_features,
        )
    elif compare_baselines:
        report = compare_saliency_baselines(
            clips,
            frame_provider,
            frame_ms=frame_ms,
            positive_labels=positive_labels,
        )
    else:
        report = harness.evaluate_corpus(clips, frame_provider)
    report["label_csv"] = label_csv
    report["clip_loaded"] = len(clips)
    report["input_mode"] = input_mode
    return report


def _register_sources(args: list[str]) -> tuple[list[tuple[str, str]], str | None]:
    sources: list[tuple[str, str]] = []
    out: str | None = None
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--source":
            if "::" not in value:
                raise ValueError("--source must be ROOT::LABEL_CSV")
            root, label_csv = value.split("::", 1)
            if not root or not label_csv:
                raise ValueError("--source must be ROOT::LABEL_CSV")
            sources.append((root, label_csv))
        elif option == "--out":
            out = value
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    if not sources:
        raise ValueError("at least one --source ROOT::LABEL_CSV is required")
    return sources, out


def fit_register_scorer_cli(args: list[str]) -> dict[str, object]:
    """Fit the transparent speaker-adaptive register model from spent data."""

    sources, out = _register_sources(args)
    if out is None:
        raise ValueError("--out model JSON is required")
    examples = []
    source_reports: list[dict[str, object]] = []
    for root, label_csv in sources:
        clips = CorpusLoader(root).load_labels(label_csv)
        source_examples = [build_register_example(clip) for clip in clips]
        examples.extend(source_examples)
        source_reports.append(
            {
                "root": str(Path(root).resolve()),
                "label_csv": str(Path(label_csv).resolve()),
                "clip_count": len(source_examples),
                "clip_ids": [example.clip_id for example in source_examples],
            }
        )
    design_report = evaluate_leave_one_fieldsite_out(examples)
    threshold_report = select_register_thresholds(examples)
    model = replace(
        SpeakerAdaptiveRegisterModel.fit(examples),
        speech_score_minimum=float(threshold_report["speech_score_minimum"]),
        ids_score_minimum=float(threshold_report["ids_score_minimum"]),
    )
    payload = {
        "mode": "speaker_adaptive_register_model_artifact",
        "training_sources": source_reports,
        "training_clip_count": len(examples),
        "training_fieldsites": sorted({example.fieldsite for example in examples}),
        "design_evaluation": {
            key: value
            for key, value in design_report.items()
            if key != "predictions"
        },
        "threshold_selection": threshold_report,
        "model": model.to_dict(),
    }
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "mode": "fit_speaker_adaptive_register_model",
        "out": str(output_path.resolve()),
        "training_clip_count": len(examples),
        "training_fieldsites": payload["training_fieldsites"],
        "design_metrics": design_report["metrics"],
        "selected_decision_thresholds": {
            "speech_score_minimum": model.speech_score_minimum,
            "ids_score_minimum": model.ids_score_minimum,
        },
        "calibration_boundary": design_report["calibration_boundary"],
    }


def _load_register_model_artifact(model_path: str | Path) -> SpeakerAdaptiveRegisterModel:
    """Load either a self-contained model or a hash-pinned gate composition."""

    path = Path(model_path)
    payload = json.loads(path.read_text())
    if payload.get("mode") != "register_pipeline_composition":
        model_payload = payload.get("model", payload)
        return SpeakerAdaptiveRegisterModel.from_dict(model_payload)

    base_name = payload.get("base_register_model_artifact")
    expected_sha = payload.get("base_register_model_sha256")
    gate_payload = payload.get("vocal_gate")
    if not isinstance(base_name, str) or not isinstance(expected_sha, str):
        raise ValueError("invalid composed register model reference")
    if not isinstance(gate_payload, dict):
        raise ValueError("invalid composed vocal gate")
    base_path = path.parent / base_name
    base_bytes = base_path.read_bytes()
    if hashlib.sha256(base_bytes).hexdigest() != expected_sha:
        raise ValueError("composed register model hash mismatch")
    base_artifact = json.loads(base_bytes)
    base_payload = base_artifact.get("model", base_artifact)
    base_model = SpeakerAdaptiveRegisterModel.from_dict(base_payload)
    decision_thresholds = payload.get("decision_thresholds")
    thresholds = (
        decision_thresholds if isinstance(decision_thresholds, dict) else {}
    )
    return replace(
        base_model,
        vocal_gate=VocalActivityGate.from_dict(gate_payload),
        speech_score_minimum=float(
            thresholds.get("speech_score_minimum", base_model.speech_score_minimum)
        ),
        ids_score_minimum=float(
            thresholds.get("ids_score_minimum", base_model.ids_score_minimum)
        ),
    )


def eval_register_scorer_cli(args: list[str]) -> dict[str, object]:
    """Evaluate a frozen register model on a batch of unseen speakers."""

    model_path: str | None = None
    root: str | None = None
    label_csv: str | None = None
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--model":
            model_path = value
        elif option == "--root":
            root = value
        elif option == "--label-csv":
            label_csv = value
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    if model_path is None or root is None or label_csv is None:
        raise ValueError("--model, --root, and --label-csv are required")
    model = _load_register_model_artifact(model_path)
    clips = CorpusLoader(root).load_labels(label_csv)
    examples = [build_register_example(clip) for clip in clips]
    report = model.evaluate_batch(examples)
    report["model_path"] = str(Path(model_path).resolve())
    report["root"] = str(Path(root).resolve())
    report["label_csv"] = str(Path(label_csv).resolve())
    return report


def eval_register_pipeline_cli(args: list[str]) -> dict[str, object]:
    """Evaluate vocal retention, noise rejection, and register scoring jointly."""

    values: dict[str, str] = {}
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        if option not in {
            "--model",
            "--vocal-root",
            "--vocal-label-csv",
            "--environmental-root",
            "--environmental-label-csv",
        }:
            raise ValueError(f"unknown option: {option}")
        values[option] = args[i + 1]
        i += 2
    required = {
        "--model",
        "--vocal-root",
        "--vocal-label-csv",
        "--environmental-root",
        "--environmental-label-csv",
    }
    if required - values.keys():
        raise ValueError("model, vocal source, and environmental source are required")
    model = _load_register_model_artifact(values["--model"])
    vocal_clips = CorpusLoader(values["--vocal-root"]).load_labels(
        values["--vocal-label-csv"]
    )
    environmental_clips = CorpusLoader(values["--environmental-root"]).load_labels(
        values["--environmental-label-csv"]
    )
    report = evaluate_register_pipeline(
        model,
        [build_register_example(clip) for clip in vocal_clips],
        [build_register_example(clip) for clip in environmental_clips],
    )
    report["model_path"] = str(Path(values["--model"]).resolve())
    report["vocal_root"] = str(Path(values["--vocal-root"]).resolve())
    report["vocal_label_csv"] = str(
        Path(values["--vocal-label-csv"]).resolve()
    )
    report["environmental_root"] = str(
        Path(values["--environmental-root"]).resolve()
    )
    report["environmental_label_csv"] = str(
        Path(values["--environmental-label-csv"]).resolve()
    )
    return report


def structured_prediction_cli(args: list[str]) -> dict[str, object]:
    """Plan or run the explicitly non-real structured prediction harness."""

    out_dir: str | None = None
    fake_io = False
    real_io = False
    bounded_playback_authorized = False
    input_device = ":0"
    output_device_index = 2
    speaker_distance_cm: float | None = None
    speaker_orientation: str | None = None
    quiet_room_confirmed = False
    variant = "original"
    i = 1
    while i < len(args):
        option = args[i]
        if option == "--fake-io":
            fake_io = True
            i += 1
            continue
        if option == "--real-io":
            real_io = True
            i += 1
            continue
        if option == "--authorize-bounded-playback":
            bounded_playback_authorized = True
            i += 1
            continue
        if option == "--quiet-room-confirmed":
            quiet_room_confirmed = True
            i += 1
            continue
        if option == "--out-dir":
            if i + 1 >= len(args):
                raise ValueError("missing value for --out-dir")
            out_dir = args[i + 1]
            i += 2
            continue
        if option in {
            "--input-device",
            "--output-device-index",
            "--speaker-distance-cm",
            "--speaker-orientation",
            "--variant",
        }:
            if i + 1 >= len(args):
                raise ValueError(f"missing value for {option}")
            value = args[i + 1]
            if option == "--input-device":
                input_device = value
            elif option == "--output-device-index":
                try:
                    output_device_index = int(value)
                except ValueError as exc:
                    raise ValueError("invalid output device index") from exc
            elif option == "--speaker-distance-cm":
                try:
                    speaker_distance_cm = float(value)
                except ValueError as exc:
                    raise ValueError("invalid speaker distance") from exc
            elif option == "--speaker-orientation":
                speaker_orientation = value
            else:
                if value not in {
                    "original",
                    "calibrated_v2",
                    "robustness_v3",
                    "dropout_v4",
                    "dropout_v5",
                }:
                    raise ValueError("unknown structured protocol variant")
                variant = value
            i += 2
            continue
        raise ValueError(f"unknown option: {option}")
    if out_dir is None:
        raise ValueError("--out-dir is required")
    if variant == "dropout_v4" and real_io:
        raise ValueError(
            "dropout_v4 failed its fresh synthetic design gate; real I/O is prohibited"
        )
    geometry: dict[str, object] | None = None
    playback_fn = None
    capture_fn = None
    if real_io and bounded_playback_authorized:
        if speaker_distance_cm is None or speaker_distance_cm <= 0.0:
            raise ValueError("real structured I/O requires --speaker-distance-cm")
        if not speaker_orientation:
            raise ValueError("real structured I/O requires --speaker-orientation")
        if not quiet_room_confirmed:
            raise ValueError("real structured I/O requires --quiet-room-confirmed")
        geometry = {
            "input_device": input_device,
            "output_device_index": output_device_index,
            "speaker_distance_cm": speaker_distance_cm,
            "speaker_orientation": speaker_orientation,
            "quiet_room_confirmed": True,
            "host_channel_count": 2,
        }
        playback_fn = lambda path: play_wav_to_device(
            path,
            device_index=output_device_index,
            sample_rate=16000,
            volume_db=-18.0,
        )
        capture_fn = lambda path, **kwargs: capture_to_wav(
            path,
            device=input_device,
            **kwargs,
        )
    calibrated = variant in {
        "calibrated_v2",
        "robustness_v3",
        "dropout_v4",
        "dropout_v5",
    }
    robust = variant == "robustness_v3"
    dropout_robust = variant == "dropout_v4"
    dropout_masked = variant == "dropout_v5"
    return run_structured_prediction_protocol(
        out_dir=out_dir,
        fake_io=fake_io,
        real_io=real_io,
        bounded_playback_authorized=bounded_playback_authorized,
        playback_fn=playback_fn,
        capture_fn=capture_fn,
        geometry=geometry,
        real_hardware_claim=bool(real_io and bounded_playback_authorized),
        development_seeds=(
            DROPOUT_V5_DEVELOPMENT_SEEDS
            if dropout_masked
            else DROPOUT_V4_DEVELOPMENT_SEEDS if dropout_robust
            else ROBUST_V3_DEVELOPMENT_SEEDS if robust
            else CALIBRATED_V2_DEVELOPMENT_SEEDS if calibrated else None
        ),
        held_out_seeds=(
            DROPOUT_V5_HELD_OUT_SEEDS
            if dropout_masked
            else DROPOUT_V4_HELD_OUT_SEEDS if dropout_robust
            else ROBUST_V3_HELD_OUT_SEEDS if robust
            else CALIBRATED_V2_HELD_OUT_SEEDS if calibrated else None
        ),
        protocol_variant=(variant if calibrated else None),
        calibration_normalization_std_floor=(
            CALIBRATED_V2_STD_FLOOR if calibrated else None
        ),
        acceptance_profile=(
            "h2_condition_robustness_v3"
            if robust or dropout_robust or dropout_masked
            else "h2_predictive_advantage_v2" if calibrated else "original"
        ),
        trial_conditions=(
            build_dropout_v5_trial_conditions()
            if dropout_masked
            else build_dropout_v4_trial_conditions() if dropout_robust
            else build_robust_v3_trial_conditions() if robust else None
        ),
        signature_mode=(
            "efference_masked_mean_30_5"
            if dropout_masked
            else "temporal_median_60_30" if dropout_robust else "full_spectrum"
        ),
        representation_selection_boundary=(
            "selected on spent dropout-v4 synthetic matrix only"
            if dropout_masked
            else "selected on robustness-v3 development captures only" if dropout_robust
            else "selected on attempt-2 development captures only"
        ),
    )


def natural_prediction_design_cli(args: list[str]) -> dict[str, object]:
    """Run the label-blind natural-stream design grid on external WAVs."""

    root: str | None = None
    participants: tuple[str, ...] | None = None
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--root":
            root = value
        elif option == "--participants":
            participants = tuple(
                item.strip() for item in value.split(",") if item.strip()
            )
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    if root is None or participants is None:
        raise ValueError("--root and --participants are required")
    return audit_natural_prediction_grid(root=root, participants=participants)


def natural_event_cli(args: list[str]) -> dict[str, object]:
    """Run the fixed event design audit or frozen cross-fieldsite transfer."""

    values: dict[str, str] = {}
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        values[option] = args[i + 1]
        i += 2
    command = args[0]
    if command == "--evaluate-natural-event-design":
        if {"--root", "--participants"} - values.keys():
            raise ValueError("--root and --participants are required")
        return audit_natural_event_grid(
            root=values["--root"],
            participants=tuple(values["--participants"].split(",")),
        )
    required = {
        "--development-root",
        "--development-participants",
        "--held-out-root",
        "--held-out-participants",
    }
    if required - values.keys():
        raise ValueError("development and held-out roots/participants are required")
    evaluator = (
        evaluate_natural_event_ablations
        if command == "--evaluate-natural-event-ablations"
        else evaluate_natural_event_cross_fieldsite
    )
    evaluation_arguments = {
        "development_root": values["--development-root"],
        "development_participants": tuple(
            values["--development-participants"].split(",")
        ),
        "config": NaturalEventConfig(),
    }
    if command == "--evaluate-natural-event-ablations":
        return evaluator(
            **evaluation_arguments,
            evaluation_root=values["--held-out-root"],
            evaluation_participants=tuple(
                values["--held-out-participants"].split(",")
            ),
        )
    return evaluator(
        **evaluation_arguments,
        held_out_root=values["--held-out-root"],
        held_out_participants=tuple(values["--held-out-participants"].split(",")),
    )


def real_workspace_design_cli(args: list[str]) -> dict[str, object]:
    """Run the spent Corollary H3 binding/interference design audit."""

    if len(args) != 3 or args[1] != "--root":
        raise ValueError("--evaluate-real-workspace-design requires --root DIR")
    return evaluate_real_workspace_binding_design(root=args[2])


def recorded_vowel_cli(args: list[str]) -> dict[str, object]:
    """Evaluate the preregistered held-out VocalSet vowel subset."""

    if len(args) != 5 or args[1] != "--root" or args[3] != "--manifest":
        raise ValueError("--evaluate-recorded-vowels requires --root DIR --manifest FILE")
    return evaluate_recorded_vowels(corpus_root=args[2], manifest_path=args[4])


def structured_workspace_binding_cli(args: list[str]) -> dict[str, object]:
    """Plan, inject, or run the preregistration-gated structured H3 protocol."""

    if args == ["--evaluate-structured-workspace-binding"]:
        return build_workspace_binding_protocol()
    if args == ["--evaluate-structured-workspace-binding", "--fake-io"]:
        return run_fake_workspace_binding_protocol()
    if "--real-io" not in args:
        raise ValueError(
            "structured workspace binding accepts --fake-io or the gated --real-io route"
        )

    values: dict[str, str] = {}
    authorized = False
    index = 1
    while index < len(args):
        option = args[index]
        if option == "--real-io":
            index += 1
            continue
        if option == "--authorize-bounded-playback":
            authorized = True
            index += 1
            continue
        if option in {"--out-dir", "--preregistration"}:
            if index + 1 >= len(args):
                raise ValueError(f"{option} requires a value")
            values[option] = args[index + 1]
            index += 2
            continue
        raise ValueError(f"unknown structured workspace binding option: {option}")
    for required in ("--out-dir", "--preregistration"):
        if required not in values:
            raise ValueError(f"structured workspace real I/O requires {required}")

    preregistration_path = Path(values["--preregistration"])
    preregistration = json.loads(preregistration_path.read_text())
    root = Path(__file__).resolve().parents[2]
    hash_paths = {
        "implementation_sha256": root / "src/cora_infans/structured_workspace_binding.py",
        "cli_sha256": root / "src/cora_infans/cli.py",
        "io_sha256": root / "src/cora_infans/live_speak_smoke.py",
        "protocol_sha256": root
        / "docs/evidence/2026-07-10-structured-workspace-binding-protocol.json",
        "standing_world_geometry_sha256": root
        / "docs/evidence/2026-07-10-h3-standing-world-geometry.json",
        "standing_self_placement_sha256": root
        / "docs/evidence/2026-07-10-h3-standing-self-placement.json",
    }
    current_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in hash_paths.items()
    }
    hardware = preregistration.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("structured workspace preregistration hardware mapping is missing")
    input_route = hardware.get("input")
    self_route = hardware.get("self_source")
    world_route = hardware.get("world_source")
    if not all(isinstance(route, dict) for route in (input_route, self_route, world_route)):
        raise ValueError("structured workspace preregistration routes are missing")
    assert isinstance(input_route, dict)
    assert isinstance(self_route, dict)
    assert isinstance(world_route, dict)
    config = WorkspaceBindingConfig()
    return run_workspace_binding_acquisition(
        out_dir=values["--out-dir"],
        self_playback=lambda path: play_wav_to_device(
            path,
            device_index=int(self_route["output_device_index"]),
            sample_rate=config.sample_rate,
            volume_db=float(self_route["playback_gain_db"]),
        ),
        world_playback=lambda path: play_wav_to_device(
            path,
            device_index=int(world_route["output_device_index"]),
            sample_rate=config.sample_rate,
            volume_db=float(world_route["playback_gain_db"]),
        ),
        capture_fn=lambda path, **kwargs: capture_to_wav(
            path,
            device=str(input_route["avfoundation_device"]),
            **kwargs,
        ),
        config=config,
        real_hardware_claim=True,
        bounded_playback_authorized=authorized,
        hardware_preregistration=preregistration,
        current_hashes=current_hashes,
    )


def synthesize_vowel_cli(args: list[str] | None = None) -> dict[str, object]:
    """Option C CLI: synthesize a vowel (motor-plan production) to a WAV file.

    Usage: --synthesize-vowel a --out /tmp/a.wav [--duration-ms N]
           [--sample-rate N] [--f0-hz F] [--analyze]

    If ``--analyze`` is given, the written WAV is read back and analyzed
    (perception) so the analysis-by-synthesis loop is closed in one command.
    """
    parsed = args or ["--synthesize-vowel", "a"]
    vowel: str | None = None
    out_path: str | None = None
    duration_ms = 250
    sample_rate = 16000
    f0_hz = 110.0
    analyze = False
    i = 0
    while i < len(parsed):
        option = parsed[i]
        if option == "--analyze":
            analyze = True
            i += 1
            continue
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {option}")
        value = parsed[i + 1]
        if option == "--synthesize-vowel":
            vowel = value
        elif option == "--out":
            out_path = value
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--sample-rate":
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise ValueError("invalid sample rate") from exc
        elif option == "--f0-hz":
            try:
                f0_hz = float(value)
            except ValueError as exc:
                raise ValueError("invalid f0") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2

    if vowel is None:
        raise ValueError("--synthesize-vowel is required")
    if vowel not in VOWEL_FORMANTS_HZ:
        raise ValueError(f"unknown vowel: {vowel!r}")
    if out_path is None:
        raise ValueError("--out WAV path is required")

    wav = synthesize_vowel(
        vowel,  # type: ignore[arg-type]
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        f0_hz=f0_hz,
    )
    write_wav(out_path, wav, sample_rate)

    report: dict[str, object] = {
        "vowel": vowel,
        "out": out_path,
        "samples": len(wav),
        "sample_rate": sample_rate,
        "formant_target_hz": list(VOWEL_FORMANTS_HZ[vowel]),  # type: ignore[index]
        "motor_plan": ANALYSIS_BY_SYNTHESIS_TABLE[vowel],  # type: ignore[index]
    }
    if analyze:
        import wave

        with wave.open(out_path, "rb") as w:
            frames = w.readframes(w.getnframes())
            import numpy as np

            pcm = (
                np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
            )
        feat = analyze_vowel(pcm, sample_rate=sample_rate)
        report["perceived_formants_hz"] = list(feat.formant_hz)
        report["perceived_nearest_vowel"] = feat.nearest_vowel
        planner = MotorPlanner()
        plan = planner.plan_from_analysis(feat)
        report["motor_plan_from_perception"] = {
            "vowel": plan.vowel,
            "f1_hz": plan.f1_hz,
            "f2_hz": plan.f2_hz,
            "f3_hz": plan.f3_hz,
        }
    return report


def evaluate_vowel_loop_cli(args: list[str] | None = None) -> dict[str, object]:
    parsed = args or ["--evaluate-vowel-loop"]
    duration_ms = 300
    minimum_accuracy = 0.9
    i = 1
    while i < len(parsed):
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {parsed[i]}")
        option, value = parsed[i], parsed[i + 1]
        if option == "--duration-ms":
            duration_ms = int(value)
        elif option == "--minimum-accuracy":
            minimum_accuracy = float(value)
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return evaluate_vowel_round_trip(
        duration_ms=duration_ms,
        minimum_accuracy=minimum_accuracy,
    )


def benchmark_core_cli(args: list[str] | None = None) -> dict[str, object]:
    parsed = args or ["--benchmark-core"]
    iterations = 1000
    frame_ms = 25
    i = 1
    while i < len(parsed):
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {parsed[i]}")
        option, value = parsed[i], parsed[i + 1]
        if option == "--iterations":
            iterations = int(value)
        elif option == "--frame-ms":
            frame_ms = int(value)
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return benchmark_event_pipeline(iterations=iterations, frame_ms=frame_ms)


def prepare_ids_manifest_cli(args: list[str] | None = None) -> dict[str, object]:
    parsed = args or ["--prepare-ids-manifest"]
    root: str | None = None
    output: str | None = None
    held_out_fraction = 0.2
    include_song = False
    i = 1
    while i < len(parsed):
        option = parsed[i]
        if option == "--include-song":
            include_song = True
            i += 1
            continue
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {option}")
        value = parsed[i + 1]
        if option == "--root":
            root = value
        elif option == "--out":
            output = value
        elif option == "--held-out-fraction":
            held_out_fraction = float(value)
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    if root is None or output is None:
        raise ValueError("--root and --out are required")
    return build_ids_manifest(
        root,
        output,
        held_out_fraction=held_out_fraction,
        include_song=include_song,
    )


def prepare_corollary_manifest_cli(args: list[str] | None = None) -> dict[str, object]:
    parsed = args or ["--prepare-corollary-manifest"]
    root: str | None = None
    output: str | None = None
    sample_rate = 16000
    held_out_seeds = (2,)
    i = 1
    while i < len(parsed):
        if i + 1 >= len(parsed):
            raise ValueError(f"missing value for {parsed[i]}")
        option, value = parsed[i], parsed[i + 1]
        if option == "--root":
            root = value
        elif option == "--out":
            output = value
        elif option == "--sample-rate":
            sample_rate = int(value)
        elif option == "--held-out-seeds":
            held_out_seeds = tuple(int(seed) for seed in value.split(","))
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    if root is None or output is None:
        raise ValueError("--root and --out are required")
    return build_corollary_capture_manifest(
        root,
        output,
        sample_rate=sample_rate,
        held_out_seeds=held_out_seeds,
    )


def simulate_self_suppression_cli(args: list[str]) -> dict[str, object]:
    vowel = "a"
    world_vowel: str | None = None
    sample_rate = 16000
    duration_ms = 300
    world_gain = 0.0
    svsr_pass_db = 20.0
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown vowel: {value!r}")
            vowel = value
        elif option == "--world-vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown world vowel: {value!r}")
            world_vowel = value
        elif option == "--sample-rate":
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise ValueError("invalid sample rate") from exc
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--world-gain":
            try:
                world_gain = float(value)
            except ValueError as exc:
                raise ValueError("invalid world gain") from exc
        elif option == "--svsr-pass-db":
            try:
                svsr_pass_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid SVSR threshold") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    params = SelfSuppressionParams(
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        world_gain=world_gain,
        svsr_pass_db=svsr_pass_db,
    )
    return run_self_suppression_trial(
        vowel=vowel,  # type: ignore[arg-type]
        world_vowel=world_vowel,  # type: ignore[arg-type]
        params=params,
    )



def learn_self_forward_model_cli(args: list[str]) -> dict[str, object]:
    vowel = "a"
    world_vowel: str | None = None
    sample_rate = 16000
    duration_ms = 300
    observed_delay_ms = 18.0
    observed_attenuation = 0.55
    observed_echo_decay = 0.22
    world_gain = 0.0
    svsr_pass_db = 20.0
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown vowel: {value!r}")
            vowel = value
        elif option == "--world-vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown world vowel: {value!r}")
            world_vowel = value
        elif option == "--sample-rate":
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise ValueError("invalid sample rate") from exc
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--observed-delay-ms":
            try:
                observed_delay_ms = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed delay") from exc
        elif option == "--observed-attenuation":
            try:
                observed_attenuation = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed attenuation") from exc
        elif option == "--observed-echo-decay":
            try:
                observed_echo_decay = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed echo decay") from exc
        elif option == "--world-gain":
            try:
                world_gain = float(value)
            except ValueError as exc:
                raise ValueError("invalid world gain") from exc
        elif option == "--svsr-pass-db":
            try:
                svsr_pass_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid SVSR threshold") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return run_forward_model_learning_trial(
        vowel=vowel,  # type: ignore[arg-type]
        world_vowel=world_vowel,  # type: ignore[arg-type]
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        observed_delay_ms=observed_delay_ms,
        observed_attenuation=observed_attenuation,
        observed_echo_decay=observed_echo_decay,
        world_gain=world_gain,
        svsr_pass_db=svsr_pass_db,
    )



def _parse_float_tuple(raw: str, *, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in raw.split(",") if part)
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc
    if not values:
        raise ValueError(f"invalid {label}")
    return values


def sweep_self_forward_model_cli(args: list[str]) -> dict[str, object]:
    vowel = "a"
    world_vowel: str | None = None
    sample_rate = 16000
    duration_ms = 300
    observed_delay_ms = 18.0
    observed_attenuation = 0.55
    observed_echo_decay = 0.22
    delay_errors_ms = (-20.0, -10.0, 0.0, 10.0, 20.0)
    attenuation_scales = (0.75, 1.0, 1.25)
    echo_decay_scales = (0.5, 1.0, 1.5)
    world_gains = (0.0,)
    svsr_pass_db = 20.0
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown vowel: {value!r}")
            vowel = value
        elif option == "--world-vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown world vowel: {value!r}")
            world_vowel = value
        elif option == "--sample-rate":
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise ValueError("invalid sample rate") from exc
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--observed-delay-ms":
            try:
                observed_delay_ms = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed delay") from exc
        elif option == "--observed-attenuation":
            try:
                observed_attenuation = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed attenuation") from exc
        elif option == "--observed-echo-decay":
            try:
                observed_echo_decay = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed echo decay") from exc
        elif option == "--delay-errors-ms":
            delay_errors_ms = _parse_float_tuple(value, label="delay errors")
        elif option == "--attenuation-scales":
            attenuation_scales = _parse_float_tuple(value, label="attenuation scales")
        elif option == "--echo-decay-scales":
            echo_decay_scales = _parse_float_tuple(value, label="echo decay scales")
        elif option == "--world-gain":
            world_gains = (0.0,) + _parse_float_tuple(value, label="world gain")
        elif option == "--world-gains":
            world_gains = _parse_float_tuple(value, label="world gains")
        elif option == "--svsr-pass-db":
            try:
                svsr_pass_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid SVSR threshold") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return run_forward_model_robustness_sweep(
        vowel=vowel,  # type: ignore[arg-type]
        world_vowel=world_vowel,  # type: ignore[arg-type]
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        observed_delay_ms=observed_delay_ms,
        observed_attenuation=observed_attenuation,
        observed_echo_decay=observed_echo_decay,
        delay_errors_ms=delay_errors_ms,
        attenuation_scales=attenuation_scales,
        echo_decay_scales=echo_decay_scales,
        world_gains=world_gains,
        svsr_pass_db=svsr_pass_db,
    )



def promote_self_forward_model_cli(args: list[str]) -> dict[str, object]:
    vowel = "a"
    world_vowel: str | None = "o"
    sample_rate = 16000
    duration_ms = 300
    observed_delay_ms = 18.0
    observed_attenuation = 0.55
    observed_echo_decay = 0.22
    world_gain = 0.25
    min_learning_improvement_db = 3.0
    min_exact_svsr_db = 20.0
    min_delay_degradation_db = 10.0
    min_world_correlation = 0.2
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown vowel: {value!r}")
            vowel = value
        elif option == "--world-vowel":
            if value not in VOWEL_FORMANTS_HZ:
                raise ValueError(f"unknown world vowel: {value!r}")
            world_vowel = value
        elif option == "--sample-rate":
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise ValueError("invalid sample rate") from exc
        elif option == "--duration-ms":
            try:
                duration_ms = int(value)
            except ValueError as exc:
                raise ValueError("invalid duration") from exc
        elif option == "--observed-delay-ms":
            try:
                observed_delay_ms = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed delay") from exc
        elif option == "--observed-attenuation":
            try:
                observed_attenuation = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed attenuation") from exc
        elif option == "--observed-echo-decay":
            try:
                observed_echo_decay = float(value)
            except ValueError as exc:
                raise ValueError("invalid observed echo decay") from exc
        elif option == "--world-gain":
            try:
                world_gain = float(value)
            except ValueError as exc:
                raise ValueError("invalid world gain") from exc
        elif option == "--min-learning-improvement-db":
            try:
                min_learning_improvement_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid learning threshold") from exc
        elif option == "--min-exact-svsr-db":
            try:
                min_exact_svsr_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid exact SVSR threshold") from exc
        elif option == "--min-delay-degradation-db":
            try:
                min_delay_degradation_db = float(value)
            except ValueError as exc:
                raise ValueError("invalid delay degradation threshold") from exc
        elif option == "--min-world-correlation":
            try:
                min_world_correlation = float(value)
            except ValueError as exc:
                raise ValueError("invalid world correlation threshold") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return run_forward_model_promotion_gate(
        vowel=vowel,  # type: ignore[arg-type]
        world_vowel=world_vowel,  # type: ignore[arg-type]
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        observed_delay_ms=observed_delay_ms,
        observed_attenuation=observed_attenuation,
        observed_echo_decay=observed_echo_decay,
        world_gain=world_gain,
        min_learning_improvement_db=min_learning_improvement_db,
        min_exact_svsr_db=min_exact_svsr_db,
        min_delay_degradation_db=min_delay_degradation_db,
        min_world_correlation=min_world_correlation,
    )



def promote_speaking_policy_cli(args: list[str]) -> dict[str, object]:
    min_reward_ratio = 0.8
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--min-reward-ratio":
            try:
                min_reward_ratio = float(value)
            except ValueError as exc:
                raise ValueError("invalid reward ratio threshold") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return run_speaking_policy_promotion_gate(min_reward_ratio=min_reward_ratio)


def promote_speaking_readiness_cli(args: list[str]) -> dict[str, object]:
    min_speaking_reward_ratio = 0.8
    min_self_world_correlation = 0.2
    i = 1
    while i < len(args):
        option = args[i]
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--min-speaking-reward-ratio":
            try:
                min_speaking_reward_ratio = float(value)
            except ValueError as exc:
                raise ValueError("invalid speaking reward ratio threshold") from exc
        elif option == "--min-self-world-correlation":
            try:
                min_self_world_correlation = float(value)
            except ValueError as exc:
                raise ValueError("invalid self/world correlation threshold") from exc
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    return run_speaking_readiness_gate(
        min_speaking_reward_ratio=min_speaking_reward_ratio,
        min_self_world_correlation=min_self_world_correlation,
    )


def speak_live_smoke_cli(args: list[str]) -> dict[str, object]:
    """Safe-by-default CLI for the bounded live SPEAK smoke helper.

    No real playback is ever performed. Without ``--fake-io`` the helper writes
    only the planned efference WAV and reports ``blocked_missing_injected_io``.
    With ``--fake-io`` it runs the in-process fake playback/capture harness,
    still with ``real_hardware=false`` and ``live_playback_allowed=false``.
    """
    out_dir: str | None = None
    vowel: str = "a"
    duration_ms = 300
    max_capture_ms = 1500
    trials = 1
    use_fake_io = False
    use_real_io = False
    input_device = ":0"
    output_device_index = 2
    stimulus_kind = "vowel"
    volume_db = 12.0
    tone_hz = 1000.0
    chirp_f0 = 200.0
    chirp_f1 = 4000.0
    causal_window_ms = 20.0
    near_field = False
    save_calibration: str | None = None
    i = 1
    while i < len(args):
        option = args[i]
        if option == "--fake-io":
            use_fake_io = True
            i += 1
            continue
        if option == "--real-io":
            use_real_io = True
            i += 1
            continue
        if option == "--near-field":
            # Proven-stable near-field preset (validated 2026-07-09 at 4" speaker
            # proximity, max volume): chirp @24dB + tight causal window so the
            # loopback latency can actually cross the stability gate by default.
            near_field = True
            i += 1
            continue
        if i + 1 >= len(args):
            raise ValueError(f"missing value for {option}")
        value = args[i + 1]
        if option == "--out-dir":
            out_dir = value
        elif option == "--vowel":
            vowel = value
        elif option == "--duration-ms":
            duration_ms = int(value)
        elif option == "--max-capture-ms":
            max_capture_ms = int(value)
        elif option == "--input-device":
            input_device = value
        elif option == "--output-device-index":
            output_device_index = int(value)
        elif option == "--trials":
            trials = int(value)
        elif option == "--stimulus":
            stimulus_kind = value
        elif option == "--volume-db":
            volume_db = float(value)
        elif option == "--tone-hz":
            tone_hz = float(value)
        elif option == "--chirp-f0":
            chirp_f0 = float(value)
        elif option == "--chirp-f1":
            chirp_f1 = float(value)
        elif option == "--causal-window-ms":
            causal_window_ms = float(value)
        elif option == "--save-calibration":
            save_calibration = value
        else:
            raise ValueError(f"unknown option: {option}")
        i += 2
    if out_dir is None:
        raise ValueError("--out-dir is required")
    if use_fake_io and use_real_io:
        raise ValueError("--fake-io and --real-io are mutually exclusive")

    # --near-field applies the proven-stable preset unless the user overrode the
    # relevant flags explicitly. Explicit flags always win over the preset.
    if near_field:
        if stimulus_kind == "vowel":
            stimulus_kind = "chirp"
            chirp_f0 = 200.0
            chirp_f1 = 4000.0
        if volume_db == 12.0:
            volume_db = 24.0
        if causal_window_ms == 20.0:
            causal_window_ms = 4.0

    if stimulus_kind == "vowel":
        stimulus = StimulusSpec(kind="vowel", vowel=vowel)
    elif stimulus_kind == "tone":
        stimulus = StimulusSpec(kind="tone", freq_hz=tone_hz)
    elif stimulus_kind == "chirp":
        stimulus = StimulusSpec(kind="chirp", f0_hz=chirp_f0, f1_hz=chirp_f1)
    else:
        raise ValueError(f"unknown --stimulus: {stimulus_kind!r}")

    if use_real_io:
        # Real hardware path: bounded, gated, tag_only. The readiness gate is
        # consulted inside run_real_io_speak_trial and aborts before any I/O if
        # not promoted. Plays the generated efference to the output device and
        # captures the mic; the playback Popen is awaited so the trial blocks
        # until output finishes.
        def _real_playback(path: str) -> dict[str, object]:
            return play_wav_to_device(
                path,
                device_index=output_device_index,
                sample_rate=48000,
                volume_db=volume_db,
            )

        def _real_capture(
            path: str,
            *,
            sample_rate: int,
            duration_ms: int,
            ready_event,
            cancel_event,
            capture_timeline,
        ) -> dict[str, object]:
            return capture_to_wav(
                path,
                device=input_device,
                sample_rate=sample_rate,
                duration_ms=duration_ms,
                ready_event=ready_event,
                cancel_event=cancel_event,
                capture_timeline=capture_timeline,
            )

        if trials > 1:
            return run_real_io_speak_trials(
                vowel=vowel,  # type: ignore[arg-type]
                out_dir=out_dir,
                playback_fn=_real_playback,
                capture_fn=_real_capture,
                trials=trials,
                stimulus=stimulus,
                causal_window_ms=causal_window_ms,
                volume_db=volume_db,
                save_calibration=save_calibration,
                sample_rate=48000,
                duration_ms=duration_ms,
                max_capture_ms=max_capture_ms,
                bounded_live_io_authorized=True,
            )

        return run_real_io_speak_trial(
            vowel=vowel,  # type: ignore[arg-type]
            out_dir=out_dir,
            playback_fn=_real_playback,
            capture_fn=_real_capture,
            stimulus=stimulus,
            causal_window_ms=causal_window_ms,
            sample_rate=48000,
            duration_ms=duration_ms,
            max_capture_ms=max_capture_ms,
            bounded_live_io_authorized=True,
        )

    if use_fake_io:
        # In-process fakes: write a real efference WAV, capture silent frames.
        def _fake_playback(path: str) -> dict[str, object]:
            return {"backend": "fake_playback", "started": True, "completed": True}

        def _fake_capture(path: str, *, sample_rate: int, duration_ms: int) -> dict[str, object]:
            import wave

            import numpy as np

            samples = int(sample_rate * duration_ms / 1000)
            pcm = np.zeros(samples, dtype="<i2")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(pcm.tobytes())
            return {"backend": "fake_capture", "frames": samples, "completed": True}

        return run_live_speak_smoke_trial(
            vowel=vowel,  # type: ignore[arg-type]
            out_dir=out_dir,
            playback_fn=_fake_playback,
            capture_fn=_fake_capture,
            duration_ms=duration_ms,
            max_capture_ms=max_capture_ms,
        )
    return run_live_speak_smoke_trial(
        vowel=vowel,  # type: ignore[arg-type]
        out_dir=out_dir,
        duration_ms=duration_ms,
        max_capture_ms=max_capture_ms,
    )


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args == ["--describe"]:
        print(json.dumps(describe()))
        return 0
    if args and args[0] == "--simulate-frame":
        try:
            simulation_params, output_mode = parse_simulation_cli_args(args)
            payload = build_simulation_report(simulation_params)
            indent = 2 if output_mode == "pretty" else None
            print(json.dumps(payload, indent=indent))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args == ["--simulate-sequence"] or (args and args[0] == "--simulate-sequence"):
        try:
            print(json.dumps(simulate_sequence(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--capture-live":
        try:
            print(json.dumps(capture_live(args)))
            return 0
        except (ValueError, LiveMicError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--listen-live":
        try:
            print(json.dumps(listen_live(args)))
            return 0
        except (ValueError, LiveMicError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--eval-corpus":
        try:
            print(json.dumps(eval_corpus(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--fit-register-scorer":
        try:
            print(json.dumps(fit_register_scorer_cli(args)))
            return 0
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--eval-register-scorer":
        try:
            print(json.dumps(eval_register_scorer_cli(args)))
            return 0
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--eval-register-pipeline":
        try:
            print(json.dumps(eval_register_pipeline_cli(args)))
            return 0
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--evaluate-structured-prediction":
        try:
            print(json.dumps(structured_prediction_cli(args)))
            return 0
        except (ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--evaluate-natural-prediction-design":
        try:
            print(json.dumps(natural_prediction_design_cli(args)))
            return 0
        except (ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] in {
        "--evaluate-natural-event-design",
        "--evaluate-natural-event-transfer",
        "--evaluate-natural-event-ablations",
    }:
        try:
            print(json.dumps(natural_event_cli(args)))
            return 0
        except (ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--evaluate-real-workspace-design":
        try:
            print(json.dumps(real_workspace_design_cli(args)))
            return 0
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--evaluate-recorded-vowels":
        try:
            print(json.dumps(recorded_vowel_cli(args)))
            return 0
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--evaluate-structured-workspace-binding":
        try:
            print(json.dumps(structured_workspace_binding_cli(args)))
            return 0
        except (ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--synthesize-vowel":
        try:
            print(json.dumps(synthesize_vowel_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--evaluate-vowel-loop":
        try:
            print(json.dumps(evaluate_vowel_loop_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--benchmark-core":
        try:
            print(json.dumps(benchmark_core_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args == ["--evaluate-workspace-ablation"]:
        print(json.dumps(run_workspace_ablation()))
        return 0
    if args and args[0] == "--prepare-ids-manifest":
        try:
            print(json.dumps(prepare_ids_manifest_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--prepare-corollary-manifest":
        try:
            print(json.dumps(prepare_corollary_manifest_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--simulate-self-suppression":
        try:
            print(json.dumps(simulate_self_suppression_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--learn-self-forward-model":
        try:
            print(json.dumps(learn_self_forward_model_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--sweep-self-forward-model":
        try:
            print(json.dumps(sweep_self_forward_model_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--promote-self-forward-model":
        try:
            print(json.dumps(promote_self_forward_model_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--promote-speaking-policy":
        try:
            print(json.dumps(promote_speaking_policy_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--promote-speaking-readiness":
        try:
            print(json.dumps(promote_speaking_readiness_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] == "--speak-live-smoke":
        try:
            print(json.dumps(speak_live_smoke_cli(args)))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args and args[0] in ("--snapshot-firmware", "--tune-raw-linear"):
        from .io.hardware_tuner import HardwareTuner, HardwareTunerError

        try:
            tuner = HardwareTuner()
            if args[0] == "--tune-raw-linear":
                result = {"applied": tuner.apply_raw_linear(), "snapshot": tuner.snapshot()}
            else:
                result = {"snapshot": tuner.snapshot()}
            print(json.dumps(result))
            return 0
        except (HardwareTunerError, Exception) as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            return 2

    print(
        "usage: python -m cora_infans --describe | --simulate-frame | --simulate-sequence "
        "[--timestamp-ms N --dominant-frequency-hz F --channel-energies a,b,c,d --initial-prediction-hz F --mic-geometry l0,l1,l2,l3 --saliency-threshold F --output json|pretty] "
        "[--family sweep|zigzag|gap|multi_gap|delayed_recovery] [--with-scheduling] "
        "[--capture-live [--device :0] [--duration-ms N]] "
        "[--listen-live [--duration-ms N] [--frame-ms N] [--capture-backend auto|native|ffmpeg] [--with-stimulus] [--with-attention] [--with-scheduling] [--saliency-threshold F] [--with-motor-imitation [--real-io [--output-device-index 2] [--imitation-device-index 3] [--with-self-subtraction] [--volume-db F] [--latency-anchor PATH]]]] "
        "[--eval-corpus --root DIR --label-csv FILE [--saliency-threshold F] [--frame-ms N] [--score-mode peak_rms|spatial_contrast] [--input-mode wav|recorded_npy|synthetic_fixture] [--compare-baselines] [--evaluate-prediction FEATURE] [--evaluate-online-safety FEATURE[,FEATURE]] [--positive-label LABEL[,LABEL]]] "
        "[--synthesize-vowel a|e|i|o|u --out PATH.wav [--duration-ms N] [--sample-rate N] [--f0-hz F] [--analyze]] "
        "[--evaluate-vowel-loop [--duration-ms N] [--minimum-accuracy F]] "
        "[--benchmark-core [--iterations N] [--frame-ms N]] "
        "[--evaluate-workspace-ablation] "
        "[--prepare-ids-manifest --root DIR --out labels.csv [--held-out-fraction F] [--include-song]] "
        "[--prepare-corollary-manifest --root DIR --out labels.csv [--sample-rate N] [--held-out-seeds csv]] "
        "[--fit-register-scorer --source ROOT::LABEL_CSV [--source ROOT::LABEL_CSV] --out MODEL.json] "
        "[--eval-register-scorer --model MODEL.json --root DIR --label-csv FILE] "
        "[--eval-register-pipeline --model MODEL.json --vocal-root DIR --vocal-label-csv FILE --environmental-root DIR --environmental-label-csv FILE] "
        "[--evaluate-structured-prediction --out-dir DIR [--variant original|calibrated_v2|robustness_v3|dropout_v4|dropout_v5] [--fake-io | --real-io --authorize-bounded-playback --input-device :0 --output-device-index 2 --speaker-distance-cm N --speaker-orientation TEXT --quiet-room-confirmed]] "
        "[--evaluate-natural-prediction-design --root DIR --participants ID,ID,ID] "
        "[--evaluate-natural-event-design --root DIR --participants ID,ID,ID] "
        "[--evaluate-natural-event-transfer --development-root DIR --development-participants ID,ID --held-out-root DIR --held-out-participants ID,ID] "
        "[--evaluate-natural-event-ablations --development-root DIR --development-participants ID,ID --held-out-root DIR --held-out-participants ID,ID] "
        "[--evaluate-real-workspace-design --root COROLLARY_M2_BASELINE_DIR] "
        "[--evaluate-recorded-vowels --root VOCALSET_SUBSET_DIR --manifest FILE] "
        "[--evaluate-structured-workspace-binding [--fake-io | --real-io --out-dir DIR --preregistration FILE --authorize-bounded-playback]] "
        "[--simulate-self-suppression [--vowel a|e|i|o|u] [--world-vowel a|e|i|o|u] [--world-gain F] [--duration-ms N] [--sample-rate N]] "
        "[--learn-self-forward-model [--vowel a|e|i|o|u] [--observed-delay-ms F] [--observed-attenuation F] [--observed-echo-decay F] [--world-vowel a|e|i|o|u] [--world-gain F]] "
        "[--sweep-self-forward-model [--vowel a|e|i|o|u] [--delay-errors-ms csv] [--attenuation-scales csv] [--echo-decay-scales csv] [--world-vowel a|e|i|o|u] [--world-gain F]] "
        "[--promote-self-forward-model [--vowel a|e|i|o|u] [--world-vowel a|e|i|o|u] [--world-gain F] [--min-world-correlation F]] "
        "[--promote-speaking-policy [--min-reward-ratio F]] "
        "[--promote-speaking-readiness [--min-speaking-reward-ratio F] [--min-self-world-correlation F]] "
        "[--speak-live-smoke --out-dir DIR [--vowel a|e|i|o|u] [--duration-ms N] [--max-capture-ms N] [--fake-io] [--real-io [--near-field] [--input-device :0] [--output-device-index 2] [--trials N] [--stimulus vowel|tone|chirp] [--volume-db F] [--tone-hz F] [--chirp-f0 F] [--chirp-f1 F] [--causal-window-ms F] [--save-calibration PATH]]]",
        file=sys.stderr,
    )
    return 2
