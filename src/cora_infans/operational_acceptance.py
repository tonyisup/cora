"""Preflight and bounded live acceptance for the operational VISION."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Mapping, Sequence

from .operational_contract import (
    OPERATIONAL_CONTRACT,
    evaluate_stage6_operating_point,
)
from .operational_readiness import (
    build_operational_readiness_report,
    repository_evidence_dir,
)


DeviceProbe = Callable[[], Mapping[str, object]]
LiveRunner = Callable[[str], Mapping[str, object]]


def canonical_operating_point(
    operator_attestations: Mapping[str, object],
) -> dict[str, object]:
    """Build the only accepted physical metadata envelope."""

    stage6 = OPERATIONAL_CONTRACT["stage6"]
    return {
        "schema_version": stage6["operating_point_schema_version"],
        "host": deepcopy(stage6["host"]),
        "input": deepcopy(stage6["input"]),
        "self_source": deepcopy(stage6["self_source"]),
        "world_source": deepcopy(stage6["world_source"]),
        "operator_attestations": dict(operator_attestations),
    }


def load_operating_point_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid operating-point manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("operating-point manifest must be a JSON object")
    return payload


def probe_local_operating_environment() -> dict[str, object]:
    """Inspect dependencies and devices without opening an audio stream."""

    ffmpeg = shutil.which("ffmpeg")
    ffmpeg_has_audiotoolbox = False
    ffmpeg_error: str | None = None
    if ffmpeg is not None:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-h", "muxer=audiotoolbox"],
            capture_output=True,
            check=False,
            text=True,
        )
        combined = result.stdout + result.stderr
        ffmpeg_has_audiotoolbox = (
            result.returncode == 0
            and "AudioToolbox output device" in combined
            and "audio_device_index" in combined
        )
        if not ffmpeg_has_audiotoolbox:
            ffmpeg_error = combined.strip()[-1000:]

    devices: Sequence[Mapping[str, object]] = ()
    input_probe_error: str | None = None
    input_rate_supported = False
    input_device: dict[str, object] | None = None
    portaudio_output_device_names: list[str] = []
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        for index, device in enumerate(devices):
            if int(device.get("max_output_channels", 0)) > 0:
                portaudio_output_device_names.append(str(device.get("name", "")))
            if (
                "respeaker xvf3800" in str(device.get("name", "")).lower()
                and int(device.get("max_input_channels", 0)) >= 2
            ):
                input_device = {
                    "index": index,
                    "name": str(device["name"]),
                    "max_input_channels": int(device["max_input_channels"]),
                }
                try:
                    sd.check_input_settings(
                        device=index,
                        channels=2,
                        dtype="float32",
                        samplerate=16000,
                    )
                    input_rate_supported = True
                except Exception as exc:  # PortAudio reports host-specific errors.
                    input_probe_error = str(exc)
                break
    except Exception as exc:  # Import/device discovery must become a report.
        input_probe_error = str(exc)

    host_model: str | None = None
    host_name: str | None = None
    host_chip: str | None = None
    coreaudio_output_device_names: list[str] = []
    try:
        hardware = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            check=False,
            text=True,
        )
        if hardware.returncode == 0:
            overview = json.loads(hardware.stdout)["SPHardwareDataType"][0]
            host_name = str(overview.get("machine_name") or "") or None
            host_model = str(overview.get("machine_model") or "") or None
            host_chip = str(overview.get("chip_type") or "") or None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass

    try:
        audio = subprocess.run(
            ["system_profiler", "SPAudioDataType", "-json"],
            capture_output=True,
            check=False,
            text=True,
        )
        if audio.returncode == 0:
            groups = json.loads(audio.stdout)["SPAudioDataType"]
            for group in groups:
                for device in group.get("_items", ()):
                    if int(device.get("coreaudio_device_output", 0)) > 0:
                        coreaudio_output_device_names.append(str(device["_name"]))
    except (OSError, ValueError, KeyError, TypeError):
        pass

    return {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "host_name": host_name,
        "host_model": host_model,
        "host_chip": host_chip,
        "ffmpeg_path": ffmpeg,
        "ffmpeg_has_audiotoolbox": ffmpeg_has_audiotoolbox,
        "ffmpeg_error": ffmpeg_error,
        "xvf3800_input": input_device,
        "xvf3800_16khz_two_channel_supported": input_rate_supported,
        "input_probe_error": input_probe_error,
        "enumerated_device_count": len(devices),
        "portaudio_output_device_names": portaudio_output_device_names,
        "coreaudio_output_device_names": coreaudio_output_device_names,
        "output_index_verification": (
            "operator_attested; ffmpeg AudioToolbox indices are verified only "
            "when the bounded playback process opens each named route"
        ),
    }


def build_operational_preflight(
    manifest: Mapping[str, object],
    *,
    device_probe: DeviceProbe = probe_local_operating_environment,
    readiness_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Combine retained-evidence, manifest, host, and device readiness."""

    operating_point = manifest.get("operating_point")
    if not isinstance(operating_point, Mapping):
        raise ValueError("manifest is missing operating_point")
    # Reuse the authoritative evaluator with an otherwise clean synthetic
    # summary. Only its metadata checks are consumed here; no synthetic value is
    # represented as a live measurement.
    profile = OPERATIONAL_CONTRACT["stage6"]["acceptance_profile"]
    zero_counts = {
        field: 0 for field in OPERATIONAL_CONTRACT["stage6"]["required_zero_counts"]
    }
    metadata_evaluation = evaluate_stage6_operating_point(
        {
            "operating_point": operating_point,
            "real_capture": True,
            "capture_backend": OPERATIONAL_CONTRACT["stage6"]["capture_backend"],
            "frame_ms": profile["frame_ms"],
            "frame_count": profile["target_frame_count"],
            "salient_frame_count": profile["minimum_salient_frame_count"],
            "active_playback_frame_count": 1,
            "software_playback_gain_db": profile["software_playback_gain_db"],
            "stimulus_sha256": profile["stimulus_sha256"],
            OPERATIONAL_CONTRACT["stage6"]["latency_metric"]: 0.0,
            **zero_counts,
        }
    )
    metadata_checks = {
        key: value
        for key, value in metadata_evaluation["criteria"].items()
        if key.endswith("_matches") or key.endswith("_confirmed")
    }
    probe = dict(device_probe())
    readiness = dict(
        readiness_report
        or build_operational_readiness_report(repository_evidence_dir())
    )
    checks = {
        **metadata_checks,
        "retained_evidence_ready": readiness.get("operational_evidence_ready")
        is True,
        "host_is_macos": probe.get("platform_system") == "Darwin",
        "host_is_mac_mini": probe.get("host_name") == "Mac mini",
        "ffmpeg_audiotoolbox_available": probe.get("ffmpeg_has_audiotoolbox")
        is True,
        "xvf3800_input_discovered": probe.get("xvf3800_input") is not None,
        "xvf3800_16khz_two_channel_supported": probe.get(
            "xvf3800_16khz_two_channel_supported"
        )
        is True,
        "external_headphones_discovered": "External Headphones"
        in probe.get("coreaudio_output_device_names", ()),
        "mac_mini_speakers_discovered": "Mac mini Speakers"
        in probe.get("coreaudio_output_device_names", ()),
    }
    return {
        "mode": "bounded_operational_preflight",
        "checks": checks,
        "preflight_passed": all(checks.values()),
        "operating_point": dict(operating_point),
        "device_probe": probe,
        "retained_evidence_sha256": readiness.get("evidence_sha256"),
        "claim_boundary": (
            "Read-only preflight: no audio stream is opened and no playback or "
            "capture is represented as completed"
        ),
    }


def stage6_report_from_live_listener(
    raw_report: Mapping[str, object],
    operating_point: Mapping[str, object],
) -> dict[str, object]:
    """Normalize a live-listener report into the strict Stage 6 schema."""

    performance = raw_report["runtime_performance"]
    workloads = performance["workload_latency_ms"]
    salient_workload = workloads.get("salient")
    salient_latency = (
        salient_workload.get("adc_frame_end_to_workspace_decision")
        if isinstance(salient_workload, Mapping)
        else None
    )
    frames = raw_report["frames"]
    active_frames = [
        frame for frame in frames if frame.get("stimulus_playback_active") is True
    ]
    stimulus = raw_report["stimulus"]
    fallback = raw_report.get("capture_stream_fallback_reason")
    return {
        "operating_point": dict(operating_point),
        "real_capture": performance.get("real_capture") is True,
        "capture_backend": performance.get("capture_backend"),
        "frame_ms": int(round(float(performance["deadline_ms"]))),
        "frame_count": int(raw_report["frame_count"]),
        "salient_frame_count": int(raw_report["salient_frame_count"]),
        "active_playback_frame_count": len(active_frames),
        "software_playback_gain_db": float(stimulus["volume_db"]),
        "stimulus_sha256": stimulus["source_wav_sha256"],
        "salient_adc_frame_end_to_workspace_p95_ms": (
            float(salient_latency["p95"])
            if isinstance(salient_latency, Mapping)
            else None
        ),
        "device_deadline_miss_count": int(
            performance["device_timestamped_deadline_miss_count"]
        ),
        "overflow_count": int(performance["input_overflow_count"]),
        "callback_status_count": int(raw_report["capture_callback_status_count"]),
        "timestamp_discontinuity_count": int(
            raw_report["capture_callback_timestamp_discontinuity_count"]
        ),
        "fallback_frame_count": 0 if fallback is None else len(frames),
        "active_playback_clipped_sample_count": sum(
            int(frame.get("clipped_sample_count", 0)) for frame in active_frames
        ),
        "whole_capture_clipped_sample_count": int(
            raw_report["capture_quality"]["clipped_sample_count"]
        ),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_operational_acceptance(
    *,
    manifest: Mapping[str, object],
    stimulus_wav: str | Path,
    output_dir: str | Path,
    live_runner: LiveRunner,
    device_probe: DeviceProbe = probe_local_operating_environment,
) -> dict[str, object]:
    """Run one bounded, already-spent Stage 6 physical acceptance trial."""

    preflight = build_operational_preflight(manifest, device_probe=device_probe)
    if preflight["preflight_passed"] is not True:
        raise ValueError("operational preflight failed; live audio was not started")
    stimulus_path = Path(stimulus_wav).resolve()
    raw_stimulus = stimulus_path.read_bytes()
    actual_hash = hashlib.sha256(raw_stimulus).hexdigest()
    expected_hash = OPERATIONAL_CONTRACT["stage6"]["acceptance_profile"][
        "stimulus_sha256"
    ]
    if actual_hash != expected_hash:
        raise ValueError("operational acceptance stimulus hash mismatch")

    raw_report = dict(live_runner(str(stimulus_path)))
    output = Path(output_dir).resolve()
    raw_path = output / "stage6-live-raw.json"
    _write_json(raw_path, raw_report)
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    try:
        stage6_report = stage6_report_from_live_listener(
            raw_report, preflight["operating_point"]
        )
        stage6_evaluation = evaluate_stage6_operating_point(stage6_report)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        failed_report = {
            "mode": "bounded_operational_acceptance",
            "preflight": preflight,
            "live_raw_report": {"path": str(raw_path), "sha256": raw_sha256},
            "stage6_report": None,
            "stage6_evaluation": None,
            "operational_acceptance_passed": False,
            "error": f"invalid live report: {exc}",
            "claim_boundary": (
                "Live I/O completed but could not satisfy the operational "
                "report contract; raw telemetry is retained"
            ),
        }
        _write_json(output / "operational-acceptance.json", failed_report)
        raise ValueError(
            "live report failed operational validation; raw and failure reports "
            f"were retained in {output}"
        ) from exc
    report = {
        "mode": "bounded_operational_acceptance",
        "preflight": preflight,
        "live_raw_report": {"path": str(raw_path), "sha256": raw_sha256},
        "stage6_report": stage6_report,
        "stage6_evaluation": stage6_evaluation,
        "retained_stage4_exact_efference": (
            "hash-pinned readiness evidence; not re-executed by this Stage 6 run"
        ),
        "operational_acceptance_passed": stage6_evaluation[
            "operating_point_passed"
        ]
        is True,
        "claim_boundary": (
            "One current Stage 6 run at the named physical point plus hash-pinned "
            "retained stage evidence; not a new scientific promotion"
        ),
    }
    _write_json(output / "operational-acceptance.json", report)
    return report
