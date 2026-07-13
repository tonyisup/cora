from __future__ import annotations

import struct
import wave

import numpy as np
import pytest

from cora_infans.io.offline_eval import CorpusClip, OfflineEvalHarness
from cora_infans.io.wav_frames import (
    WavDecodeError,
    build_wav_corpus_frames,
    read_pcm_wav,
    read_wav_metadata,
)


def _write_pcm16(path, samples: np.ndarray, sample_rate: int) -> None:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(data.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def _write_float_wav(path, samples: np.ndarray, sample_rate: int, bits: int) -> None:
    data = np.asarray(samples, dtype="<f4" if bits == 32 else "<f8")
    if data.ndim == 1:
        data = data[:, None]
    raw = data.tobytes()
    channels = data.shape[1]
    block_align = channels * bits // 8
    fmt = struct.pack(
        "<HHIIHH",
        3,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
    )
    riff_size = 4 + (8 + len(fmt)) + (8 + len(raw))
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(raw))
        + raw
    )


def test_read_pcm_wav_preserves_real_channel_geometry(tmp_path) -> None:
    sample_rate = 8000
    left = np.full(800, 0.25, dtype=np.float32)
    right = np.full(800, -0.5, dtype=np.float32)
    path = tmp_path / "stereo.wav"
    _write_pcm16(path, np.column_stack([left, right]), sample_rate)

    decoded, decoded_rate = read_pcm_wav(path)

    assert decoded_rate == sample_rate
    assert decoded.shape == (800, 2)
    assert float(np.mean(decoded[:, 0])) == pytest.approx(0.25, abs=1e-3)
    assert float(np.mean(decoded[:, 1])) == pytest.approx(-0.5, abs=1e-3)


@pytest.mark.parametrize("bits", [32, 64])
def test_read_pcm_wav_accepts_ieee_float_corpus_files(tmp_path, bits) -> None:
    path = tmp_path / f"float{bits}.wav"
    samples = np.column_stack(
        [np.linspace(-0.5, 0.5, 20), np.linspace(0.25, -0.25, 20)]
    )
    _write_float_wav(path, samples, sample_rate=44100, bits=bits)

    decoded, sample_rate = read_pcm_wav(path)
    metadata = read_wav_metadata(path)

    assert sample_rate == 44100
    assert decoded == pytest.approx(samples, abs=1e-6)
    assert metadata == (44100, 2, 20)


def test_wav_provider_measures_frames_without_using_annotations(tmp_path) -> None:
    sample_rate = 8000
    frame_samples = sample_rate // 4
    first = 0.4 * np.sin(2.0 * np.pi * 200.0 * np.arange(frame_samples) / sample_rate)
    second = np.zeros(frame_samples, dtype=np.float32)
    path = tmp_path / "clip.wav"
    _write_pcm16(path, np.concatenate([first, second]), sample_rate)
    clip = CorpusClip(clip_id="clip", wav_path=path, annotations=())

    frames = build_wav_corpus_frames(clip, frame_ms=250)

    assert [frame.timestamp_ms for frame in frames] == [0, 250]
    assert frames[0].source == "pcm_wav"
    assert frames[0].sample_rate == sample_rate
    assert frames[0].channel_count == 1
    assert frames[0].channel_energies[0] == pytest.approx(0.4 / np.sqrt(2), rel=0.02)
    assert frames[1].channel_energies[0] == 0.0
    assert frames[0].features is not None
    assert frames[0].features["dominant_frequency_hz"] == pytest.approx(200.0, abs=4.0)
    assert frames[0].features["pitch_hz"] == pytest.approx(200.0, rel=0.03)


def test_real_wav_evaluation_reports_confusion_and_provenance(tmp_path) -> None:
    from cora_infans.io.offline_eval import Annotation

    sample_rate = 8000
    quarter = sample_rate // 4
    tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * np.arange(quarter) / sample_rate)
    path = tmp_path / "clip.wav"
    _write_pcm16(path, np.concatenate([tone, np.zeros(quarter)]), sample_rate)
    clip = CorpusClip(
        clip_id="clip",
        wav_path=path,
        annotations=(Annotation(0.0, 0.25, "salient_speech"),),
    )

    report = OfflineEvalHarness(
        saliency_threshold=0.05,
        frame_ms=250,
    ).evaluate_corpus([clip], build_wav_corpus_frames)

    assert report["evidence_kind"] == "real_audio"
    assert report["true_positive_frames"] == 1
    assert report["true_negative_frames"] == 1
    assert report["false_positive_frames"] == 0
    assert report["false_negative_frames"] == 0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["f1"] == 1.0
    assert report["per_clip"][0]["input_source"] == "pcm_wav"
    assert report["per_clip"][0]["feature_summary"]["pitch_hz"]["count"] == 1


def test_invalid_wav_is_rejected_instead_of_falling_back(tmp_path) -> None:
    path = tmp_path / "broken.wav"
    path.write_bytes(b"RIFF")

    with pytest.raises(WavDecodeError, match="invalid RIFF/WAVE header"):
        read_pcm_wav(path)
