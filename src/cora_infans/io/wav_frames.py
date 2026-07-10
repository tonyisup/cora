"""Real uncompressed-WAV decoding and frame-level acoustic observations.

This is the Stage-1 evidence path. It reads samples from each clip; it never
consults annotations and never generates input from labels.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .offline_eval import CorpusClip, FrameObservation


class WavDecodeError(ValueError):
    """Raised when a WAV cannot be decoded as uncompressed sample data."""


def _read_wav_container(
    path: Path, *, include_data: bool
) -> tuple[int, int, int, int, int, bytes | None]:
    """Return format, channels, rate, bits, frames, and optional sample bytes.

    Python's :mod:`wave` rejects IEEE-float WAVs. The selected IDS corpus uses
    both integer PCM and float64, so this small RIFF reader handles the two
    uncompressed formats explicitly while rejecting compressed codecs.
    """

    try:
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                raise WavDecodeError(f"invalid RIFF/WAVE header in {path}")
            format_tag = channel_count = sample_rate = block_align = bits = None
            data_size: int | None = None
            data: bytes | None = None
            while True:
                chunk_header = handle.read(8)
                if not chunk_header:
                    break
                if len(chunk_header) != 8:
                    raise WavDecodeError(f"truncated WAV chunk header in {path}")
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
                if chunk_id == b"fmt ":
                    payload = handle.read(chunk_size)
                    if len(payload) != chunk_size or chunk_size < 16:
                        raise WavDecodeError(f"invalid WAV fmt chunk in {path}")
                    (
                        format_tag,
                        channel_count,
                        sample_rate,
                        _byte_rate,
                        block_align,
                        bits,
                    ) = struct.unpack_from("<HHIIHH", payload)
                    if format_tag == 0xFFFE:
                        if len(payload) < 40:
                            raise WavDecodeError(
                                f"invalid extensible WAV fmt chunk in {path}"
                            )
                        format_tag = struct.unpack_from("<H", payload, 24)[0]
                elif chunk_id == b"data":
                    data_size = chunk_size
                    if include_data:
                        data = handle.read(chunk_size)
                        if len(data) != chunk_size:
                            raise WavDecodeError(f"truncated WAV data in {path}")
                    else:
                        handle.seek(chunk_size, 1)
                else:
                    handle.seek(chunk_size, 1)
                if chunk_size % 2:
                    handle.seek(1, 1)
    except OSError as exc:
        raise WavDecodeError(f"could not read WAV {path}: {exc}") from exc

    if None in (format_tag, channel_count, sample_rate, block_align, bits, data_size):
        raise WavDecodeError(f"WAV is missing fmt or data chunks: {path}")
    assert format_tag is not None
    assert channel_count is not None
    assert sample_rate is not None
    assert block_align is not None
    assert bits is not None
    assert data_size is not None
    if channel_count <= 0 or sample_rate <= 0 or block_align <= 0:
        raise WavDecodeError(f"invalid WAV geometry in {path}")
    if data_size % block_align:
        raise WavDecodeError(f"WAV data is not frame-aligned in {path}")
    expected_align = channel_count * ((bits + 7) // 8)
    if block_align != expected_align:
        raise WavDecodeError(f"unsupported WAV block alignment in {path}")
    if not (
        (format_tag == 1 and bits in (8, 16, 24, 32))
        or (format_tag == 3 and bits in (32, 64))
    ):
        raise WavDecodeError(f"unsupported WAV format {format_tag} at {bits} bits")
    return (
        format_tag,
        channel_count,
        sample_rate,
        bits,
        data_size // block_align,
        data,
    )


def _decode_pcm_bytes(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise WavDecodeError(f"unsupported PCM sample width: {sample_width} bytes")


def read_wav_metadata(path: str | Path) -> tuple[int, int, int]:
    """Return ``(sample_rate, channel_count, frame_count)`` without decoding."""

    wav_path = Path(path)
    _format, channels, sample_rate, _bits, frames, _data = _read_wav_container(
        wav_path, include_data=False
    )
    return sample_rate, channels, frames


def read_pcm_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Return normalized float32 samples shaped ``(sample, channel)``."""

    wav_path = Path(path)
    format_tag, channel_count, sample_rate, bits, _frames, raw = _read_wav_container(
        wav_path, include_data=True
    )
    assert raw is not None
    sample_width = bits // 8
    if format_tag == 1:
        decoded = _decode_pcm_bytes(raw, sample_width)
    elif format_tag == 3 and sample_width in (4, 8):
        dtype = "<f4" if sample_width == 4 else "<f8"
        decoded = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if not bool(np.all(np.isfinite(decoded))):
            raise WavDecodeError(f"non-finite IEEE-float samples in {wav_path}")
    else:
        raise WavDecodeError(
            f"unsupported WAV format {format_tag} at {sample_width} bytes/sample"
        )
    if decoded.size % channel_count:
        raise WavDecodeError(f"sample count is not channel-aligned in {wav_path}")
    return decoded.reshape(-1, channel_count), sample_rate


def _dominant_frequency_hz(mono: np.ndarray, sample_rate: int) -> float:
    centered = mono.astype(np.float64) - float(np.mean(mono))
    if centered.size < 2 or float(np.sqrt(np.mean(centered * centered))) <= 1e-7:
        return 0.0
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / sample_rate)
    if spectrum.size:
        spectrum[0] = 0.0
    return float(frequencies[int(np.argmax(spectrum))])


def _pitch_hz_and_confidence(
    mono: np.ndarray,
    sample_rate: int,
    *,
    min_hz: float = 70.0,
    max_hz: float = 500.0,
) -> tuple[float | None, float]:
    centered = mono.astype(np.float64) - float(np.mean(mono))
    energy = float(np.dot(centered, centered))
    if centered.size < 2 or energy <= 1e-10:
        return None, 0.0
    windowed = centered * np.hanning(centered.size)
    fft_size = 1 << max(1, (2 * centered.size - 1).bit_length())
    spectrum = np.fft.rfft(windowed, fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)
    autocorrelation = autocorrelation[: centered.size]
    minimum_lag = max(1, int(sample_rate / max_hz))
    maximum_lag = min(centered.size - 1, int(sample_rate / min_hz))
    if maximum_lag <= minimum_lag or autocorrelation[0] <= 1e-12:
        return None, 0.0
    search = autocorrelation[minimum_lag : maximum_lag + 1]
    lag = minimum_lag + int(np.argmax(search))
    confidence = max(0.0, min(1.0, float(autocorrelation[lag] / autocorrelation[0])))
    if confidence < 0.1:
        return None, confidence
    return float(sample_rate / lag), confidence


def _spectral_features(
    mono: np.ndarray,
    sample_rate: int,
    previous_spectrum: np.ndarray | None,
) -> tuple[float, float, np.ndarray]:
    centered = mono.astype(np.float64) - float(np.mean(mono))
    magnitude = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / sample_rate)
    magnitude_sum = float(np.sum(magnitude))
    centroid = (
        float(np.sum(frequencies * magnitude) / magnitude_sum)
        if magnitude_sum > 1e-12
        else 0.0
    )
    normalized = magnitude / magnitude_sum if magnitude_sum > 1e-12 else magnitude
    if previous_spectrum is None or previous_spectrum.shape != normalized.shape:
        flux = 0.0
    else:
        flux = float(np.sqrt(np.mean((normalized - previous_spectrum) ** 2)))
    return centroid, flux, normalized


def build_wav_corpus_frames(
    clip: CorpusClip,
    frame_ms: int,
) -> list[FrameObservation]:
    """Decode a corpus clip into non-overlapping measured observations."""

    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    samples, sample_rate = read_pcm_wav(clip.wav_path)
    return build_pcm_observations(
        samples,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        source="pcm_wav",
    )


def build_pcm_observations(
    samples: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: int,
    source: str,
) -> list[FrameObservation]:
    """Extract the shared measured feature contract from normalized PCM."""

    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[1] <= 0:
        raise ValueError(f"PCM must have shape (sample, channel), got {samples.shape}")
    frame_samples = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    observations: list[FrameObservation] = []
    previous_spectrum: np.ndarray | None = None

    for start in range(0, len(samples), frame_samples):
        frame = samples[start : start + frame_samples]
        if not len(frame):
            continue
        channel_energies = [
            float(np.sqrt(np.mean(channel.astype(np.float64) ** 2)))
            for channel in frame.T
        ]
        mono = np.mean(frame, axis=1).astype(np.float32)
        dominant = _dominant_frequency_hz(mono, sample_rate)
        pitch_hz, pitch_confidence = _pitch_hz_and_confidence(mono, sample_rate)
        centroid, spectral_flux, previous_spectrum = _spectral_features(
            mono, sample_rate, previous_spectrum
        )
        peak = max(channel_energies, default=0.0)
        mean = sum(channel_energies) / len(channel_energies) if channel_energies else 0.0
        observations.append(
            FrameObservation(
                timestamp_ms=int(round(start * 1000.0 / sample_rate)),
                channel_energies=channel_energies,
                features={
                    "peak_rms": peak,
                    "spatial_contrast": max(0.0, peak - mean),
                    "dominant_frequency_hz": dominant,
                    "pitch_hz": pitch_hz,
                    "pitch_confidence": pitch_confidence,
                    "spectral_centroid_hz": centroid,
                    "spectral_flux": spectral_flux,
                },
                source=source,
                sample_rate=sample_rate,
                channel_count=samples.shape[1],
            )
        )
    return observations
