--- 
name: audacity-aup3
description: |
  Read and extract audio from Audacity .aup3 project files (SQLite-based format).
  Decodes zlib-compressed float32 samples, segments by silence, and exports mono 16-bit PCM WAV.
  Use when analyzing recorded audio from Audacity, extracting TTS or recordings from .aup3 projects,
  or building audio analysis pipelines that need to work with Audacity project files.
version: 1.0.0
author: Hermes Agent (adapted from M9 audio workflow)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [audio, audacity, aup3, sqlite, wav, media]
    related_skills: [media]
---

# Audacity AUP3 Reader

## Overview

Audacity 3.x stores projects as `.aup3` files — SQLite databases containing zlib-compressed
float32 audio samples. This skill provides a Python module for reading, segmenting, and exporting
these files without needing Audacity installed.

## When to Use

- User provides an `.aup3` file and wants to analyze the audio content
- You need to extract TTS recordings, captured audio, or other signals from Audacity projects
- Building audio analysis, spectrogram, or ML preprocessing pipelines on Audacity recordings
- Converting `.aup3` to `.wav` for playback or further processing

## Usage

### Quick Extract (CLI)

```python
python scripts/aup3_read.py input.aup3                    # Export to input.wav
python scripts/aup3_read.py input.aup3 -o output.wav     # Custom output path
python scripts/aup3_read.py input.aup3 --segment         # Silence-segmented output
python scripts/aup3_read.py input.aup3 --segment --min-gap 0.3  # Min gap between segments
```

### Python Module

```python
from aup3_read import AUP3Reader

reader = AUP3Reader("project.aup3")

# Get sample rate
sr = reader.sample_rate  # e.g. 16000

# Get all audio as a single array
audio = reader.get_audio()  # shape: (total_samples,)

# Get segmented audio (split by silence)
segments = reader.get_segments(
    silence_threshold=0.01,  # RMS threshold for silence (auto-estimated if 0)
    min_gap_seconds=0.3,    # Minimum silence gap to split on
    min_segment_seconds=0.1 # Minimum segment duration
)
# Returns list of np.ndarray, each mono float32

# Export to WAV
reader.export_wav("output.wav", audio=None)  # None = all audio
reader.export_wav("output.wav", segments[0])  # Specific segment

# Iterate segments with metadata
for seg_info in reader.iter_segments():
    print(f"Segment: {seg_info.duration:.2f}s, "
          f"rms={seg_info.rms:.4f}, "
          f"start={seg_info.start_time:.2f}s")
    audio = seg_info.audio

# Extract individual tracks (AUP3 can have multiple tracks)
track = reader.get_track(blockid=2)  # Get specific track by block ID
```

### Programmatic Pipeline Example

```python
from aup3_read import AUP3Reader
from scipy.signal import butter, sosfilt

reader = AUP3Reader("project.aup3")
sr = reader.sample_rate

# Get segments
segments = reader.get_segments(silence_threshold=0.005, min_gap_seconds=0.2)

hp = butter(4, 100, btype='high', fs=sr, output='sos')

for i, seg_audio in enumerate(segments):
    # Preprocess
    audio = sosfilt(hp, seg_audio).astype(np.float32)
    
    # Your analysis here (decode, classify, etc.)
    decoded = decode_vowel_transition_audio(audio, frame_size=480, center_mel=True)
    vowels = [s['vowel'] for s in decoded.get('vowel_segments', [])]
    print(f"Segment {i+1}: {len(seg_audio)/sr:.2f}s, vowels={vowels}")
```

## File Format Reference

### AUP3 SQLite Schema

```
project table:  { id, dict, doc }        # Project metadata (JSON in dict/doc)
sampleblocks:    { blockid, sampleformat, summin, summax, sumrms, 
                   summary256, summary64k, samples }   # Audio data
sqlite_sequence: { name, seq }             # Auto-increment tracker
```

### sampleformat Encoding

| Value | Encoding |
|-------|----------|
| 1 | 16-bit signed integer LE |
| 2 | 32-bit float LE |
| 3 | 24-bit signed integer LE |
| 262159 (0x4000F) | 32-bit float LE + compressed (most common in modern Audacity) |

### Audio Pipeline (inside AUP3)

1. Raw samples stored as zlib-compressed binary blobs
2. Decompress → numpy array
3. Cast to float32 if needed (normalize int16: `/ 32768.0`, int24: `/ 2147483648.0`)
4. Optional: `sampleformat & 0xFF == 15` indicates compressed blocks

## API Reference

### AUP3Reader

#### Constructor

```python
AUP3Reader(path: str | Path)
```

Opens the .aup3 file. Raises `FileNotFoundError` or `ValueError` on invalid format.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `sample_rate` | int | Default 16000 (M9 convention) |
| `n_tracks` | int | Number of audio tracks in project |
| `track_ids` | list[int] | Block IDs of all tracks |
| `duration` | float | Duration of longest track in seconds |

#### Methods

**`get_audio(track_id=None, normalize=True) -> np.ndarray`**
Returns mono float32 audio. If `track_id` is None, concatenates all tracks.

**`get_segments(track_id=None, silence_threshold=0.0, min_gap_seconds=0.3, min_segment_seconds=0.1) -> list[np.ndarray]`**
Splits audio by silence. Auto-estimates threshold from 10th percentile RMS if `silence_threshold=0`.

**`export_wav(path, audio=None, sr=None)`**
Writes mono 16-bit PCM WAV. If `audio` is None, exports all audio.

**`iter_segments(**kwargs) -> Iterator[SegmentInfo]`**
Yields `SegmentInfo(audio, start_frame, end_frame, duration, rms)` for each segment.

## Dependencies

- Python 3.10+
- numpy
- scipy (for WAV export and optional preprocessing)
- Standard library: sqlite3, zlib, json, wave, struct

## References

- `references/format-spec.md` — Detailed AUP3 binary format specification
- `references/audacity-source.md` — Links to Audacity source code for sampleblock encoding

## Scripts

- `scripts/aup3_read.py` — CLI tool for extracting/segmenting AUP3 files
- `scripts/aup3_info.py` — Inspect AUP3 file metadata without extracting audio
