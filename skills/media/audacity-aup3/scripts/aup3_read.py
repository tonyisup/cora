#!/usr/bin/env python3
"""AUP3 Reader — extract audio from Audacity .aup3 project files.

Usage:
    python aup3_read.py input.aup3
    python aup3_read.py input.aup3 -o output.wav
    python aup3_read.py input.aup3 --segment --min-gap 0.3
"""

import sys, json, zlib, wave, struct, argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SegmentInfo:
    audio: np.ndarray
    start_frame: int
    end_frame: int
    duration: float
    rms: float


class AUP3Reader:
    """Read and extract audio from Audacity .aup3 project files."""
    
    def __init__(self, path, sr=16000):
        self.path = Path(path)
        self.sr = sr
        self._conn = None
        self._tracks = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def open(self):
        import sqlite3
        self._conn = sqlite3.connect(self.path)
        self._load_tracks()
    
    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def _load_tracks(self):
        cur = self._conn.cursor()
        cur.execute("SELECT blockid, sampleformat, length(samples) FROM sampleblocks;")
        self._tracks = cur.fetchall()
    
    def _ensure_loaded(self):
        if self._tracks is None:
            self.open()
    
    @property
    def n_tracks(self):
        self._ensure_loaded()
        return len(self._tracks) if self._tracks else 0
    
    @property
    def track_ids(self):
        self._ensure_loaded()
        return [t[0] for t in self._tracks] if self._tracks else []
    
    @property
    def duration(self):
        self._ensure_loaded()
        if not self._tracks:
            return 0.0
        return max(t[2] / self.sr for t in self._tracks if t[2] > 0)
    
    def _decompress_samples(self, blob):
        """Decompress and decode sample blob to float32."""
        try:
            raw = zlib.decompress(blob)
        except zlib.error:
            raw = blob
        
        # Try float32 first (most common)
        n_samples = len(raw) // 4
        if n_samples > 0:
            audio = np.frombuffer(raw, dtype=np.float32).copy()
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                raise ValueError("Invalid float32 data")
            return audio
        
        # Fallback: int16
        n_samples = len(raw) // 2
        if n_samples > 0:
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Fallback: int24
        n_samples = len(raw) // 3
        if n_samples > 0:
            padded = np.zeros(len(raw) + 4 - (len(raw) % 4), dtype=np.uint8)
            padded[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
            audio32 = np.frombuffer(padded.tobytes(), dtype=np.int32)[:n_samples]
            return audio32.astype(np.float32) / 2147483648.0
        
        return np.zeros(0, dtype=np.float32)
    
    def get_track(self, blockid):
        """Get audio for a specific track by block ID."""
        cur = self._conn.cursor()
        cur.execute("SELECT samples FROM sampleblocks WHERE blockid=?;", (blockid,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Track {blockid} not found")
        return self._decompress_samples(row[0])
    
    def get_audio(self, track_id=None):
        """Get audio. If track_id is None, concatenates all tracks."""
        self.open()
        try:
            if track_id is not None:
                return self.get_track(track_id)
            segments = []
            for blockid, _, _ in self._tracks:
                segments.append(self.get_track(blockid))
            if not segments:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(segments)
        finally:
            self.close()
    
    def _estimate_silence_rms(self, audio, frame_size=480):
        """Auto-estimate silence threshold from 10th percentile frame RMS."""
        n_frames = len(audio) // frame_size
        if n_frames == 0:
            return 0.001
        rms_values = []
        for i in range(n_frames):
            frame = audio[i*frame_size:(i+1)*frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            rms_values.append(rms)
        return float(np.percentile(rms_values, 10))
    
    def get_segments(self, track_id=None, silence_threshold=0.0, 
                     min_gap_seconds=0.3, min_segment_seconds=0.1):
        """Split audio by silence into segments."""
        audio = self.get_audio(track_id)
        frame_size = 480
        n_frames = len(audio) // frame_size
        if n_frames == 0:
            return []
        
        # Compute frame RMS
        rms_values = []
        for i in range(n_frames):
            frame = audio[i*frame_size:(i+1)*frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            rms_values.append(rms)
        
        # Auto-estimate silence threshold
        if silence_threshold == 0.0:
            silence_threshold = self._estimate_silence_rms(audio, frame_size)
            silence_threshold *= 3.0
        
        min_gap_frames = int(min_gap_seconds * self.sr / frame_size)
        min_seg_frames = int(min_segment_seconds * self.sr / frame_size)
        
        # Find segments
        segments = []
        in_seg = False
        seg_start = 0
        silent_count = 0
        
        for i in range(n_frames):
            if rms_values[i] < silence_threshold:
                if in_seg:
                    silent_count += 1
                    if silent_count >= min_gap_frames:
                        seg_end = i - silent_count + 1
                        if seg_end - seg_start >= min_seg_frames:
                            seg_audio = audio[seg_start*frame_size:seg_end*frame_size]
                            seg_rms = float(np.sqrt(np.mean(seg_audio ** 2)))
                            segments.append(seg_audio)
                        in_seg = False
            else:
                if not in_seg:
                    seg_start = i
                    in_seg = True
                silent_count = 0
        
        if in_seg:
            seg_end = n_frames
            if seg_end - seg_start >= min_seg_frames:
                seg_audio = audio[seg_start*frame_size:seg_end*frame_size]
                segments.append(seg_audio)
        
        return segments
    
    def iter_segments(self, **kwargs):
        """Yield SegmentInfo for each silence-separated segment."""
        audio = self.get_audio(kwargs.get('track_id'))
        frame_size = 480
        segments = self.get_segments(**kwargs)
        
        pos = 0
        for seg_audio in segments:
            n_frames = len(seg_audio) // frame_size
            rms = float(np.sqrt(np.mean(seg_audio ** 2)))
            duration = len(seg_audio) / self.sr
            yield SegmentInfo(
                audio=seg_audio,
                start_frame=pos,
                end_frame=pos + n_frames,
                duration=duration,
                rms=rms,
            )
            pos += n_frames
    
    def export_wav(self, path, audio=None, sr=None):
        """Export audio to mono 16-bit PCM WAV."""
        sr = sr or self.sr
        if audio is None:
            audio = self.get_audio()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio_int16 = audio.astype(np.int16)
        with wave.open(str(path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Extract audio from Audacity .aup3 files")
    parser.add_argument("input", type=Path, help="Input .aup3 file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output WAV path")
    parser.add_argument("--segment", action="store_true", help="Split by silence")
    parser.add_argument("--min-gap", type=float, default=0.3, help="Min silence gap (s)")
    parser.add_argument("--min-dur", type=float, default=0.1, help="Min segment duration (s)")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--track", type=int, default=None, help="Specific track blockid")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input.with_suffix('.wav')
    
    reader = AUP3Reader(args.input, sr=args.sr)
    
    if args.segment:
        segments = reader.get_segments(
            track_id=args.track,
            min_gap_seconds=args.min_gap,
            min_segment_seconds=args.min_dur,
        )
        print(f"Found {len(segments)} segments")
        for i, seg in enumerate(segments):
            seg_path = args.output.parent / f"{args.output.stem}_seg{i+1:02d}.wav"
            reader.export_wav(seg_path, seg)
            dur = len(seg) / args.sr
            rms = float(np.sqrt(np.mean(seg ** 2)))
            print(f"  {seg_path.name}: {dur:.2f}s, rms={rms:.4f}")
    else:
        audio = reader.get_audio(track_id=args.track)
        reader.export_wav(args.output, audio)
        dur = len(audio) / args.sr
        print(f"Exported: {args.output} ({dur:.2f}s)")


if __name__ == "__main__":
    main()
