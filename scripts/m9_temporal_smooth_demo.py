#!/usr/bin/env python3
"""M9 Smooth Temporal Demo — segment-based frequency interpolation.

Detects tone boundaries from controller output, then smoothly interpolates
between target frequencies. This reduces the "many tones" effect by
creating cleaner transitions.
"""

import sys; sys.path.insert(0, 'packages')
import json, numpy as np
from m9_sound import AcousticSubstrate, OscillatorBank, NeuralController
import sounddevice as sd

SR = 16000

# Load controller
ckpt = json.load(open('data/m9_controller_temporal_v3.ckpt.json'))
controller = NeuralController(hidden_dim=32, sr=SR, controller_hidden_dim=8)
controller._W1 = np.array(ckpt['W1'], dtype=np.float32)
controller._b1 = np.array(ckpt['b1'], dtype=np.float32)
controller._W2 = np.array(ckpt['W2'], dtype=np.float32)
controller._b2 = np.array(ckpt['b2'], dtype=np.float32)

# Load live substrate
sub_ckpt = json.load(open('data/m9_substrate_live.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

osc = OscillatorBank(sr=SR)
import time

def generate_sequence(tones):
    parts = []
    for freq, dur in tones:
        t = np.arange(int(SR*dur), dtype=np.float32)/SR
        phase = np.random.uniform(0, 2*np.pi)
        parts.append((0.5*np.sin(2*np.pi*freq*t + phase)).astype(np.float32))
    return np.concatenate(parts)

def generate_smooth_imitation(tones, crossfade_frames=3):
    """Generate imitation with smooth frequency interpolation between tone segments."""
    stim = generate_sequence(tones)
    frame_size = 480
    
    # Encode through substrate and get per-frame params
    substrate.reset()
    params_list = []
    for i in range(len(stim)//frame_size):
        substrate.step(stim[i*frame_size:(i+1)*frame_size])
        params_list.append(controller.decode(substrate.hidden_state))
    
    # Detect tone segments from frequency changes
    freqs = [p.freq_hz for p in params_list]
    segments = []
    current_start = 0
    current_freq = np.mean(freqs[:3])  # average first few frames
    
    for i in range(1, len(freqs)):
        # Check if frequency has shifted significantly
        recent_mean = np.mean(freqs[max(0,i-2):i+1])
        if abs(recent_mean - current_freq) > 40:  # threshold for new segment
            segments.append((current_start, i, current_freq))
            current_start = i
            current_freq = recent_mean
    segments.append((current_start, len(freqs), current_freq))
    
    # Merge segments that are too short (< 3 frames)
    merged = [segments[0]]
    for start, end, freq in segments[1:]:
        if end - start < 3:
            # Merge with previous
            prev_start, prev_end, prev_freq = merged[-1]
            merged[-1] = (prev_start, end, (prev_freq + freq) / 2)
        else:
            merged.append((start, end, freq))
    segments = merged
    
    # Generate audio with smooth interpolation between segments
    total_frames = len(params_list)
    output = np.zeros(total_frames * frame_size, dtype=np.float32)
    cum_phase = 0.0
    
    for frame_idx in range(total_frames):
        # Find which segment this frame belongs to
        seg_freq = segments[0][2]
        next_seg_freq = seg_freq
        seg_progress = 0.0
        
        for seg_idx, (start, end, freq) in enumerate(segments):
            if start <= frame_idx < end:
                seg_freq = freq
                seg_size = end - start
                seg_progress = (frame_idx - start) / max(seg_size, 1)
                # Look ahead to next segment
                if seg_idx + 1 < len(segments):
                    next_seg_freq = segments[seg_idx + 1][2]
                break
        
        # Apply crossfade near segment boundaries
        freq = seg_freq
        if seg_progress > (1 - crossfade_frames / max(segments[0][1]-segments[0][0], 1)):
            # In crossfade zone — interpolate to next segment's frequency
            fade = (seg_progress - (1 - crossfade_frames)) / max(crossfade_frames, 1)
            freq = seg_freq + fade * (next_seg_freq - seg_freq)
        
        t_f = np.arange(frame_size, dtype=np.float32) / SR
        frame = 0.5 * np.sin(2*np.pi*freq*t_f + cum_phase)
        output[frame_idx*frame_size:(frame_idx+1)*frame_size] = frame.astype(np.float32)
        cum_phase += 2*np.pi*freq*(frame_size/SR)
        cum_phase = cum_phase % (2*np.pi)
    
    return output[:len(stim)], segments, freqs

demos = [
    ("2-tone up", [(220, 0.3), (440, 0.3)]),
    ("2-tone down", [(440, 0.3), (220, 0.3)]),
    ("3-tone up", [(220, 0.2), (330, 0.2), (440, 0.2)]),
    ("up-down", [(330, 0.2), (550, 0.2), (330, 0.2)]),
]

print("=" * 72)
print("M9 Smooth Temporal Demo — Segment Interpolation")
print("=" * 72)

for name, tones in demos:
    print(f"\nDemo: {name}")
    stim = generate_sequence(tones)
    imit, segments, raw_freqs = generate_smooth_imitation(tones)
    print(f"  Detected segments: {[(s, e, f'{f:.0f}Hz') for s, e, f in segments]}")
    print(f"  Raw freq range: [{min(raw_freqs):.0f}, {max(raw_freqs):.0f}] Hz")
    print(f"  Playing stimulus...")
    sd.play(stim, samplerate=SR, device=2, blocking=True)
    time.sleep(0.3)
    print(f"  Playing smooth imitation...")
    sd.play(imit, samplerate=SR, device=2, blocking=True)
    time.sleep(0.5)

print("\nDone!")
