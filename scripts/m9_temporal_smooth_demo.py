#!/usr/bin/env python3
"""M9 Smooth Temporal Demo — with frequency smoothing.

Applies exponential moving average to controller output to reduce
frame-to-frame jitter. Makes the tonal transitions sound smoother
and more like natural articulation.
"""

import sys, json
import numpy as np
from scipy.signal import butter, filtfilt
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank, AcousticProducer
import sounddevice as sd

SR = 16000

# Load temporal controller v3
producer = AcousticProducer(sr=SR, hidden_dim=32)
ckpt = json.load(open('data/m9_controller_temporal_v3.ckpt.json'))
producer.controller._W1 = np.array(ckpt['W1'], dtype=np.float32)
producer.controller._b1 = np.array(ckpt['b1'], dtype=np.float32)
producer.controller._W2 = np.array(ckpt['W2'], dtype=np.float32)
producer.controller._b2 = np.array(ckpt['b2'], dtype=np.float32)

sub_ckpt = json.load(open('data/m9_substrate_live.ckpt.json'))
producer.substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
producer.substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
producer.substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

osc = OscillatorBank(sr=SR)

def generate_sequence(tones, sr=SR):
    parts = []
    for freq, dur in tones:
        t = np.arange(int(sr * dur), dtype=np.float32) / sr
        phase = np.random.uniform(0, 2*np.pi)
        parts.append((0.5 * np.sin(2*np.pi*freq*t + phase)).astype(np.float32))
    return np.concatenate(parts)

def get_controller_params(producer, seq, frame_size=480):
    """Get per-frame controller params from stimulus-encoded hidden states."""
    producer.substrate.reset()
    params_list = []
    for i in range(len(seq) // frame_size):
        producer.substrate.step(seq[i*frame_size:(i+1)*frame_size])
        params_list.append(producer.controller.decode(producer.substrate.hidden_state))
    return params_list

def smooth_params(params_list, alpha=0.3):
    """Apply exponential moving average to smooth frequency transitions.
    alpha=0.3 means ~30% new value, 70% previous — smooths over ~3 frames.
    """
    smoothed = []
    prev_freq = params_list[0].freq_hz
    prev_amp = params_list[0].amplitude
    prev_phase = params_list[0].phase
    
    for p in params_list:
        freq = alpha * p.freq_hz + (1-alpha) * prev_freq
        amp = alpha * p.amplitude + (1-alpha) * prev_amp
        phase = alpha * p.phase + (1-alpha) * prev_phase
        from m9_sound.controller import ControllerParams
        smoothed.append(ControllerParams(freq_hz=freq, amplitude=amp, phase=phase, waveform=p.waveform))
        prev_freq, prev_amp, prev_phase = freq, amp, phase
    
    return smoothed

def params_to_audio(params_list, frame_size=480, sr=SR):
    """Convert params list to audio."""
    frames = []
    for params in params_list:
        t_f = np.arange(frame_size, dtype=np.float32) / sr
        frame = params.amplitude * np.sin(2*np.pi*params.freq_hz*t_f + params.phase)
        frames.append(frame.astype(np.float32))
    return np.concatenate(frames)

# Comparison: raw vs smoothed
print("=" * 72)
print("M9 Temporal Demo — Raw vs Smoothed Comparison")
print("=" * 72)
print()

demos = [
    ("2-tone up", [(220, 0.3), (440, 0.3)]),
    ("3-tone", [(220, 0.2), (330, 0.2), (440, 0.2)]),
]

import time

for name, tones in demos:
    print(f"Demo: {name}")
    stim = generate_sequence(tones)
    raw_params = get_controller_params(producer, stim)
    
    # Show raw frequencies
    raw_freqs = [p.freq_hz for p in raw_params]
    print(f"  Raw freq range: [{min(raw_freqs):.0f}, {max(raw_freqs):.0f}] Hz, std={np.std(raw_freqs):.1f}")
    
    # Smooth with different alphas
    for alpha in [0.5, 0.3, 0.1]:
        smooth = smooth_params(raw_params, alpha=alpha)
        smooth_freqs = [p.freq_hz for p in smooth]
        print(f"  Smoothed (α={alpha}): [{min(smooth_freqs):.0f}, {max(smooth_freqs):.0f}] Hz, std={np.std(smooth_freqs):.1f}")
    
    # Play stimulus
    print(f"  Playing stimulus...")
    sd.play(stim, samplerate=SR, device=2, blocking=True)
    time.sleep(0.3)
    
    # Play raw imitation
    print(f"  Playing raw imitation...")
    raw_audio = params_to_audio(raw_params)
    sd.play(raw_audio[:len(stim)], samplerate=SR, device=2, blocking=True)
    time.sleep(0.3)
    
    # Play smoothed imitation (alpha=0.3)
    print(f"  Playing smoothed imitation (α=0.3)...")
    smooth = smooth_params(raw_params, alpha=0.3)
    smooth_audio = params_to_audio(smooth)
    sd.play(smooth_audio[:len(stim)], samplerate=SR, device=2, blocking=True)
    time.sleep(0.3)
    
    # Play heavily smoothed (alpha=0.1)
    print(f"  Playing heavily smoothed (α=0.1)...")
    smooth2 = smooth_params(raw_params, alpha=0.1)
    smooth_audio2 = params_to_audio(smooth2)
    sd.play(smooth_audio2[:len(stim)], samplerate=SR, device=2, blocking=True)
    time.sleep(0.5)
    print()

print("Done!")
