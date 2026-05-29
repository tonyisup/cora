#!/usr/bin/env python3
"""M9 Temporal Demo with noise-filtered input.

Applies high-pass filter to captured audio before encoding through substrate.
This removes low-frequency room noise that was confusing the controller.
"""

import sys, json
import numpy as np
from scipy.signal import butter, sosfilt
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank, AcousticProducer
import sounddevice as sd

SR = 16000

# Load temporal controller + live substrate
producer = AcousticProducer(sr=SR, hidden_dim=32)
ckpt = json.load(open('data/m9_controller_temporal_v1.ckpt.json'))
producer.controller._W1 = np.array(ckpt['W1'], dtype=np.float32)
producer.controller._b1 = np.array(ckpt['b1'], dtype=np.float32)
producer.controller._W2 = np.array(ckpt['W2'], dtype=np.float32)
producer.controller._b2 = np.array(ckpt['b2'], dtype=np.float32)

sub_ckpt = json.load(open('data/m9_substrate_live.ckpt.json'))
producer.substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
producer.substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
producer.substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

osc = OscillatorBank(sr=SR)

# High-pass filter to remove room noise below 100 Hz
def highpass_filter(signal, cutoff=100, sr=SR, order=4):
    """Remove low-frequency room noise."""
    nyq = sr / 2
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', output='sos')
    return sosfilt(sos, signal).astype(np.float32)

def generate_sequence(tones, sr=SR):
    parts = []
    for freq, dur in tones:
        t = np.arange(int(sr * dur), dtype=np.float32) / sr
        phase = np.random.uniform(0, 2*np.pi)
        parts.append((0.5 * np.sin(2*np.pi*freq*t + phase)).astype(np.float32))
    return np.concatenate(parts)

def get_params(producer, seq, frame_size=480):
    producer.substrate.reset()
    params = []
    for i in range(len(seq) // frame_size):
        raw_frame = seq[i*frame_size:(i+1)*frame_size]
        # Filter each frame before feeding to substrate
        filtered = highpass_filter(raw_frame)
        producer.substrate.step(filtered)
        params.append(producer.controller.decode(producer.substrate.hidden_state))
    return params

def smooth_params(params_list, alpha=0.3):
    from m9_sound.controller import ControllerParams
    smoothed = []
    prev_f = params_list[0].freq_hz
    prev_a = params_list[0].amplitude
    prev_p = params_list[0].phase
    for p in params_list:
        f = alpha * p.freq_hz + (1-alpha) * prev_f
        a = alpha * p.amplitude + (1-alpha) * prev_a
        ph = alpha * p.phase + (1-alpha) * prev_p
        smoothed.append(ControllerParams(freq_hz=f, amplitude=a, phase=ph, waveform=p.waveform))
        prev_f, prev_a, prev_p = f, a, ph
    return smoothed

def params_to_audio(params_list, frame_size=480, sr=SR):
    frames = []
    for p in params_list:
        t_f = np.arange(frame_size, dtype=np.float32) / sr
        frame = (p.amplitude * np.sin(2*np.pi*p.freq_hz*t_f + p.phase)).astype(np.float32)
        frames.append(frame)
    return np.concatenate(frames)

import time

print("=" * 72)
print("M9 Temporal Demo — with noise filtering")
print("=" * 72)

demos = [
    ("2-tone 220->440", [(220, 0.3), (440, 0.3)]),
    ("2-tone 440->220", [(440, 0.3), (220, 0.3)]),
    ("3-tone up", [(220, 0.2), (330, 0.2), (440, 0.2)]),
    ("3-tone down", [(440, 0.2), (330, 0.2), (220, 0.2)]),
    ("up-down", [(330, 0.2), (550, 0.2), (330, 0.2)]),
]

for name, tones in demos:
    print(f"\nDemo: {name}")
    
    # Generate stimulus
    stim = generate_sequence(tones)
    
    # Get controller params (with noise filtering)
    raw_params = get_params(producer, stim)
    raw_freqs = [p.freq_hz for p in raw_params]
    print(f"  Raw params: [{min(raw_freqs):.0f}, {max(raw_freqs):.0f}] Hz, mean={np.mean(raw_freqs):.0f}")
    
    # Smooth
    smooth = smooth_params(raw_params, alpha=0.3)
    smooth_freqs = [p.freq_hz for p in smooth]
    print(f"  Smoothed:   [{min(smooth_freqs):.0f}, {max(smooth_freqs):.0f}] Hz, mean={np.mean(smooth_freqs):.0f}")
    
    # Play stimulus
    print(f"  Playing stimulus...")
    sd.play(stim, samplerate=SR, device=2, blocking=True)
    time.sleep(0.2)
    
    # Play imitation (smoothed, with noise filtering)
    print(f"  Playing imitation...")
    imit_audio = params_to_audio(smooth)
    sd.play(imit_audio[:len(stim)], samplerate=SR, device=2, blocking=True)
    time.sleep(0.3)

print("\nDone!")
