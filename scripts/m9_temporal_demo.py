#!/usr/bin/env python3
"""M9 Temporal Live Demo — play back controller output without feedback loop.

The controller was trained on stimulus-encoded hidden states.
For imitation, we reuse those same params in sequence — no feedback loop.
This avoids the garbled output from unstable controller-substrate feedback.
"""

import sys, json
import numpy as np
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

# Load live substrate
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

def get_controller_params_for_sequence(producer, seq, frame_size=480):
    """Run controller on stimulus-encoded hidden states, return per-frame params."""
    producer.substrate.reset()
    all_params = []
    for i in range(len(seq) // frame_size):
        frame = seq[i*frame_size:(i+1)*frame_size]
        producer.substrate.step(frame)
        h = producer.substrate.hidden_state
        params = producer.controller.decode(h)
        all_params.append(params)
    return all_params

def play_params_as_audio(params_list, frame_size=480, sr=SR):
    """Convert controller params to audio frames and play.
    Uses cumulative phase for smooth frame transitions.
    """
    frames = []
    cum_phase = 0.0
    for params in params_list:
        t_f = np.arange(frame_size, dtype=np.float32) / sr
        phase_off = cum_phase + params.phase
        frame = params.amplitude * np.sin(2*np.pi*params.freq_hz*t_f + phase_off)
        frames.append(frame.astype(np.float32))
        cum_phase += 2 * np.pi * params.freq_hz * (frame_size / sr)
        cum_phase = cum_phase % (2 * np.pi)
    audio = np.concatenate(frames)
    sd.play(audio, samplerate=sr, device=2, blocking=True)

print("=" * 72)
print("M9 Temporal Live Demo — Clean Playback")
print("=" * 72)
print()

demos = [
    ("2-tone up", [(220, 0.3), (440, 0.3)]),
    ("2-tone down", [(440, 0.3), (220, 0.3)]),
    ("3-tone up", [(220, 0.2), (330, 0.2), (440, 0.2)]),
    ("3-tone down", [(440, 0.2), (330, 0.2), (220, 0.2)]),
    ("up-down", [(330, 0.2), (550, 0.2), (330, 0.2)]),
    ("low-high-low", [(220, 0.15), (550, 0.15), (220, 0.15)]),
]

import time

for name, tones in demos:
    print(f"Demo: {name}")
    
    # Generate stimulus
    stim = generate_sequence(tones)
    
    # Get controller params for this sequence
    params = get_controller_params_for_sequence(producer, stim)
    
    # Print what the controller produced
    freqs = [p.freq_hz for p in params]
    print(f"  Stimulus: {' -> '.join(f'{f} Hz' for f, d in tones)}")
    print(f"  Imitation: {len(params)} frames, freq range [{min(freqs):.0f}, {max(freqs):.0f}] Hz")
    
    # Play stimulus
    print(f"  Playing stimulus...")
    sd.play(stim, samplerate=SR, device=2, blocking=True)
    time.sleep(0.3)
    
    # Play imitation (controller output, no feedback loop)
    print(f"  Playing imitation...")
    play_params_as_audio(params)
    time.sleep(0.5)
    print()

print("Done!")
