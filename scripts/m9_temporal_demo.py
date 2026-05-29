#!/usr/bin/env python3
"""M9 Temporal Live Demo — smooth multi-tone imitation.

Demonstrates the controller's ability to produce smooth, time-varying
oscillator params that match multi-tone stimulus sequences.
"""

import sys, json
import numpy as np
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank, AcousticProducer
import sounddevice as sd

SR = 16000

# Load temporal controller v3 + live substrate
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

def play_and_listen(pcm, label="", out_dev=2, in_dev=1):
    """Play audio on output, capture on input. Returns capture."""
    sd.play(pcm.astype(np.float32), samplerate=SR, device=out_dev, blocking=True)
    return None  # we don't need to capture for demo

def generate_sequence(tones, sr=SR):
    """Generate a multi-tone stimulus sequence."""
    parts = []
    for freq, dur in tones:
        t = np.arange(int(sr * dur), dtype=np.float32) / sr
        phase = np.random.uniform(0, 2*np.pi)
        parts.append((0.5 * np.sin(2*np.pi*freq*t + phase)).astype(np.float32))
    return np.concatenate(parts)

def generate_imitation(producer, n_frames_per_tone, n_tones, substrate):
    """Generate imitation frame by frame using controller."""
    sub = AcousticSubstrate(hidden_dim=32, sr=SR, frame_size=480)
    sub._W_h = substrate._W_h.copy()
    sub._W_x = substrate._W_x.copy()
    sub._b = substrate._b.copy()
    sub.reset()
    
    output_frames = []
    for tone_idx in range(n_tones):
        for f in range(n_frames_per_tone):
            h = sub.hidden_state
            params = producer.controller.decode(h)
            t_f = np.arange(480, dtype=np.float32) / SR
            frame = params.amplitude * np.sin(2*np.pi*params.freq_hz*t_f + params.phase)
            frame = frame.astype(np.float32)
            output_frames.append(frame)
            sub.step(frame)
    
    return np.concatenate(output_frames)

# Pre-encode stimulus sequences into substrate
def encode_stimulus(seq):
    """Feed stimulus through substrate, return final state."""
    sub = AcousticSubstrate(hidden_dim=32, sr=SR, frame_size=480)
    sub._W_h = producer.substrate._W_h.copy()
    sub._W_x = producer.substrate._W_x.copy()
    sub._b = producer.substrate._b.copy()
    sub.reset()
    for i in range(len(seq) // 480):
        sub.step(seq[i*480:(i+1)*480])
    return sub

print("=" * 72)
print("M9 Temporal Live Demo — Smooth Multi-Tone Imitation")
print("=" * 72)
print()

# Demo 1: 2-tone up
seq1 = generate_sequence([(220, 0.3), (440, 0.3)])
sub1 = encode_stimulus(seq1)
imit1 = generate_imitation(producer, 10, 2, sub1)

print("Demo 1: 2-tone ascending (220 -> 440 Hz)")
print("  Playing stimulus...")
sd.play(seq1, samplerate=SR, device=2, blocking=True)
print("  Playing imitation...")
sd.play(imit1[:len(seq1)], samplerate=SR, device=2, blocking=True)

import time
time.sleep(0.5)

# Demo 2: 2-tone down
seq2 = generate_sequence([(440, 0.3), (220, 0.3)])
sub2 = encode_stimulus(seq2)
imit2 = generate_imitation(producer, 10, 2, sub2)

print("Demo 2: 2-tone descending (440 -> 220 Hz)")
print("  Playing stimulus...")
sd.play(seq2, samplerate=SR, device=2, blocking=True)
print("  Playing imitation...")
sd.play(imit2[:len(seq2)], samplerate=SR, device=2, blocking=True)
time.sleep(0.5)

# Demo 3: 3-tone
seq3 = generate_sequence([(220, 0.2), (330, 0.2), (440, 0.2)])
sub3 = encode_stimulus(seq3)
imit3 = generate_imitation(producer, 7, 3, sub3)

print("Demo 3: 3-tone ascending (220 -> 330 -> 440 Hz)")
print("  Playing stimulus...")
sd.play(seq3, samplerate=SR, device=2, blocking=True)
print("  Playing imitation...")
sd.play(imit3[:len(seq3)], samplerate=SR, device=2, blocking=True)
time.sleep(0.5)

# Demo 4: up-down
seq4 = generate_sequence([(330, 0.2), (550, 0.2), (330, 0.2)])
sub4 = encode_stimulus(seq4)
imit4 = generate_imitation(producer, 7, 3, sub4)

print("Demo 4: up-down pattern (330 -> 550 -> 330 Hz)")
print("  Playing stimulus...")
sd.play(seq4, samplerate=SR, device=2, blocking=True)
print("  Playing imitation...")
sd.play(imit4[:len(seq4)], samplerate=SR, device=2, blocking=True)

print("\nDone! Each demo plays stimulus then imitation.")
print("The imitation should match the pitch pattern of the stimulus.")
