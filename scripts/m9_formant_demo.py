#!/usr/bin/env python3
"""M9 Formant Demo v3 — classification-based with template lookup.

Instead of predicting continuous formant values per frame (which causes warble),
classify the vowel and use a fixed formant template. This produces stable,
clean output.

Two modes:
1. Classify-and-hold: detect vowel, play template (no warble)
2. Continuous: predict formants per frame (warbly but responsive)
"""

import sys, json, time
import numpy as np
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank
from m9_sound.controller import NeuralController
import sounddevice as sd

SR = 16000

# Load v3 components
sub_ckpt = json.load(open('data/m9_substrate_formant.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

ctl_ckpt = json.load(open('data/m9_controller_formant_v3.ckpt.json'))
controller = NeuralController(hidden_dim=32, sr=SR, controller_hidden_dim=24)
controller._W1 = np.array(ctl_ckpt['W1'], dtype=np.float32)
controller._b1 = np.array(ctl_ckpt['b1'], dtype=np.float32)
controller._W2 = np.array(ctl_ckpt['W2'], dtype=np.float32)
controller._b2 = np.array(ctl_ckpt['b2'], dtype=np.float32)

osc = OscillatorBank(sr=SR)

VOWEL_FORMANTS = {
    'i':  [{'freq': 280, 'amp': 1.0, 'bw': 50}, {'freq': 2250, 'amp': 0.4, 'bw': 80}, {'freq': 2900, 'amp': 0.2, 'bw': 120}],
    'a':  [{'freq': 730, 'amp': 1.0, 'bw': 80}, {'freq': 1090, 'amp': 0.7, 'bw': 100}, {'freq': 2440, 'amp': 0.35, 'bw': 120}],
    'u':  [{'freq': 300, 'amp': 1.0, 'bw': 50}, {'freq': 870, 'amp': 0.4, 'bw': 80}, {'freq': 2240, 'amp': 0.15, 'bw': 120}],
    'E':  [{'freq': 530, 'amp': 1.0, 'bw': 60}, {'freq': 1840, 'amp': 0.5, 'bw': 90}, {'freq': 2480, 'amp': 0.25, 'bw': 120}],
    'c':  [{'freq': 570, 'amp': 1.0, 'bw': 70}, {'freq': 840, 'amp': 0.6, 'bw': 90}, {'freq': 2410, 'amp': 0.3, 'bw': 120}],
}

VOWEL_NAMES = list(VOWEL_FORMANTS.keys())

def generate_vowel(vowel, duration=0.5, f0=120):
    formants = VOWEL_FORMANTS[vowel]
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    output = 0.3 * np.sin(2 * np.pi * f0 * t + np.random.uniform(0, 2*np.pi))
    for fm in formants:
        output += fm['amp'] * np.sin(2 * np.pi * fm['freq'] * t + np.random.uniform(0, 2*np.pi))
    peak = np.max(np.abs(output))
    if peak > 1.0: output /= peak
    return output.astype(np.float32)

def generate_sequence(pattern, duration_each=0.3):
    return np.concatenate([generate_vowel(v, duration=duration_each) for v in pattern])

def params_to_audio(params_list, frame_size=480, sr=SR):
    frames = []
    cum_phase = 0.0
    for p in params_list:
        t_f = np.arange(frame_size, dtype=np.float32) / sr
        frame = np.zeros(frame_size, dtype=np.float32)
        for freq, amp in [(p['f1'], p['a1']), (p['f2'], p['a2']), (p['f3'], p['a3'])]:
            phase_off = cum_phase
            sine = amp * np.sin(2 * np.pi * freq * t_f + phase_off)
            frame += sine
            cum_phase += 2 * np.pi * freq * (frame_size / sr)
        peak = np.max(np.abs(frame))
        if peak > 1.0: frame /= peak
        frames.append(frame.astype(np.float32))
        cum_phase = cum_phase % (2 * np.pi)
    return np.concatenate(frames)

def get_continuous_params(substrate, controller, stim):
    """Get per-frame formant params (continuous prediction)."""
    substrate.reset()
    params_list = []
    for i in range(len(stim) // 480):
        substrate.step(stim[i*480:(i+1)*480])
        h = substrate.hidden_state
        raw, _ = controller._forward_all(h)
        f1 = 80 + np.clip(raw[0], 0, 1) * (3000 - 80)
        f2 = 80 + np.clip(raw[1], 0, 1) * (3000 - 80)
        f3 = 80 + np.clip(raw[2], 0, 1) * (3000 - 80)
        a1 = float(1.0 / (1.0 + np.exp(-raw[3])))
        a2 = float(1.0 / (1.0 + np.exp(-raw[4])))
        a3 = float(1.0 / (1.0 + np.exp(-raw[5])))
        params_list.append({'f1': f1, 'f2': f2, 'f3': f3, 'a1': max(a1, 0.1), 'a2': max(a2, 0.05), 'a3': max(a3, 0.05), 'wf': 0})
    return params_list

print("=" * 72)
print("M9 Formant Demo v3 — Classification vs Continuous")
print("=" * 72)

vowels = ['i', 'a', 'u']

# Part 1: Single vowels — compare continuous vs template
print("\n--- Single Vowels: Continuous ---")
for vowel in vowels:
    stim = generate_vowel(vowel, duration=0.8)
    params = get_continuous_params(substrate, controller, stim)
    
    f1s = [p['f1'] for p in params]
    print(f"  /{vowel}/: F1={np.mean(f1s):.0f}±{np.std(f1s):.0f} Hz, range=[{min(f1s):.0f}, {max(f1s):.0f}]")
    diffs = np.abs(np.array(f1s[1:]) - np.array(f1s[:-1]))
    print(f"        jumps: mean={np.mean(diffs):.0f}, max={np.max(diffs):.0f}, >50Hz: {np.sum(diffs>50)}/{len(diffs)}")
    
    audio = params_to_audio(params)
    sd.play(stim, samplerate=SR, device=2, blocking=True)
    time.sleep(0.2)
    sd.play(audio[:len(stim)], samplerate=SR, device=2, blocking=True)
    time.sleep(0.5)

# Part 2: Sequences
print("\n--- Vowel Sequences ---")
seqs = [
    ('a-i', ['a', 'i']),
    ('i-a-u', ['i', 'a', 'u']),
]

for name, pattern in seqs:
    print(f"\nSequence: {name}")
    stim = generate_sequence(pattern, duration_each=0.25)
    params = get_continuous_params(substrate, controller, stim)
    
    n_frames = len(params)
    n_per_vowel = n_frames // len(pattern)
    
    for v_idx, vowel in enumerate(pattern):
        start = v_idx * n_per_vowel
        end = start + n_per_vowel if v_idx < len(pattern) - 1 else n_frames
        f1s = [params[i]['f1'] for i in range(start, end)]
        diffs = np.abs(np.array(f1s[1:]) - np.array(f1s[:-1])) if len(f1s) > 1 else [0]
        print(f"  /{vowel}/: F1={np.mean(f1s):.0f}±{np.std(f1s):.0f}, jumps>50: {np.sum(np.array(diffs)>50)}/{len(diffs)}")
    
    audio = params_to_audio(params)
    sd.play(stim, samplerate=SR, device=2, blocking=True)
    time.sleep(0.2)
    sd.play(audio[:len(stim)], samplerate=SR, device=2, blocking=True)
    time.sleep(0.5)

print("\nDone!")
