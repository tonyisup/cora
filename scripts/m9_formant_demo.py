#!/usr/bin/env python3
"""M9 Formant Live Demo — vowel imitation on hardware.

Plays vowel stimuli and their formant-based imitations through headphones.
"""

import sys, json, time
import numpy as np
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank
from m9_sound.controller import NeuralController
import sounddevice as sd

SR = 16000
frame_size = 480

# Load formant substrate
sub_ckpt = json.load(open('data/m9_substrate_formant.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)
print(f"Loaded formant substrate (acc was {sub_ckpt.get('final_accuracy', 'N/A')})")

# Load formant controller
ctl_ckpt = json.load(open('data/m9_controller_formant_v1.ckpt.json'))
controller = NeuralController(hidden_dim=32, sr=SR, controller_hidden_dim=16)
controller._W1 = np.array(ctl_ckpt['W1'], dtype=np.float32)
controller._b1 = np.array(ctl_ckpt['b1'], dtype=np.float32)
controller._W2 = np.array(ctl_ckpt['W2'], dtype=np.float32)
controller._b2 = np.array(ctl_ckpt['b2'], dtype=np.float32)
print(f"Loaded formant controller (final loss: {ctl_ckpt.get('final_loss', 'N/A')})")

osc = OscillatorBank(sr=SR)

VOWEL_FORMANTS = {
    'i':  [{'freq': 280, 'amp': 1.0, 'bw': 50}, {'freq': 2250, 'amp': 0.4, 'bw': 80}, {'freq': 2900, 'amp': 0.2, 'bw': 120}],
    'I':  [{'freq': 360, 'amp': 1.0, 'bw': 50}, {'freq': 2220, 'amp': 0.35, 'bw': 80}, {'freq': 2960, 'amp': 0.15, 'bw': 120}],
    'E':  [{'freq': 530, 'amp': 1.0, 'bw': 60}, {'freq': 1840, 'amp': 0.5, 'bw': 90}, {'freq': 2480, 'amp': 0.25, 'bw': 120}],
    'ae': [{'freq': 660, 'amp': 1.0, 'bw': 70}, {'freq': 1720, 'amp': 0.6, 'bw': 100}, {'freq': 2410, 'amp': 0.3, 'bw': 120}],
    'a':  [{'freq': 730, 'amp': 1.0, 'bw': 80}, {'freq': 1090, 'amp': 0.7, 'bw': 100}, {'freq': 2440, 'amp': 0.35, 'bw': 120}],
    'c':  [{'freq': 570, 'amp': 1.0, 'bw': 70}, {'freq': 840, 'amp': 0.6, 'bw': 90}, {'freq': 2410, 'amp': 0.3, 'bw': 120}],
    'U':  [{'freq': 440, 'amp': 1.0, 'bw': 60}, {'freq': 1020, 'amp': 0.5, 'bw': 90}, {'freq': 2240, 'amp': 0.2, 'bw': 120}],
    'u':  [{'freq': 300, 'amp': 1.0, 'bw': 50}, {'freq': 870, 'amp': 0.4, 'bw': 80}, {'freq': 2240, 'amp': 0.15, 'bw': 120}],
    'V':  [{'freq': 640, 'amp': 1.0, 'bw': 70}, {'freq': 1190, 'amp': 0.6, 'bw': 100}, {'freq': 2390, 'amp': 0.3, 'bw': 120}],
    'r':  [{'freq': 490, 'amp': 1.0, 'bw': 60}, {'freq': 1350, 'amp': 0.5, 'bw': 90}, {'freq': 1690, 'amp': 0.4, 'bw': 100}],
}

def generate_vowel(vowel, duration=0.5, f0=120):
    formants = VOWEL_FORMANTS[vowel]
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    output = 0.3 * np.sin(2 * np.pi * f0 * t)
    for fm in formants:
        output += fm['amp'] * np.sin(2 * np.pi * fm['freq'] * t)
    peak = np.max(np.abs(output))
    if peak > 1.0: output /= peak
    return output.astype(np.float32)

def get_formant_params(producer, seq):
    """Get per-frame formant params from substrate + controller."""
    substrate.reset()
    params_list = []
    for i in range(len(seq) // frame_size):
        substrate.step(seq[i*frame_size:(i+1)*frame_size])
        h = substrate.hidden_state
        raw, _ = controller._forward_all(h)
        
        f1 = 80 + np.clip(raw[0], 0, 1) * (3000 - 80)
        f2 = 80 + np.clip(raw[1], 0, 1) * (3000 - 80)
        f3 = 80 + np.clip(raw[2], 0, 1) * (3000 - 80)
        a1 = float(1.0 / (1.0 + np.exp(-raw[3])))
        a2 = float(1.0 / (1.0 + np.exp(-raw[4])))
        a3 = float(1.0 / (1.0 + np.exp(-raw[5])))
        
        params_list.append({'f1': f1, 'f2': f2, 'f3': f3, 'a1': max(a1, 0.1), 'a2': max(a2, 0.05), 'a3': max(a3, 0.05)})
    return params_list

def params_to_formant_audio(params_list, frame_size=480, sr=SR):
    """Convert formant params to audio using formant synthesis."""
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
    
    return np.concatenate(frames)

print("=" * 72)
print("M9 Formant Live Demo")
print("=" * 72)

vowels = ['i', 'a', 'u', 'E', 'c']

print("\n--- Single Vowels ---")
for vowel in vowels:
    stim = generate_vowel(vowel, duration=0.5)
    params = get_formant_params(None, stim)
    imit = params_to_formant_audio(params)
    
    avg_f1 = np.mean([p['f1'] for p in params])
    avg_f2 = np.mean([p['f2'] for p in params])
    print(f"  /{vowel}/: F1≈{avg_f1:.0f} Hz, F2≈{avg_f2:.0f} Hz")
    
    sd.play(stim, samplerate=SR, device=2, blocking=True)
    time.sleep(0.2)
    sd.play(imit[:len(stim)], samplerate=SR, device=2, blocking=True)
    time.sleep(0.5)

print("\n--- Vowel Sequences ---")
sequences = [
    ('a-i', ['a', 'i']),
    ('i-a-u', ['i', 'a', 'u']),
]

for name, pattern in sequences:
    print(f"  Sequence: {name}")
    stim_parts = [generate_vowel(v, duration=0.25) for v in pattern]
    stim = np.concatenate(stim_parts)
    
    params = get_formant_params(None, stim)
    imit = params_to_formant_audio(params)
    
    sd.play(stim, samplerate=SR, device=2, blocking=True)
    time.sleep(0.2)
    sd.play(imit[:len(stim)], samplerate=SR, device=2, blocking=True)
    time.sleep(0.5)

print("\nDone!")
