#!/usr/bin/env python3
"""M9 Formant Demo — with output smoothing to reduce warble.

Applies dead-zone smoothing: only update formant params when the
controller output changes significantly. Small jitters are ignored,
genuine vowel transitions pass through.
"""

import sys, json, time
import numpy as np
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank
from m9_sound.controller import NeuralController
import sounddevice as sd

SR = 16000

# Load v2 components
sub_ckpt = json.load(open('data/m9_substrate_formant.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

ctl_ckpt = json.load(open('data/m9_controller_formant_v2.ckpt.json'))
controller = NeuralController(hidden_dim=32, sr=SR, controller_hidden_dim=24)
controller._W1 = np.array(ctl_ckpt['W1'], dtype=np.float32)
controller._b1 = np.array(ctl_ckpt['b1'], dtype=np.float32)
controller._W2 = np.array(ctl_ckpt['W2'], dtype=np.float32)
controller._b2 = np.array(ctl_ckpt['b2'], dtype=np.float32)

osc = OscillatorBank(sr=SR)

def params_to_audio(params_list, frame_size=480, sr=SR):
    """Convert formant params to audio with phase continuity."""
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

VOWEL_FORMANTS = {
    'i':  [{'freq': 280, 'amp': 1.0, 'bw': 50}, {'freq': 2250, 'amp': 0.4, 'bw': 80}, {'freq': 2900, 'amp': 0.2, 'bw': 120}],
    'a':  [{'freq': 730, 'amp': 1.0, 'bw': 80}, {'freq': 1090, 'amp': 0.7, 'bw': 100}, {'freq': 2440, 'amp': 0.35, 'bw': 120}],
    'u':  [{'freq': 300, 'amp': 1.0, 'bw': 50}, {'freq': 870, 'amp': 0.4, 'bw': 80}, {'freq': 2240, 'amp': 0.15, 'bw': 120}],
    'E':  [{'freq': 530, 'amp': 1.0, 'bw': 60}, {'freq': 1840, 'amp': 0.5, 'bw': 90}, {'freq': 2480, 'amp': 0.25, 'bw': 120}],
    'c':  [{'freq': 570, 'amp': 1.0, 'bw': 70}, {'freq': 840, 'amp': 0.6, 'bw': 90}, {'freq': 2410, 'amp': 0.3, 'bw': 120}],
}

def generate_vowel(vowel, duration=0.5, f0=120):
    formants = VOWEL_FORMANTS[vowel]
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    output = 0.3 * np.sin(2 * np.pi * f0 * t + np.random.uniform(0, 2*np.pi))
    for fm in formants:
        output += fm['amp'] * np.sin(2 * np.pi * fm['freq'] * t + np.random.uniform(0, 2*np.pi))
    peak = np.max(np.abs(output))
    if peak > 1.0: output /= peak
    return output.astype(np.float32)

def smooth_params(raw_params, freq_threshold=30, ema_alpha=0.3):
    """Smooth controller output with dead zone + EMA.
    
    Only update frequency when change exceeds threshold.
    This preserves sharp transitions while eliminating jitter.
    """
    smoothed = []
    prev_f1 = raw_params[0]['f1']
    prev_f2 = raw_params[0]['f2']
    prev_f3 = raw_params[0]['f3']
    prev_a1 = raw_params[0]['a1']
    prev_a2 = raw_params[0]['a2']
    prev_a3 = raw_params[0]['a3']
    
    for p in raw_params:
        # Dead zone: only update if change is significant
        if abs(p['f1'] - prev_f1) > freq_threshold:
            prev_f1 = ema_alpha * p['f1'] + (1-ema_alpha) * prev_f1
        if abs(p['f2'] - prev_f2) > freq_threshold:
            prev_f2 = ema_alpha * p['f2'] + (1-ema_alpha) * prev_f2
        if abs(p['f3'] - prev_f3) > freq_threshold:
            prev_f3 = ema_alpha * p['f3'] + (1-ema_alpha) * prev_f3
            
        # Amplitudes always smooth (they vary less)
        prev_a1 = ema_alpha * p['a1'] + (1-ema_alpha) * prev_a1
        prev_a2 = ema_alpha * p['a2'] + (1-ema_alpha) * prev_a2
        prev_a3 = ema_alpha * p['a3'] + (1-ema_alpha) * prev_a3
        
        smoothed.append({
            'f1': prev_f1, 'f2': prev_f2, 'f3': prev_f3,
            'a1': prev_a1, 'a2': prev_a2, 'a3': prev_a3,
            'wf': p['wf']
        })
    
    return smoothed

# Test vowels
vowels = ['i', 'a', 'u', 'E', 'c']

print("Smoothed Formant Demo")
print("=" * 72)

for vowel in vowels:
    print(f"\nVowel /{vowel}/")
    
    # Generate stimulus
    stim = generate_vowel(vowel, duration=0.8)
    
    # Get raw controller params
    substrate.reset()
    raw_params = []
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
        
        raw_params.append({'f1': f1, 'f2': f2, 'f3': f3, 'a1': max(a1, 0.1), 'a2': max(a2, 0.05), 'a3': max(a3, 0.05), 'wf': 0})
    
    raw_freqs = [p['f1'] for p in raw_params]
    print(f"  Raw F1: mean={np.mean(raw_freqs):.0f}, std={np.std(raw_freqs):.0f}, range=[{min(raw_freqs):.0f}, {max(raw_freqs):.0f}]")
    
    # Smooth
    smooth = smooth_params(raw_params, freq_threshold=30, ema_alpha=0.3)
    smooth_freqs = [p['f1'] for p in smooth]
    print(f"  Smoothed F1: mean={np.mean(smooth_freqs):.0f}, std={np.std(smooth_freqs):.0f}, range=[{min(smooth_freqs):.0f}, {max(smooth_freqs):.0f}]")
    
    # Generate raw imitation
    raw_audio = params_to_audio(raw_params)
    
    # Generate smoothed imitation
    smooth_audio = params_to_audio(smooth)
    
    # Play stimulus then both imitations
    sd.play(stim, samplerate=SR, device=2, blocking=True)
    time.sleep(0.2)
    
    print("  Playing raw imitation...")
    sd.play(raw_audio[:len(stim)], samplerate=SR, device=2, blocking=True)
    time.sleep(0.2)
    
    print("  Playing smoothed imitation...")
    sd.play(smooth_audio[:len(stim)], samplerate=SR, device=2, blocking=True)
    time.sleep(0.4)

# Play vowel sequence with smoothing
print("\n" + "=" * 72)
print("Vowel Sequence: a -> i -> u")
stim_seq = np.concatenate([generate_vowel(v, duration=0.25) for v in ['a', 'i', 'u']])

substrate.reset()
seq_params = []
for i in range(len(stim_seq) // 480):
    substrate.step(stim_seq[i*480:(i+1)*480])
    h = substrate.hidden_state
    raw, _ = controller._forward_all(h)
    f1 = 80 + np.clip(raw[0], 0, 1) * (3000 - 80)
    f2 = 80 + np.clip(raw[1], 0, 1) * (3000 - 80)
    f3 = 80 + np.clip(raw[2], 0, 1) * (3000 - 80)
    a1 = float(1.0 / (1.0 + np.exp(-raw[3])))
    a2 = float(1.0 / (1.0 + np.exp(-raw[4])))
    a3 = float(1.0 / (1.0 + np.exp(-raw[5])))
    seq_params.append({'f1': f1, 'f2': f2, 'f3': f3, 'a1': max(a1, 0.1), 'a2': max(a2, 0.05), 'a3': max(a3, 0.05), 'wf': 0})

seq_smooth = smooth_params(seq_params, freq_threshold=30, ema_alpha=0.3)

sd.play(stim_seq, samplerate=SR, device=2, blocking=True)
time.sleep(0.3)
print("  Playing raw sequence imitation...")
sd.play(params_to_audio(seq_params)[:len(stim_seq)], samplerate=SR, device=2, blocking=True)
time.sleep(0.3)
print("  Playing smoothed sequence imitation...")
sd.play(params_to_audio(seq_smooth)[:len(stim_seq)], samplerate=SR, device=2, blocking=True)

print("\nDone!")
