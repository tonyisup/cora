#!/usr/bin/env python3
"""M9 Formant Controller — lookup table approach.

Instead of a neural network that can produce any output,
use a simple nearest-neighbor lookup: cluster substrate hidden states,
assign each cluster a formant template.

This guarantees zero warble within a segment since each cluster maps
to ONE set of formant values.
"""

import sys, json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank
from m9_sound.controller import NeuralController

SR = 16000

VOWEL_FORMANTS = {
    'i':  [{'freq': 280, 'amp': 1.0, 'bw': 50}, {'freq': 2250, 'amp': 0.4, 'bw': 80}, {'freq': 2900, 'amp': 0.2, 'bw': 120}],
    'a':  [{'freq': 730, 'amp': 1.0, 'bw': 80}, {'freq': 1090, 'amp': 0.7, 'bw': 100}, {'freq': 2440, 'amp': 0.35, 'bw': 120}],
    'u':  [{'freq': 300, 'amp': 1.0, 'bw': 50}, {'freq': 870, 'amp': 0.4, 'bw': 80}, {'freq': 2240, 'amp': 0.15, 'bw': 120}],
    'E':  [{'freq': 530, 'amp': 1.0, 'bw': 60}, {'freq': 1840, 'amp': 0.5, 'bw': 90}, {'freq': 2480, 'amp': 0.25, 'bw': 120}],
    'c':  [{'freq': 570, 'amp': 1.0, 'bw': 70}, {'freq': 840, 'amp': 0.6, 'bw': 90}, {'freq': 2410, 'amp': 0.3, 'bw': 120}],
}
VOWEL_NAMES = list(VOWEL_FORMANTS.keys())

oscil = OscillatorBank(sr=SR)

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

def train_lookup(substrate, n_samples_per_vowel=100):
    """Build a lookup table: hidden state cluster -> formant template."""
    
    # Collect hidden states for each vowel
    vowel_hidden = {v: [] for v in VOWEL_NAMES}
    vowel_targets = {}
    
    print("Collecting hidden states...")
    for vowel in VOWEL_NAMES:
        formants = VOWEL_FORMANTS[vowel]
        target = {
            'f1': formants[0]['freq'], 'f2': formants[1]['freq'], 'f3': formants[2]['freq'],
            'a1': formants[0]['amp'], 'a2': formants[1]['amp'], 'a3': formants[2]['amp'],
        }
        vowel_targets[vowel] = target
        
        for _ in range(n_samples_per_vowel):
            f0 = np.random.uniform(100, 180)
            stim = generate_vowel(vowel, duration=0.5, f0=f0)
            substrate.reset()
            h_states = []
            for i in range(len(stim) // 480):
                substrate.step(stim[i*480:(i+1)*480])
                h_states.append(substrate.hidden_state.copy())
            
            # Average hidden state over the middle half of the stimulus
            mid_start = len(h_states) // 4
            mid_end = 3 * len(h_states) // 4
            avg_h = np.mean(h_states[mid_start:mid_end], axis=0)
            vowel_hidden[vowel].append(avg_h)
    
    # Build KD-tree for each vowel
    trees = {}
    hidden_arrays = {}
    for vowel in VOWEL_NAMES:
        h_arr = np.array(vowel_hidden[vowel], dtype=np.float32)
        hidden_arrays[vowel] = h_arr
        trees[vowel] = cKDTree(h_arr)
    
    # Classify: for each vowel, check which vowel's tree is nearest
    print("\nClassification accuracy:")
    correct = 0
    total = 0
    for true_vowel in VOWEL_NAMES:
        for h in vowel_hidden[true_vowel]:
            # Find nearest neighbor across all vowels
            best_dist = float('inf')
            best_vowel = None
            for v in VOWEL_NAMES:
                dist, _ = trees[v].query(k=1, x=h)
                if dist < best_dist:
                    best_dist = dist
                    best_vowel = v
            if best_vowel == true_vowel:
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"  {accuracy:.1%} ({correct}/{total})")
    
    # Build lookup: cluster centroids -> templates
    lookup = {}
    for vowel in VOWEL_NAMES:
        h_arr = hidden_arrays[vowel]
        centroid = np.mean(h_arr, axis=0)
        lookup[vowel] = {
            'centroid': centroid,
            'template': vowel_targets[vowel],
        }
    
    # Save
    save_data = {
        'lookup': {v: {'centroid': lookup[v]['centroid'].tolist(), 'template': lookup[v]['template']} for v in VOWEL_NAMES},
        'hidden_arrays': {v: hidden_arrays[v].tolist() for v in VOWEL_NAMES},
        'vowel_names': VOWEL_NAMES,
        'accuracy': accuracy,
    }
    Path('data/m9_formant_lookup.json').write_text(json.dumps(save_data))
    print(f"\nSaved to data/m9_formant_lookup.json")
    
    return lookup, trees

def classify_and_synthesize(substrate, stim, lookup, trees):
    """Classify each frame and synthesize from template — zero warble."""
    substrate.reset()
    params_list = []
    
    for i in range(len(stim) // 480):
        substrate.step(stim[i*480:(i+1)*480])
        h = substrate.hidden_state
        
        # Find nearest vowel
        best_dist = float('inf')
        best_vowel = VOWEL_NAMES[0]
        for v in VOWEL_NAMES:
            dist, _ = trees[v].query(k=1, x=h)
            if dist < best_dist:
                best_dist = dist
                best_vowel = v
        
        template = lookup[best_vowel]['template']
        params_list.append({
            'f1': template['f1'], 'f2': template['f2'], 'f3': template['f3'],
            'a1': template['a1'], 'a2': template['a2'], 'a3': template['a3'],
            'wf': 0
        })
    
    return params_list

def main():
    # Load substrate
    sub_ckpt = json.load(open('data/m9_substrate_formant.ckpt.json'))
    substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
    substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
    substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
    substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)
    
    # Train lookup table
    lookup, trees = train_lookup(substrate, n_samples_per_vowel=200)
    
    # Demo
    import sounddevice as sd, time
    
    print("\n" + "=" * 72)
    print("M9 Formant Demo — Lookup Table (Zero Warble)")
    print("=" * 72)
    
    # Part 1: Single vowels
    for vowel in VOWEL_NAMES:
        stim = generate_vowel(vowel, duration=0.8)
        
        # Classify frames to detect which vowel was recognized
        substrate.reset()
        recognized = []
        for i in range(len(stim) // 480):
            substrate.step(stim[i*480:(i+1)*480])
            h = substrate.hidden_state
            best_v = VOWEL_NAMES[0]
            best_d = float('inf')
            for v in VOWEL_NAMES:
                d, _ = trees[v].query(k=1, x=h)
                if d < best_d:
                    best_d = d; best_v = v
            recognized.append(best_v)
        
        # Stability: how often does the classification change?
        changes = sum(1 for i in range(1, len(recognized)) if recognized[i] != recognized[i-1])
        
        params = classify_and_synthesize(substrate, stim, lookup, trees)
        audio = params_to_audio(params)
        
        f1s = [p['f1'] for p in params]
        diffs = np.abs(np.array(f1s[1:]) - np.array(f1s[:-1])) if len(f1s) > 1 else [0]
        
        print(f"\n/{vowel}/: recognized={recognized[0] if recognized else '?'} "
              f"F1={np.mean(f1s):.0f}±{np.std(f1s):.0f} "
              f"jumps>50={np.sum(np.array(diffs)>50)}/{len(diffs)} "
              f"class_changes={changes}/{len(recognized)-1}")
        
        sd.play(stim, samplerate=SR, device=2, blocking=True)
        time.sleep(0.2)
        sd.play(audio[:len(stim)], samplerate=SR, device=2, blocking=True)
        time.sleep(0.4)
    
    # Part 2: Sequences
    print("\n--- Vowel Sequences ---")
    for name, pattern in [('a-i', ['a', 'i']), ('i-a-u', ['i', 'a', 'u'])]:
        stim = generate_sequence(pattern, duration_each=0.25)
        params = classify_and_synthesize(substrate, stim, lookup, trees)
        audio = params_to_audio(params)
        
        f1s = [p['f1'] for p in params]
        print(f"  {name}: F1 range=[{min(f1s):.0f}, {max(f1s):.0f}], "
              f"changes={sum(1 for i in range(1,len(f1s)) if abs(f1s[i]-f1s[i-1])>50)}")
        
        sd.play(stim, samplerate=SR, device=2, blocking=True)
        time.sleep(0.2)
        sd.play(audio[:len(stim)], samplerate=SR, device=2, blocking=True)
        time.sleep(0.4)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
