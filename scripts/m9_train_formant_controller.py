#!/usr/bin/env python3
"""M9 Formant Controller Training.

Trains a controller to output formant parameters (F1, F2, F3, A1, A2, A3)
from substrate hidden states. The controller learns to map
"what I heard" -> "what formants to produce".

Uses the same supervised approach as the temporal controller,
but with 7 output dims instead of 4.
"""

import sys, json, time
import numpy as np
from pathlib import Path
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank
from m9_sound.controller import NeuralController

SR = 16000

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
VOWEL_NAMES = list(VOWEL_FORMANTS.keys())
osc = OscillatorBank(sr=SR)
frame_size = 480

def generate_vowel_audio(vowel, duration=0.5, f0=120):
    formants = VOWEL_FORMANTS[vowel]
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    output = 0.3 * np.sin(2 * np.pi * f0 * t + np.random.uniform(0, 2*np.pi))
    for fm in formants:
        output += fm['amp'] * np.sin(2 * np.pi * fm['freq'] * t + np.random.uniform(0, 2*np.pi))
    peak = np.max(np.abs(output))
    if peak > 1.0: output /= peak
    return output.astype(np.float32)

def generate_sequence(pattern, duration_each=0.3):
    parts = []
    for vowel in pattern:
        parts.append(generate_vowel_audio(vowel, duration=duration_each))
    return np.concatenate(parts)

def train_formant_controller(
    n_epochs=500,
    lr=0.001,
    hidden_dim=16,
    checkpoint_path='data/m9_controller_formant_v1.ckpt.json'
):
    """Train controller to output formant parameters from substrate hidden states."""
    
    # Load formant substrate
    sub_ckpt = json.load(open('data/m9_substrate_formant.ckpt.json'))
    substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
    substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
    substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
    substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)
    print(f"Loaded formant substrate (acc was {sub_ckpt.get('final_accuracy', 'N/A')})")
    
    # Create controller: 32-dim hidden state -> 7 outputs (F1_norm, F2_norm, F3_norm, A1, A2, A3, wf)
    # Output: [F1_norm, F2_norm, F3_norm, A1, A2, A3, waveform_logits]
    n_out = 7
    controller = NeuralController(hidden_dim=32, sr=SR, controller_hidden_dim=hidden_dim)
    print(f"Controller: {controller.n_params} params, {n_out} outputs")
    
    # Prepare training data
    print("Generating formant training data...")
    seq_defs = [
        ('a-i', [('a', 0.3), ('i', 0.3)]),
        ('i-a', [('i', 0.3), ('a', 0.3)]),
        ('a-u', [('a', 0.3), ('u', 0.3)]),
        ('u-a', [('u', 0.3), ('a', 0.3)]),
        ('i-a-u', [('i', 0.2), ('a', 0.2), ('u', 0.2)]),
        ('a-i-u', [('a', 0.2), ('i', 0.2), ('u', 0.2)]),
        ('E-a-i', [('E', 0.2), ('a', 0.2), ('i', 0.2)]),
        ('a-U-a', [('a', 0.2), ('U', 0.2), ('a', 0.2)]),
        ('ae-i-ae', [('ae', 0.2), ('i', 0.2), ('ae', 0.2)]),
        ('c-u-c', [('c', 0.2), ('u', 0.2), ('c', 0.2)]),
        ('V-E-V', [('V', 0.2), ('E', 0.2), ('V', 0.2)]),
        ('r-V-r', [('r', 0.2), ('V', 0.2), ('r', 0.2)]),
    ]
    
    # For each sequence, encode through substrate, get per-frame hidden states + target params
    all_hidden = []
    all_targets = []
    
    for name, pattern in seq_defs:
        for rep in range(10):
            # Generate stimulus
            seq = []
            for vowel, dur in pattern:
                seq.append(generate_vowel_audio(vowel, duration=dur, f0=np.random.uniform(100, 150)))
            stim = np.concatenate(seq)
            
            # Encode through substrate
            substrate.reset()
            target_idx = 0
            tone_start = 0
            
            for i in range(len(stim) // frame_size):
                frame = stim[i*frame_size:(i+1)*frame_size]
                substrate.step(frame)
                h = substrate.hidden_state.copy()
                all_hidden.append(h)
                
                # Target: formants for the CURRENT tone segment
                frame_time = (i * frame_size) / SR
                elapsed = 0
                current_vowel = pattern[0][0]
                for vowel, dur in pattern:
                    if frame_time < elapsed + dur:
                        current_vowel = vowel
                        break
                    elapsed += dur
                
                # Build target: [F1_norm, F2_norm, F3_norm, A1, A2, A3, wf_onehot...]
                formants = VOWEL_FORMANTS[current_vowel]
                f1n = np.clip((formants[0]['freq'] - 80) / (3000 - 80), 0, 1)
                f2n = np.clip((formants[1]['freq'] - 80) / (3000 - 80), 0, 1)
                f3n = np.clip((formants[2]['freq'] - 80) / (3000 - 80), 0, 1)
                a1 = formants[0]['amp']
                a2 = formants[1]['amp']
                a3 = formants[2]['amp']
                
                target = np.zeros(8, dtype=np.float32)
                target[0] = f1n
                target[1] = f2n
                target[2] = f3n
                target[3] = a1
                target[4] = a2
                target[5] = a3
                target[6] = 1.0  # sine waveform
                all_targets.append(target)
    
    hidden_states = np.array(all_hidden, dtype=np.float32)
    target_params = np.array(all_targets, dtype=np.float32)
    print(f"Training data: {len(hidden_states)} samples")
    
    # Train controller with closed-loop feedback
    history = []
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        indices = np.random.permutation(len(hidden_states))
        
        for idx in indices:
            h = hidden_states[idx]
            target = target_params[idx]
            
            # Forward
            raw, a1 = controller._forward_all(h)
            
            # Loss: MSE on normalized formants + amplitudes
            # F1, F2, F3 (normalized 0-1)
            err_f1 = raw[0] - target[0]
            err_f2 = raw[1] - target[1]
            err_f3 = raw[2] - target[2]
            
            # Amplitudes (sigmoid output)
            sig_1 = 1.0 / (1.0 + np.exp(-raw[3]))
            sig_2 = 1.0 / (1.0 + np.exp(-raw[4]))
            sig_3 = 1.0 / (1.0 + np.exp(-raw[5]))
            err_a1 = sig_1 - target[3]
            err_a2 = sig_2 - target[4]
            err_a3 = sig_3 - target[5]
            
            # Waveform (softmax)
            wf_logits = raw[6:6+controller._n_waveforms]
            wf_sm = np.exp(wf_logits - np.max(wf_logits))
            wf_sm /= wf_sm.sum()
            target_wf = target[6:6+controller._n_waveforms]
            err_wf = wf_sm - target_wf
            
            loss = (err_f1**2 + err_f2**2 + err_f3**2 + 
                    err_a1**2 + err_a2**2 + err_a3**2 + 
                    np.sum(err_wf**2))
            total_loss += loss
            
            # Backprop
            delta = np.zeros(4 + controller._n_waveforms, dtype=np.float32)
            delta[0] = err_f1
            delta[1] = err_f2
            delta[2] = err_f3
            delta[3] = err_a1 * sig_1 * (1 - sig_1)
            delta[4] = err_a2 * sig_2 * (1 - sig_2)
            delta[5] = err_a3 * sig_3 * (1 - sig_3)
            delta[6:6+controller._n_waveforms] = err_wf
            
            h_flat = h.flatten()[:controller._hidden_dim]
            if len(h_flat) < controller._hidden_dim:
                h_flat = np.pad(h_flat, (0, controller._hidden_dim - len(h_flat)))
            
            dW2 = np.outer(delta, a1)
            db2 = delta
            da1 = controller._W2.T @ delta
            z1 = controller._W1 @ h_flat + controller._b1
            dz1 = da1 * (1.0 - np.tanh(z1)**2)
            dW1 = np.outer(dz1, h_flat)
            db1 = dz1
            
            max_grad = 5.0
            for g in [dW2, db2, dW1, db1]:
                gn = np.linalg.norm(g)
                if gn > max_grad: g *= max_grad/gn
            
            controller._W2 -= lr * dW2
            controller._b2 -= lr * db2
            controller._W1 -= lr * dW1
            controller._b1 -= lr * db1
        
        avg_loss = total_loss / len(hidden_states)
        history.append(float(avg_loss))
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.6f}")
    
    # Save
    Path(checkpoint_path).write_text(json.dumps({
        'W1': controller._W1.tolist(), 'b1': controller._b1.tolist(),
        'W2': controller._W2.tolist(), 'b2': controller._b2.tolist(),
        'final_loss': history[-1], 'seq_defs': seq_defs,
    }))
    print(f"\nSaved to {checkpoint_path}")
    
    # Evaluate: check formant output for each vowel
    print("\n=== Evaluation ===")
    for vowel in VOWEL_NAMES[:5]:
        stim = generate_vowel_audio(vowel, duration=0.5)
        substrate.reset()
        for i in range(len(stim)//frame_size):
            substrate.step(stim[i*frame_size:(i+1)*frame_size])
        
        h = substrate.hidden_state
        raw, _ = controller._forward_all(h)
        
        f1 = 80 + np.clip(raw[0], 0, 1) * (3000 - 80)
        f2 = 80 + np.clip(raw[1], 0, 1) * (3000 - 80)
        f3 = 80 + np.clip(raw[2], 0, 1) * (3000 - 80)
        a1 = float(1.0 / (1.0 + np.exp(-raw[3])))
        a2 = float(1.0 / (1.0 + np.exp(-raw[4])))
        a3 = float(1.0 / (1.0 + np.exp(-raw[5])))
        
        target = VOWEL_FORMANTS[vowel]
        print(f"  /{vowel}/: F1={f1:.0f}({target[0]['freq']}), F2={f2:.0f}({target[1]['freq']}), F3={f3:.0f}({target[2]['freq']})")
    
    return controller, history

if __name__ == '__main__':
    print("=" * 72)
    print("M9 Formant Controller Training")
    print("=" * 72)
    controller, history = train_formant_controller(n_epochs=500, lr=0.001, hidden_dim=16)
    print(f"\nFinal loss: {history[-1]:.6f}")
    print("Done!")
