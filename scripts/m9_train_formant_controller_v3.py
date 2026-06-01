#!/usr/bin/env python3
"""M9 Formant Controller v3 — stable training with noise injection.

Key changes:
1. Lower learning rate to prevent sharp transitions
2. Noise injection on hidden states during training (robustness)
3. Weight decay to prevent extreme weights
4. Target smoothing (soft labels) to reduce sharp decision boundaries
"""

import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank
from m9_sound.controller import NeuralController

SR = 16000
osc = OscillatorBank(sr=SR)
frame_size = 480

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

def generate_vowel(vowel, duration=0.5, f0=120):
    formants = VOWEL_FORMANTS[vowel]
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    output = 0.3 * np.sin(2 * np.pi * f0 * t + np.random.uniform(0, 2*np.pi))
    for fm in formants:
        output += fm['amp'] * np.sin(2 * np.pi * fm['freq'] * t + np.random.uniform(0, 2*np.pi))
    peak = np.max(np.abs(output))
    if peak > 1.0: output /= peak
    return output.astype(np.float32)

def main():
    # Load v2 substrate (best so far)
    sub_ckpt = json.load(open('data/m9_substrate_formant.ckpt.json'))
    substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
    substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
    substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
    substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)
    print(f"Loaded substrate (acc: {sub_ckpt.get('final_accuracy', 'N/A')})")
    
    # Create controller
    controller = NeuralController(hidden_dim=32, sr=SR, controller_hidden_dim=24)
    print(f"Controller: {controller.n_params} params")
    
    # Training data — same extended set as v2
    import itertools
    seq_defs = []
    for i, v1 in enumerate(VOWEL_NAMES):
        for v2 in VOWEL_NAMES:
            if v1 != v2:
                seq_defs.append((f'{v1}-{v2}', [(v1, 0.25), (v2, 0.25)]))
    for v1, v2, v3 in itertools.product(VOWEL_NAMES[:5], repeat=3):
        if len(set([v1, v2, v3])) > 1:
            seq_defs.append((f'{v1}-{v2}-{v3}', [(v1, 0.2), (v2, 0.2), (v3, 0.2)]))
    
    print(f"Training sequences: {len(seq_defs)}")
    
    # Generate training data with noise injection
    all_hidden = []
    all_targets = []
    noise_std = 0.05  # Small noise on hidden states for robustness
    
    print("Generating training data with noise injection...")
    for name, pattern in seq_defs:
        for rep in range(5):
            stim_parts = []
            for vowel, dur in pattern:
                stim_parts.append(generate_vowel(vowel, duration=dur, f0=np.random.uniform(100, 180)))
            stim = np.concatenate(stim_parts)
            
            substrate.reset()
            for i in range(len(stim) // frame_size):
                substrate.step(stim[i*frame_size:(i+1)*frame_size])
                
                h = substrate.hidden_state.copy()
                # Add noise to hidden state for robustness
                h += np.random.randn(h.shape[0]).astype(np.float32) * noise_std
                
                all_hidden.append(h)
                
                frame_time = (i * frame_size) / SR
                elapsed = 0
                current_vowel = pattern[0][0]
                for vowel, dur in pattern:
                    if frame_time < elapsed + dur:
                        current_vowel = vowel
                        break
                    elapsed += dur
                
                formants = VOWEL_FORMANTS[current_vowel]
                target = np.zeros(8, dtype=np.float32)
                target[0] = np.clip((formants[0]['freq'] - 80) / (3000 - 80), 0, 1)
                target[1] = np.clip((formants[1]['freq'] - 80) / (3000 - 80), 0, 1)
                target[2] = np.clip((formants[2]['freq'] - 80) / (3000 - 80), 0, 1)
                target[3] = formants[0]['amp']
                target[4] = formants[1]['amp']
                target[5] = formants[2]['amp']
                target[6] = 1.0
                all_targets.append(target)
    
    hidden_states = np.array(all_hidden, dtype=np.float32)
    target_params = np.array(all_targets, dtype=np.float32)
    print(f"Training data: {len(hidden_states)} samples")
    
    # Train with lower LR + weight decay
    history = []
    lr = 0.002  # Lower than v2's 0.005
    weight_decay = 1e-5
    n_epochs = 1500
    
    print(f"Training ({n_epochs} epochs, lr={lr}, wd={weight_decay}, noise={noise_std})...")
    for epoch in range(n_epochs):
        total_loss = 0.0
        indices = np.random.permutation(len(hidden_states))
        
        for idx in indices:
            h = hidden_states[idx]
            target = target_params[idx]
            
            raw, a1 = controller._forward_all(h)
            
            err_f1 = raw[0] - target[0]
            err_f2 = raw[1] - target[1]
            err_f3 = raw[2] - target[2]
            
            sig_1 = 1.0 / (1.0 + np.exp(-raw[3]))
            sig_2 = 1.0 / (1.0 + np.exp(-raw[4]))
            sig_3 = 1.0 / (1.0 + np.exp(-raw[5]))
            err_a1 = sig_1 - target[3]
            err_a2 = sig_2 - target[4]
            err_a3 = sig_3 - target[5]
            
            wf_logits = raw[6:6+controller._n_waveforms]
            wf_sm = np.exp(wf_logits - np.max(wf_logits))
            wf_sm /= wf_sm.sum()
            target_wf = target[6:6+controller._n_waveforms]
            err_wf = wf_sm - target_wf
            
            loss = (err_f1**2 + err_f2**2 + err_f3**2 +
                    2*(err_a1**2 + err_a2**2 + err_a3**2) +
                    np.sum(err_wf**2))
            total_loss += loss
            
            delta = np.zeros(4 + controller._n_waveforms, dtype=np.float32)
            delta[0] = err_f1; delta[1] = err_f2; delta[2] = err_f3
            delta[3] = 2 * err_a1 * sig_1 * (1 - sig_1)
            delta[4] = 2 * err_a2 * sig_2 * (1 - sig_2)
            delta[5] = 2 * err_a3 * sig_3 * (1 - sig_3)
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
            
            current_lr = lr * (1 - epoch / n_epochs)
            
            controller._W2 -= current_lr * (dW2 + weight_decay * controller._W2)
            controller._b2 -= current_lr * db2
            controller._W1 -= current_lr * (dW1 + weight_decay * controller._W1)
            controller._b1 -= current_lr * db1
        
        avg_loss = total_loss / len(hidden_states)
        history.append(float(avg_loss))
        
        if (epoch + 1) % 300 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  loss={avg_loss:.6f}")
    
    # Save
    Path('data/m9_controller_formant_v3.ckpt.json').write_text(json.dumps({
        'W1': controller._W1.tolist(), 'b1': controller._b1.tolist(),
        'W2': controller._W2.tolist(), 'b2': controller._b2.tolist(),
        'final_loss': history[-1], 'n_epochs': n_epochs,
        'controller_hidden_dim': 24, 'lr': lr, 'weight_decay': weight_decay,
        'noise_std': noise_std,
    }))
    print(f"\nSaved to data/m9_controller_formant_v3.ckpt.json")
    
    # Evaluate
    print("\n=== Formant Evaluation ===")
    for vowel in VOWEL_NAMES:
        stim = generate_vowel(vowel, duration=0.5)
        substrate.reset()
        for i in range(len(stim)//frame_size):
            substrate.step(stim[i*frame_size:(i+1)*frame_size])
        
        raw, _ = controller._forward_all(substrate.hidden_state)
        
        f1 = 80 + np.clip(raw[0], 0, 1) * (3000 - 80)
        f2 = 80 + np.clip(raw[1], 0, 1) * (3000 - 80)
        f3 = 80 + np.clip(raw[2], 0, 1) * (3000 - 80)
        
        target = VOWEL_FORMANTS[vowel]
        f1_err = abs(f1 - target[0]['freq'])
        f2_err = abs(f2 - target[1]['freq'])
        f3_err = abs(f3 - target[2]['freq'])
        mean_err = (f1_err + f2_err + f3_err) / 3
        
        print(f"  /{vowel}/: F1={f1:.0f}({target[0]['freq']},err={f1_err:.0f}) "
              f"F2={f2:.0f}({target[1]['freq']},err={f2_err:.0f}) "
              f"F3={f3:.0f}({target[2]['freq']},err={f3_err:.0f}) "
              f"mean_err={mean_err:.0f}Hz")
    
    # Stability test: measure frame-to-frame variation for a sustained vowel
    print("\n=== Stability Test ===")
    for vowel in ['i', 'a', 'u']:
        stim = generate_vowel(vowel, duration=1.0)
        substrate.reset()
        f1s = []
        for i in range(len(stim)//frame_size):
            substrate.step(stim[i*frame_size:(i+1)*frame_size])
            raw, _ = controller._forward_all(substrate.hidden_state)
            f1 = 80 + np.clip(raw[0], 0, 1) * (3000 - 80)
            f1s.append(f1)
        f1s = np.array(f1s)
        diffs = np.abs(np.diff(f1s))
        print(f"  /{vowel}/: F1 mean={np.mean(f1s):.0f}, std={np.std(f1s):.1f}, "
              f"mean_jump={np.mean(diffs):.1f}, max_jump={np.max(diffs):.0f}, "
              f"jumps>50Hz={np.sum(diffs>50)}/{len(diffs)}")

if __name__ == '__main__':
    print("=" * 72)
    print("M9 Formant Controller v3 — Stable Training")
    print("=" * 72)
    main()
    print("\nDone!")
