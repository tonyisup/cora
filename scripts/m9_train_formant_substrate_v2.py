#!/usr/bin/env python3
"""M9 Formant Substrate v2 — proper vowel classification training.

Key changes from v1:
- Train ONLY on vowel classification (no frequency classification mixed in)
- More epochs (500)
- Larger hidden dim (64 instead of 32)
- Centered hidden states (subtract mean) for better clustering
"""

import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, 'packages')
from m9_sound import AcousticSubstrate, OscillatorBank

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
N_VOWELS = len(VOWEL_NAMES)
osc = OscillatorBank(sr=SR)
frame_size = 480

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
    print("=" * 72)
    print("M9 Formant Substrate v2 — Proper Vowel Classification")
    print("=" * 72)
    
    # Larger hidden dim for better separability
    hidden_dim = 64
    substrate = AcousticSubstrate(hidden_dim=hidden_dim, sr=SR)
    
    # Classification head
    W_cls = np.random.randn(N_VOWELS, hidden_dim).astype(np.float32) * 0.1
    b_cls = np.zeros(N_VOWELS, dtype=np.float32)
    
    print(f"Hidden dim: {hidden_dim}")
    print(f"Training: {N_VOWELS} vowels x 200 repeats x 500 epochs")
    
    history = []
    
    for epoch in range(500):
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle vowel order
        vowel_order = np.random.permutation(N_VOWELS)
        
        for vowel_idx in vowel_order:
            vowel = VOWEL_NAMES[vowel_idx]
            
            for rep in range(4):  # 4 reps per epoch per vowel
                f0 = np.random.uniform(100, 180)
                stim = generate_vowel(vowel, duration=0.5, f0=f0)
                
                # Encode through substrate
                substrate.reset()
                for i in range(len(stim) // frame_size):
                    frame = stim[i*frame_size:(i+1)*frame_size].flatten()
                    substrate.step(frame)
                
                h = substrate.hidden_state
                
                # Classify
                logits = W_cls @ h + b_cls
                logits_s = logits - np.max(logits)
                exp_logits = np.exp(logits_s)
                probs = exp_logits / exp_logits.sum()
                
                loss = -np.log(probs[vowel_idx] + 1e-10)
                total_loss += loss
                
                if np.argmax(probs) == vowel_idx:
                    correct += 1
                total += 1
                
                # Backprop
                dlogits = probs.copy()
                dlogits[vowel_idx] -= 1.0
                
                dW_cls = np.outer(dlogits, h)
                db_cls = dlogits
                
                # Backprop to substrate
                dh = W_cls.T @ dlogits
                last_frame = stim[(len(stim)//frame_size - 1)*frame_size:len(stim)].flatten()
                x = np.zeros(frame_size, dtype=np.float32)
                n = min(len(last_frame), frame_size)
                x[:n] = last_frame[:n]
                r = np.sqrt(np.mean(x**2))
                if r > 1e-6: x /= r
                
                z = substrate._W_h @ substrate._h + substrate._W_x @ x + substrate._b
                dz = dh * (1.0 - np.tanh(z)**2)
                
                dW_x = np.outer(dz, x)
                db_rnn = dz
                
                lr = 0.002
                max_grad = 5.0
                for g in [dW_cls, db_cls]:
                    gn = np.linalg.norm(g)
                    if gn > max_grad: g *= max_grad/gn
                
                W_cls -= lr * dW_cls
                b_cls -= lr * db_cls
                substrate._W_x -= lr * 0.1 * dW_x
                substrate._b -= lr * 0.1 * db_rnn
        
        avg_loss = total_loss / total
        accuracy = correct / total
        history.append({'epoch': epoch+1, 'loss': float(avg_loss), 'accuracy': float(accuracy)})
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/500  loss={avg_loss:.4f}  acc={accuracy:.3f}")
    
    # Save
    data = {
        'W_h': substrate._W_h.tolist(),
        'W_x': substrate._W_x.tolist(),
        'b': substrate._b.tolist(),
        'W_cls': W_cls.tolist(),
        'b_cls': b_cls.tolist(),
        'config': {'hidden_dim': hidden_dim, 'sr': SR, 'n_vowels': N_VOWELS, 'vowel_names': VOWEL_NAMES},
        'epoch': 500,
        'final_accuracy': history[-1]['accuracy'],
    }
    Path('data/m9_substrate_v2.ckpt.json').write_text(json.dumps(data))
    print(f"\nSaved to data/m9_substrate_v2.ckpt.json")
    
    # Evaluate: check centroid separation
    print("\n=== Centroid separation ===")
    centroids = {}
    for vowel in VOWEL_NAMES:
        stim = generate_vowel(vowel, duration=0.5)
        substrate.reset()
        h_states = []
        for i in range(len(stim) // frame_size):
            substrate.step(stim[i*frame_size:(i+1)*frame_size].flatten())
            h_states.append(substrate.hidden_state.copy())
        mid_start = len(h_states) // 4
        mid_end = 3 * len(h_states) // 4
        centroids[vowel] = np.mean(h_states[mid_start:mid_end], axis=0)
    
    print("Pairwise distances:")
    for i, v1 in enumerate(VOWEL_NAMES):
        for v2 in VOWEL_NAMES[i+1:]:
            dist = np.linalg.norm(centroids[v1] - centroids[v2])
            print(f"  {v1}-{v2}: {dist:.3f}")

if __name__ == '__main__':
    main()
