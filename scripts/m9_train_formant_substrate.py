#!/usr/bin/env python3
"""M9 Formant Substrate Training.

Retrain the substrate (RNN) to classify formant-rich vowel stimuli.
Uses the same architecture as the live-trained substrate but with
formant-generated stimuli instead of pure tones.
"""

import sys, json, time
import numpy as np
from pathlib import Path
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank, NeuralController

SR = 16000

# Vowel formant definitions
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

def generate_vowel_audio(vowel, duration=0.5, f0=120):
    """Generate a vowel-like sound with formants + fundamental."""
    formants = VOWEL_FORMANTS[vowel]
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    
    # Fundamental (voicing)
    output = 0.3 * np.sin(2 * np.pi * f0 * t + np.random.uniform(0, 2*np.pi))
    
    # Add formants
    for fm in formants:
        output += fm['amp'] * np.sin(2 * np.pi * fm['freq'] * t + np.random.uniform(0, 2*np.pi))
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output /= peak
    
    return output.astype(np.float32)

def generate_training_data(n_repeats=50):
    """Generate labeled training data for all vowels."""
    all_clips = []
    all_labels = []
    
    print(f"Generating training data ({N_VOWELS} vowels x {n_repeats} repeats)...")
    for vowel_idx, vowel in enumerate(VOWEL_NAMES):
        for rep in range(n_repeats):
            # Random duration and f0 for variety
            duration = np.random.uniform(0.4, 0.8)
            f0 = np.random.uniform(100, 150)
            audio = generate_vowel_audio(vowel, duration=duration, f0=f0)
            
            all_clips.append(audio)
            all_labels.append(vowel_idx)
    
    print(f"  Generated {len(all_clips)} clips")
    return all_clips, all_labels

def train_substrate_classifier(
    n_epochs=300,
    lr=0.001,
    hidden_dim=32,
    checkpoint_path='data/m9_substrate_formant.ckpt.json'
):
    """Train substrate for vowel classification."""
    
    substrate = AcousticSubstrate(hidden_dim=hidden_dim, sr=SR)
    
    # Classification head
    W_cls = np.random.randn(N_VOWELS, hidden_dim).astype(np.float32) * 0.1
    b_cls = np.zeros(N_VOWELS, dtype=np.float32)
    
    # Generate training data
    clips, labels = generate_training_data(n_repeats=50)
    
    history = []
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        indices = np.random.permutation(len(clips))
        
        for idx in indices:
            clip = clips[idx]
            label = labels[idx]
            
            # Process full clip through substrate
            substrate.reset()
            n_frames = len(clip) // frame_size
            for i in range(n_frames):
                frame = clip[i*frame_size:(i+1)*frame_size]
                substrate.step(frame)
            
            h = substrate.hidden_state
            
            # Classify
            logits = W_cls @ h + b_cls
            logits_s = logits - np.max(logits)
            exp_logits = np.exp(logits_s)
            probs = exp_logits / exp_logits.sum()
            
            loss = -np.log(probs[label] + 1e-10)
            total_loss += loss
            
            if np.argmax(probs) == label:
                correct += 1
            total += 1
            
            # Backprop through classification head
            dlogits = probs.copy()
            dlogits[label] -= 1.0
            
            dW_cls = np.outer(dlogits, h)
            db_cls = dlogits
            
            # Backprop through substrate (truncated BPTT)
            dh = W_cls.T @ dlogits
            last_frame = clip[(n_frames-1)*frame_size:n_frames*frame_size]
            x = np.zeros(frame_size, dtype=np.float32)
            n = min(len(last_frame), frame_size)
            x[:n] = last_frame[:n]
            r = np.sqrt(np.mean(x**2))
            if r > 1e-6: x /= r
            
            z = substrate._W_h @ substrate._h + substrate._W_x @ x + substrate._b
            dz = dh * (1.0 - np.tanh(z)**2)
            
            dW_x = np.outer(dz, x)
            db_rnn = dz
            
            max_grad = 5.0
            for g in [dW_cls, db_cls]:
                gn = np.linalg.norm(g)
                if gn > max_grad: g *= max_grad / gn
            
            W_cls -= lr * dW_cls
            b_cls -= lr * db_cls
            substrate._W_x -= lr * 0.1 * dW_x
            substrate._b -= lr * 0.1 * db_rnn
        
        avg_loss = total_loss / total
        accuracy = correct / total
        history.append({'epoch': epoch+1, 'loss': float(avg_loss), 'accuracy': float(accuracy)})
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  acc={accuracy:.3f}")
    
    # Save checkpoint
    data = {
        'W_h': substrate._W_h.tolist(),
        'W_x': substrate._W_x.tolist(),
        'b': substrate._b.tolist(),
        'W_cls': W_cls.tolist(),
        'b_cls': b_cls.tolist(),
        'config': {'hidden_dim': hidden_dim, 'sr': SR, 'n_vowels': N_VOWELS, 'vowel_names': VOWEL_NAMES},
        'epoch': n_epochs,
        'final_accuracy': history[-1]['accuracy'],
    }
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).write_text(json.dumps(data))
    print(f"\nSaved checkpoint: {checkpoint_path}")
    
    return substrate, history

if __name__ == '__main__':
    print("=" * 72)
    print("M9 Formant Substrate Training")
    print("=" * 72)
    
    substrate, history = train_substrate_classifier(
        n_epochs=300,
        lr=0.001,
        hidden_dim=32,
    )
    
    print(f"\nFinal accuracy: {history[-1]['accuracy']:.3f}")
    print("Done!")
