#!/usr/bin/env python3
"""Train and evaluate temporal controller with smaller capacity."""

import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank, AcousticSimilarity
from m9_sound.controller import NeuralController

SR = 16000
frame_size = 480

# Load live-trained substrate
sub_ckpt = json.load(open('data/m9_substrate_live.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

osc = OscillatorBank(sr=SR)

# Create small controller (336 params — hard to overfit)
controller = NeuralController(hidden_dim=32, sr=SR, controller_hidden_dim=8)
print(f"Controller: {controller.n_params} params")

# Generate training sequences
seq_defs = [
    [(220, 0.3), (330, 0.3)], [(330, 0.3), (220, 0.3)],
    [(262, 0.3), (392, 0.3)], [(392, 0.3), (262, 0.3)],
    [(440, 0.3), (550, 0.3)], [(550, 0.3), (440, 0.3)],
    [(220, 0.2), (330, 0.2), (440, 0.2)],
    [(440, 0.2), (330, 0.2), (220, 0.2)],
    [(262, 0.2), (550, 0.2)], [(550, 0.2), (262, 0.2)],
]

all_h = []
all_targets = []

for tones in seq_defs:
    for rep in range(10):
        parts = []
        for freq, dur in tones:
            t = np.arange(int(SR * dur), dtype=np.float32) / SR
            phase = np.random.uniform(0, 2 * np.pi)
            parts.append((0.5 * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32))
        seq = np.concatenate(parts)

        substrate.reset()
        for i in range(len(seq) // frame_size):
            substrate.step(seq[i*frame_size:(i+1)*frame_size])
            h = substrate.hidden_state.copy()
            all_h.append(h)

            # Target frequency for this frame
            frame_time = (i * frame_size) / SR
            elapsed = 0
            target_freq = tones[0][0]
            for f, d in tones:
                if frame_time < elapsed + d:
                    target_freq = f
                    break
                elapsed += d

            freq_norm = np.clip((target_freq - 80.0) / (2000.0 - 80.0), 0.0, 1.0)
            target = np.zeros(8, dtype=np.float32)
            target[0] = freq_norm; target[1] = 0.5; target[3] = 1.0
            all_targets.append(target)

hidden_states = np.array(all_h, dtype=np.float32)
target_params = np.array(all_targets, dtype=np.float32)
print(f"Training: {len(all_h)} samples")

history = controller.train_supervised(hidden_states, target_params, lr=0.005, n_epochs=1000)
print(f"Final loss: {history[-1]:.6f}")

# Save
Path('data/m9_controller_temporal_v3.ckpt.json').write_text(json.dumps({
    'W1': controller._W1.tolist(), 'b1': controller._b1.tolist(),
    'W2': controller._W2.tolist(), 'b2': controller._b2.tolist(),
}))

# Evaluate: check per-frame output
print("\n=== Evaluation ===")

sim = AcousticSimilarity(sr=SR)

# Build test sequences
def make_seq(tones):
    parts = []
    for freq, dur in tones:
        t = np.arange(int(SR*dur), dtype=np.float32)/SR
        parts.append((0.5*np.sin(2*np.pi*freq*t)).astype(np.float32))
    return np.concatenate(parts)

def eval_seq(seq, n_tones):
    substrate.reset()
    all_params = []
    for i in range(len(seq)//frame_size):
        substrate.step(seq[i*frame_size:(i+1)*frame_size])
        p = controller.decode(substrate.hidden_state)
        all_params.append(p)

    # Generate imitation
    imitation_frames = []
    sub2 = AcousticSubstrate(hidden_dim=32, sr=SR, frame_size=frame_size)
    sub2._W_h = substrate._W_h.copy()
    sub2._W_x = substrate._W_x.copy()
    sub2._b = substrate._b.copy()
    sub2.reset()

    n_frames = len(seq)//frame_size
    for tone_idx in range(n_tones):
        for f in range(n_frames//n_tones):
            h = sub2.hidden_state
            p = controller.decode(h)
            t_f = np.arange(frame_size,dtype=np.float32)/SR
            frame = (p.amplitude*np.sin(2*np.pi*p.freq_hz*t_f + p.phase)).astype(np.float32)
            imitation_frames.append(frame)
            sub2.step(frame)

    target = make_seq([(220,0.3),(440,0.3)])[:len(seq)]
    imitation = np.concatenate(imitation_frames)[:len(target)]
    score = sim.overall_score(target, imitation)

    # Show per-frame
    for i, p in enumerate(all_params[:n_frames]):
        tone = 'A' if i < n_frames//2 else 'B'
        print(f"  Frame {i} ({tone}): {p.freq_hz:.1f} Hz")

    first = np.mean([p.freq_hz for p in all_params[:n_frames//2]])
    second = np.mean([p.freq_hz for p in all_params[n_frames//2:]])
    print(f"  First half: {first:.1f} Hz  Second half: {second:.1f} Hz  Diff: {second-first:.1f} Hz  Score: {score:.4f}")
    return score

test_seq = make_seq([(220, 0.3), (440, 0.3)])
print("\nTest: 220->440 Hz")
score1 = eval_seq(test_seq, 2)

test_seq2 = make_seq([(440, 0.3), (220, 0.3)])
print("\nTest: 440->220 Hz")
score2 = eval_seq(test_seq2, 2)

test_seq3 = make_seq([(220, 0.2), (330, 0.2), (440, 0.2)])
print("\nTest: 220->330->440 Hz (3-tone)")
score3 = eval_seq(test_seq3, 3)

print(f"\nMean score: {(score1+score2+score3)/3:.4f}")
