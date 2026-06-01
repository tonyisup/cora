#!/usr/bin/env python3
"""M9 Temporal Training with Closed-Loop Feedback.

Key insight: the controller was trained on hidden states from CLEAN synthetic audio,
but during imitation it sees hidden states from its own noisy output.
This distribution shift causes frame-to-frame jitter.

Solution: train with closed-loop feedback where the controller's own output
feeds back through the substrate during training.
"""

import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank, AcousticProducer, AcousticSimilarity
from m9_sound.controller import NeuralController

SR = 16000

# Load live substrate
sub_ckpt = json.load(open('data/m9_substrate_live.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

# Create controller
controller = NeuralController(hidden_dim=32, sr=SR, controller_hidden_dim=64)
print(f"Controller: {controller.n_params} params")

# Generate temporal training targets
# Each sequence: list of (freq, amplitude, phase, waveform) for each frame
frame_size = 480

def make_sequence(freqs_and_durs, sr=SR, frame_size=480):
    """Generate a temporal sequence with per-frame target params."""
    frames = []
    for freq, dur in freqs_and_durs:
        n_frames = int(sr * dur) // frame_size
        for _ in range(n_frames):
            freq_norm = np.clip((freq - 80.0) / (2000.0 - 80.0), 0.0, 1.0)
            frames.append({
                'freq_norm': freq_norm,
                'amplitude': 0.5,
                'phase_norm': 0.0,
                'waveform': 0,  # sine
            })
    return frames

# Training sequences
seq_defs = [
    # 2-tone patterns
    [(220, 0.3), (330, 0.3)],
    [(330, 0.3), (220, 0.3)],
    [(262, 0.3), (392, 0.3)],
    [(392, 0.3), (262, 0.3)],
    [(440, 0.3), (550, 0.3)],
    [(550, 0.3), (440, 0.3)],
    [(220, 0.2), (330, 0.2), (440, 0.2)],
    [(440, 0.2), (330, 0.2), (220, 0.2)],
    [(220, 0.15), (262, 0.15), (330, 0.15), (440, 0.15)],
    [(330, 0.3), (330, 0.3)],  # same tone
    [(440, 0.4), (220, 0.2)],  # long-short
    [(220, 0.2), (440, 0.4)],  # short-long
    [(550, 0.25), (660, 0.25)],  # high freq pair
    [(660, 0.25), (550, 0.25)],
    [(262, 0.2), (550, 0.2)],  # wide jump
    [(550, 0.2), (262, 0.2)],
]

target_sequences = [make_sequence(sd) for sd in seq_defs]
n_frames = len(target_sequences[0])
n_samples = sum(len(s) for s in target_sequences)
print(f"Training: {len(target_sequences)} sequences, {n_frames} frames each, {n_samples} total samples")

# Train with closed-loop feedback
history = []
lr = 0.01
n_epochs = 500

print(f"\nTraining (lr={lr}, epochs={n_epochs})...")
for epoch in range(n_epochs):
    total_loss = 0.0
    n_total = 0

    for seq_idx, target_seq in enumerate(target_sequences):
        substrate.reset()

        for frame_idx, target in enumerate(target_seq):
            h = substrate.hidden_state.copy()

            # Forward pass through controller
            raw, a1 = controller._forward_all(h)

            # Compute loss
            target_freq = target['freq_norm']
            target_amp = target['amplitude']
            target_phase = target['phase_norm']

            # Frequency: linear output
            err_freq = raw[0] - target_freq

            # Amplitude: sigmoid
            amp_sig = 1.0 / (1.0 + np.exp(-raw[1]))
            err_amp = amp_sig - target_amp

            # Phase: tanh
            phase_tanh = np.tanh(raw[2])
            err_phase = phase_tanh - target_phase

            # Waveform: softmax CE
            wf_logits = raw[3:3+controller._n_waveforms]
            wf_sm = np.exp(wf_logits - np.max(wf_logits))
            wf_sm /= wf_sm.sum()
            target_wf = np.zeros(controller._n_waveforms, dtype=np.float32)
            target_wf[target['waveform']] = 1.0
            err_wf = wf_sm - target_wf

            loss = err_freq**2 + err_amp**2 + err_phase**2 + np.sum(err_wf**2)
            total_loss += loss
            n_total += 1

            # Backprop through controller
            delta = np.zeros(4 + controller._n_waveforms, dtype=np.float32)
            delta[0] = err_freq
            delta[1] = err_amp * amp_sig * (1.0 - amp_sig)
            delta[2] = err_phase * (1.0 - phase_tanh**2)
            delta[3:3+controller._n_waveforms] = err_wf

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
                if gn > max_grad: g *= max_grad / gn

            controller._W2 -= lr * dW2; controller._b2 -= lr * db2
            controller._W1 -= lr * dW1; controller._b1 -= lr * db1

            # CRITICAL: generate output frame and feed back through substrate
            with np.errstate(all='ignore'):
                freq_hz = 80.0 + float(np.clip(raw[0], 0, 1)) * (2000.0 - 80.0)
                amp = float(1.0 / (1.0 + np.exp(-raw[1])))
                phase = float(np.tanh(raw[2]) * np.pi)
                t_f = np.arange(frame_size, dtype=np.float32)/SR
                audio_frame = (amp * np.sin(2 * np.pi * freq_hz * t_f + phase)).astype(np.float32)

            # Feed back through substrate (closed loop)
            substrate.step(audio_frame)

    avg_loss = total_loss / n_total
    history.append(float(avg_loss))
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.6f}")

# Save checkpoint
Path('data/m9_controller_temporal_v2.ckpt.json').write_text(json.dumps({
    'W1': controller._W1.tolist(), 'b1': controller._b1.tolist(),
    'W2': controller._W2.tolist(), 'b2': controller._b2.tolist(),
    'seq_defs': [(f, d) for f, d in [(220, 0.3), (330, 0.3)]],  # just for reference
    'final_loss': history[-1],
}))
print(f"\nSaved to data/m9_controller_temporal_v2.ckpt.json")

# === Evaluate ===
print(f"\n=== Evaluation ===")
osc2 = OscillatorBank(sr=SR)
sim = AcousticSimilarity(sr=SR)

# Helper: evaluate a 2-tone sequence
def evaluate_seq(freq1, freq2, dur=0.3):
    # Build target stimulus
    t1 = np.arange(int(SR*dur), dtype=np.float32)/SR
    t2 = np.arange(int(SR*dur), dtype=np.float32)/SR
    stim = np.concatenate([
        0.5*np.sin(2*np.pi*freq1*t1),
        0.5*np.sin(2*np.pi*freq2*t2),
    ]).astype(np.float32)

    # Encode stimulus through substrate
    substrate.reset()
    for i in range(len(stim)//frame_size):
        substrate.step(stim[i*frame_size:(i+1)*frame_size])

    # Generate imitation using controller (frame by frame, closed loop)
    n_frames = int(SR*dur)//frame_size
    imit_frames = []
    for frame_idx in range(n_frames*2):
        h = substrate.hidden_state
        raw, a1 = controller._forward_all(h)
        freq_hz = 80.0 + float(np.clip(raw[0], 0, 1))*(2000.0-80.0)
        amp = float(1.0/(1.0+np.exp(-raw[1])))
        phase = float(np.tanh(raw[2])*np.pi)
        t_f = np.arange(frame_size, dtype=np.float32)/SR
        frame = (amp*np.sin(2*np.pi*freq_hz*t_f + phase)).astype(np.float32)
        imit_frames.append(frame)
        substrate.step(frame)

    # Score
    target_single = np.concatenate([
        0.5*np.sin(2*np.pi*freq1*np.arange(int(SR*dur),dtype=np.float32)/SR),
        0.5*np.sin(2*np.pi*freq2*np.arange(int(SR*dur),dtype=np.float32)/SR),
    ]).astype(np.float32)

    imitation = np.concatenate([imit_frames[:n_frames], imit_frames[n_frames:]])
    # Compare first half to first target, second half to second target
    score = sim.overall_score(target_single, imitation[:len(target_single)])
    return score

test_pairs = [
    (220, 330), (330, 220), (440, 550), (550, 440), (220, 440),
    (262, 392), (392, 262), (330, 440),
]

for f1, f2 in test_pairs:
    score = evaluate_seq(f1, f2)
    print(f"  {f1}->{f2} Hz: score={score:.4f}")

print("\nDone.")
