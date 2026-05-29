#!/usr/bin/env python3
"""M9 Temporal Sequence Training — 2-tone stimulus imitation.

Trains the controller to produce time-varying oscillator params that match
a multi-tone stimulus sequence (e.g., 220 Hz -> 330 Hz).

Key idea: the substrate's hidden state evolves during the sequence, encoding
temporal context. The controller learns: hidden_state_at_time_t -> params_at_time_t.
"""

import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank, AcousticProducer, AcousticSimilarity, NeuralController

SR = 16000

# Load live-trained substrate
sub_data = json.load(open('data/m9_substrate_live.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_data['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_data['W_x'], dtype=np.float32)
substrate._b = np.array(sub_data['b'], dtype=np.float32)

osc = OscillatorBank(sr=SR)

def generate_2tone_sequence(freq1, freq2, duration_each=0.3, sr=SR):
    """Generate a 2-tone sequence: freq1 for duration_each, then freq2."""
    t1 = np.arange(int(sr * duration_each), dtype=np.float32) / sr
    t2 = np.arange(int(sr * duration_each), dtype=np.float32) / sr
    tone1 = 0.5 * np.sin(2 * np.pi * freq1 * t1)
    tone2 = 0.5 * np.sin(2 * np.pi * freq2 * t2 + np.pi)  # phase flip for clarity
    return np.concatenate([tone1, tone2]).astype(np.float32)

def play_capture(pcm, out_dev=2, in_dev=1, sr=SR):
    import sounddevice as sd
    pcm = pcm.astype(np.float32).flatten()
    info = sd.query_devices(in_dev)
    in_sr = int(info['default_samplerate'] or sr)
    lead = int(0.1 * in_sr)
    total = int((0.1 + len(pcm)/sr + 0.1) * in_sr)
    captured = np.zeros(total, dtype=np.float32)
    pos = 0
    started = False
    stream = sd.InputStream(device=in_dev, channels=1, samplerate=in_sr, dtype='float32', blocksize=1024)
    stream.start()
    try:
        while pos < total:
            if not started and pos >= lead:
                sd.play(pcm, samplerate=sr, device=out_dev, blocking=False)
                started = True
            data, _ = stream.read(min(1024, total - pos))
            n = min(len(data), total - pos)
            captured[pos:pos+n] = data[:n, 0]
            pos += n
    finally:
        stream.stop()
        stream.close()
        sd.wait()
    if in_sr != sr:
        from scipy.signal import resample_poly
        captured = resample_poly(captured, sr, in_sr).astype(np.float32)
    return captured

# Define 2-tone sequences to train on
tone_pairs = [
    (220, 330),
    (330, 220),
    (262, 392),
    (392, 262),
    (440, 550),
    (550, 440),
    (330, 440),
    (220, 440),
    (440, 220),
    (550, 660),
    (660, 550),
    (262, 550),
    (392, 660),
]

frame_size = 480
n_frames_per_tone = int(SR * 0.3) // frame_size  # ~10 frames per tone

print(f"Training on {len(tone_pairs)} 2-tone sequences...")
print(f"  {n_frames_per_tone} frames per tone, {frame_size} samples per frame")

# Collect training data: for each sequence, record hidden states at each frame
# and the target oscillator params at that point
all_hidden = []
all_targets = []

for pair_idx, (freq1, freq2) in enumerate(tone_pairs):
    for repeat in range(5):  # 5 variants per pair with random phase
        # Generate sequence with random phase offset
        phase_offset = np.random.uniform(0, 2*np.pi)
        seq = generate_2tone_sequence(freq1, freq2, duration_each=0.3)
        
        # Play and capture
        captured = play_capture(seq, out_dev=2, in_dev=1)
        
        # Encode through substrate frame by frame, recording hidden state at each frame
        substrate.reset()
        for frame_idx in range(len(captured) // frame_size):
            frame = captured[frame_idx*frame_size:(frame_idx+1)*frame_size]
            substrate.step(frame)
            h = substrate.hidden_state.copy()
            all_hidden.append(h)
            
            # Target: which tone are we in?
            if frame_idx < n_frames_per_tone:
                target_freq = freq1
            else:
                target_freq = freq2
            
            freq_norm = np.clip((target_freq - 80.0) / (2000.0 - 80.0), 0.0, 1.0)
            target = np.zeros(8, dtype=np.float32)
            target[0] = freq_norm
            target[1] = 0.5  # amplitude
            target[2] = 0.0  # phase
            target[3] = 1.0  # sine
            all_targets.append(target)
    
    if (pair_idx + 1) % 5 == 0:
        print(f"  {pair_idx + 1}/{len(tone_pairs)} sequences collected")

hidden_states = np.array(all_hidden, dtype=np.float32)
target_params = np.array(all_targets, dtype=np.float32)
print(f"\nTraining data: {len(all_hidden)} frame-level samples")

# Train controller
controller = NeuralController(hidden_dim=32, sr=SR, controller_hidden_dim=64)
print(f"Training controller ({controller.n_params} params)...")
history = controller.train_supervised(hidden_states, target_params, lr=0.01, n_epochs=500)

# Save
Path('data/m9_controller_temporal_v1.ckpt.json').write_text(json.dumps({
    'W1': controller._W1.tolist(), 'b1': controller._b1.tolist(),
    'W2': controller._W2.tolist(), 'b2': controller._b2.tolist(),
    'tone_pairs': tone_pairs,
    'final_loss': history[-1],
}))
print(f"Saved to data/m9_controller_temporal_v1.ckpt.json")

# === Evaluate: test on live 2-tone sequences ===
print(f"\n=== Live 2-Tone Evaluation ===")
sim = AcousticSimilarity(sr=SR)

# Helper: play sequence and capture at each frame, running controller live
def test_live_2tone(freq1, freq2):
    # Play stimulus and capture
    seq = generate_2tone_sequence(freq1, freq2)
    captured_stim = play_capture(seq, out_dev=2, in_dev=1)
    
    # Encode full stimulus through substrate
    substrate.reset()
    for i in range(len(captured_stim) // frame_size):
        substrate.step(captured_stim[i*frame_size:(i+1)*frame_size])
    
    # Run imitation: controller reads evolving substrate state each frame
    imitation_parts = []
    test_sub = AcousticSubstrate(hidden_dim=32, sr=SR, frame_size=frame_size)
    test_sub._W_h = substrate._W_h.copy()
    test_sub._W_x = substrate._W_x.copy()
    test_sub._b = substrate._b.copy()
    test_sub.reset()
    
    # Process stimulus frames to build up substrate state
    for i in range(len(captured_stim) // frame_size):
        frame = captured_stim[i*frame_size:(i+1)*frame_size]
        test_sub.step(frame)
    
    # Now produce imitation: run frame by frame with controller
    test_producer = AcousticProducer(sr=SR, hidden_dim=32)
    test_producer.substrate = test_sub
    test_producer.controller = controller
    test_producer.similarity = sim
    
    n_total_frames = len(seq) // frame_size
    for frame_idx in range(n_total_frames):
        h = test_producer.substrate.hidden_state
        params = controller.decode(h)
        
        # Generate one frame
        t_f = np.arange(frame_size, dtype=np.float32) / SR
        frame_audio = params.amplitude * np.sin(2 * np.pi * params.freq_hz * t_f + params.phase)
        frame_audio = frame_audio.astype(np.float32)
        
        imitation_parts.append(frame_audio)
        test_producer.substrate.step(frame_audio)
    
    imitation = np.concatenate(imitation_parts)
    
    # Play and capture the imitation
    captured_imit = play_capture(imitation, out_dev=2, in_dev=1)
    
    # Score
    min_len = min(len(captured_stim), len(captured_imit))
    score = sim.detailed_score(captured_stim[:min_len], captured_imit[:min_len])
    return score

# Test on a few pairs
test_pairs = [(220, 330), (330, 220), (440, 550), (550, 440), (220, 440)]
for freq1, freq2 in test_pairs:
    score = test_live_2tone(freq1, freq2)
    print(f"  {freq1}->{freq2} Hz: score={score.overall:.4f}")

print("\nDone.")
