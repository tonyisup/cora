#!/usr/bin/env python3
"""Train controller on pre-collected live data."""

import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank, AcousticSimilarity
from m9_sound.controller import NeuralController

SR = 16000

# Load live substrate
sub_data = json.load(open('data/m9_substrate_live.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_data['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_data['W_x'], dtype=np.float32)
substrate._b = np.array(sub_data['b'], dtype=np.float32)

# Load collected training data
data = json.load(open('data/live_training_data.json'))
hidden_states = np.array(data['hidden_states'], dtype=np.float32)
target_params = np.array(data['targets'], dtype=np.float32)
freq_bins = data['freq_bins']
print(f"Loaded {len(hidden_states)} training samples for {len(freq_bins)} frequencies")

# Play evaluation stimuli and capture
osc = OscillatorBank(sr=SR)

def play_capture(pcm):
    import sounddevice as sd
    pcm = pcm.astype(np.float32).flatten()
    info = sd.query_devices(1)
    in_sr = int(info['default_samplerate'] or SR)
    lead = int(0.1 * in_sr)
    total = int((0.1 + len(pcm)/SR + 0.1) * in_sr)
    captured = np.zeros(total, dtype=np.float32)
    pos = 0
    started = False
    stream = sd.InputStream(device=1, channels=1, samplerate=in_sr, dtype='float32', blocksize=1024)
    stream.start()
    try:
        while pos < total:
            if not started and pos >= lead:
                sd.play(pcm, samplerate=SR, device=2, blocking=False)
                started = True
            data, _ = stream.read(min(1024, total - pos))
            n = min(len(data), total - pos)
            captured[pos:pos+n] = data[:n, 0]
            pos += n
    finally:
        stream.stop()
        stream.close()
        sd.wait()
    if in_sr != SR:
        from scipy.signal import resample_poly
        captured = resample_poly(captured, SR, in_sr).astype(np.float32)
    return captured

# Try different controller configs
for hidden_dim, n_epochs in [(32, 500), (64, 500)]:
    print(f"\n=== Training controller: {hidden_dim} hidden, {n_epochs} epochs ===")
    controller = NeuralController(hidden_dim=32, sr=SR, controller_hidden_dim=hidden_dim)
    print(f"  {controller.n_params} params, {len(hidden_states)} training samples")
    history = controller.train_supervised(hidden_states, target_params, lr=0.01, n_epochs=n_epochs)

    # Evaluate on live captures
    sim = AcousticSimilarity(sr=SR)
    print(f"\n  Live evaluation:")
    for freq in freq_bins:
        phase = np.random.uniform(0, 2*np.pi)
        stim = osc.sine(freq, 1.0, amplitude=0.5, phase=phase)
        captured = play_capture(stim)
        substrate.reset()
        for i in range(len(captured) // 480):
            substrate.step(captured[i*480:(i+1)*480])
        params = controller.decode(substrate.hidden_state)
        imitation = osc.sine(params.freq_hz, 1.0, amplitude=params.amplitude)
        imit_capt = play_capture(imitation)
        min_len = min(len(captured), len(imit_capt))
        score = sim.detailed_score(captured[:min_len], imit_capt[:min_len])
        error = abs(freq - params.freq_hz)
        print(f"    {freq:3d} Hz -> {params.freq_hz:6.1f} Hz (err={error:5.1f}, score={score.overall:.4f})")

    # Save
    ckpt_path = f'data/m9_controller_live_h{hidden_dim}.ckpt.json'
    Path(ckpt_path).write_text(json.dumps({
        'W1': controller._W1.tolist(), 'b1': controller._b1.tolist(),
        'W2': controller._W2.tolist(), 'b2': controller._b2.tolist(),
    }))
    print(f"  Saved to {ckpt_path}")
