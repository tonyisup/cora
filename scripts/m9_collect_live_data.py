#!/usr/bin/env python3
"""Collect live training data and save to disk for reuse."""

import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank

SR = 16000

# Load live-trained substrate
sub_data = json.load(open('data/m9_substrate_live.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_data['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_data['W_x'], dtype=np.float32)
substrate._b = np.array(sub_data['b'], dtype=np.float32)

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

freq_bins = [220, 262, 330, 392, 440, 550, 660]
n_repeats = 15  # 7 x 15 = 105 captures

print(f"Collecting {len(freq_bins) * n_repeats} live training captures...")
all_h = []
all_targets = []
for i, freq in enumerate(freq_bins):
    for r in range(n_repeats):
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(0.3, 0.7)
        stim = osc.sine(freq, 1.0, amplitude=amp, phase=phase)
        captured = play_capture(stim)
        substrate.reset()
        for j in range(len(captured) // 480):
            substrate.step(captured[j*480:(j+1)*480])
        h = substrate.hidden_state.copy()
        all_h.append(h)
        freq_norm = np.clip((freq - 80.0) / (2000.0 - 80.0), 0.0, 1.0)
        target = np.zeros(8, dtype=np.float32)
        target[0] = freq_norm; target[1] = 0.5; target[2] = 0.0; target[3] = 1.0
        all_targets.append(target)
    print(f"  {freq} Hz done ({(i+1)*n_repeats}/{len(freq_bins)*n_repeats})")

data = {
    'hidden_states': np.array(all_h).tolist(),
    'targets': np.array(all_targets).tolist(),
    'freq_bins': freq_bins,
    'n_samples': len(all_h),
}
Path('data/live_training_data.json').write_text(json.dumps(data))
print(f"\nSaved {len(all_h)} training samples to data/live_training_data.json")
