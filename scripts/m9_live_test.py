#!/usr/bin/env python3
"""M9 Live Hardware Test — Multi-frequency imitation with supervised controller.

Tests the full pipeline on real audio hardware:
1. Play stimulus on device 3 (Mac mini Speakers)
2. Capture through device 1 (BL-2021USB mic)
3. Encode through pre-trained substrate
4. Controller predicts oscillator params from hidden state
5. Produce imitation on device 2 (External Headphones)
6. Capture and score the imitation
"""

import sys, json
import numpy as np
sys.path.insert(0, 'packages')

from m9_sound import AcousticProducer, OscillatorBank, AcousticSimilarity

SR = 16000

# Load the supervised controller
producer = AcousticProducer(sr=SR, hidden_dim=32)
ckpt = json.load(open('data/m9_controller_v8.ckpt.json'))
producer.controller._W1 = np.array(ckpt['controller']['W1'], dtype=np.float32)
producer.controller._b1 = np.array(ckpt['controller']['b1'], dtype=np.float32)
producer.controller._W2 = np.array(ckpt['controller']['W2'], dtype=np.float32)
producer.controller._b2 = np.array(ckpt['controller']['b2'], dtype=np.float32)

# Load pre-trained substrate
sub_ckpt = json.load(open('data/m9_substrate_classifier.ckpt.json'))
producer.substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
producer.substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
producer.substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

osc = OscillatorBank(sr=SR)
sim = AcousticSimilarity(sr=SR)

# Play and capture stimulus
def play_and_capture(pcm, input_device=1, output_device=3, sr=SR):
    import sounddevice as sd
    import time
    if pcm.ndim != 1:
        pcm = pcm.reshape(-1)
    pcm = pcm.astype(np.float32)
    input_info = sd.query_devices(input_device)
    input_sr = int(input_info['default_samplerate'] or sr)
    playback_duration = len(pcm) / sr
    total_duration = 0.25 + playback_duration + 0.25
    total_samples = int(total_duration * input_sr)
    captured = np.zeros(total_samples, dtype=np.float32)
    lead_in_samples = int(0.25 * input_sr)
    pos = 0
    stream = sd.InputStream(device=input_device, channels=1, samplerate=input_sr, dtype='float32', blocksize=1024)
    stream.start()
    while pos < total_samples:
        data, _ = stream.read(min(1024, total_samples - pos))
        n = min(len(data), total_samples - pos)
        captured[pos:pos+n] = data[:n, 0]
        if pos >= lead_in_samples and pos == lead_in_samples + n:
            sd.play(pcm, samplerate=sr, device=output_device, blocking=False)
        pos += n
    stream.stop()
    stream.close()
    sd.wait()
    if input_sr != sr:
        from scipy.signal import resample_poly
        captured = resample_poly(captured, sr, input_sr).astype(np.float32)
    return captured

print("=" * 72)
print("M9 Live Hardware Test — Supervised Multi-Frequency Imitation")
print("=" * 72)

freqs = [220, 330, 440, 550, 660]

for freq in freqs:
    print(f"\n-- {freq} Hz --")
    
    # 1. Generate and play stimulus
    stimulus = osc.sine(freq, 1.0, amplitude=0.5)
    print(f"  Playing {freq} Hz stimulus on device 3...")
    stimulus_capture = play_and_capture(stimulus, input_device=1, output_device=3)
    
    # 2. Encode through substrate
    producer.substrate.reset()
    frame_size = 480
    n_frames = len(stimulus_capture) // frame_size
    for i in range(n_frames):
        frame = stimulus_capture[i*frame_size:(i+1)*frame_size]
        producer.substrate.step(frame)
    
    # 3. Controller predicts params
    params = producer.controller.decode(producer.substrate.hidden_state)
    print(f"  Controller: freq={params.freq_hz:.1f} Hz, amp={params.amplitude:.3f}, wf={params.waveform}")
    
    # 4. Generate and play imitation
    osc2 = OscillatorBank(sr=SR)
    wf_map = [osc2.sine, osc2.triangle, osc2.sawtooth, osc2.square]
    imitation = wf_map[params.waveform](params.freq_hz, 1.0, amplitude=params.amplitude)
    print(f"  Playing imitation on device 2...")
    imitation_capture = play_and_capture(imitation, input_device=1, output_device=2)
    
    # 5. Score: compare captured imitation to captured stimulus
    # Trim to same length
    min_len = min(len(stimulus_capture), len(imitation_capture))
    stim_trim = stimulus_capture[:min_len]
    imit_trim = imitation_capture[100:min_len]  # skip lead-in
    stim_trim = stim_trim[:len(imit_trim)]
    
    score = sim.detailed_score(stim_trim, imit_trim)
    print(f"  Score: {score.overall:.4f}")

print("\nDone.")
