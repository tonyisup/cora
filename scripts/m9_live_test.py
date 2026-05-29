#!/usr/bin/env python3
"""M9 Live Hardware Test — Multi-frequency imitation with supervised controller.

Tests the full pipeline on real audio hardware:
1. Play stimulus on device 2 (External Headphones) so user can hear
2. Capture through device 1 (BL-2021USB mic)
3. Encode through pre-trained substrate
4. Controller predicts oscillator params from hidden state
5. Produce imitation on device 2 (same headphones)
6. Play a beep between stimulus and imitation for clarity
"""

import sys, json
import numpy as np
sys.path.insert(0, 'packages')

from m9_sound import AcousticProducer, OscillatorBank, AcousticSimilarity

SR = 16000

# Load controller trained on live data (h64 — best live result)
producer = AcousticProducer(sr=SR, hidden_dim=32)
ckpt = json.load(open('data/m9_controller_live_h64.ckpt.json'))
producer.controller._W1 = np.array(ckpt['W1'], dtype=np.float32)
producer.controller._b1 = np.array(ckpt['b1'], dtype=np.float32)
producer.controller._W2 = np.array(ckpt['W2'], dtype=np.float32)
producer.controller._b2 = np.array(ckpt['b2'], dtype=np.float32)

# Load LIVE-trained substrate (frequency classifier)
sub_ckpt = json.load(open('data/m9_substrate_live.ckpt.json'))
producer.substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
producer.substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
producer.substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)
producer.substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
producer.substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
producer.substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

osc = OscillatorBank(sr=SR)
sim = AcousticSimilarity(sr=SR)

def play_and_capture(pcm, output_device=2, input_device=1, sr=SR):
    import sounddevice as sd
    if pcm.ndim != 1:
        pcm = pcm.reshape(-1)
    pcm = pcm.astype(np.float32)
    input_info = sd.query_devices(input_device)
    input_sr = int(input_info['default_samplerate'] or sr)
    playback_duration = len(pcm) / sr
    total_duration = 0.1 + playback_duration + 0.1
    total_samples = int(total_duration * input_sr)
    captured = np.zeros(total_samples, dtype=np.float32)
    lead_in_samples = int(0.1 * input_sr)
    pos = 0
    playback_started = False
    stream = sd.InputStream(device=input_device, channels=1, samplerate=input_sr, dtype='float32', blocksize=1024)
    stream.start()
    try:
        while pos < total_samples:
            if not playback_started and pos >= lead_in_samples:
                sd.play(pcm, samplerate=sr, device=output_device, blocking=False)
                playback_started = True
            data, _ = stream.read(min(1024, total_samples - pos))
            n = min(len(data), total_samples - pos)
            captured[pos:pos+n] = data[:n, 0]
            pos += n
    finally:
        stream.stop()
        stream.close()
        sd.wait()
    return captured

# Short beep for separation
beep = 0.3 * np.sin(2 * np.pi * 800 * np.arange(int(0.1 * SR)) / SR).astype(np.float32)

print("=" * 72)
print("M9 Live Hardware Test — Supervised Multi-Frequency Imitation")
print("=" * 72)
print()
print("Playing everything on device 2 (External Headphones)")
print("Listening on device 1 (BL-2021USB mic)")
print()

freqs = [220, 330, 440, 550, 660]

for freq in freqs:
    print(f"-- {freq} Hz --")
    
    # 1. Generate stimulus
    stimulus = osc.sine(freq, 1.0, amplitude=0.5)
    
    # 2. Play stimulus and capture
    print(f"  Playing {freq} Hz stimulus...")
    stim_capture = play_and_capture(stimulus, output_device=2)
    
    # Small pause
    import time
    time.sleep(0.3)
    
    # 3. Encode through substrate
    producer.substrate.reset()
    frame_size = 480
    n_frames = len(stim_capture) // frame_size
    for i in range(n_frames):
        frame = stim_capture[i*frame_size:(i+1)*frame_size]
        producer.substrate.step(frame)
    
    # 4. Controller predicts params
    params = producer.controller.decode(producer.substrate.hidden_state)
    print(f"  Controller: freq={params.freq_hz:.1f} Hz, amp={params.amplitude:.3f}")
    
    # 5. Generate and play imitation
    osc2 = OscillatorBank(sr=SR)
    wf_map = [osc2.sine, osc2.triangle, osc2.sawtooth, osc2.square]
    imitation = wf_map[params.waveform](params.freq_hz, 1.0, amplitude=params.amplitude)
    
    print(f"  Playing imitation...")
    imit_capture = play_and_capture(imitation, output_device=2)
    
    # 7. Score: compare captured imitation to captured stimulus
    min_len = min(len(stim_capture), len(imit_capture))
    stim_trim = stim_capture[:min_len]
    imit_trim = imit_capture[:min_len]
    
    score = sim.detailed_score(stim_trim, imit_trim)
    error = abs(freq - params.freq_hz)
    print(f"  Produced: {params.freq_hz:.1f} Hz (error: {error:.1f} Hz), score: {score.overall:.4f}")
    print()

print("Done.")
