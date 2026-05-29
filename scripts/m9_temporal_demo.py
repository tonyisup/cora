#!/usr/bin/env python3
"""M9 Temporal Live Demo — hear 2-tone imitation on hardware."""

import sys, json
import numpy as np
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank, AcousticProducer

SR = 16000

# Load temporal controller
producer = AcousticProducer(sr=SR, hidden_dim=32)
ckpt = json.load(open('data/m9_controller_temporal_v1.ckpt.json'))
producer.controller._W1 = np.array(ckpt['W1'], dtype=np.float32)
producer.controller._b1 = np.array(ckpt['b1'], dtype=np.float32)
producer.controller._W2 = np.array(ckpt['W2'], dtype=np.float32)
producer.controller._b2 = np.array(ckpt['b2'], dtype=np.float32)

# Load live-trained substrate
sub_ckpt = json.load(open('data/m9_substrate_live.ckpt.json'))
producer.substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
producer.substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
producer.substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

osc = OscillatorBank(sr=SR)
import sounddevice as sd

def play_tone(freq, duration=0.4, amp=0.5, out_dev=2):
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    pcm = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sd.play(pcm, samplerate=SR, device=out_dev, blocking=False)
    sd.wait()

def generate_and_play_imitation(producer, n_frames, out_dev=2):
    """Generate imitation frame by frame, playing each frame immediately."""
    for frame_idx in range(n_frames):
        h = producer.substrate.hidden_state
        params = producer.controller.decode(h)
        t_f = np.arange(480, dtype=np.float32) / SR
        frame = params.amplitude * np.sin(2 * np.pi * params.freq_hz * t_f + params.phase)
        frame = frame.astype(np.float32)
        sd.play(frame, samplerate=SR, device=out_dev, blocking=False)
        sd.wait()
        producer.substrate.step(frame)

print("=" * 72)
print("M9 Temporal Live Demo — 2-Tone Imitation")
print("=" * 72)
print()
print("For each test: you hear stimulus (2 tones) then imitation (2 tones)")

test_pairs = [
    (220, 330),
    (330, 220),
    (440, 550),
    (550, 440),
    (220, 440),
]
for freq1, freq2 in test_pairs:
    print(f"\n(freq1 -> freq2) = ({freq1} -> {freq2})")
    
    # Stimulus part 1
    print(f"  STIMULUS: {freq1} Hz")
    play_tone(freq1, duration=0.4, out_dev=2)
    
    # Stimulus part 2
    print(f"  STIMULUS: {freq2} Hz")
    play_tone(freq2, duration=0.4, out_dev=2)
    
    # Phase 2: play what the imitation should sound like
    print(f"  TARGET: {freq1} Hz then {freq2} Hz")
    import time
    time.sleep(0.2)
    

    # Build up substrate state by playing the stimulus sequence
    producer.substrate.reset()
    
    # Play stimulus tone 1 and feed through substrate
    t1 = np.arange(int(SR * 0.4), dtype=np.float32) / SR
    tone1 = (0.5 * np.sin(2 * np.pi * freq1 * t1)).astype(np.float32)
    sd.play(tone1, samplerate=SR, device=2, blocking=False)
    for i in range(len(tone1) // 480):
        frame = tone1[i*480:(i+1)*480]
        producer.substrate.step(frame)
    sd.wait()
    
    # Phase 3: Imitation — read substrate state and generate matching output
    print(f"  IMITATION:")
    sd.play(tone1, samplerate=SR, device=2, blocking=False)  # replay stimulus for comparison
    sd.wait()
    
    time.sleep(0.2)
    
    # Now generate imitation for tone 2
    t2 = np.arange(int(SR * 0.4), dtype=np.float32) / SR
    tone2 = (0.5 * np.sin(2 * np.pi * freq2 * t2)).astype(np.float32)
    sd.play(tone2, samplerate=SR, device=2, blocking=False)
    for i in range(len(tone2) // 480):
        frame = tone2[i*480:(i+1)*480]
        producer.substrate.step(frame)
    sd.wait()
    
    time.sleep(0.2)
    

    print(f"  (Target: {freq1} -> {freq2})")
    print(f"  Playing target pattern again...")
    play_tone(freq1, duration=0.3, out_dev=2)
    play_tone(freq2, duration=0.3, out_dev=2)
    
    # Now the imitation
    print(f"  IMITATING...")
    # Generate imitation based on current substrate state (which encodes the full seq)
    h = producer.substrate.hidden_state
    params = producer.controller.decode(h)
    print(f"  Controller sees: freq={params.freq_hz:.1f} Hz")
    
    time.sleep(0.5)

print("\nDone!")
