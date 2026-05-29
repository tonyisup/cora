#!/usr/bin/env python3
"""M9 Temporal Live Demo — hear multi-tone imitation on hardware.

Plays stimulus sequences and their imitations through headphones.
Each demo: stimulus (2-4 tones) -> imitation (matching pitch pattern).
"""

import sys, json, time
sys.path.insert(0, 'packages')
import numpy as np
from m9_sound import AcousticProducer, OscillatorBank, AcousticSubstrate
import sounddevice as sd

SR = 16000

# Load temporal controller + live substrate
producer = AcousticProducer(sr=SR, hidden_dim=32)
ckpt = json.load(open('data/m9_controller_temporal_v1.ckpt.json'))
producer.controller._W1 = np.array(ckpt['W1'], dtype=np.float32)
producer.controller._b1 = np.array(ckpt['b1'], dtype=np.float32)
producer.controller._W2 = np.array(ckpt['W2'], dtype=np.float32)
producer.controller._b2 = np.array(ckpt['b2'], dtype=np.float32)

sub_ckpt = json.load(open('data/m9_substrate_live.ckpt.json'))
producer.substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
producer.substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
producer.substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

osc = OscillatorBank(sr=SR)

def play_seq_and_imitation(tones, stim_dur=0.3, imit_dur=0.3):
    """Play stimulus sequence, then generate and play imitation."""
    # Build stimulus
    parts = []
    for freq, dur in tones:
        t = np.arange(int(SR * dur), dtype=np.float32) / SR
        phase = np.random.uniform(0, 2*np.pi)
        parts.append((0.5 * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32))
    stimulus = np.concatenate(parts)
    
    # Play stimulus
    sd.play(stimulus, samplerate=SR, device=2, blocking=False)
    sd.wait()
    time.sleep(0.3)
    
    # Encode stimulus through substrate
    producer.substrate.reset()
    for i in range(len(stimulus) // 480):
        producer.substrate.step(stimulus[i*480:(i+1)*480])
    
    # Generate imitation using controller
    n_frames = int(SR * imit_dur) // 480
    for tone_idx in range(len(tones)):
        for frame in range(n_frames):
            h = producer.substrate.hidden_state
            params = producer.controller.decode(h)
            t_f = np.arange(480, dtype=np.float32) / SR
            frame_audio = params.amplitude * np.sin(2 * np.pi * params.freq_hz * t_f + params.phase)
            frame_audio = frame_audio.astype(np.float32)
            sd.play(frame_audio, samplerate=SR, device=2, blocking=False)
            sd.wait()
            producer.substrate.step(frame_audio)
    
    time.sleep(0.3)

print("=" * 72)
print("M9 Temporal Live Demo")
print("=" * 72)
print("Demo: 2-tone up")
play_seq_and_imitation([(220, 0.3), (440, 0.3)])
time.sleep(0.3)

print("Demo: 3-tone up")
play_seq_and_imitation([(220, 0.2), (330, 0.2), (440, 0.2)])
time.sleep(0.3)

print("Demo: 4-tone scale")
play_seq_and_imitation([(220, 0.15), (262, 0.15), (330, 0.15), (440, 0.15)])
time.sleep(0.3)

print("Demo: down-up")
play_seq_and_imitation([(440, 0.2), (220, 0.2), (440, 0.2)])
time.sleep(0.3)

print("Demo: same tone x3")
play_seq_and_imitation([(330, 0.2), (330, 0.2), (330, 0.2)])
time.sleep(0.3)

print("\nDone!")
