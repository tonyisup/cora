#!/usr/bin/env python3
"""M9 Formant Live — simplified debug version."""

import sys, json, time
import numpy as np
sys.path.insert(0, 'packages')
from m9_sound import AcousticSubstrate, OscillatorBank
import sounddevice as sd

SR = 16000
FRAME_SIZE = 480

sub_ckpt = json.load(open('data/m9_substrate_formant.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

lookup_data = json.load(open('data/m9_formant_lookup_v3.json'))
vowel_names = lookup_data['vowel_names']
lookup = lookup_data['lookup']

print(f"Vowels: {vowel_names}")
n_callback = [0]
rms_vals = []

def callback(indata, outdata, frames, time_info, status):
    n_callback[0] += 1
    
    mic = indata[:, 0].astype(np.float32)
    rms = np.sqrt(np.mean(mic ** 2))
    rms_vals.append(rms)
    
    if rms > 0.003:
        substrate.reset()
        n_frames = len(mic) // FRAME_SIZE
        for i in range(n_frames):
            frame = mic[i*FRAME_SIZE:(i+1)*FRAME_SIZE]
            if len(frame) == FRAME_SIZE:
                substrate.step(frame.flatten())
        
        h = substrate.hidden_state
        best_v = vowel_names[0]
        best_d = float('inf')
        for v in vowel_names:
            c = np.array(lookup[v]['centroid'], dtype=np.float32)
            d = np.linalg.norm(h - c)
            if d < best_d:
                best_d = d; best_v = v
        
        conf = max(0, 1 - best_d / 6.0)
        
        template = lookup[best_v]['template']
        cum_phase = 0.0
        audio_frames = []
        for _ in range(max(n_frames, 1)):
            t_f = np.arange(FRAME_SIZE, dtype=np.float32) / SR
            frame_audio = np.zeros(FRAME_SIZE, dtype=np.float32)
            for fk, ak in [('f1', 'a1'), ('f2', 'a2'), ('f3', 'a3')]:
                frame_audio += template[ak] * np.sin(2*np.pi*template[fk]*t_f + cum_phase)
                cum_phase += 2*np.pi*template[fk]*(FRAME_SIZE/SR)
            peak = np.max(np.abs(frame_audio))
            if peak > 1.0: frame_audio /= peak
            audio_frames.append(frame_audio.astype(np.float32))
            cum_phase = cum_phase % (2 * np.pi)
        
        audio = np.concatenate(audio_frames)
        n_out = len(outdata)
        if len(audio) >= n_out:
            outdata[:] = audio[:n_out].reshape(-1, 1)
        else:
            outdata[:len(audio)] = audio.reshape(-1, 1)
            outdata[len(audio):] = 0
        
        print(f"  #{n_callback[0]} rms={rms:.3f} -> {best_v} (conf={conf:.2f})", flush=True)
    else:
        outdata[:] = 0

print("Starting stream (10s)... speak into mic", flush=True)
try:
    with sd.Stream(
        device=(1, 2),
        samplerate=SR,
        blocksize=SR // 2,  # 0.5s blocks for more audio per callback
        channels=(1, 1),
        callback=callback,
        dtype='float32',
    ):
        time.sleep(10)
except KeyboardInterrupt:
    pass

print(f"\nCallbacks: {n_callback[0]}")
if rms_vals:
    print(f"RMS: min={min(rms_vals):.4f}, max={max(rms_vals):.4f}, mean={np.mean(rms_vals):.4f}")
