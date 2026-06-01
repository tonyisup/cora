#!/usr/bin/env python3
"""M9 Live Vowel Imitation — real-time mic input.

Listens to mic input, encodes through substrate, classifies vowel,
and synthesizes imitation through headphones in real-time.
"""

import sys, json, time
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.signal import butter, sosfilt
sys.path.insert(0, 'packages')

from m9_sound import AcousticSubstrate, OscillatorBank
import sounddevice as sd

SR = 16000
FRAME_SIZE = 480  # 30ms per frame

# Load substrate
sub_ckpt = json.load(open('data/m9_substrate_formant.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)
print(f"Loaded substrate (acc: {sub_ckpt.get('final_accuracy', 'N/A')})")

# Load lookup table
lookup_data = json.load(open('data/m9_formant_lookup.json'))
vowel_names = lookup_data['vowel_names']
lookup = lookup_data['lookup']

# Build centroid tree for fast classification
all_centroids = []
all_vowels = []
for vowel in vowel_names:
    centroid = np.array(lookup[vowel]['centroid'], dtype=np.float32)
    all_centroids.append(centroid)
    all_vowels.append(vowel)
centroid_tree = cKDTree(np.array(all_centroids, dtype=np.float32))

# High-pass filter
hp_sos = butter(4, 100, btype='high', fs=SR, output='sos')

def highpass(signal):
    return sosfilt(hp_sos, signal).astype(np.float32)

def classify_vowel(h):
    dist, idx = centroid_tree.query(k=1, x=h)
    return all_vowels[idx]

def synthesize_vowel_frames(vowel, n_frames, prev_phase=0.0):
    """Generate audio frames for a vowel from its template."""
    template = lookup[vowel]['template']
    
    cum_phase = prev_phase
    frames = []
    
    for _ in range(n_frames):
        t_f = np.arange(FRAME_SIZE, dtype=np.float32) / SR
        frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        
        for freq_key, amp_key in [('f1', 'a1'), ('f2', 'a2'), ('f3', 'a3')]:
            freq = template[freq_key]
            amp = template[amp_key]
            phase_off = cum_phase
            sine = amp * np.sin(2 * np.pi * freq * t_f + phase_off)
            frame += sine
            cum_phase += 2 * np.pi * freq * (FRAME_SIZE / SR)
        
        peak = np.max(np.abs(frame))
        if peak > 1.0: frame /= peak
        frames.append(frame.astype(np.float32))
        cum_phase = cum_phase % (2 * np.pi)
    
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32), cum_phase

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-device', type=int, default=1)
    parser.add_argument('--output-device', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--hold', type=int, default=3)
    args = parser.parse_args()
    
    print(f"Input: device {args.input_device} (mic)")
    print(f"Output: device {args.output_device} (headphones)")
    print(f"Vowels: {vowel_names}")
    print(f"Threshold: {args.threshold}")
    print("\nSpeak vowels into the mic. Ctrl+C to stop.\n")
    
    state = {'prev_vowel': None, 'cum_phase': 0.0, 'hold': 0}
    
    def callback(indata, outdata, frames_count, time_info, status):
        if status:
            print(f"Status: {status}")
        
        mic = indata[:, 0].astype(np.float32)
        mic = highpass(mic)
        rms = np.sqrt(np.mean(mic ** 2))
        
        out_vowel = None
        
        if rms > args.threshold:
            substrate.reset()
            n_frames = len(mic) // FRAME_SIZE
            for i in range(n_frames):
                frame = mic[i*FRAME_SIZE:(i+1)*FRAME_SIZE]
                if len(frame) == FRAME_SIZE:
                    substrate.step(frame)
            
            h = substrate.hidden_state
            vowel = classify_vowel(h)
            out_vowel = vowel
            state['hold'] = args.hold
            
            if vowel != state['prev_vowel']:
                print(f"  [{rms:.3f}] {vowel}", end='\r')
            
            state['prev_vowel'] = vowel
            
        elif state['hold'] > 0:
            state['hold'] -= 1
            out_vowel = state['prev_vowel']
        
        if out_vowel is not None:
            n_out = max(1, len(mic) // FRAME_SIZE)
            audio, state['cum_phase'] = synthesize_vowel_frames(
                out_vowel, n_frames=n_out, prev_phase=state['cum_phase']
            )
            n_samples = len(outdata)
            if len(audio) >= n_samples:
                outdata[:] = audio[:n_samples].reshape(-1, 1)
            else:
                outdata[:len(audio)] = audio.reshape(-1, 1)
                outdata[len(audio):] = 0
        else:
            outdata[:] = 0
    
    try:
        with sd.Stream(
            device=(args.input_device, args.output_device),
            samplerate=SR,
            blocksize=FRAME_SIZE * 4,
            channels=(1, 1),
            callback=callback,
            dtype='float32',
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()
