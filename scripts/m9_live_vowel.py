#!/usr/bin/env python3
"""M9 Live Vowel Imitation — real-time mic input."""

import sys, json, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import butter, sosfilt
sys.path.insert(0, 'packages')
from m9_sound import AcousticSubstrate, OscillatorBank
import sounddevice as sd

SR = 16000

# Load substrate
sub_ckpt = json.load(open('data/m9_substrate_formant.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

# Load lookup
lookup_data = json.load(open('data/m9_formant_lookup_v3.json'))
vowel_names = lookup_data['vowel_names']
lookup = lookup_data['lookup']

print(f"Loaded substrate + lookup ({vowel_names})")

# Build centroid tree
centroids = np.array([lookup[v]['centroid'] for v in vowel_names], dtype=np.float32)
tree = cKDTree(centroids)

# High-pass filter
hp_sos = butter(4, 100, btype='high', fs=SR, output='sos')
def hp(x): return sosfilt(hp_sos, x).astype(np.float32)

def synth(vowel, n_frames, phase=0.0):
    t = lookup[vowel]['template']
    frames = []
    for _ in range(n_frames):
        tf = np.arange(480, dtype=np.float32) / SR
        fr = np.zeros(480, dtype=np.float32)
        for fk, ak in [('f1','a1'),('f2','a2'),('f3','a3')]:
            fr += t[ak] * np.sin(2*np.pi*t[fk]*tf + phase)
            phase += 2*np.pi*t[fk]*(480/SR)
        peak = np.max(np.abs(fr))
        if peak > 1.0: fr /= peak
        frames.append(fr.astype(np.float32))
        phase %= 2*np.pi
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32), phase

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-device', type=int, default=1)
    parser.add_argument('--output-device', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.01)
    args = parser.parse_args()
    
    print(f"Mic: device {args.input_device}, Output: device {args.output_device}")
    print(f"Threshold: {args.threshold}")
    print("Speak vowels into mic. Ctrl+C to stop.\n")
    
    state = {'prev': None, 'phase': 0.0, 'hold': 0}
    
    def callback(indata, outdata, frames, time_info, status):
        mic = hp(indata[:, 0].astype(np.float32))
        rms = np.sqrt(np.mean(mic ** 2))
        
        out_v = None
        if rms > args.threshold:
            substrate.reset()
            nf = len(mic) // 480
            for i in range(nf):
                substrate.step(mic[i*480:(i+1)*480].flatten())
            
            _, idx = tree.query(k=1, x=substrate.hidden_state)
            v = vowel_names[idx]
            out_v = v
            state['hold'] = 3
            if v != state['prev']:
                print(f"  {v}", end='\r')
            state['prev'] = v
        elif state['hold'] > 0:
            state['hold'] -= 1
            out_v = state['prev']
        
        if out_v:
            n_out = max(1, len(mic) // 480)
            audio, state['phase'] = synth(out_v, n_out, state['phase'])
            n = len(outdata)
            if len(audio) >= n:
                outdata[:] = audio[:n].reshape(-1, 1)
            else:
                outdata[:len(audio)] = audio.reshape(-1, 1)
                outdata[len(audio):] = 0
        else:
            outdata[:] = 0
    
    try:
        sd.Stream(
            device=(args.input_device, args.output_device),
            samplerate=SR, blocksize=SR // 2, channels=(1, 1),
            callback=callback, dtype='float32'
        ).__enter__()
        while True: time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()
