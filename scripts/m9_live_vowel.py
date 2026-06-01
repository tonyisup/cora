#!/usr/bin/env python3
"""M9 Live Vowel Imitation — real-time mic input with feedback prevention."""

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
centroids = np.array([lookup[v]['centroid'] for v in vowel_names], dtype=np.float32)
tree = cKDTree(centroids)

hp = butter(4, 100, btype='high', fs=SR, output='sos')

def synth(vowel, n_frames, phase=0.0):
    t = lookup[vowel]['template']
    frames, cp = [], phase
    for _ in range(n_frames):
        tf = np.arange(480, dtype=np.float32) / SR
        fr = np.zeros(480, dtype=np.float32)
        for fk, ak in [('f1','a1'),('f2','a2'),('f3','a3')]:
            fr += t[ak] * np.sin(2*np.pi*t[fk]*tf + cp)
            cp += 2*np.pi*t[fk]*(480/SR)
        pk = np.max(np.abs(fr))
        if pk > 1.0: fr /= pk
        frames.append(fr.astype(np.float32))
        cp %= 2*np.pi
    return np.concatenate(frames), cp

def main():
    import argparse
    parser = argparse.ArgumentParser(description='M9 Live Vowel Imitation')
    parser.add_argument('--input-device', type=int, default=1, help='Mic input device')
    parser.add_argument('--output-device', type=int, default=2, help='Headphone output device')
    parser.add_argument('--threshold', type=float, default=0.003, help='RMS threshold for speech detection')
    parser.add_argument('--feedback-frames', type=int, default=5, 
                        help='Frames to ignore input after producing output (prevents feedback loop)')
    parser.add_argument('--hold-frames', type=int, default=3, help='Frames to hold vowel after input stops')
    args = parser.parse_args()
    
    print(f"Cora Vowel Imitation")
    print(f"  Input: device {args.input_device} (mic)")
    print(f"  Output: device {args.output_device} (headphones)")
    print(f"  Vowels: {vowel_names}")
    print(f"  Feedback gate: {args.feedback_frames} frames ({args.feedback_frames * 30}ms)")
    print(f"\nSpeak clearly. Hold each vowel 0.5s+. Ctrl+C to stop.\n")
    
    # State: prev vowel, synthesis phase, hold counter, feedback gate
    state = {
        'prev': None,
        'phase': 0.0,
        'hold': 0,
        'silence_count': args.feedback_frames,  # Start ready to listen
    }
    
    def callback(indata, outdata, frames, time_info, status):
        mic = sosfilt(hp, indata[:, 0].astype(np.float32))
        rms = np.sqrt(np.mean(mic ** 2))
        
        out_v = None
        
        # Only classify if we've been silent long enough (feedback gate)
        # This prevents: output -> mic picks it up -> re-classify -> more output (infinite loop)
        if rms > args.threshold and state['silence_count'] >= args.feedback_frames:
            # Real speech detected — encode through substrate
            substrate.reset()
            for i in range(len(mic) // 480):
                substrate.step(mic[i*480:(i+1)*480].flatten())
            
            # Classify via nearest centroid
            _, idx = tree.query(k=1, x=substrate.hidden_state)
            v = vowel_names[idx]
            out_v = v
            state['hold'] = args.hold_frames
            state['silence_count'] = 0  # Reset silence counter — we're now producing output
            
            if v != state['prev']:
                print(f"  {v}", end='\r', flush=True)
            state['prev'] = v
            
        elif state['hold'] > 0:
            # Hold the previous vowel for smooth transitions
            state['hold'] -= 1
            out_v = state['prev']
            state['silence_count'] = 0  # Still producing output
        else:
            # No output — increment silence counter (feedback gate)
            state['silence_count'] = min(state['silence_count'] + 1, args.feedback_frames * 2)
        
        # Produce audio output
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
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped.")

if __name__ == '__main__':
    main()
