#!/usr/bin/env python3
"""M9 Live Vowel Imitation — simplified version."""

import sys, json, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import butter, sosfilt
sys.path.insert(0, 'packages')
from m9_sound import AcousticSubstrate, OscillatorBank
import sounddevice as sd

SR = 16000

sub_ckpt = json.load(open('data/m9_substrate_formant.ckpt.json'))
substrate = AcousticSubstrate(hidden_dim=32, sr=SR)
substrate._W_h = np.array(sub_ckpt['W_h'], dtype=np.float32)
substrate._W_x = np.array(sub_ckpt['W_x'], dtype=np.float32)
substrate._b = np.array(sub_ckpt['b'], dtype=np.float32)

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

state = {'prev': None, 'phase': 0.0, 'hold': 0}

def callback(indata, outdata, frames, time_info, status):
    mic = sosfilt(hp, indata[:, 0].astype(np.float32))
    rms = np.sqrt(np.mean(mic ** 2))
    
    out_v = None
    if rms > 0.003:
        substrate.reset()
        for i in range(len(mic) // 480):
            substrate.step(mic[i*480:(i+1)*480])
        _, idx = tree.query(k=1, x=substrate.hidden_state)
        v = vowel_names[idx]
        out_v = v; state['hold'] = 3
        if v != state['prev']: print(f'  {v}', end='\r', flush=True)
        state['prev'] = v
    elif state['hold'] > 0:
        state['hold'] -= 1
        out_v = state['prev']
    
    if out_v:
        audio, state['phase'] = synth(out_v, max(1, len(mic)//480), state['phase'])
        n = len(outdata)
        if len(audio) >= n:
            outdata[:] = audio[:n].reshape(-1, 1)
        else:
            outdata[:len(audio)] = audio.reshape(-1, 1)
            outdata[len(audio):] = 0
    else:
        outdata[:] = 0

print('Speak into mic now... (Ctrl+C to stop)')
try:
    with sd.Stream(device=(1, 2), samplerate=SR, blocksize=SR//2, channels=(1, 1), callback=callback, dtype='float32'):
        while True: time.sleep(0.1)
except KeyboardInterrupt:
    print('\nStopped.')
