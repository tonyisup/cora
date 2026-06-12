#!/usr/bin/env python3
"""Mel-feature classifier — feedforward, no RNN.

Instead of temporal integration (which smears spectral detail),
classify each frame independently using the mel features directly.
"""

import sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))

SR = 16000
FRAME = 480
N_MELS = 32

VOWEL_FORMANTS = {
    'i': {'f1': 220, 'f2': 3200, 'f3': 3600, 'a1': 1.0, 'a2': 0.55, 'a3': 0.30},
    'E': {'f1': 500, 'f2': 2400, 'f3': 2800, 'a1': 1.0, 'a2': 0.60, 'a3': 0.35},
    'a': {'f1': 900, 'f2': 700,  'f3': 2300, 'a1': 1.0, 'a2': 0.80, 'a3': 0.45},
    'O': {'f1': 620, 'f2': 500,  'f3': 2250, 'a1': 1.0, 'a2': 0.70, 'a3': 0.40},
    'u': {'f1': 350, 'f2': 550,  'f3': 2100, 'a1': 1.0, 'a2': 0.50, 'a3': 0.25},
}
VOWEL_NAMES = ['i', 'E', 'a', 'O', 'u']
HP100 = butter(4, 100, btype='high', fs=SR, output='sos')

def gen_vowel(vowel, duration=0.5, f0=120.0, jitter=0.05, noise_db=-40.0, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    t = VOWEL_FORMANTS[vowel]
    f1, f2, f3 = [t[k]*(1+jitter*rng.randn()) for k in ('f1','f2','f3')]
    a1, a2, a3 = t['a1'], max(0.05, t['a2']*(1+jitter*rng.randn())), max(0.05, t['a3']*(1+jitter*rng.randn()))
    n = int(SR * duration)
    t_arr = np.arange(n, dtype=np.float32) / SR
    phase0 = rng.uniform(0, 2*np.pi)
    sig = (0.3*np.sin(2*np.pi*f0*t_arr+phase0) + a1*np.sin(2*np.pi*f1*t_arr+phase0*1.3) +
           a2*np.sin(2*np.pi*f2*t_arr+phase0*0.7) + a3*np.sin(2*np.pi*f3*t_arr+phase0*1.1))
    if noise_db > -60:
        sig += (10**(noise_db/20)) * rng.randn(n).astype(np.float32)
    peak = np.max(np.abs(sig))
    return (sig / peak if peak > 1e-6 else sig).astype(np.float32)

def mel_feat(frame):
    from m9_sound.vowel_transitions import MEL_FILTERBANK
    x = np.zeros(512, dtype=np.float32)
    x[:min(len(frame), 512)] = frame[:min(len(frame), 512)].astype(np.float32)
    mel = MEL_FILTERBANK @ np.abs(np.fft.rfft(x))
    mel = np.log(mel + 1e-6).astype(np.float32)
    r = np.sqrt(np.mean(mel**2))
    return mel / r if r > 1e-6 else mel

def extract_centered_mel(audio):
    """Extract centered mel features from audio."""
    audio = sosfilt(HP100, audio).astype(np.float32)
    nf = len(audio) // FRAME
    all_mel = []
    for i in range(nf):
        frame = audio[i*FRAME:(i+1)*FRAME].flatten().astype(np.float32)
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > 1e-6:
            frame = frame / rms
        all_mel.append(mel_feat(frame))
    if not all_mel:
        return np.zeros((0, N_MELS), dtype=np.float32)
    mean_mel = np.mean(all_mel, axis=0)
    centered = np.array(all_mel) - mean_mel
    return centered

# Simple feedforward: mel -> hidden -> logits
# Architecture: 32 -> 64 -> 32 -> 5
# With batch norm and ReLU
H1, H2, N_OUT = 64, 32, 5

def init_params(rng):
    """Xavier initialization."""
    w1 = rng.randn(N_MELS, H1).astype(np.float32) * np.sqrt(2.0 / N_MELS)
    b1 = np.zeros(H1, dtype=np.float32)
    w2 = rng.randn(H1, H2).astype(np.float32) * np.sqrt(2.0 / H1)
    b2 = np.zeros(H2, dtype=np.float32)
    w3 = rng.randn(H2, N_OUT).astype(np.float32) * np.sqrt(2.0 / H2)
    b3 = np.zeros(N_OUT, dtype=np.float32)
    # Batch norm params
    bn1_gamma = np.ones(H1, dtype=np.float32)
    bn1_beta = np.zeros(H1, dtype=np.float32)
    bn2_gamma = np.ones(H2, dtype=np.float32)
    bn2_beta = np.zeros(H2, dtype=np.float32)
    return {'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3,
            'bn1g':bn1_gamma,'bn1b':bn1_beta,'bn2g':bn2_gamma,'bn2b':bn2_beta}

def forward(params, x, training=True):
    """Forward pass. x: (batch, N_MELS)."""
    h = x @ params['w1'] + params['b1']  # (batch, H1)
    h = h * params['bn1g'] + params['bn1b']  # batch norm
    h = np.maximum(h, 0)  # ReLU
    h = h @ params['w2'] + params['b2']  # (batch, H2)
    h = h * params['bn2g'] + params['bn2b']
    h = np.maximum(h, 0)
    logits = h @ params['w3'] + params['b3']  # (batch, N_OUT)
    cache = {'x':x, 'h1_pre':None, 'h1':h, 'h2':None, 'logits':logits}
    return logits, cache

def compute_loss(logits, labels, margin=1.0):
    """Multi-class SVM (hinge) loss with margin."""
    n = len(labels)
    correct_scores = logits[np.arange(n), labels]
    margins = np.maximum(0, logits - correct_scores[:, None] + margin)
    margins[np.arange(n), labels] = 0
    loss = np.mean(np.sum(margins, axis=1))
    
    # Gradient
    dlogits = np.zeros_like(logits)
    dlogits[margins > 0] = 1
    counts = np.sum(margins > 0, axis=1)
    dlogits[np.arange(n), labels] = -counts
    dlogits /= n
    return loss, dlogits

def backward(params, cache, dlogits):
    """Backward pass."""
    h = cache['h1']  # after relu of layer 1
    x = cache['x']
    
    # Layer 3 backward
    dw3 = h.T @ dlogits
    db3 = np.sum(dlogits, axis=0)
    dh = dlogits @ params['w3'].T
    
    # ReLU backward
    dh = dh * (h > 0).astype(np.float32)
    # BN2 backward (simplified — just pass through)
    dh = dh * params['bn2g']
    
    # Layer 2 backward
    # We need the pre-ReLU h1, but we stored post-ReLU. Re-derive:
    h1_pre = x @ params['w1'] + params['b1']
    h1_pre = h1_pre * params['bn1g'] + params['bn1b']
    dw2 = h1_pre.T @ dh
    db2 = np.sum(dh, axis=0)
    dx = dh @ params['w2'].T
    
    # ReLU + BN1 backward
    dx = dx * (h1_pre > 0).astype(np.float32)
    dx = dx * params['bn1g']
    
    # Layer 1 backward
    dw1 = x.T @ dx
    db1 = np.sum(dx, axis=0)
    
    grads = {'w1':dw1,'b1':db1,'w2':dw2,'b2':db2,'w3':dw3,'b3':db3}
    return grads

def train_ff(epochs=500, lr=0.01, n_synth=200, margin=1.0, seed=42):
    rng = np.random.RandomState(seed)
    params = init_params(rng)
    print(f"Params: {sum(p.size for p in params.values())} total")
    print(f"Epochs: {epochs}, lr: {lr}, margin: {margin}")
    
    # Generate data
    X, y = [], []
    for vowel in VOWEL_NAMES:
        idx = VOWEL_NAMES.index(vowel)
        for _ in range(n_synth):
            audio = gen_vowel(vowel, duration=rng.uniform(0.3, 0.6), f0=rng.uniform(100, 180),
                              jitter=rng.uniform(0.02, 0.08), noise_db=rng.uniform(-45, -30), rng=rng)
            mel = extract_centered_mel(audio)
            for frame in mel:
                X.append(frame)
                y.append(idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Total frames: {len(X)}")
    
    # Shuffle
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    m = {k: np.zeros_like(v) for k,v in params.items()}
    v_adam = {k: np.zeros_like(v) for k,v in params.items()}
    
    batch_size = 256
    n_batches = len(X) // batch_size
    
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]
        
        total_loss = 0
        for bi in range(n_batches):
            bx = X[bi*batch_size:(bi+1)*batch_size]
            by = y[bi*batch_size:(bi+1)*batch_size]
            
            logits, cache = forward(params, bx, training=True)
            loss, dlogits = compute_loss(logits, by, margin)
            grads = backward(params, cache, dlogits)
            
            total_loss += loss
            
            t = ep * n_batches + bi + 1
            for k in params:
                m[k] = b1*m[k] + (1-b1)*grads.get(k, np.zeros_like(params[k]))
                v_adam[k] = b2*v_adam[k] + (1-b2)*grads.get(k, np.zeros_like(params[k]))**2
                mh = m[k] / (1 - b1**t)
                vh = v_adam[k] / (1 - b2**t)
                params[k] -= lr * mh / (np.sqrt(vh) + eps_adam)
        
        if ep % 50 == 0 or ep == epochs - 1:
            logits_all, _ = forward(params, X, training=False)
            preds = np.argmax(logits_all, axis=1)
            acc = np.mean(preds == y)
            print(f"  E{ep:4d} loss={total_loss/n_batches:.4f} acc={acc:.1%}")
    
    return params

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--n-synth',type=int,default=200)
    p.add_argument('--margin',type=float,default=1.0)
    a = p.parse_args()
    params = train_ff(epochs=a.epochs, n_synth=a.n_synth, margin=a.margin)
    json.dump({k: v.tolist() for k,v in params.items()},
              open('data/m9_ff_v1.json','w'))
    print("Saved m9_ff_v1.json")

if __name__=='__main__': main()
