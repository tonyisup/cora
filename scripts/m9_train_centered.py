#!/usr/bin/env python3
"""M9 Substrate Training — with centered mel features."""

import sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))

SR = 16000
FRAME = 480
HIDDEN_DIM = 32

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
    sig = (0.3*np.sin(2*np.pi*f0*t_arr + phase0) + a1*np.sin(2*np.pi*f1*t_arr + phase0*1.3) +
           a2*np.sin(2*np.pi*f2*t_arr + phase0*0.7) + a3*np.sin(2*np.pi*f3*t_arr + phase0*1.1))
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

def fwd(W_h, W_x, b, audio):
    audio = sosfilt(HP100, audio)
    nf = (len(audio) - FRAME) // FRAME + 1 if len(audio) >= FRAME else 0
    if nf == 0: return np.zeros(HIDDEN_DIM, dtype=np.float32), []
    h = np.zeros(HIDDEN_DIM, dtype=np.float32)
    hl, pl, xl = [], [], []
    for i in range(nf):
        frame = audio[i*FRAME:(i+1)*FRAME].flatten().astype(np.float32)
        mel = mel_feat(frame)
        pre = W_h @ h + W_x @ mel + b
        h = np.tanh(pre)
        hl.append(h.copy()); pl.append(pre.copy()); xl.append(mel.copy())
    lo, hi = nf//3, 2*nf//3
    if lo >= hi: lo, hi = 0, nf
    return np.mean(hl[lo:hi], axis=0).astype(np.float32), {'hl':hl,'pl':pl,'xl':xl,'lo':lo,'hi':hi,'n':nf}

def fwd_centered(W_h, W_x, b, audio):
    """Forward with centered mel features."""
    audio = sosfilt(HP100, audio)
    nf = (len(audio) - FRAME) // FRAME + 1 if len(audio) >= FRAME else 0
    if nf == 0: return np.zeros(HIDDEN_DIM, dtype=np.float32), []
    # First pass: compute all mel features
    all_mel = []
    for i in range(nf):
        frame = audio[i*FRAME:(i+1)*FRAME].flatten().astype(np.float32)
        all_mel.append(mel_feat(frame))
    mean_mel = np.mean(all_mel, axis=0)
    # Second pass: classify with centered mel
    h = np.zeros(HIDDEN_DIM, dtype=np.float32)
    hl, pl, xl = [], [], []
    for i in range(nf):
        mel = all_mel[i] - mean_mel
        pre = W_h @ h + W_x @ mel + b
        h = np.tanh(pre)
        hl.append(h.copy()); pl.append(pre.copy()); xl.append(mel.copy())
    lo, hi = nf//3, 2*nf//3
    if lo >= hi: lo, hi = 0, nf
    return np.mean(hl[lo:hi], axis=0).astype(np.float32), {'hl':hl,'pl':pl,'xl':xl,'lo':lo,'hi':hi,'n':nf}

def bwd(dh, cache, W_h):
    hl, pl, xl, lo, hi, n = cache['hl'], cache['pl'], cache['xl'], cache['lo'], cache['hi'], cache['n']
    na = hi - lo
    dWh,dWx,db = np.zeros_like(W_h),np.zeros((HIDDEN_DIM,26),dtype=np.float32),np.zeros(HIDDEN_DIM,dtype=np.float32)
    dhn = np.zeros(HIDDEN_DIM, dtype=np.float32)
    for t in range(n-1,-1,-1):
        dh_t = (dh/na + dhn) if lo<=t<hi else dhn
        dp = dh_t*(1-np.tanh(pl[t])**2)
        db += dp
        if t>0: dWh += np.outer(dp, hl[t-1])
        dWx += np.outer(dp, xl[t])
        dhn = W_h.T @ dp
    return dWh.astype(np.float32), dWx.astype(np.float32), db.astype(np.float32)

def triplet_loss(emb, labels, margin=2.0):
    n = len(emb)
    cents, cnts = {}, {}
    for v in VOWEL_NAMES:
        idx = [i for i,l in enumerate(labels) if l==v]
        cents[v] = np.mean(emb[idx], axis=0); cnts[v] = len(idx)
    loss = 0.0; de = np.zeros_like(emb)
    for i in range(n):
        va = labels[i]; ha = emb[i]
        for vn in VOWEL_NAMES:
            if vn==va: continue
            dp = np.sum((ha-cents[va])**2); dn = np.sum((ha-cents[vn])**2)
            v = dp - dn + margin
            if v>0:
                loss += v; de[i] += 2*(ha-cents[va])-2*(ha-cents[vn])
                for j in range(n):
                    if labels[j]==va: de[j] += -2*(ha-cents[va])/cnts[va]
                    if labels[j]==vn: de[j] += 2*(ha-cents[vn])/cnts[vn]
    na = sum(1 for i in range(n) for vn in VOWEL_NAMES if vn!=va and
             np.sum((emb[i]-cents[labels[i]])**2)-np.sum((emb[i]-cents[vn])**2)+margin>0)
    if na>0: loss/=na; de/=na
    return loss, de, cents

def gen_dataset(n_per=300, seed=42):
    rng = np.random.RandomState(seed)
    ds = []
    for v in VOWEL_NAMES:
        for _ in range(n_per):
            ds.append((v, gen_vowel(v, duration=rng.uniform(0.4,0.8), f0=rng.uniform(100,180),
                              jitter=rng.uniform(0.02,0.08), noise_db=rng.uniform(-45,-30), rng=rng)))
    return ds

def train(epochs=500, lr=0.001, n_synth=300, margin=3.0, seed=42):
    # Start from best checkpoint
    for ckpt_name in ['data/m9_substrate_spectral_v10.ckpt.json', 'data/m9_substrate_spectral_v9.ckpt.json']:
        if Path(ckpt_name).exists():
            ckpt = json.load(open(ckpt_name))
            Wh = np.array(ckpt['W_h'])*0.5; Wx = np.array(ckpt['W_x'])*0.5; b = np.array(ckpt['b'])
            print(f"Init from {ckpt_name} (0.5x)")
            break
    else:
        rng = np.random.RandomState(seed)
        Wh = rng.randn(HIDDEN_DIM,HIDDEN_DIM).astype(np.float32)*0.05
        Wx = rng.randn(HIDDEN_DIM,26).astype(np.float32)*0.05
        b = np.zeros(HIDDEN_DIM,dtype=np.float32)
        print("Init from random")

    ds = gen_dataset(n_synth, seed)
    print(f"Total: {len(ds)} samples")
    print(f"Epochs: {epochs}, lr: {lr}, margin: {margin}, center_mel=True")
    print()

    mWh,vWh = np.zeros_like(Wh),np.zeros_like(Wh); mWx,vWx = np.zeros_like(Wx),np.zeros_like(Wx); mb,vb = np.zeros_like(b),np.zeros_like(b)
    b1,b2,eps = 0.9,0.999,1e-8; bl = float('inf'); bWh,bWx,bB = Wh.copy(),Wx.copy(),b.copy()

    for ep in range(epochs):
        emb,lab,cache = [],[],[]
        for v,a in ds:
            h,c = fwd_centered(Wh,Wx,b,a)
            emb.append(h); lab.append(v); cache.append(c)
        emb = np.array(emb,dtype=np.float32)
        tl,dt,cents = triplet_loss(emb,lab,margin)
        dl = 0.1*np.mean(emb**4)
        tot = tl+dl+1e-4*(np.sum(Wh**2)+np.sum(Wx**2)+np.sum(b**2))
        de = dt+dl
        dWh,dWx,db = np.zeros_like(Wh),np.zeros_like(Wx),np.zeros_like(b)
        for i in range(len(ds)):
            dw,dx,dd = bwd(de[i],cache[i],Wh); dWh+=dw; dWx+=dx; db+=dd
        dWh+=2e-4*Wh; dWx+=2e-4*Wx; db+=2e-4*b
        gn = np.linalg.norm(np.concatenate([dWh.ravel(),dWx.ravel(),db.ravel()]))
        if gn>1: c=1/gn; dWh*=c; dWx*=c; db*=c
        t = ep+1
        for pn,g,m,v in [('Wh',dWh,mWh,vWh),('Wx',dWx,mWx,vWx),('b',db,mb,vb)]:
            m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g**2
            mh=m/(1-b1**t); vh=v/(1-b2**t)
            if pn=='Wh': Wh-=lr*mh/(np.sqrt(vh)+eps)
            elif pn=='Wx': Wx-=lr*mh/(np.sqrt(vh)+eps)
            else: b-=lr*mh/(np.sqrt(vh)+eps)
        if tot<bl: bl=tot; bWh,bWx,bB=Wh.copy(),Wx.copy(),b.copy()
        if ep%50==0 or ep==epochs-1:
            bt=[np.linalg.norm(cents[VOWEL_NAMES[i]]-cents[VOWEL_NAMES[j]]) for i in range(5) for j in range(i+1,5)]
            wm=max(np.mean([np.linalg.norm(emb[k]-cents[v]) for k in range(len(lab)) if lab[k]==v]) for v in VOWEL_NAMES)
            print(f"  E{ep:4d} loss={tot:.4f} (tri={tl:.4f}) bmin={min(bt):.2f} wmax={wm:.2f} r={min(bt)/wm:.2f}")
    return bWh,bWx,bB,ds

def build_lookup(W_h, W_x, b, ds, n_augment=200):
    from scipy.spatial import cKDTree
    embeddings = {v: [] for v in VOWEL_NAMES}
    for v,a in ds:
        h,_ = fwd_centered(W_h,Wx,b,a); embeddings[v].append(h)
    rng = np.random.RandomState(123)
    for v in VOWEL_NAMES:
        for _ in range(n_augment):
            a = gen_vowel(v, duration=0.5, f0=rng.uniform(100,180),
                          jitter=rng.uniform(0.02,0.08), noise_db=rng.uniform(-45,-30), rng=rng)
            h,_ = fwd_centered(W_h,Wx,b,a); embeddings[v].append(h)
    cents = {v: np.mean(embeddings[v],axis=0).astype(np.float32) for v in VOWEL_NAMES}
    all_h, all_labels = [], []
    for v in VOWEL_NAMES:
        all_h.extend(embeddings[v]); all_labels.extend([v]*len(embeddings[v]))
    centroid_arr = np.array([cents[v] for v in VOWEL_NAMES], dtype=np.float32)
    tree = cKDTree(centroid_arr)
    _,idxs = tree.query(np.array(all_h,dtype=np.float32),k=1)
    correct = sum(1 for i,idx in enumerate(idxs) if VOWEL_NAMES[idx]==all_labels[i])
    acc = correct/len(all_labels)
    between = [np.linalg.norm(cents[VOWEL_NAMES[i]]-cents[VOWEL_NAMES[j]]) for i in range(5) for j in range(i+1,5)]
    within = {}
    for v in VOWEL_NAMES:
        h_arr = np.array(embeddings[v],dtype=np.float32)
        dv = np.linalg.norm(h_arr-cents[v],axis=1)
        within[v] = {'mean':float(np.mean(dv)),'std':float(np.std(dv)),'max':float(np.max(dv))}
    print(f"\n=== Lookup Table ===")
    print(f"Accuracy: {acc:.1%} ({correct}/{len(all_labels)})")
    print(f"Between: {', '.join(f'{d:.2f}' for d in between)} (min={min(between):.2f})")
    for v in VOWEL_NAMES:
        s=within[v]; print(f"  Within {v}: mean={s['mean']:.2f} std={s['std']:.2f} max={s['max']:.2f}")
    print(f"Separation ratio: {min(between)/max(s['max'] for s in within.values()):.2f}")
    return cents, within, acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--n-synth',type=int,default=300)
    p.add_argument('--margin',type=float,default=3.0)
    a = p.parse_args()
    Wh,Wx,b,ds = train(epochs=a.epochs, n_synth=a.n_synth, margin=a.margin)
    json.dump({'Wh':Wh.tolist(),'Wx':Wx.tolist(),'b':b.tolist(),'hidden_dim':HIDDEN_DIM,'frame_size':FRAME,'sr':SR},
              open('data/m9_substrate_spectral_v11.ckpt.json','w'))
    print("Saved v11")
    cents,within,acc = build_lookup(Wh,Wx,b,ds)
    json.dump({'lookup':{v: {'centroid':cents[v].tolist(),'template':VOWEL_FORMANTS[v]} for v in VOWEL_NAMES},
              'vowel_names':VOWEL_NAMES,'accuracy':float(acc),'within_stats':within},
              open('data/m9_formant_lookup_v15.json','w'))
    print("Saved v15")

if __name__=='__main__': main()
