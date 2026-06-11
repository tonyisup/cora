#!/usr/bin/env python3
"""M9 Substrate Training — real captures + synthetic with HP100."""

import sys, json, argparse, wave
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
    f1 = t['f1'] * (1 + jitter * rng.randn())
    f2 = t['f2'] * (1 + jitter * rng.randn())
    f3 = t['f3'] * (1 + jitter * rng.randn())
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

def bwd(dh, cache, W_h):
    hl, pl, xl, lo, hi, n = cache['hl'], cache['pl'], cache['xl'], cache['lo'], cache['hi'], cache['n']
    na = hi - lo
    dWh = np.zeros_like(W_h); dWx = np.zeros((HIDDEN_DIM, 26), dtype=np.float32); db = np.zeros(HIDDEN_DIM, dtype=np.float32)
    dhn = np.zeros(HIDDEN_DIM, dtype=np.float32)
    for t in range(n-1, -1, -1):
        dh_t = (dh/na + dhn) if lo <= t < hi else dhn
        dp = dh_t * (1 - np.pl(pl[t])**2)
        db += dp
        if t > 0: dWh += np.outer(dp, hl[t-1])
        dWx += np.outer(dp, xl[t])
        dhn = W_h.T @ dp
    return dWh.astype(np.float32), dWx.astype(np.float32), db.astype(np.float32)

def triplet_loss emb, labels, margin=2.0):
    n = len(emb)
    cents, cnts = {}, {}
    for v in VOWEL_NAMES:
        idx = [i for i,l in enumerate(labels) if l == v]
        cents[v] = np.mean(emb[idx], axis=0); cnts[v] = len(idx)
    loss = 0.0; de = np.zeros_like(emb)
    for i in range(n):
        va = labels[i]; ha = emb[i]
        for vn in VOWEL_NAMES:
            if vn == va: continue
            dp = np.sum((ha-cents[va])**2); dn = np.sum((ha-cents[vn])**2)
            v = dp - dn + margin
            if v > 0:
                loss += v
                de[i] += 2*(ha-cents[va]) - 2*(ha-cents[vn])
                for j in range(n):
                    if labels[j] == va: de[j] += -2*(ha-cents[va])/cnts[va]
                    if labels[j] == vn: de[j] += 2*(ha-cents[vn])/cnts[vn]
    na = sum(1 for i in range(n) for vn in VOWEL_NAMES if vn!=va and np.sum((emb[i]-cents[labels[i]])**2)-np.sum((emb[i]-cents[vn])**2)+margin>0)
    if na > 0: loss /= na; de /= na
    return loss, de, cents

def load_real(data_dir="data/m9_real_vowels"):
    data_dir = Path(data_dir)
    ds = []
    for vowel in VOWEL_NAMES:
        for p in sorted(data_dir.glob(f"{vowel}_*.wav")):
            with wave.open(str(p), 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
            audio = sosfilt(HP100, audio).astype(np.float32)
            chunk = int(0.4*SR); step = int(0.1*SR)
            for s in range(0, len(audio)-chunk+1, step):
                c = audio[s:s+chunk]
                if len(c) >= FRAME: ds.append((vowel, c))
    return ds

def gen_synth(n_per=300, seed=42):
    rng = np.random.RandomState(seed)
    ds = []
    for v in VOWEL_NAMES:
        for _ in range(n_per):
            ds.append((v, gen_vowel(v, duration=rng.uniform(0.4,0.8), f0=rng.uniform(100,180),
                              jitter=rng.uniform(0.02,0.08), noise_db=rng.uniform(-45,-30), rng=rng)))
    return ds

def train(epochs=500, lr=0.001, n_synth=300, margin=3.0, seed=42):
    ckpt = json.load(open('data/m9_substrate_spectral_v9.ckpt.json'))
    Wh = np.array(ckpt['Wh'])*0.5; Wx = np.array(ckpt['Wx'])*0.5; b = np.array(ckpt['b'])
    print("Init from v9 (0.5x)")
    real = load_real(); synth = gen_synth(n_synth, seed)
    ds = real + synth
    print(f"Real: {len(real)}, Synth: {len(synth)}, Total: {len(ds)}")
    mWh,vWh = np.zeros_like(Wh),np.zeros_like(Wh); mWx,vWx = np.zeros_like(Wx),np.zeros_like(Wx); mb,vb = np.zeros_like(b),np.zeros_like(b)
    b1,b2,eps = 0.9,0.999,1e-8; bl = float('inf'); bWh,bWx,bB = Wh.copy(),Wx.copy(),b.copy()
    for ep in range(epochs):
        emb,lab,cache = [],[],[]
        for v,a in ds: h,c = fwd(Wh,Wx,b,a); emb.append(h); lab.append(v); cache.append(c)
        emb = np.array(emb,dtype=np.float32)
        tl,dt,cents = triplet_loss(emb,lab,margin)
        dl,_ = 0.1*np.mean(emb**4), 4*0.1*emb**3/emb.size
        tot = tl+dl+1e-4*(np.sum(Wh**2)+np.sum(Wx**2)+np.sum(b**2))
        de = dt+dl; dWh,dWx,db = np.zeros_like(Wh),np.zeros_like(Wx),np.zeros_like(b)
        for i in range(len(ds)):
            dw,dx,dd = bwd(de[i],cache[i],Wh); dWh+=dw; dWx+=dx; db+=dd
        dWh+=2e-4*Wh; dWx+=2e-4*Wx; db+=2e-4*b
        gn = np.linalg.norm(np.concatenate([dWh.ravel(),dWx.ravel(),db.ravel()]))
        if gn>1: c=1/gn; dWh*=c; dWx*=c; db*=c
        for pn,g,m,v in [('Wh',dWh,mWh,vWh),('Wx',dWx,mWx,vWx),('b',db,mb,vb)]:
            m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g**2
            mh=m/(1-b1**(ep+1)); vh=v/(1-b2**(ep+1))
            if pn=='Wh': Wh-=lr*mh/(np.sqrt(vh)+eps)
            elif pn=='Wx': Wx-=lr*mh/(np.sqrt(vh)+eps)
            else: b-=lr*mh/(np.sqrt(vh)+eps)
        if tot<bl: bl=tot; bWh,bWx,bB=Wh.copy(),Wx.copy(),b.copy()
        if ep%50==0 or ep==epochs-1:
            bt=[np.linalg.norm(cents[VOWEL_NAMES[i]]-cents[VOWEL_NAMES[j]]) for i in range(5) for j in range(i+1,5)]
            wm=max(np.mean([np.linalg.norm(emb[k]-cents[v]) for k in range(len(lab)) if lab[k]==v]) for v in VOWEL_NAMES)
            print(f"  E{ep:4d} loss={tot:.4f} (tri={tl:.4f}) bmin={min(bt):.2f} wmax={wm:.2f} r={min(bt)/wm:.2f}")
    return bWh,bWx,bB,ds

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--n-synth',type=int,default=300)
    a = p.parse_args()
    Wh,Wx,b,ds = train(epochs=a.epochs,n_synth=a.n_synth)
    json.dump({'Wh':Wh.tolist(),'Wx':Wx.tolist(),'b':b.tolist(),'hidden_dim':HIDDEN_DIM,'frame_size':FRAME,'sr':SR},
              open('data/m9_substrate_spectral_v10.ckpt.json','w'))
    print("Saved v10")

if __name__=='__main__': main()
