#!/usr/bin/env python3
"""M10 Consonant Training v2 — real-data-focused.

Trains with real TTS captures as primary data, minimal synthetic augmentation.
Uses class weighting to balance vowel/consonant performance.

Usage:
    ./.venv/bin/python scripts/m10_train_v2.py --epochs 300
"""

import sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))

SR = 16000
FRAME = 480
HIDDEN_DIM = 64
N_MELS = 32

VOWEL_FORMANTS = {
    'i': {'f1': 220, 'f2': 3200, 'f3': 3600, 'a1': 1.0, 'a2': 0.55, 'a3': 0.30},
    'E': {'f1': 500, 'f2': 2400, 'f3': 2800, 'a1': 1.0, 'a2': 0.60, 'a3': 0.35},
    'a': {'f1': 900, 'f2': 700,  'f3': 2300, 'a1': 1.0, 'a2': 0.80, 'a3': 0.45},
    'O': {'f1': 620, 'f2': 500,  'f3': 2250, 'a1': 1.0, 'a2': 0.70, 'a3': 0.40},
    'u': {'f1': 350, 'f2': 550,  'f3': 2100, 'a1': 1.0, 'a2': 0.50, 'a3': 0.25},
}

CONSONANT_PARAMS = {
    'p': {'type': 'stop', 'noise_ms': 20},
    't': {'type': 'stop', 'noise_ms': 15},
    'k': {'type': 'stop', 'noise_ms': 25},
    's': {'type': 'fricative', 'noise_ms': 120, 'center_freq': 6000, 'amp': 0.3},
    'f': {'type': 'fricative', 'noise_ms': 100, 'center_freq': 8000, 'amp': 0.2},
    'sh': {'type': 'fricative', 'noise_ms': 140, 'center_freq': 3500, 'amp': 0.25},
    'm': {'type': 'nasal', 'duration_ms': 80, 'f0': 250, 'amp': 0.4},
    'n': {'type': 'nasal', 'duration_ms': 80, 'f0': 280, 'amp': 0.35},
}

VOWEL_NAMES = ['i', 'E', 'a', 'O', 'u']
CONSONANT_NAMES = ['p', 't', 'k', 's', 'f', 'sh', 'm', 'n']
ALL_NAMES = VOWEL_NAMES + CONSONANT_NAMES
N_CLASSES = len(ALL_NAMES)
HP100 = butter(4, 100, btype='high', fs=SR, output='sos')


def gen_vowel(vowel, duration=0.5, f0=120.0, jitter=0.05, noise_db=-40.0, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    t = VOWEL_FORMANTS[vowel]
    f1, f2, f3 = [t[k]*(1+jitter*rng.randn()) for k in ('f1','f2','f3')]
    a1, a2, a3 = [t[k]*(1+0.1*rng.randn()) for k in ('a1','a2','a3')]
    n = int(duration * SR)
    s = np.arange(n, dtype=np.float32) / SR
    audio = (a1*np.sin(2*np.pi*f1*s) + a2*np.sin(2*np.pi*f2*s) + a3*np.sin(2*np.pi*f3*s)).astype(np.float32)
    audio *= (0.5 + 0.5*np.sin(2*np.pi*f0*s)).astype(np.float32)
    noise_amp = 10**(noise_db/20)
    audio += rng.randn(n).astype(np.float32) * noise_amp
    audio = sosfilt(HP100, audio).astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def gen_consonant(consonant, duration=0.3, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    params = CONSONANT_PARAMS.get(consonant, {'type': 'stop', 'noise_ms': 20})
    n = int(duration * SR)
    audio = np.zeros(n, dtype=np.float32)
    if params['type'] == 'stop':
        burst = min(n, int(SR * params['noise_ms'] / 1000.0))
        audio[:burst] = rng.randn(burst).astype(np.float32) * 0.3
    elif params['type'] == 'fricative':
        noise = np.diff(rng.randn(n).astype(np.float32), prepend=0.0)
        s = np.arange(n, dtype=np.float32) / SR
        audio = (noise * np.sin(2*np.pi*params.get('center_freq', 6000)*s) * params.get('amp', 0.3)).astype(np.float32)
    elif params['type'] == 'nasal':
        s = np.arange(n, dtype=np.float32) / SR
        audio = (params.get('amp', 0.4) * np.sin(2*np.pi*params.get('f0', 250)*s)).astype(np.float32)
    audio = sosfilt(HP100, audio).astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


_MEL_FB = None

def _get_mel_fb():
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _build_mel_fb(N_MELS, 512, SR)
    return _MEL_FB


def _build_mel_fb(n_mels, n_fft, sr):
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(80), hz_to_mel(7600), n_mels + 2)
    bin_points = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        for k in range(left, center):
            if k < n_bins and center > left:
                fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if k < n_bins and right > center:
                fb[m, k] = (right - k) / (right - center)
    return fb


def mel_feat(frame):
    x = np.zeros(512, dtype=np.float32)
    n = min(len(frame), 512)
    x[:n] = frame[:n]
    spectrum = np.abs(np.fft.rfft(x))
    mel_spec = _get_mel_fb() @ spectrum
    mel_spec = np.log(mel_spec + 1e-6).astype(np.float32)
    rms = float(np.sqrt(np.mean(mel_spec**2)))
    if rms > 1e-6:
        mel_spec /= rms
    return mel_spec


def extract_mel_rms(audio):
    audio = sosfilt(HP100, np.asarray(audio, dtype=np.float32).reshape(-1))
    n_frames = len(audio) // FRAME
    return np.array([mel_feat(audio[i*FRAME:(i+1)*FRAME]) for i in range(n_frames)], dtype=np.float32)


def _split_key(stem):
    import hashlib
    return int(hashlib.sha1(stem.encode()).digest()[0]) % 100


def load_real_vowel_captures(split=None, holdout_pct=20):
    import wave
    data_dir = Path("data/m9_tts_captures")
    phrase_vowels = {
        "beat_bet_hot_caught": ["i", "E", "a", "O", "u"],
        "boot_caught_hot_bet": ["u", "O", "a", "E", "i"],
        "beat_hot_bet_boot": ["i", "a", "E", "u", "O"],
    }
    ds = []
    for wav_path in sorted(data_dir.glob("*.wav")):
        name = wav_path.stem.lower()
        phrase = next((p for p in phrase_vowels if p in name), None)
        if not phrase:
            continue
        sk = _split_key(wav_path.stem)
        if split == "train" and sk < holdout_pct:
            continue
        if split == "eval" and sk >= holdout_pct:
            continue
        with wave.open(str(wav_path), 'rb') as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
        mel = extract_mel_rms(audio)
        if len(mel) < 5:
            continue
        chunk = len(mel) // 5
        for idx, v in enumerate(phrase_vowels[phrase]):
            mid = (idx * chunk + ((idx + 1) * chunk if idx < 4 else len(mel))) // 2
            ds.append((v, mel[mid]))
    return ds


def load_real_consonant_captures(split=None, holdout_pct=20):
    import wave, hashlib
    manifest_path = Path("data/m10/manifests/captures.jsonl")
    if not manifest_path.exists():
        return []
    ds = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        label = row.get("label", "")
        if label not in ALL_NAMES:
            continue
        sk = int(hashlib.sha1(row.get("capture_id", "").encode()).digest()[0]) % 100
        if split == "train" and sk < holdout_pct:
            continue
        if split == "eval" and sk >= holdout_pct:
            continue
        wav_path = Path(row["wav_path"])
        if not wav_path.exists():
            continue
        with wave.open(str(wav_path), 'rb') as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
        mel = extract_mel_rms(audio)
        if len(mel) > 0:
            ds.append((label, mel[len(mel)//2]))
    return ds


def gen_synth_vowels(n_per=300, seed=42):
    rng = np.random.RandomState(seed)
    ds = []
    for v in VOWEL_NAMES:
        for _ in range(n_per):
            audio = gen_vowel(v, duration=rng.uniform(0.3, 0.6), f0=rng.uniform(100, 180),
                              jitter=rng.uniform(0.02, 0.08), noise_db=rng.uniform(-45, -30), rng=rng)
            mel = extract_mel_rms(audio)
            if len(mel) > 0:
                ds.append((v, mel[len(mel)//2]))
    return ds


def gen_synth_consonants(n_per=50, seed=42):
    rng = np.random.RandomState(seed)
    ds = []
    for c in CONSONANT_NAMES:
        for _ in range(n_per):
            audio = gen_consonant(c, duration=rng.uniform(0.15, 0.4), rng=rng)
            mel = extract_mel_rms(audio)
            if len(mel) > 0:
                ds.append((c, mel[len(mel)//2]))
    return ds


def _adam_step(param, grad, m, v, beta1, beta2, eps, t, lr):
    m[:] = beta1 * m + (1.0 - beta1) * grad
    v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
    param -= lr * m / (1 - beta1**t) / (np.sqrt(v / (1 - beta2**t)) + eps)


def train(epochs=300, lr=0.01, n_synth_vowels=300, n_synth_consonants=50, seed=42,
          real_weight=2.0, consonant_weight=1.5):
    """Train with real-data focus and class weighting."""
    rng = np.random.RandomState(seed)

    W1 = rng.randn(N_MELS, HIDDEN_DIM).astype(np.float32) * np.sqrt(2.0 / N_MELS)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = rng.randn(HIDDEN_DIM, N_CLASSES).astype(np.float32) * np.sqrt(2.0 / HIDDEN_DIM)
    b2 = np.zeros(N_CLASSES, dtype=np.float32)

    # Load data
    real_v = load_real_vowel_captures(split="train")
    real_c = load_real_consonant_captures(split="train")
    eval_v = load_real_vowel_captures(split="eval")
    eval_c = load_real_consonant_captures(split="eval")
    synth_v = gen_synth_vowels(n_synth_vowels, seed)
    synth_c = gen_synth_consonants(n_synth_consonants, seed)

    # Combine: real data weighted by duplication
    ds = (real_v * int(real_weight) + real_c * int(real_weight) +
          synth_v + synth_c)
    eval_ds = eval_v + eval_c

    X = np.array([m for _, m in ds], dtype=np.float32)
    y = np.array([ALL_NAMES.index(v) for v, _ in ds], dtype=np.int32)

    # Sample weights: boost consonants
    sample_weights = np.ones(len(y), dtype=np.float32)
    for i, label in enumerate(ALL_NAMES):
        if label in CONSONANT_NAMES:
            sample_weights[y == i] = consonant_weight

    print(f"Real vowels: {len(real_v)}, Real consonants: {len(real_c)}")
    print(f"Synth vowels: {len(synth_v)}, Synth consonants: {len(synth_c)}")
    print(f"Total train: {len(ds)}, Eval: {len(eval_ds)}")
    print(f"Real weight: {real_weight}, Consonant weight: {consonant_weight}")
    print(f"Epochs: {epochs}, lr: {lr}")
    print()

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)

    bs = min(256, len(X))
    nb = max(1, len(X) // bs)

    for ep in range(epochs):
        perm = rng.permutation(len(X))
        Xs, ys, ws = X[perm], y[perm], sample_weights[perm]

        total_loss = 0.0
        correct = 0

        for bi in range(nb):
            s, e = bi * bs, min((bi + 1) * bs, len(X))
            bx, by, bw = Xs[s:e], ys[s:e], ws[s:e]

            h = np.maximum(bx @ W1 + b1, 0)
            logits = h @ W2 + b2

            # Softmax + weighted CE
            lm = logits - np.max(logits, axis=1, keepdims=True)
            el = np.exp(lm)
            p = el / np.sum(el, axis=1, keepdims=True)
            cp = p[np.arange(len(by)), by]
            loss = -np.mean(bw * np.log(cp + 1e-8))
            total_loss += float(loss)

            preds = np.argmax(logits, axis=1)
            correct += int(np.sum(preds == by))

            # Backward with sample weights
            dl = p.copy()
            dl[np.arange(len(by)), by] -= 1
            dl *= bw[:, None]
            dl /= len(by)

            dW2 = h.T @ dl
            db2 = np.sum(dl, axis=0)
            dh = (dl @ W2.T) * (h > 0).astype(np.float32)
            dW1 = bx.T @ dh
            db1 = np.sum(dh, axis=0)

            t = ep * nb + bi + 1
            _adam_step(W1, dW1, mW1, vW1, beta1, beta2, eps, t, lr)
            _adam_step(b1, db1, mb1, vb1, beta1, beta2, eps, t, lr)
            _adam_step(W2, dW2, mW2, vW2, beta1, beta2, eps, t, lr)
            _adam_step(b2, db2, mb2, vb2, beta1, beta2, eps, t, lr)

        acc = correct / len(X)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  E{ep:4d} loss={total_loss/nb:.4f} acc={acc:.1%}")

    # Eval
    if eval_ds:
        Xe = np.array([m for _, m in eval_ds], dtype=np.float32)
        ye = np.array([ALL_NAMES.index(v) for v, _ in eval_ds], dtype=np.int32)
        h = np.maximum(Xe @ W1 + b1, 0)
        logits = h @ W2 + b2
        preds = np.argmax(logits, axis=1)
        eval_acc = float(np.mean(preds == ye))
        print(f"\nEval accuracy: {eval_acc:.1%} ({int(np.sum(preds==ye))}/{len(ye)})")
        for i, name in enumerate(ALL_NAMES):
            mask = ye == i
            if mask.sum() > 0:
                ca = float(np.mean(preds[mask] == i))
                print(f"  {name:4s}: {ca:.1%} (n={mask.sum()})")

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--n-synth-vowels', type=int, default=300)
    p.add_argument('--n-synth-consonants', type=int, default=50)
    p.add_argument('--real-weight', type=float, default=2.0)
    p.add_argument('--consonant-weight', type=float, default=1.5)
    a = p.parse_args()

    params = train(epochs=a.epochs, lr=a.lr,
                   n_synth_vowels=a.n_synth_vowels, n_synth_consonants=a.n_synth_consonants,
                   real_weight=a.real_weight, consonant_weight=a.consonant_weight)

    out = Path('data/m10_v1.json')
    with open(out, 'w') as f:
        json.dump({k: v.tolist() for k, v in params.items()}, f)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
