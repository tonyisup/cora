#!/usr/bin/env python3
"""M10 Consonant Training — extended cross-entropy + softmax.

Trains a classifier on both vowels and consonants using:
- real vowel captures (from m9_tts_captures)
- real consonant captures (from m10/raw)
- synthetic vowel and consonant examples

Usage:
    ./.venv/bin/python scripts/m10_train.py --epochs 200
    ./.venv/bin/python scripts/m10_train.py --epochs 500 --n-synth 300 --speaker your_name
"""

import sys, json, argparse, hashlib
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))

SR = 16000
FRAME = 480
HIDDEN_DIM = 64
N_MELS = 32

# Vowel formants (from M9)
VOWEL_FORMANTS = {
    'i': {'f1': 220, 'f2': 3200, 'f3': 3600, 'a1': 1.0, 'a2': 0.55, 'a3': 0.30},
    'E': {'f1': 500, 'f2': 2400, 'f3': 2800, 'a1': 1.0, 'a2': 0.60, 'a3': 0.35},
    'a': {'f1': 900, 'f2': 700,  'f3': 2300, 'a1': 1.0, 'a2': 0.80, 'a3': 0.45},
    'O': {'f1': 620, 'f2': 500,  'f3': 2250, 'a1': 1.0, 'a2': 0.70, 'a3': 0.40},
    'u': {'f1': 350, 'f2': 550,  'f3': 2100, 'a1': 1.0, 'a2': 0.50, 'a3': 0.25},
}

# Consonant synthesis parameters
CONSONANT_PARAMS = {
    'p': {'type': 'stop', 'noise_ms': 20, 'f0': 120},
    't': {'type': 'stop', 'noise_ms': 15, 'f0': 120},
    'k': {'type': 'stop', 'noise_ms': 25, 'f0': 120},
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
    # Add f0 modulation
    audio *= (0.5 + 0.5*np.sin(2*np.pi*f0*s)).astype(np.float32)
    # Add noise
    noise_amp = 10**(noise_db/20)
    audio += rng.randn(n).astype(np.float32) * noise_amp
    # HP filter
    audio = sosfilt(HP100, audio).astype(np.float32)
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def gen_consonant(consonant, duration=0.3, rng=None):
    """Generate synthetic consonant audio."""
    if rng is None:
        rng = np.random.RandomState()
    params = CONSONANT_PARAMS.get(consonant, {'type': 'stop', 'noise_ms': 20})
    n = int(duration * SR)
    audio = np.zeros(n, dtype=np.float32)

    if params['type'] == 'stop':
        # Short noise burst
        burst_samples = min(n, int(SR * params['noise_ms'] / 1000.0))
        audio[:burst_samples] = rng.randn(burst_samples).astype(np.float32) * 0.3
    elif params['type'] == 'fricative':
        # Colored noise
        noise = rng.randn(n).astype(np.float32)
        # High-pass via differencing
        noise = np.diff(noise, prepend=0.0).astype(np.float32)
        # Modulate with center frequency
        s = np.arange(n, dtype=np.float32) / SR
        audio = (noise * np.sin(2*np.pi*params.get('center_freq', 6000)*s) * params.get('amp', 0.3)).astype(np.float32)
    elif params['type'] == 'nasal':
        # Low-frequency resonance
        s = np.arange(n, dtype=np.float32) / SR
        audio = (params.get('amp', 0.4) * np.sin(2*np.pi*params.get('f0', 250)*s)).astype(np.float32)
        # Add nasal formant
        audio += (params.get('amp', 0.4) * 0.3 * np.sin(2*np.pi*params.get('f0', 250)*2.5*s)).astype(np.float32)

    # HP filter
    audio = sosfilt(HP100, audio).astype(np.float32)
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def mel_feat(frame, n_mels=N_MELS, n_fft=512, sr=SR):
    x = np.zeros(n_fft, dtype=np.float32)
    n = min(len(frame), n_fft)
    x[:n] = np.asarray(frame, dtype=np.float32)[:n]
    spectrum = np.abs(np.fft.rfft(x))
    # Simple mel filterbank
    mel_fb = _mel_filterbank(n_mels, n_fft, sr)
    mel_spec = mel_fb @ spectrum
    mel_spec = np.log(mel_spec + 1e-6).astype(np.float32)
    rms = float(np.sqrt(np.mean(mel_spec**2)))
    if rms > 1e-6:
        mel_spec /= rms
    return mel_spec


@staticmethod
def _mel_filterbank(n_mels, n_fft, sr):
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


# Cache the mel filterbank
_MEL_FB = None

def _get_mel_fb():
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(N_MELS, 512, SR)
    return _MEL_FB


def mel_feat(frame, n_mels=N_MELS, n_fft=512, sr=SR):
    x = np.zeros(n_fft, dtype=np.float32)
    n = min(len(frame), n_fft)
    x[:n] = np.asarray(frame, dtype=np.float32)[:n]
    spectrum = np.abs(np.fft.rfft(x))
    mel_fb = _get_mel_fb()
    mel_spec = mel_fb @ spectrum
    mel_spec = np.log(mel_spec + 1e-6).astype(np.float32)
    rms = float(np.sqrt(np.mean(mel_spec**2)))
    if rms > 1e-6:
        mel_spec /= rms
    return mel_spec


def extract_mel_rms(audio, frame_size=FRAME, hp_sos=HP100, sr=SR):
    audio = sosfilt(hp_sos, np.asarray(audio, dtype=np.float32).reshape(-1))
    n_frames = len(audio) // frame_size
    all_mel = []
    for i in range(n_frames):
        frame = audio[i*frame_size:(i+1)*frame_size]
        if len(frame) < frame_size:
            frame = np.zeros(frame_size, dtype=np.float32)
            frame[:len(frame)] = audio[i*frame_size:]
        all_mel.append(mel_feat(frame))
    return np.array(all_mel, dtype=np.float32)


def _real_capture_split_key(stem: str) -> int:
    """Stable per-file split key in [0, 99]."""
    digest = hashlib.sha1(stem.encode('utf-8')).digest()
    return digest[0] % 100


def load_real_vowel_captures(data_dir="data/m9_tts_captures", split=None, holdout_pct=20):
    """Load real vowel captures from the M9 dataset."""
    import wave
    data_dir = Path(data_dir)
    ds = []
    phrase_vowels = {
        "beat_bet_hot_caught": ["i", "E", "a", "O", "u"],
        "boot_caught_hot_bet": ["u", "O", "a", "E", "i"],
        "beat_hot_bet_boot": ["i", "a", "E", "u", "O"],
    }
    for wav_path in sorted(data_dir.glob("*.wav")):
        name = wav_path.stem.lower()
        phrase = None
        for p in phrase_vowels:
            if p in name:
                phrase = p
                break
        if not phrase:
            continue
        if split and _real_capture_split_key(wav_path.stem) >= holdout_pct:
            if split == "train":
                continue
        elif split == "eval" and _real_capture_split_key(wav_path.stem) < holdout_pct:
            continue

        vowels = phrase_vowels[phrase]
        with wave.open(str(wav_path), 'rb') as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
        mel = extract_mel_rms(audio)
        nf = len(mel)
        if nf < 5:
            continue
        chunk_size = nf // 5
        for idx in range(5):
            s = idx * chunk_size
            e = (idx + 1) * chunk_size if idx < 4 else nf
            mid = (s + e) // 2
            ds.append((vowels[idx], mel[mid]))
    return ds


def load_real_consonant_captures(data_dir="data/m10", split=None, holdout_pct=20):
    """Load real consonant captures from the M10 dataset."""
    manifest_path = Path(data_dir) / "manifests" / "captures.jsonl"
    if not manifest_path.exists():
        return []

    ds = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label", "")
            wav_path = row.get("wav_path", "")
            if not label or not wav_path:
                continue
            if label not in ALL_NAMES:
                continue

            # Deterministic split by capture_id
            split_key = _real_capture_split_key(row.get("capture_id", wav_path))
            if split == "train" and split_key < holdout_pct:
                continue
            if split == "eval" and split_key >= holdout_pct:
                continue

            full_path = Path(data_dir).parent / wav_path
            if not full_path.exists():
                continue

            import wave
            with wave.open(str(full_path), 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
            mel = extract_mel_rms(audio)
            if len(mel) == 0:
                continue
            mid = len(mel) // 2
            ds.append((label, mel[mid]))
    return ds


def gen_synth_vowels(n_per=300, seed=42):
    """Generate synthetic vowel training data."""
    rng = np.random.RandomState(seed)
    ds = []
    for v in VOWEL_NAMES:
        for _ in range(n_per):
            audio = gen_vowel(v, duration=rng.uniform(0.3, 0.6), f0=rng.uniform(100, 180),
                              jitter=rng.uniform(0.02, 0.08), noise_db=rng.uniform(-45, -30), rng=rng)
            mel = extract_mel_rms(audio)
            if len(mel) > 0:
                mid = len(mel) // 2
                ds.append((v, mel[mid]))
    return ds


def gen_synth_consonants(n_per=100, seed=42):
    """Generate synthetic consonant training data."""
    rng = np.random.RandomState(seed)
    ds = []
    for c in CONSONANT_NAMES:
        for _ in range(n_per):
            duration = rng.uniform(0.15, 0.4)
            audio = gen_consonant(c, duration=duration, rng=rng)
            mel = extract_mel_rms(audio)
            if len(mel) > 0:
                mid = len(mel) // 2
                ds.append((c, mel[mid]))
    return ds


def _adam_step(param, grad, m, v, beta1, beta2, eps, t, lr):
    """In-place Adam update with separate first/second-moment buffers."""
    m[:] = beta1 * m + (1.0 - beta1) * grad
    v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
    mh = m / (1.0 - beta1 ** t)
    vh = v / (1.0 - beta2 ** t)
    param -= lr * mh / (np.sqrt(vh) + eps)


def train_m10(epochs=500, lr=0.01, n_synth_vowels=200, n_synth_consonants=100, seed=42):
    """Train with cross-entropy + softmax on vowels + consonants."""
    rng = np.random.RandomState(seed)

    # Feedforward: N_MELS -> HIDDEN_DIM -> N_CLASSES
    W1 = rng.randn(N_MELS, HIDDEN_DIM).astype(np.float32) * np.sqrt(2.0 / N_MELS)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = rng.randn(HIDDEN_DIM, N_CLASSES).astype(np.float32) * np.sqrt(2.0 / HIDDEN_DIM)
    b2 = np.zeros(N_CLASSES, dtype=np.float32)

    # Load data
    real_vowels_train = load_real_vowel_captures(split="train")
    real_vowels_eval = load_real_vowel_captures(split="eval")
    real_consonants_train = load_real_consonant_captures(split="train")
    real_consonants_eval = load_real_consonant_captures(split="eval")
    synth_vowels = gen_synth_vowels(n_synth_vowels, seed)
    synth_consonants = gen_synth_consonants(n_synth_consonants, seed)

    ds = real_vowels_train + real_consonants_train + synth_vowels + synth_consonants
    eval_ds = real_vowels_eval + real_consonants_eval

    X = np.array([m for _, m in ds], dtype=np.float32)
    y = np.array([ALL_NAMES.index(v) for v, _ in ds], dtype=np.int32)

    print(f"Real vowel train: {len(real_vowels_train)}, Real consonant train: {len(real_consonants_train)}")
    print(f"Synth vowels: {len(synth_vowels)}, Synth consonants: {len(synth_consonants)}")
    print(f"Total train: {len(ds)}, Eval: {len(eval_ds)}")
    print(f"Classes: {ALL_NAMES}")
    print(f"Epochs: {epochs}, lr: {lr}")
    print()

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)

    batch_size = min(256, len(X))
    n_batches = max(1, len(X) // batch_size)

    for ep in range(epochs):
        # Shuffle
        perm = rng.permutation(len(X))
        X_shuf = X[perm]
        y_shuf = y[perm]

        total_loss = 0.0
        correct = 0

        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, len(X))
            bx = X_shuf[start:end]
            by = y_shuf[start:end]

            # Forward
            h = np.maximum(bx @ W1 + b1, 0)
            logits = h @ W2 + b2

            # Softmax
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Cross-entropy loss
            correct_probs = probs[np.arange(len(by)), by]
            loss = -np.mean(np.log(correct_probs + 1e-8))
            total_loss += float(loss)

            # Accuracy
            preds = np.argmax(logits, axis=1)
            correct += int(np.sum(preds == by))

            # Backward
            dlogits = probs.copy()
            dlogits[np.arange(len(by)), by] -= 1
            dlogits /= len(by)

            dW2 = h.T @ dlogits
            db2 = np.sum(dlogits, axis=0)
            dh = dlogits @ W2.T
            dh = dh * (h > 0).astype(np.float32)

            dW1 = bx.T @ dh
            db1 = np.sum(dh, axis=0)

            # Adam update
            t = ep * n_batches + bi + 1
            _adam_step(W1, dW1, mW1, vW1, beta1, beta2, eps, t, lr)
            _adam_step(b1, db1, mb1, vb1, beta1, beta2, eps, t, lr)
            _adam_step(W2, dW2, mW2, vW2, beta1, beta2, eps, t, lr)
            _adam_step(b2, db2, mb2, vb2, beta1, beta2, eps, t, lr)

        acc = correct / len(X)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  E{ep:4d} loss={total_loss/n_batches:.4f} acc={acc:.1%}")
            if not np.isfinite(total_loss) or not np.isfinite(acc):
                raise FloatingPointError("Non-finite loss/accuracy detected during training")

    # Eval
    if eval_ds:
        X_eval = np.array([m for _, m in eval_ds], dtype=np.float32)
        y_eval = np.array([ALL_NAMES.index(v) for v, _ in eval_ds], dtype=np.int32)
        h = np.maximum(X_eval @ W1 + b1, 0)
        logits = h @ W2 + b2
        preds = np.argmax(logits, axis=1)
        eval_acc = float(np.mean(preds == y_eval))
        print(f"\nEval accuracy: {eval_acc:.1%} ({int(np.sum(preds == y_eval))}/{len(y_eval)})")

        # Per-class accuracy
        print("Per-class:")
        for i, name in enumerate(ALL_NAMES):
            mask = y_eval == i
            if mask.sum() > 0:
                class_acc = float(np.mean(preds[mask] == i))
                print(f"  {name:4s}: {class_acc:.1%} (n={mask.sum()})")

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def main():
    p = argparse.ArgumentParser(description="M10 consonant training")
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--n-synth-vowels', type=int, default=200)
    p.add_argument('--n-synth-consonants', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.01)
    a = p.parse_args()

    params = train_m10(
        epochs=a.epochs,
        lr=a.lr,
        n_synth_vowels=a.n_synth_vowels,
        n_synth_consonants=a.n_synth_consonants,
    )

    out_path = Path('data/m10_v1.json')
    with open(out_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in params.items()}, f)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
