#!/usr/bin/env python3
"""M10 Consonant Training v3 — 48kHz with wideband mel features.

Uses 48kHz speaker→mic captures with 64-bin mel filterbank (fmax=20kHz)
to capture high-frequency discriminative information lost at 16kHz.

Usage:
    ./.venv/bin/python scripts/m10_train_v3.py --epochs 300
"""

import sys, json, argparse, hashlib
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))

# 48kHz configuration
SR = 48000
FRAME = 1440  # 30ms at 48kHz
HIDDEN_DIM = 128
N_MELS = 96  # Even more bins for finer frequency resolution
FMAX = 22000
HIDDEN_DIM = 192
VOWEL_NAMES = ['i', 'E', 'a', 'O', 'u']
CONSONANT_NAMES = ['p', 't', 'k', 's', 'f', 'sh', 'm', 'n']
ALL_NAMES = VOWEL_NAMES + CONSONANT_NAMES
N_CLASSES = len(ALL_NAMES)

# Build wideband mel filterbank
def _build_mel_fb():
    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    n_fft = 2048
    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(80), hz_to_mel(FMAX), N_MELS + 2)
    bin_points = np.floor((n_fft + 1) * mel_to_hz(mel_points) / SR).astype(int)
    fb = np.zeros((N_MELS, n_bins), dtype=np.float32)
    for m in range(N_MELS):
        left, center, right = bin_points[m], bin_points[m+1], bin_points[m+2]
        for k in range(left, center):
            if k < n_bins and center > left: fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if k < n_bins and right > center: fb[m, k] = (right - k) / (right - center)
    return fb, n_fft

MEL_FB, N_FFT = _build_mel_fb()
HP200 = butter(4, 200, btype='high', fs=SR, output='sos')


def mel_feat(frame):
    x = np.zeros(N_FFT, dtype=np.float32)
    n = min(len(frame), N_FFT)
    x[:n] = frame[:n]
    spec = np.abs(np.fft.rfft(x))
    mel = MEL_FB @ spec
    mel = np.log(mel + 1e-6).astype(np.float32)
    rms = float(np.sqrt(np.mean(mel**2)))
    if rms > 1e-6: mel /= rms
    return mel


def extract_mel_rms(audio):
    audio = sosfilt(HP200, np.asarray(audio, dtype=np.float32).reshape(-1))
    n_frames = len(audio) // FRAME
    if n_frames == 0:
        # Pad to at least one frame
        audio = np.pad(audio, (0, FRAME - len(audio)))
        n_frames = 1
    return np.array([mel_feat(audio[i*FRAME:(i+1)*FRAME]) for i in range(n_frames)], dtype=np.float32)


def split_key(stem):
    return int(hashlib.sha1(stem.encode()).digest()[0]) % 100


def load_48k_consonants(split=None, holdout_pct=20):
    """Load 48kHz speaker→mic consonant captures."""
    import wave
    ds = []
    manifest_path = Path("data/m10/manifests/captures.jsonl")
    if not manifest_path.exists():
        return ds
    
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line: continue
        row = json.loads(line)
        label = row.get("label", "")
        if label not in ALL_NAMES: continue
        
        # Only use speaker→mic data (not tts_direct)
        if row.get("source") == "tts_direct": continue
        
        sk = split_key(row.get("capture_id", ""))
        if split == "train" and sk < holdout_pct: continue
        if split == "eval" and sk >= holdout_pct: continue
        
        # Try 48k path first, fall back to 16k
        wav_path_48k = row.get("wav_path_48k", "")
        if wav_path_48k:
            wav_path = Path(wav_path_48k)
        else:
            # Check tts48 directory for 48k version
            wav_16k = Path(row["wav_path"])
            stem = wav_16k.stem.replace("_16k", "_48k")
            wav_path = wav_16k.parent / f"{stem}.wav"
            if not wav_path.exists():
                wav_path = wav_16k  # Fall back to 16k
        
        if not wav_path.exists(): continue
        
        with wave.open(str(wav_path), 'rb') as wf:
            actual_sr = wf.getframerate()
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
        
        # Resample to 48kHz if needed
        if actual_sr != SR:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, SR // actual_sr, 1)
        
        mel = extract_mel_rms(audio)
        if len(mel) > 0:
            ds.append((label, mel[len(mel)//2]))
    
    return ds


def load_vowels_48k(split=None, holdout_pct=20):
    """Load vowel captures, upsampled to 48kHz."""
    import wave
    from scipy.signal import resample_poly
    ds = []
    vowel_dir = Path("data/m9_tts_captures")
    phrase_vowels = {
        "beat_bet_hot_caught": ["i", "E", "a", "O", "u"],
        "boot_caught_hot_bet": ["u", "O", "a", "E", "i"],
        "beat_hot_bet_boot": ["i", "a", "E", "u", "O"],
    }
    for wav_path in sorted(vowel_dir.glob("*.wav")):
        name = wav_path.stem.lower()
        phrase = next((p for p in phrase_vowels if p in name), None)
        if not phrase: continue
        sk = split_key(wav_path.stem)
        if split == "train" and sk < holdout_pct: continue
        if split == "eval" and sk >= holdout_pct: continue
        with wave.open(str(wav_path), 'rb') as wf:
            audio_16k = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
        # Upsample to 48kHz
        audio_48k = resample_poly(audio_16k, 3, 1)
        mel = extract_mel_rms(audio_48k)
        if len(mel) < 5: continue
        chunk = len(mel) // 5
        for idx, v in enumerate(phrase_vowels[phrase]):
            mid = (idx * chunk + ((idx + 1) * chunk if idx < 4 else len(mel))) // 2
            ds.append((v, mel[mid]))
    return ds


def main():
    parser = argparse.ArgumentParser(description="M10 48kHz training")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--holdout", type=int, default=20)
    a = parser.parse_args()

    print(f"M10 48kHz Training — {N_MELS} mel bins, fmax={FMAX//1000}kHz, SR={SR//1000}kHz")
    print()

    # Load data
    train_v = load_vowels_48k(split="train", holdout_pct=a.holdout)
    eval_v = load_vowels_48k(split="eval", holdout_pct=a.holdout)
    train_c = load_48k_consonants(split="train", holdout_pct=a.holdout)
    eval_c = load_48k_consonants(split="eval", holdout_pct=a.holdout)

    train_ds = train_v + train_c
    eval_ds = eval_v + eval_c

    print(f"Train: {len(train_ds)} (vowels={len(train_v)}, consonants={len(train_c)})")
    print(f"Eval: {len(eval_ds)} (vowels={len(eval_v)}, consonants={len(eval_c)})")

    from collections import Counter
    print("Train labels:", dict(Counter(r[0] for r in train_ds)))
    print("Eval labels:", dict(Counter(r[0] for r in eval_ds)))
    print()

    # Train
    rng = np.random.RandomState(42)
    W1 = rng.randn(N_MELS, HIDDEN_DIM).astype(np.float32) * np.sqrt(2.0 / N_MELS)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = rng.randn(HIDDEN_DIM, N_CLASSES).astype(np.float32) * np.sqrt(2.0 / HIDDEN_DIM)
    b2 = np.zeros(N_CLASSES, dtype=np.float32)

    X = np.array([m for _, m in train_ds], dtype=np.float32)
    y = np.array([ALL_NAMES.index(v) for v, _ in train_ds], dtype=np.int32)

    # Sample weights: boost consonants
    sw = np.ones(len(y), dtype=np.float32)
    for i, n in enumerate(ALL_NAMES):
        if n in CONSONANT_NAMES:
            sw[y == i] = 2.0

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)

    bs = min(256, len(X))
    nb = max(1, len(X) // bs)

    for ep in range(a.epochs):
        perm = rng.permutation(len(X))
        Xs, ys, ws = X[perm], y[perm], sw[perm]
        total_loss = 0.0
        correct = 0

        for bi in range(nb):
            s, e = bi * bs, min((bi + 1) * bs, len(X))
            bx, by, bw = Xs[s:e], ys[s:e], ws[s:e]

            h = np.maximum(bx @ W1 + b1, 0)
            logits = h @ W2 + b2

            lm = logits - np.max(logits, axis=1, keepdims=True)
            el = np.exp(lm)
            p = el / np.sum(el, axis=1, keepdims=True)
            cp = p[np.arange(len(by)), by]
            loss = -np.mean(bw * np.log(cp + 1e-8))
            total_loss += float(loss)

            preds = np.argmax(logits, axis=1)
            correct += int(np.sum(preds == by))

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
            mW1[:] = beta1 * mW1 + (1 - beta1) * dW1
            vW1[:] = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
            mW2[:] = beta1 * mW2 + (1 - beta1) * dW2
            vW2[:] = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
            W1 -= a.lr * mW1 / (1 - beta1**t) / (np.sqrt(vW1 / (1 - beta2**t)) + eps)
            W2 -= a.lr * mW2 / (1 - beta1**t) / (np.sqrt(vW2 / (1 - beta2**t)) + eps)
            b1 -= a.lr * db1 / len(by)
            b2 -= a.lr * db2 / len(by)

        if ep % 50 == 0 or ep == a.epochs - 1:
            acc = correct / len(X)
            print(f"  E{ep:4d} loss={total_loss/nb:.4f} acc={acc:.1%}")

    # Eval
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

    # Save
    out = Path('data/m10_v1.json')
    with open(out, 'w') as f:
        json.dump({
            'W1': W1.tolist(), 'b1': b1.tolist(),
            'W2': W2.tolist(), 'b2': b2.tolist(),
            'config': {
                'sr': SR, 'frame': FRAME, 'n_mels': N_MELS,
                'fmax': FMAX, 'hidden_dim': HIDDEN_DIM,
                'n_classes': N_CLASSES, 'class_names': ALL_NAMES
            }
        }, f)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
