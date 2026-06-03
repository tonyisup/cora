#!/usr/bin/env python3
"""M9 Live Capture + Substrate Retrain.

Records real vowel audio from the mic, combines with synthetic data,
and retrains the spectral substrate for better live classification.

Steps:
1. Record N samples of each vowel from live mic
2. Combine with synthetic training data
3. Retrain substrate with contrastive loss
4. Rebuild lookup table
5. Verify accuracy improvement

Usage:
    python scripts/m9_live_retrain.py
    python scripts/m9_live_retrain.py --skip-capture  # reuse previous captures
"""

import sys, json, argparse, time
import numpy as np
from scipy.signal import butter, sosfilt
from pathlib import Path
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))

SR = 16000
FRAME = 480
HIDDEN_DIM = 32
N_MEL = 26

VOWEL_FORMANTS = {
    'i':  {'f1': 280, 'f2': 2250, 'f3': 2900, 'a1': 1.0, 'a2': 0.4, 'a3': 0.2},
    'E':  {'f1': 530, 'f2': 1840, 'f3': 2480, 'a1': 1.0, 'a2': 0.5, 'a3': 0.25},
    'a':  {'f1': 730, 'f2': 1090, 'f3': 2440, 'a1': 1.0, 'a2': 0.7, 'a3': 0.35},
    'O':  {'f1': 570, 'f2': 840,  'f3': 2410, 'a1': 1.0, 'a2': 0.6, 'a3': 0.3},
    'u':  {'f1': 300, 'f2': 870,  'f3': 2240, 'a1': 1.0, 'a2': 0.4, 'a3': 0.15},
}
VOWEL_NAMES = list(VOWEL_FORMANTS.keys())

VOWEL_IPA = {'i': '/i/', 'E': '/ɛ/', 'a': '/ɑ/', 'O': '/ɔ/', 'u': '/u/'}

CAPTURE_DIR = Path('data/m9_live_captures')
HP_SOS = butter(4, 100, btype='high', fs=SR, output='sos')


def summarize_live_captures(captures):
    """Return a compact count summary for a captures dict."""
    counts = {v: len(captures.get(v, [])) for v in VOWEL_NAMES}
    return {
        'vowel_counts': counts,
        'live_captured': sum(counts.values()),
    }

# ── Mel filterbank (same as training) ──────────────────────────────────────

def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def make_mel_filterbank(n_mels=N_MEL, n_fft=512, sr=SR, fmin=80, fmax=7600):
    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bin_points = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m+1], bin_points[m+2]
        for k in range(left, center):
            if k < n_bins and center > left: fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if k < n_bins and right > center: fb[m, k] = (right - k) / (right - center)
    return fb

MEL_FB = make_mel_filterbank()

def compute_mel(frame, n_fft=512):
    x = np.zeros(n_fft, dtype=np.float32)
    n = min(len(frame), n_fft)
    x[:n] = frame[:n]
    ms = MEL_FB @ np.abs(np.fft.rfft(x))
    ms = np.log(ms + 1e-6).astype(np.float32)
    r = np.sqrt(np.mean(ms**2))
    if r > 1e-6: ms /= r
    return ms

def audio_to_mel_frames(audio, hp_sos=HP_SOS):
    if hp_sos is not None:
        audio = sosfilt(hp_sos, audio)
    n_frames = max(0, (len(audio) - FRAME) // FRAME + 1) if len(audio) >= FRAME else 0
    mel_frames = np.zeros((n_frames, N_MEL), dtype=np.float32)
    for i in range(n_frames):
        raw_frame = audio[i * FRAME:(i + 1) * FRAME].flatten()
        mel_frames[i] = compute_mel(raw_frame)
    return mel_frames

# ── Capture ───────────────────────────────────────────────────────────────

def capture_vowels(n_per_vowel=10, duration=1.5, input_device=1, threshold=0.003, capture_dir=CAPTURE_DIR):
    """Record live vowel samples from the mic."""
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)

    print(f"Live Vowel Capture")
    print(f"  {n_per_vowel} samples per vowel, {duration}s each")
    print(f"  Input device: {input_device}")
    print(f"  RMS threshold: {threshold}")
    print()

    captures = {v: [] for v in VOWEL_NAMES}

    for v in VOWEL_NAMES:
        ipa = VOWEL_IPA[v]
        print(f"── {ipa} ({v}) ──")
        print(f"  Say and hold '{ipa}' when recording starts.")

        captured = 0
        existing = list(capture_dir.glob(f'{v}_*.json'))
        if existing:
            captured = len(existing)
            print(f"  Found {captured} existing captures.")

        while captured < n_per_vowel:
            input(f"  Press Enter for sample {captured + 1}/{n_per_vowel}...")

            # Record
            print(f"  ● Recording {duration}s...", end=' ', flush=True)
            rec = sd.rec(int(SR * duration), samplerate=SR, channels=1,
                         device=input_device, dtype='float32')
            sd.wait()

            audio = sosfilt(HP_SOS, rec.flatten()).astype(np.float32)
            rms = np.sqrt(np.mean(audio ** 2))

            if rms < threshold:
                print(f"too quiet (RMS={rms:.4f}), try again")
                continue

            # Save
            mel_frames = audio_to_mel_frames(audio, hp_sos=None)  # Already filtered
            sample = {
                'audio_rms': float(rms),
                'mel_frames': mel_frames.tolist(),
                'n_frames': len(mel_frames),
            }
            path = capture_dir / f'{v}_{captured:03d}.json'
            path.write_text(json.dumps(sample))
            captures[v].append(mel_frames)
            captured += 1
            print(f"OK (RMS={rms:.4f}, {len(mel_frames)} frames)")

        print()

    return captures


def load_captures(capture_dir=CAPTURE_DIR):
    """Load previously captured vowel samples."""
    capture_dir = Path(capture_dir)
    captures = {v: [] for v in VOWEL_NAMES}
    for path in sorted(capture_dir.glob('*.json')):
        v = path.stem.split('_')[0]
        if v not in VOWEL_NAMES:
            continue
        data = json.load(open(path))
        mel = np.array(data['mel_frames'], dtype=np.float32)
        captures[v].append(mel)
    return captures


def train_with_live_data(live_captures, n_synthetic=200, epochs=300,
                         lr=0.001, margin=8.0, desat_weight=0.2, l2_weight=1e-4):
    """Retrain substrate using both live captures and synthetic data."""
    # Import training functions from the spectral training script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m9_train_spectral import (
        gen_vowel, forward_substrate, backward_substrate,
        compute_triplet_loss_and_grad, compute_desat_loss_and_grad,
    )

    # Generate synthetic data
    print(f"Generating {n_synthetic} synthetic samples per vowel...")
    rng = np.random.RandomState(42)
    dataset = []

    for vowel in VOWEL_NAMES:
        # Add synthetic samples
        for _ in range(n_synthetic):
            f0 = rng.uniform(100, 180)
            jitter = rng.uniform(0.02, 0.08)
            noise_db = rng.uniform(-45, -30)
            dur = rng.uniform(0.4, 0.8)
            audio = gen_vowel(vowel, duration=dur, f0=f0, jitter=jitter,
                              noise_db=noise_db, rng=rng)
            mel_frames = audio_to_mel_frames(audio)
            dataset.append((vowel, mel_frames))

        # Add live captures (weight them by repeating)
        n_live = len(live_captures.get(vowel, []))
        if n_live > 0:
            # Repeat live samples to balance with synthetic count
            n_repeat = max(1, n_synthetic // n_live)
            live_list = live_captures[vowel]
            for _ in range(n_repeat):
                for mel_frames in live_list:
                    dataset.append((vowel, mel_frames))
            print(f"  {vowel}: {n_synthetic} synthetic + {n_live} live × {n_repeat} = {n_synthetic + n_live * n_repeat}")
        else:
            print(f"  {vowel}: {n_synthetic} synthetic (no live captures)")

    total = len(dataset)
    print(f"Total training samples: {total}")

    # Initialize from existing spectral checkpoint
    ckpt_path = Path('data/m9_substrate_spectral.ckpt.json')
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        W_h = np.array(ckpt['W_h'], dtype=np.float32)
        W_x = np.array(ckpt['W_x'], dtype=np.float32)
        b = np.array(ckpt['b'], dtype=np.float32)
        print("Resumed from spectral checkpoint")
    else:
        raise FileNotFoundError("No spectral checkpoint found. Run m9_train_spectral.py first.")

    # Adam optimizer
    m_Wh = np.zeros_like(W_h); v_Wh = np.zeros_like(W_h)
    m_Wx = np.zeros_like(W_x); v_Wx = np.zeros_like(W_x)
    m_b  = np.zeros_like(b);    v_b  = np.zeros_like(b)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    best_loss = float('inf')
    best_W_h, best_W_x, best_b = W_h.copy(), W_x.copy(), b.copy()

    for epoch in range(epochs):
        # Forward
        embeddings, labels, caches = [], [], []
        for vowel, mel_frames in dataset:
            h, cache = forward_substrate(W_h, W_x, b, mel_frames)
            embeddings.append(h)
            labels.append(vowel)
            caches.append(cache)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Losses
        tri_loss, d_tri, centroids = compute_triplet_loss_and_grad(embeddings, labels, margin=margin)
        desat_loss, d_desat = compute_desat_loss_and_grad(embeddings, weight=desat_weight)
        l2_loss = l2_weight * (np.sum(W_h**2) + np.sum(W_x**2) + np.sum(b**2))
        total_loss = tri_loss + desat_loss + l2_loss

        # Backprop
        d_emb = d_tri + d_desat
        d_W_h, d_W_x, d_b = np.zeros_like(W_h), np.zeros_like(W_x), np.zeros_like(b)
        for i in range(len(dataset)):
            dwh, dwx, db = backward_substrate(d_emb[i], caches[i], W_h)
            d_W_h += dwh; d_W_x += dwx; d_b += db
        d_W_h += 2*l2_weight*W_h; d_W_x += 2*l2_weight*W_x; d_b += 2*l2_weight*b

        # Gradient clipping
        gn = np.linalg.norm(np.concatenate([d_W_h.ravel(), d_W_x.ravel(), d_b.ravel()]))
        if gn > 1.0:
            c = 1.0/gn; d_W_h *= c; d_W_x *= c; d_b *= c

        # Adam
        t_step = epoch + 1
        for pname in ['W_h', 'W_x', 'b']:
            g = d_W_h if pname == 'W_h' else (d_W_x if pname == 'W_x' else d_b)
            m = m_Wh if pname == 'W_h' else (m_Wx if pname == 'W_x' else m_b)
            v = v_Wh if pname == 'W_h' else (v_Wx if pname == 'W_x' else v_b)
            m[:] = beta1*m + (1-beta1)*g
            v[:] = beta2*v + (1-beta2)*g**2
            mh = m / (1-beta1**t_step)
            vh = v / (1-beta2**t_step)
            upd = lr*mh / (np.sqrt(vh) + adam_eps)
            if pname == 'W_h': W_h -= upd
            elif pname == 'W_x': W_x -= upd
            else: b -= upd

        if total_loss < best_loss:
            best_loss = total_loss
            best_W_h, best_W_x, best_b = W_h.copy(), W_x.copy(), b.copy()

        if epoch % 20 == 0 or epoch == epochs - 1:
            between_dists = []
            for i, vi in enumerate(VOWEL_NAMES):
                for j, vj in enumerate(VOWEL_NAMES):
                    if j <= i: continue
                    between_dists.append(np.linalg.norm(centroids[vi] - centroids[vj]))
            bm = min(between_dists)
            # Accuracy
            tree = __import__('scipy.spatial', fromlist=['cKDTree']).cKDTree(
                np.array([centroids[v] for v in VOWEL_NAMES], dtype=np.float32))
            _, idxs = tree.query(embeddings, k=1)
            correct = sum(1 for i, idx in enumerate(idxs) if VOWEL_NAMES[idx] == labels[i])
            acc = correct / len(labels)
            print(f"  E{epoch:3d} loss={total_loss:.4f} tri={tri_loss:.4f} "
                  f"between_min={bm:.2f} acc={acc:.1%}")

    return best_W_h, best_W_x, best_b, dataset, centroids


def main():
    parser = argparse.ArgumentParser(description='M9 Live Capture + Retrain')
    parser.add_argument('--skip-capture', action='store_true', help='Reuse existing captures')
    parser.add_argument('--n-per-vowel', type=int, default=10, help='Live samples per vowel')
    parser.add_argument('--input-device', type=int, default=1, help='Mic input device')
    parser.add_argument('--epochs', type=int, default=200, help='Retrain epochs')
    parser.add_argument('--n-synthetic', type=int, default=200, help='Synthetic samples per vowel')
    args = parser.parse_args()

    if args.skip_capture:
        captures = load_captures()
        total = sum(len(v) for v in captures.values())
        print(f"Loaded {total} existing live captures")
        if total == 0:
            print("No captures found. Run without --skip-capture first.")
            return
    else:
        captures = capture_vowels(n_per_vowel=args.n_per_vowel, input_device=args.input_device)

    print(f"\nRetraining with live + synthetic data...")
    W_h, W_x, b, dataset, centroids = train_with_live_data(
        captures, n_synthetic=args.n_synthetic, epochs=args.epochs)

    # Save
    ckpt = {'W_h': W_h.tolist(), 'W_x': W_x.tolist(), 'b': b.tolist(),
            'hidden_dim': HIDDEN_DIM, 'n_mel': N_MEL, 'sr': SR,
            'live_captured': sum(len(v) for v in captures.values())}
    Path('data/m9_substrate_spectral.ckpt.json').write_text(json.dumps(ckpt))
    print("\nSaved retrained substrate: data/m9_substrate_spectral.ckpt.json")
    print("Run: python scripts/m9_train_spectral.py --eval-only  # to rebuild lookup")


def run_smoke_retrain(capture_dir=CAPTURE_DIR, n_synthetic=1, epochs=1, input_device=1):
    """Run a lightweight live-capture retrain regression over local fixtures."""
    capture_dir = Path(capture_dir)
    captures = load_captures(capture_dir)
    summary = summarize_live_captures(captures)
    if summary['live_captured'] == 0:
        raise FileNotFoundError(f"No live captures found in {capture_dir}")

    W_h, W_x, b, dataset, centroids = train_with_live_data(
        captures,
        n_synthetic=n_synthetic,
        epochs=epochs,
    )

    return {
        'capture_dir': str(capture_dir),
        'trained': True,
        'n_synthetic': n_synthetic,
        'epochs': epochs,
        'dataset_size': len(dataset),
        'centroid_count': len(centroids),
        **summary,
        'checkpoint_shape': {
            'W_h': list(W_h.shape),
            'W_x': list(W_x.shape),
            'b': list(b.shape),
        },
    }


def main():
    parser = argparse.ArgumentParser(description='M9 Live Capture + Retrain')
    parser.add_argument('--skip-capture', action='store_true', help='Reuse existing captures')
    parser.add_argument('--smoke', action='store_true', help='Run a lightweight regression over an existing capture fixture')
    parser.add_argument('--capture-dir', type=str, default=str(CAPTURE_DIR), help='Directory of live capture JSON fixtures')
    parser.add_argument('--n-per-vowel', type=int, default=10, help='Live samples per vowel')
    parser.add_argument('--input-device', type=int, default=1, help='Mic input device')
    parser.add_argument('--epochs', type=int, default=200, help='Retrain epochs')
    parser.add_argument('--n-synthetic', type=int, default=200, help='Synthetic samples per vowel')
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)

    if args.smoke:
        report = run_smoke_retrain(
            capture_dir=capture_dir,
            n_synthetic=args.n_synthetic,
            epochs=args.epochs,
            input_device=args.input_device,
        )
        print(json.dumps(report, indent=2))
        return

    if args.skip_capture:
        captures = load_captures(capture_dir)
        total = sum(len(v) for v in captures.values())
        print(f"Loaded {total} existing live captures")
        if total == 0:
            print("No captures found. Run without --skip-capture first.")
            return
    else:
        captures = capture_vowels(
            n_per_vowel=args.n_per_vowel,
            input_device=args.input_device,
            capture_dir=capture_dir,
        )

    print(f"\nRetraining with live + synthetic data...")
    W_h, W_x, b, dataset, centroids = train_with_live_data(
        captures, n_synthetic=args.n_synthetic, epochs=args.epochs)

    # Save
    ckpt = {'W_h': W_h.tolist(), 'W_x': W_x.tolist(), 'b': b.tolist(),
            'hidden_dim': HIDDEN_DIM, 'n_mel': N_MEL, 'sr': SR,
            'live_captured': sum(len(v) for v in captures.values())}
    Path('data/m9_substrate_spectral.ckpt.json').write_text(json.dumps(ckpt))
    print("\nSaved retrained substrate: data/m9_substrate_spectral.ckpt.json")
    print("Run: python scripts/m9_train_spectral.py --eval-only  # to rebuild lookup")


if __name__ == '__main__':
    main()
