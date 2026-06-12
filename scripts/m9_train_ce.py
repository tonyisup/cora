#!/usr/bin/env python3
"""M9 Substrate Training — cross-entropy + softmax (classification, not contrastive)."""

import sys, json, argparse, hashlib
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

def extract_mel_rms(audio):
    """Extract mel features with per-frame RMS normalization (no centering)."""
    audio = sosfilt(HP100, audio).astype(np.float32)
    nf = len(audio) // FRAME
    all_mel = []
    for i in range(nf):
        frame = audio[i*FRAME:(i+1)*FRAME].flatten().astype(np.float32)
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > 1e-6:
            frame = frame / rms
        all_mel.append(mel_feat(frame))
    return np.array(all_mel, dtype=np.float32)

def _real_capture_split_key(wav_path: Path) -> int:
    """Stable per-file split key in [0, 99]."""
    digest = hashlib.sha1(wav_path.stem.encode("utf-8")).digest()
    return digest[0] % 100


def _load_real_capture_items(data_dir="data/m9_tts_captures", split=None, holdout_pct=20):
    import wave
    data_dir = Path(data_dir)
    items = []
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
                phrase = p; break
        if not phrase:
            continue

        split_key = _real_capture_split_key(wav_path)
        if split == "train" and split_key < holdout_pct:
            continue
        if split == "eval" and split_key >= holdout_pct:
            continue

        vowels = phrase_vowels[phrase]
        speaker = wav_path.stem.split("_")[0]
        with wave.open(str(wav_path), 'rb') as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
        mel = extract_mel_rms(audio)
        nf = len(mel)
        if nf < 5:
            continue
        chunk_size = nf // 5
        for idx in range(5):
            start = idx * chunk_size
            end = (idx + 1) * chunk_size if idx < 4 else nf
            mid = (start + end) // 2
            items.append({
                "speaker": speaker,
                "file": wav_path.name,
                "phrase": phrase,
                "split_key": split_key,
                "vowel_index": idx,
                "true_vowel": vowels[idx],
                "mel": mel[mid],
                "frames": nf,
                "chunk_size": chunk_size,
            })
    return items


def load_real_captures(data_dir="data/m9_tts_captures", split=None, holdout_pct=20):
    items = _load_real_capture_items(data_dir=data_dir, split=split, holdout_pct=holdout_pct)
    return [(item["true_vowel"], item["mel"]) for item in items]

def gen_synth(n_per=300, seed=42):
    rng = np.random.RandomState(seed)
    ds = []
    for v in VOWEL_NAMES:
        for _ in range(n_per):
            audio = gen_vowel(v, duration=rng.uniform(0.3,0.6), f0=rng.uniform(100,180),
                              jitter=rng.uniform(0.02,0.08), noise_db=rng.uniform(-45,-30), rng=rng)
            mel = extract_mel_rms(audio)
            if len(mel) > 0:
                mid = len(mel) // 2
                ds.append((v, mel[mid]))
    return ds

def _adam_step(param, grad, m, v, beta1, beta2, eps, t, lr):
    """In-place Adam update with separate first/second-moment buffers."""
    m[:] = beta1 * m + (1.0 - beta1) * grad
    v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
    mh = m / (1.0 - beta1 ** t)
    vh = v / (1.0 - beta2 ** t)
    param -= lr * mh / (np.sqrt(vh) + eps)


def train_ce(epochs=500, lr=0.01, n_synth=300, seed=42, a_weight=1.0):
    """Train with cross-entropy + softmax loss."""
    rng = np.random.RandomState(seed)
    
    # Simple feedforward: 32 -> 64 -> 5
    W1 = rng.randn(N_MELS, HIDDEN_DIM).astype(np.float32) * np.sqrt(2.0 / N_MELS)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = rng.randn(HIDDEN_DIM, 5).astype(np.float32) * np.sqrt(2.0 / HIDDEN_DIM)
    b2 = np.zeros(5, dtype=np.float32)
    
    # Load data
    real_train = load_real_captures(split="train")
    real_eval = load_real_captures(split="eval")
    synth = gen_synth(n_synth, seed)
    ds = real_train + synth
    
    X = np.array([m for _, m in ds], dtype=np.float32)
    y = np.array([VOWEL_NAMES.index(v) for v, _ in ds], dtype=np.int32)
    
    print(f"Real train: {len(real_train)}, Real eval: {len(real_eval)}, Synth: {len(synth)}, Total train: {len(ds)}")
    print(f"Epochs: {epochs}, lr: {lr}")
    print()
    
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
    
    batch_size = min(256, len(X))
    n_batches = len(X) // batch_size
    if n_batches == 0:
        raise ValueError(f"Not enough training examples ({len(X)}) for batch_size={batch_size}")
    
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]
        
        total_loss = 0.0
        correct = 0
        
        for bi in range(n_batches):
            bx = X[bi*batch_size:(bi+1)*batch_size]
            by = y[bi*batch_size:(bi+1)*batch_size]
            
            # Forward
            h = np.maximum(bx @ W1 + b1, 0)  # ReLU
            logits = h @ W2 + b2  # (batch, 5)
            
            # Softmax
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Cross-entropy loss
            correct_probs = probs[np.arange(len(by)), by]
            sample_weights = np.ones(len(by), dtype=np.float32)
            if a_weight != 1.0:
                sample_weights[by == VOWEL_NAMES.index('a')] = float(a_weight)
            loss = -np.mean(sample_weights * np.log(correct_probs + 1e-8))
            total_loss += float(loss)
            
            # Accuracy
            preds = np.argmax(logits, axis=1)
            correct += int(np.sum(preds == by))
            
            # Backward
            dlogits = probs.copy()
            dlogits[np.arange(len(by)), by] -= 1
            dlogits *= sample_weights[:, None]
            dlogits /= len(by)
            
            dW2 = h.T @ dlogits
            db2 = np.sum(dlogits, axis=0)
            dh = dlogits @ W2.T
            dh = dh * (h > 0).astype(np.float32)  # ReLU backward
            
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
    
    return {'W1':W1,'b1':b1,'W2':W2,'b2':b2}

def evaluate_real_only(W1, b1, W2, b2, holdout_pct=20):
    """Evaluate on held-out real captures only and print a confusion matrix."""
    eval_items = _load_real_capture_items(split="eval", holdout_pct=holdout_pct)
    if not eval_items:
        raise ValueError("Real evaluation split is empty")

    X_eval = np.array([item["mel"] for item in eval_items], dtype=np.float32)
    y_eval = np.array([VOWEL_NAMES.index(item["true_vowel"]) for item in eval_items], dtype=np.int32)

    h = np.maximum(X_eval @ W1 + b1, 0)
    logits = h @ W2 + b2
    preds = np.argmax(logits, axis=1)

    n_classes = len(VOWEL_NAMES)
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for yt, yp in zip(y_eval, preds):
        cm[yt, yp] += 1

    correct = int(np.trace(cm))
    total = int(cm.sum())
    acc = correct / total if total else float("nan")

    per_class = {}
    for i, v in enumerate(VOWEL_NAMES):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        pred_pos = int(cm[:, i].sum())
        fp = pred_pos - tp
        fn = support - tp
        precision = tp / pred_pos if pred_pos else float("nan")
        recall = tp / support if support else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if precision > 0 and recall > 0 else float("nan")
        per_class[v] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": int(fp),
            "fn": int(fn),
        }

    errors = []
    for item, yt, yp, logit in zip(eval_items, y_eval, preds, logits):
        if yt != yp:
            errors.append({
                "file": item["file"],
                "speaker": item["speaker"],
                "phrase": item["phrase"],
                "vowel_index": int(item["vowel_index"]),
                "true": VOWEL_NAMES[int(yt)],
                "pred": VOWEL_NAMES[int(yp)],
                "frames": int(item["frames"]),
                "chunk_size": int(item["chunk_size"]),
                "logits": [float(x) for x in np.round(logit, 2)],
            })

    print("\n--- Held-out real-capture evaluation ---")
    print(f"Eval examples: {total}")
    print(f"Accuracy: {acc:.1%} ({correct}/{total})")
    print("Per-class metrics:")
    for v in VOWEL_NAMES:
        pc = per_class[v]
        print(
            f"  {v}: precision={pc['precision']:.1%} recall={pc['recall']:.1%} f1={pc['f1']:.1%} "
            f"support={pc['support']}"
        )
    print("Confusion matrix (rows=true, cols=pred):")
    header = "       " + " ".join(f"{v:>4s}" for v in VOWEL_NAMES)
    print(header)
    for i, v in enumerate(VOWEL_NAMES):
        row = " ".join(f"{int(n):4d}" for n in cm[i])
        row_total = int(cm[i].sum())
        row_acc = (cm[i, i] / row_total) if row_total else float("nan")
        print(f"{v:>4s}  {row}   | acc={row_acc:.1%} n={row_total}")
    if errors:
        print("Misclassified examples:")
        for err in errors[:10]:
            print(
                f"  {err['file']} speaker={err['speaker']} phrase={err['phrase']} idx={err['vowel_index']} "
                f"true={err['true']} pred={err['pred']} frames={err['frames']}"
            )

    return {
        "holdout_pct": holdout_pct,
        "accuracy": float(acc),
        "correct": correct,
        "total": total,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "error_examples": errors,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--n-synth',type=int,default=300)
    p.add_argument('--a-weight',type=float,default=1.0)
    a = p.parse_args()
    params = train_ce(epochs=a.epochs, n_synth=a.n_synth, a_weight=a.a_weight)
    eval_results = evaluate_real_only(params['W1'], params['b1'], params['W2'], params['b2'])
    with open('data/m9_ce_v1.json','w') as f:
        json.dump({k: v.tolist() for k,v in params.items()}, f)
    with open('data/m9_ce_v1_eval.json','w') as f:
        json.dump(eval_results, f, indent=2)
    print("Saved m9_ce_v1.json")
    print("Saved m9_ce_v1_eval.json")

if __name__=='__main__': main()
