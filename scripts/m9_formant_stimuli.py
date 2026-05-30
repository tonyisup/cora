#!/usr/bin/env python3
"""Generate formant-rich training stimuli for M9.

Creates synthetic vowel-like sounds using standard formant frequencies.
Each "vowel" is defined by 3-4 formant frequencies with amplitudes and bandwidths.

Standard formant values (Hz) for English vowels:
  /i/ (beat):  F1=280,  F2=2250, F3=2900
  /ɪ/ (bit):   F1=360,  F2=2220, F3=2960
  /ɛ/ (bet):   F1=530,  F2=1840, F3=2480
  /æ/ (bat):   F1=660,  F2=1720, F3=2410
  /ɑ/ (bot):   F1=730,  F2=1090, F3=2440
  /ɔ/ (bought):F1=570,  F2=840,  F3=2410
  /ʊ/ (book):  F1=440,  F2=1020, F3=2240
  /u/ (boot):  F1=300,  F2=870,  F3=2240
  /ʌ/ (but):   F1=640,  F2=1190, F3=2390
  /ɝ/ (bird):  F1=490,  F2=1350, F3=1690
"""

import sys
sys.path.insert(0, 'packages')
import numpy as np
from m9_sound import OscillatorBank

SR = 16000

# Standard vowel formant definitions (F1, F2, F3) with amplitudes and bandwidths
VOWEL_FORMANTS = {
    'i': [  # /i/ as in "beat"
        {'freq': 280, 'amp': 1.0, 'bw': 50},
        {'freq': 2250, 'amp': 0.4, 'bw': 80},
        {'freq': 2900, 'amp': 0.2, 'bw': 120},
    ],
    'I': [  # /ɪ/ as in "bit"
        {'freq': 360, 'amp': 1.0, 'bw': 50},
        {'freq': 2220, 'amp': 0.35, 'bw': 80},
        {'freq': 2960, 'amp': 0.15, 'bw': 120},
    ],
    'E': [  # /ɛ/ as in "bet"
        {'freq': 530, 'amp': 1.0, 'bw': 60},
        {'freq': 1840, 'amp': 0.5, 'bw': 90},
        {'freq': 2480, 'amp': 0.25, 'bw': 120},
    ],
    'ae': [  # /æ/ as in "bat"
        {'freq': 660, 'amp': 1.0, 'bw': 70},
        {'freq': 1720, 'amp': 0.6, 'bw': 100},
        {'freq': 2410, 'amp': 0.3, 'bw': 120},
    ],
    'a': [  # /ɑ/ as in "bot"
        {'freq': 730, 'amp': 1.0, 'bw': 80},
        {'freq': 1090, 'amp': 0.7, 'bw': 100},
        {'freq': 2440, 'amp': 0.35, 'bw': 120},
    ],
    'c': [  # /ɔ/ as in "bought"
        {'freq': 570, 'amp': 1.0, 'bw': 70},
        {'freq': 840, 'amp': 0.6, 'bw': 90},
        {'freq': 2410, 'amp': 0.3, 'bw': 120},
    ],
    'U': [  # /ʊ/ as in "book"
        {'freq': 440, 'amp': 1.0, 'bw': 60},
        {'freq': 1020, 'amp': 0.5, 'bw': 90},
        {'freq': 2240, 'amp': 0.2, 'bw': 120},
    ],
    'u': [  # /u/ as in "boot"
        {'freq': 300, 'amp': 1.0, 'bw': 50},
        {'freq': 870, 'amp': 0.4, 'bw': 80},
        {'freq': 2240, 'amp': 0.15, 'bw': 120},
    ],
    'V': [  # /ʌ/ as in "but"
        {'freq': 640, 'amp': 1.0, 'bw': 70},
        {'freq': 1190, 'amp': 0.6, 'bw': 100},
        {'freq': 2390, 'amp': 0.3, 'bw': 120},
    ],
    'r': [  # /ɝ/ as in "bird"
        {'freq': 490, 'amp': 1.0, 'bw': 60},
        {'freq': 1350, 'amp': 0.5, 'bw': 90},
        {'freq': 1690, 'amp': 0.4, 'bw': 100},
    ],
}

def generate_vowel(vowel, duration=0.5, sr=SR, f0=120):
    """Generate a vowel-like sound with formants + fundamental."""
    osc = OscillatorBank(sr=sr)
    formants = VOWEL_FORMANTS[vowel]
    
    # Generate fundamental (voicing) + formants
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    
    # Fundamental frequency (pitch)
    f0_amp = 0.3
    output = f0_amp * np.sin(2 * np.pi * f0 * t)
    
    # Add formants
    for fm in formants:
        formant = fm['amp'] * np.sin(2 * np.pi * fm['freq'] * t)
        output += formant
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak
    
    return output.astype(np.float32)

def generate_sequence(pattern, duration_each=0.3):
    """Generate a sequence of vowel-like sounds."""
    parts = []
    for vowel in pattern:
        parts.append(generate_vowel(vowel, duration=duration_each))
    return np.concatenate(parts)

if __name__ == '__main__':
    import sounddevice as sd
    import time
    
    print("Formant stimulus generator")
    print(f"Vowels: {list(VOWEL_FORMANTS.keys())}")
    print()
    
    # Demo: play each vowel
    for vowel, formants in VOWEL_FORMANTS.items():
        desc = ', '.join([f"F{i+1}={f['freq']}Hz" for i, f in enumerate(formants)])
        print(f"Vowel /{vowel}/: {desc}")
        audio = generate_vowel(vowel, duration=0.5)
        sd.play(audio, samplerate=SR, device=2, blocking=True)
        time.sleep(0.2)
    
    print()
    print("Demo: vowel sequences")
    sequences = [
        ('a-i', ['a', 'i']),
        ('i-a', ['i', 'a']),
        ('a-u', ['a', 'u']),
        ('u-a', ['u', 'a']),
        ('i-a-u', ['i', 'a', 'u']),
        ('a-i-u-E', ['a', 'i', 'u', 'E']),
    ]
    
    for name, pattern in sequences:
        print(f"  Sequence: {' -> '.join(pattern)}")
        audio = generate_sequence(pattern, duration_each=0.25)
        sd.play(audio, samplerate=SR, device=2, blocking=True)
        time.sleep(0.3)
    
    print("\nDone!")
