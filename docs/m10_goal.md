# M10 — Consonant Learning Goal

Status: planned after M9 completion.

## Goal
Teach M9 to recognize and imitate consonants as first-class speech events, not just vowel-adjacent noise.

## What M10 should learn
- consonant presence vs silence
- stop consonants: p, t, k
- fricatives: s, f, sh
- nasals: m, n
- consonant-to-vowel transitions and boundaries

## First milestone
A live demo that can:
1. wait for a user to speak
2. detect consonant activity
3. separate consonants from vowels with a gate or classifier
4. imitate a short consonant-bearing phrase

## Success criteria
- consonants are detected reliably in live mic input
- consonant frames no longer collapse into vowels
- the existing vowel demo still works as a regression baseline
- the new demo produces listenable, repeatable consonant-imitation artifacts

## Non-goals for the first pass
- full phoneme coverage
- perfect accent normalization
- end-to-end speech cloning
- large-scale training before the small consonant loop works
