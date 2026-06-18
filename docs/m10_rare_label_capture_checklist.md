# M10 rare-label capture checklist

Purpose: build a better targeted pass for the rare consonant labels that are currently under-sampled in eval.

Scope:
- labels: `b,d,g,th,v,z,l,r,w,y`
- use this only as a capture guide for your own voice
- do not use Wikipedia audio directly as training data

Why not direct Wikipedia audio:
- the wiki samples are clean reference speech, not your Mac mini speaker→mic domain
- different speaker, room, and mic chain
- useful for pronunciation reference, not as a substitute for your own captures

## Recommended capture command

Dry-run first:
```bash
./.venv/bin/python scripts/m10_record_human.py \
  --speaker juicebox \
  --labels b,d,g,th,v,z,l,r,w,y \
  --reps 1 \
  --shuffle \
  --dry-run
```

Then capture:
```bash
./.venv/bin/python scripts/m10_record_human.py \
  --speaker juicebox \
  --labels b,d,g,th,v,z,l,r,w,y \
  --reps 2 \
  --shuffle
```

If you want a clean readout afterward, train from scratch:
```bash
./.venv/bin/python scripts/m10_train_v3.py --resume-from none
```

If you want a faster warm-start instead, use the current checkpoint:
```bash
./.venv/bin/python scripts/m10_train_v3.py --resume-from data/m10_v3.ckpt.json
```

## Refined prompt set

The targeted prompts now use a balanced front/back vowel spread plus one VCV context for each label.

| Label | IPA | Articulation cue | Prompts |
|---|---|---|---|
| b | /b/ | voiced bilabial stop; close lips, release cleanly into the vowel | `ba`, `bi`, `bu`, `be`, `bo`, `aba` |
| d | /d/ | voiced alveolar stop; tongue tip to ridge, quick release | `da`, `di`, `du`, `de`, `do`, `ada` |
| g | /g/ | voiced velar stop; back of tongue to soft palate | `ga`, `gi`, `gu`, `ge`, `go`, `aga` |
| th | /ð/ | voiced dental fricative; tongue between teeth, continuous airflow | `tha`, `thi`, `thu`, `the`, `tho`, `atha` |
| v | /v/ | voiced labiodental fricative; upper teeth on lower lip, steady buzz | `va`, `vi`, `vu`, `ve`, `vo`, `ava` |
| z | /z/ | voiced alveolar fricative; tongue near ridge, steady buzz | `za`, `zi`, `zu`, `ze`, `zo`, `aza` |
| l | /l/ | alveolar lateral approximant; tongue tip up, air around sides | `la`, `li`, `lu`, `le`, `lo`, `ala` |
| r | /r/ | alveolar approximant; keep it stable and non-trilled | `ra`, `ri`, `ru`, `re`, `ro`, `ara` |
| w | /w/ | labiovelar approximant; rounded lips, back of tongue raised | `wa`, `wi`, `wu`, `we`, `wo`, `awa` |
| y | /j/ | palatal approximant; tongue high/front, like the start of “yes” | `ya`, `yi`, `yu`, `ye`, `yo`, `aya` |

## Capture discipline

- speak naturally, not whispered
- keep the consonant onset crisp
- hold the vowel steady once the consonant releases
- don't over-articulate or exaggerate the sound
- if a capture is clipped or very quiet, repeat that token in a second pass
- if you see `LOW` or `QUIET` in the live readout, don't panic; just make sure the next attempt is louder/cleaner

## What to look for after capture

- each rare label should have at least 10-15 eval examples
- macro should become more meaningful after the split is balanced
- hard-label avg should stay stable or improve
- if rare labels still fail after better coverage, then revisit balancing or architecture

## Notes

- The wiki is useful for learning the IPA target.
- The actual training data should still be your own speaker→mic captures.
- This pass is about coverage, not changing the model.
