# Cora-Infans project direction — 2026-07-15

## Primary product direction

The project owner selected **bounded project-owner vocal imitation** as Cora-Infans’ primary product and engineering direction.

This direction is based on the retained V4 project-owner result, which passed its frozen development and held-confirmation gates for `/a/`, `/i/`, and `/u/` on 2026-07-13. V4 remains immutable evidence: it is not a general speech model, a reusable training set, or permission to alter the previously frozen result.

The product claim is deliberately narrow: an authorized owner target can be associated with and recalled from a fixed finite repertoire under a declared capture setup, with associations frozen before held evaluation and with reproducible provenance artifacts.

The governing product boundary is [BOUNDED_VOCAL_IMITATION_PRODUCT_SPEC.md](BOUNDED_VOCAL_IMITATION_PRODUCT_SPEC.md).

## What does not change

- Stage 2 broad cross-cultural register detection remains falsified within tested scope and inactive.
- H4 continuous perception-production closure remains falsified for the tested bounded-local representation families and inactive.
- The V4 result, previous failed candidates, their thresholds, and their artifact hashes remain retained historical evidence.
- Protected Stage 2 fieldsites, VocalSet, and CMU Arctic confirmation/reserve material remain inaccessible.
- Future owner recordings require a separately approved, versioned product-session protocol and explicit recording authorization.

## Next engineering sequence

1. The versioned product-session report/manifest contract is implemented and
   offline-only. It requires six named artifacts, separate imitation/self-model
   score families, association freeze before evaluation, and explicit
   authorization plus operating-point metadata for any future real session.
2. The deterministic `offline-replay-v1` scorer is implemented. It reads only
   pre-existing WAV artifacts, requires an explicit association-freeze
   attestation, emits unaggregated component metrics in both score families,
   and writes a report plus sibling integrity manifest.
3. The replay-only `product-session` hero source is implemented. It validates a
   completed report/manifest pair before exposing audio, preserves evidence
   class, renders the two score families separately, and displays provenance
   without claiming live behavior.
4. Draft and approve a new real-session protocol before any microphone or speaker operation.

## Verification

- Product-session contract, offline scorer, and replay-presentation tests cover
  hash verification, score-family separation, explicit evidence labeling,
  replay-only behavior, and CLI validation.
- Full local Python 3.13 suite after the presentation addition: `497 passed, 9 skipped`.
- The scorer and presentation path are statically verified as offline-only: they
  import no live or audio-device module; they read pre-existing WAV artifacts
  through the offline decoder only.

No new live recording or H4/Stage 2 experiment is authorized by this strategic direction alone.
