# External Research Data Policy

Cora-Infans is a personal, non-commercial research project.

## Permitted sources

External corpora may be used when their terms permit non-commercial research,
including attribution/share-alike licenses such as CC BY-NC-SA. Each evidence
artifact must record the source, version or DOI, license, acquisition method,
and hashes of every selected file.

## Repository boundary

- Third-party audio is stored outside the Git repository.
- Audio files are never committed or redistributed through Cora.
- The repository may retain manifests, filenames, cryptographic hashes,
  protocols, aggregate features, metrics, and code needed to reproduce a run.
- Temporary corpus subsets may be deleted after their hashes and results are
  recorded.

## Evaluation discipline

- Labels may score observations but may never generate evaluated audio input.
- Development speakers and held-out speakers or fieldsites must be disjoint.
- Confirmatory thresholds and acceptance rules are written before held-out
  audio is fetched or inspected.
- A failed criterion is retained as evidence; it is not silently retuned on the
  same held-out participants.
- Results inherit the source license's use restrictions and are not evidence
  that the underlying audio can be used in a commercial product.

## Private recordings

Personal or participant-identifiable recordings require an explicit collection
and retention decision beyond this policy. Cora does not treat access to a local
microphone as consent to retain or publish speech.
