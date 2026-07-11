import json
import re
from pathlib import Path


def test_consolidated_status_links_exist_and_preserve_pending_decisions() -> None:
    root = Path(__file__).resolve().parents[1]
    status_path = root / "docs/PROJECT_STATUS_2026-07-11.md"
    text = status_path.read_text()

    for relative in re.findall(r"\((evidence/[^)]+\.json)\)", text):
        assert (status_path.parent / relative).is_file(), relative

    predictive = json.loads(
        (root / "docs/evidence/2026-07-11-predictive-register-dynamics-result.json").read_text()
    )
    cohort = json.loads(
        (root / "docs/evidence/2026-07-11-cohort-adaptive-register-result.json").read_text()
    )
    natural_speech = json.loads(
        (
            root
            / "docs/evidence/2026-07-11-stage6-natural-speech-hardware-volume-diagnostic-result.json"
        ).read_text()
    )
    cmu = json.loads(
        (root / "docs/evidence/2026-07-11-cmu-arctic-development-result.json").read_text()
    )
    provenance = json.loads(
        (
            root
            / "docs/evidence/2026-07-11-stage2-reserved-fieldsite-provenance-correction.json"
        ).read_text()
    )

    assert predictive["development_candidate_ready"] is False
    assert predictive["confirmation_audio_or_metadata_accessed"] is False
    assert cohort["development_candidate_ready"] is False
    assert cohort["confirmation_audio_or_metadata_accessed"] is False
    assert natural_speech["overall_criteria_passed"] is True
    assert natural_speech["aggregate"]["clipped_sample_count"] == 0
    assert cmu["confirmation_audio_or_labels_accessed"] is False
    assert len(provenance["wellington_prior_access"]["clips"]) == 8
    assert provenance["repository_wide_filename_audit"][
        "san_diego_selected_clip_ids_found"
    ] is False
    assert provenance["repository_wide_filename_audit"][
        "toronto_selected_clip_ids_found"
    ] is False
    assert "Stage 2 disposition pending" in text
    assert "H4 disposition pending" in text
