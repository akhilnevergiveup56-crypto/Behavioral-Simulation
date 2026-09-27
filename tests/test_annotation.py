import pytest

from src.data.annotation import validate_annotation


def test_valid_annotation():

    annotation = {
        "example_id": "pilot_001",
        "annotator_id": "ann_001",

        "behavioral_targets": {
            "direct_confrontation": 72,
            "private_discussion": 81,
            "withdrawal": 43,
            "cooperative_resolution": 64,
            "delayed_response": 38,
            "support_seeking": 21,
        },

        "rationale": (
            "High anger and low patience increase the likelihood "
            "of addressing the issue."
        ),

        "response": (
            "The person would most likely address the issue privately."
        ),

        "confidence": 4,
    }

    assert validate_annotation(annotation)


def test_invalid_target():

    annotation = {
        "example_id": "pilot_001",
        "annotator_id": "ann_001",

        "behavioral_targets": {
            "direct_confrontation": 120,
            "private_discussion": 81,
            "withdrawal": 43,
            "cooperative_resolution": 64,
            "delayed_response": 38,
            "support_seeking": 21,
        },

        "rationale": "Test rationale.",
        "response": "Test response.",
        "confidence": 4,
    }

    with pytest.raises(ValueError):
        validate_annotation(annotation)


def test_invalid_confidence():

    annotation = {
        "example_id": "pilot_001",
        "annotator_id": "ann_001",

        "behavioral_targets": {
            "direct_confrontation": 72,
            "private_discussion": 81,
            "withdrawal": 43,
            "cooperative_resolution": 64,
            "delayed_response": 38,
            "support_seeking": 21,
        },

        "rationale": "Test rationale.",
        "response": "Test response.",
        "confidence": 8,
    }

    with pytest.raises(ValueError):
        validate_annotation(annotation)