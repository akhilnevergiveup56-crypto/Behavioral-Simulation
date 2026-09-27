import pytest

from src.data.jsonl_loader import load_jsonl
from src.data.validator import (
    validate_dataset,
    validate_example,
)


def test_real_dataset_is_valid():

    examples = load_jsonl(
        "data/raw/behavioral_data.jsonl"
    )

    assert validate_dataset(examples)


def test_invalid_emotion():

    example = {
        "id": "test",

        "emotions": {
            "anger": 120,
        },
    }

    with pytest.raises(ValueError):
        validate_example(example)