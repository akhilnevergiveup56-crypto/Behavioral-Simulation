
import json

import pytest
import torch

from src.data.goemotions_dataset import GoEmotionsDataset


class FakeTokenizer:
    """A small tokenizer substitute for testing."""

    def __call__(
        self,
        text,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


@pytest.fixture
def sample_record():
    emotion_vector = [0] * 28
    emotion_vector[19] = 1  # nervousness

    return {
        "text": "I feel nervous and worried today.",
        "emotion_vector": emotion_vector,
        "project_emotions": {
            "anxiety": 100.0,
            "fear": 40.0,
        },
        "project_emotion_mask": {
            "anger": 0.0,
            "anxiety": 1.0,
            "fear": 1.0,
            "sadness": 0.0,
            "joy": 0.0,
            "empathy": 0.0,
            "patience": 0.0,
            "frustration": 0.0,
        },
    }


def test_dataset_from_records(sample_record):
    dataset = GoEmotionsDataset(
        [sample_record],
        FakeTokenizer()
    )

    assert len(dataset) == 1

    item = dataset[0]

    assert item["input_ids"].shape == (128,)
    assert item["attention_mask"].shape == (128,)
    assert item["goemotions_targets"].shape == (28,)
    assert item["project_emotion_targets"].shape == (8,)
    assert item["project_emotion_mask"].shape == (8,)

    assert item["goemotions_targets"][19] == 1.0
    assert item["project_emotion_targets"][1] == 1.0
    assert item["project_emotion_mask"][1] == 1.0


def test_dataset_from_jsonl(tmp_path, sample_record):
    file_path = tmp_path / "sample.jsonl"

    with file_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(sample_record) + "\n")

    dataset = GoEmotionsDataset(
        file_path,
        FakeTokenizer()
    )

    assert len(dataset) == 1
    assert dataset[0]["goemotions_targets"].shape == (28,)


def test_invalid_emotion_vector(sample_record):
    sample_record["emotion_vector"] = [0, 1]

    dataset = GoEmotionsDataset(
        [sample_record],
        FakeTokenizer()
    )

    with pytest.raises(ValueError):
        dataset[0]
