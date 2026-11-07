
import torch

from src.data.loader import create_goemotions_dataloader


class FakeTokenizer:
    """Simple tokenizer substitute for testing."""

    def __call__(
        self,
        text,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        return {
            "input_ids": torch.ones(
                (1, max_length),
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                (1, max_length),
                dtype=torch.long,
            ),
        }


def create_sample_record():
    emotion_vector = [0.0] * 28
    emotion_vector[19] = 1.0  # nervousness

    return {
        "text": "I feel nervous and worried today.",
        "emotion_vector": emotion_vector,
        "project_emotions": {
            "anger": 0.0,
            "anxiety": 100.0,
            "fear": 40.0,
            "sadness": 0.0,
            "joy": 0.0,
            "empathy": 0.0,
            "patience": 0.0,
            "frustration": 0.0,
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


def test_goemotions_dataloader_batches_data():
    records = [
        create_sample_record(),
        create_sample_record(),
    ]

    dataloader = create_goemotions_dataloader(
        data=records,
        tokenizer=FakeTokenizer(),
        batch_size=2,
        max_length=128,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch["input_ids"].shape == (2, 128)
    assert batch["attention_mask"].shape == (2, 128)
    assert batch["goemotions_targets"].shape == (2, 28)
    assert batch["project_emotion_targets"].shape == (2, 8)
    assert batch["project_emotion_mask"].shape == (2, 8)

    assert batch["goemotions_targets"][0, 19] == 1.0
    assert batch["project_emotion_targets"][0, 1] == 1.0
    assert batch["project_emotion_mask"][0, 1] == 1.0


def test_goemotions_dataloader_returns_correct_batch_count():
    records = [
        create_sample_record()
        for _ in range(5)
    ]

    dataloader = create_goemotions_dataloader(
        data=records,
        tokenizer=FakeTokenizer(),
        batch_size=2,
    )

    batches = list(dataloader)

    assert len(batches) == 3
    assert batches[-1]["input_ids"].shape[0] == 1
