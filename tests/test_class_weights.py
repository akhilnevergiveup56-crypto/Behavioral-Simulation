
import json
import math

import pytest
import torch

from src.training.class_weights import (
    compute_goemotions_pos_weights,
)


def write_records(path, records):
    """Write test records in JSONL format."""

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def test_compute_class_weights(tmp_path):
    """Check that weights follow the expected formula."""

    records = []

    for row in range(4):
        vector = [0] * 28

        # Label 0 appears once.
        if row == 0:
            vector[0] = 1

        # Label 1 appears twice.
        if row < 2:
            vector[1] = 1

        # Remaining labels appear twice.
        if row < 2:
            for index in range(2, 28):
                vector[index] = 1

        records.append({
            "emotion_vector": vector
        })

    data_path = tmp_path / "train.jsonl"
    write_records(data_path, records)

    weights = compute_goemotions_pos_weights(data_path)

    assert weights.shape == (28,)
    assert weights.dtype == torch.float32

    assert weights[0].item() == pytest.approx(math.sqrt(3))
    assert weights[1].item() == pytest.approx(1.0)
    assert weights[2].item() == pytest.approx(1.0)


def test_class_weights_respect_maximum(tmp_path):
    """Ensure rare labels cannot exceed the weight cap."""

    records = []

    for row in range(102):
        vector = [0] * 28

        # Label 0 appears only once.
        if row == 0:
            vector[0] = 1

        # Other labels appear twice.
        if row < 2:
            for index in range(1, 28):
                vector[index] = 1

        records.append({
            "emotion_vector": vector
        })

    data_path = tmp_path / "train.jsonl"
    write_records(data_path, records)

    weights = compute_goemotions_pos_weights(
        data_path,
        max_weight=5.0,
    )

    assert weights[0].item() == pytest.approx(5.0)
    assert torch.all(weights <= 5.0)


def test_missing_emotion_label_raises_error(tmp_path):
    """Every emotion must have at least one positive example."""

    vector = [1] * 27 + [0]

    data_path = tmp_path / "train.jsonl"

    write_records(
        data_path,
        [{"emotion_vector": vector}],
    )

    with pytest.raises(ValueError, match="no positive examples"):
        compute_goemotions_pos_weights(data_path)


def test_invalid_vector_length_raises_error(tmp_path):
    """Reject vectors that do not contain 28 labels."""

    data_path = tmp_path / "train.jsonl"

    write_records(
        data_path,
        [{"emotion_vector": [0, 1, 0]}],
    )

    with pytest.raises(ValueError, match="Expected 28 labels"):
        compute_goemotions_pos_weights(data_path)
