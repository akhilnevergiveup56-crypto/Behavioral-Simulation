
import json
import math
from pathlib import Path

import torch


def compute_goemotions_pos_weights(
    jsonl_path,
    max_weight=10.0,
):
    """
    Calculate positive-class weights for all 28 GoEmotions labels.

    Uses only the training dataset.

    Formula:
        weight = sqrt(negative_examples / positive_examples)

    We cap weights to prevent extremely rare labels from
    receiving excessively large weights.
    """

    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {path}"
        )

    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")

    num_labels = 28

    positive_counts = [0] * num_labels
    total_examples = 0

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

            emotion_vector = record.get("emotion_vector")

            if not isinstance(emotion_vector, list):
                raise ValueError(
                    f"Missing or invalid emotion_vector "
                    f"on line {line_number}"
                )

            if len(emotion_vector) != num_labels:
                raise ValueError(
                    f"Expected {num_labels} labels on "
                    f"line {line_number}"
                )

            for index, value in enumerate(emotion_vector):

                if value not in (0, 1, 0.0, 1.0):
                    raise ValueError(
                        f"Emotion labels must be binary. "
                        f"Invalid value on line {line_number}"
                    )

                if value == 1 or value == 1.0:
                    positive_counts[index] += 1

            total_examples += 1

    if total_examples == 0:
        raise ValueError("Training dataset is empty.")

    weights = []

    for index, positive_count in enumerate(positive_counts):

        if positive_count == 0:
            raise ValueError(
                f"Emotion label {index} has no positive examples."
            )

        negative_count = total_examples - positive_count

        weight = math.sqrt(
            negative_count / positive_count
        )

        weight = min(weight, max_weight)

        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )
