
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


PROJECT_EMOTIONS = [
    "anger",
    "anxiety",
    "fear",
    "sadness",
    "joy",
    "empathy",
    "patience",
    "frustration",
]


class GoEmotionsDataset(Dataset):

    def __init__(self, data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if isinstance(data, (str, Path)):
            self.records = self._load_records(data)
        else:
            self.records = list(data)

    def _load_records(self, path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        records = []

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))

        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        # Convert text into tokens
        encoded = self.tokenizer(
            record["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Original 28 GoEmotions labels
        emotion_vector = record["emotion_vector"]

        if len(emotion_vector) != 28:
            raise ValueError(
                "Expected exactly 28 GoEmotions labels"
            )

        # Project-specific emotion scores
        project_scores = record.get(
            "project_emotions", {}
        )

        # Mask indicates which project dimensions
        # actually have mapped supervision.
        project_mask = record.get(
            "project_emotion_mask", {}
        )

        project_targets = [
            float(project_scores.get(emotion, 0.0)) / 100.0
            for emotion in PROJECT_EMOTIONS
        ]

        project_masks = [
            float(project_mask.get(emotion, 0.0))
            for emotion in PROJECT_EMOTIONS
        ]

        return {
            "input_ids": encoded["input_ids"].squeeze(0),

            "attention_mask": encoded[
                "attention_mask"
            ].squeeze(0),

            "goemotions_targets": torch.tensor(
                emotion_vector,
                dtype=torch.float32
            ),

            "project_emotion_targets": torch.tensor(
                project_targets,
                dtype=torch.float32
            ),

            "project_emotion_mask": torch.tensor(
                project_masks,
                dtype=torch.float32
            ),
        }
