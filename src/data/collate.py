import torch

from src.data.features import (
    EMOTION_FEATURES,
    BEHAVIORAL_FEATURES,
    COGNITIVE_FEATURES,
    CONTEXT_FEATURES,
    encode_features,
    encode_targets,
)


def collate_examples(batch: list[dict]):

    profile_tensors = []

    target_tensors = []
    scenarios = []

    for example in batch:

        emotions = encode_features(
            example["emotions"],
            EMOTION_FEATURES,
        )

        behavioral = encode_features(
            example["behavioral"],
            BEHAVIORAL_FEATURES,
        )

        cognitive = encode_features(
            example["cognitive"],
            COGNITIVE_FEATURES,
        )

        context = encode_features(
            example["context"],
            CONTEXT_FEATURES,
        )

        profile = torch.cat([
            emotions,
            behavioral,
            cognitive,
            context,
        ])

        profile_tensors.append(profile)

        target_tensors.append(
            encode_targets(example["targets"])
        )

        scenarios.append(example["scenario"])

    profiles = torch.stack(profile_tensors)
    targets = torch.stack(target_tensors)

    return {
        "profile": profiles,
        "scenario": scenarios,
        "targets": targets,
    }