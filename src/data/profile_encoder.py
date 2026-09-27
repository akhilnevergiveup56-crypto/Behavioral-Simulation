import torch

from src.data.features import (
    EMOTION_FEATURES,
    BEHAVIORAL_FEATURES,
    COGNITIVE_FEATURES,
    CONTEXT_FEATURES,
    encode_features,
)


def encode_profile(profile: dict) -> torch.Tensor:
    emotions = encode_features(
        profile["emotions"],
        EMOTION_FEATURES
    )

    behavioral = encode_features(
        profile["behavioral"],
        BEHAVIORAL_FEATURES
    )

    cognitive = encode_features(
        profile["cognitive"],
        COGNITIVE_FEATURES
    )

    context = encode_features(
        profile["context"],
        CONTEXT_FEATURES
    )

    return torch.cat([
        emotions,
        behavioral,
        cognitive,
        context,
    ])