import torch
from src.data.features import (
    EMOTION_FEATURES,
    BEHAVIORAL_FEATURES,
    COGNITIVE_FEATURES,
    CONTEXT_FEATURES,
    encode_features,
    encode_targets,
)


def test_feature_encoding():

    emotions = {
        "anger": 82,
        "anxiety": 65,
        "fear": 45,
        "sadness": 55,
        "joy": 40,
        "empathy": 72,
        "patience": 20,
        "frustration": 85,
    }

    tensor = encode_features(emotions, EMOTION_FEATURES)

    assert tensor.shape == (8,)
    assert torch.isclose(tensor[0], torch.tensor(0.82))
    assert torch.isclose(tensor[1], torch.tensor(0.65))
    


def test_target_encoding():

    targets = {
        "direct_confrontation": 0.72,
        "private_discussion": 0.81,
        "withdrawal": 0.43,
        "cooperative_resolution": 0.64,
        "delayed_response": 0.38,
        "support_seeking": 0.21,
    }

    tensor = encode_targets(targets)

    assert tensor.shape == (6,)
    assert torch.isclose(tensor[0], torch.tensor(0.72))