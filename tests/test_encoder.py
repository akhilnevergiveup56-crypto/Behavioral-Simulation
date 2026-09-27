import torch

from src.data.profile_encoder import encode_profile


def test_profile_encoding():

    profile = {
        "emotions": {
            "anger": 82,
            "anxiety": 65,
            "fear": 45,
            "sadness": 55,
            "joy": 40,
            "empathy": 72,
            "patience": 20,
            "frustration": 85,
        },

        "behavioral": {
            "impulsiveness": 80,
            "assertiveness": 35,
            "conflict_avoidance": 65,
            "risk_tolerance": 40,
            "self_control": 30,
            "need_for_approval": 75,
            "trust_tendency": 40,
            "adaptability": 55,
        },

        "cognitive": {
            "deliberative_thinking": 35,
            "cognitive_flexibility": 50,
            "consequence_awareness": 70,
        },

        "context": {
            "family_environment": 40,
            "social_support": 35,
            "trust_history": 70,
            "conflict_exposure": 80,
            "relationship_history": 60,
            "authority_experience": 40,
            "rejection_experience": 65,
            "environmental_stress": 50,
        },
    }

    tensor = encode_profile(profile)

    assert tensor.shape == (27,)

    assert torch.isclose(
        tensor[0],
        torch.tensor(0.82)
    )

    # First behavioral value starts after 8 emotions
    assert torch.isclose(
        tensor[8],
        torch.tensor(0.80)
    )

    # First cognitive value starts after 8 + 8 features
    assert torch.isclose(
        tensor[16],
        torch.tensor(0.35)
    )

    # First context value starts after 8 + 8 + 3 features
    assert torch.isclose(
        tensor[19],
        torch.tensor(0.40)
    )