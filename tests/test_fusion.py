import torch

from src.data.profile_encoder import encode_profile
from src.data.scenario_encoder import ScenarioEncoder
from src.data.fusion import combine_profile_and_scenario


def test_profile_scenario_fusion():

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

    profile_tensor = encode_profile(profile)

    scenarios = [
        "My friend cancelled our plans again.",
        "My manager criticized my work in front of everyone.",
    ]

    encoder = ScenarioEncoder(max_features=20)
    encoder.fit(scenarios)

    scenario_matrix = encoder.encode([scenarios[0]])

    combined = combine_profile_and_scenario(
        profile_tensor,
        scenario_matrix
    )

    assert combined.shape[0] == 1
    assert combined.shape[1] == 27 + scenario_matrix.shape[1]

    assert torch.isclose(
        combined[0, 0],
        torch.tensor(0.82)
    )