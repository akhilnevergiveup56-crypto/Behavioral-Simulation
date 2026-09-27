from src.data.dataset import BehavioralDataset


sample = {
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

    "scenario": "Your friend repeatedly cancels plans at the last minute.",

    "behavioral_targets": {
        "direct_confrontation": 0.72,
        "private_discussion": 0.81,
        "withdrawal": 0.43,
        "cooperative_resolution": 0.64,
        "delayed_response": 0.38,
        "support_seeking": 0.21,
    },
}


def test_dataset():
    dataset = BehavioralDataset([sample])

    assert len(dataset) == 1

    item = dataset[0]

    assert item["emotions"]["anger"] == 82
    assert item["scenario"] == sample["scenario"]
    assert item["targets"]["private_discussion"] == 0.81