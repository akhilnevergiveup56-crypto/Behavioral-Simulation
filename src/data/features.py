import torch


EMOTION_FEATURES = [
    "anger",
    "anxiety",
    "fear",
    "sadness",
    "joy",
    "empathy",
    "patience",
    "frustration",
]

BEHAVIORAL_FEATURES = [
    "impulsiveness",
    "assertiveness",
    "conflict_avoidance",
    "risk_tolerance",
    "self_control",
    "need_for_approval",
    "trust_tendency",
    "adaptability",
]

COGNITIVE_FEATURES = [
    "deliberative_thinking",
    "cognitive_flexibility",
    "consequence_awareness",
]

CONTEXT_FEATURES = [
    "family_environment",
    "social_support",
    "trust_history",
    "conflict_exposure",
    "relationship_history",
    "authority_experience",
    "rejection_experience",
    "environmental_stress",
]

TARGET_FEATURES = [
    "direct_confrontation",
    "private_discussion",
    "withdrawal",
    "cooperative_resolution",
    "delayed_response",
    "support_seeking",
]


def encode_features(data: dict, feature_names: list[str]) -> torch.Tensor:
    values = [data[name] / 100.0 for name in feature_names]
    return torch.tensor(values, dtype=torch.float32)


def encode_targets(data: dict) -> torch.Tensor:
    values = [data[name] for name in TARGET_FEATURES]
    return torch.tensor(values, dtype=torch.float32)