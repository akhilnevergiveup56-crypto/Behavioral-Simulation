from typing import Any
from src.schemas.input import SimulationRequest
from src.models.baseline import RuleBasedBaseline


baseline = RuleBasedBaseline()


def run_simulation(request: SimulationRequest) -> dict[str, Any]:
    profile = request.profile.model_dump()
    prediction = baseline.predict(profile, request.scenario)

    top_behavior = max(prediction.scores, key=prediction.scores.get)
    readable = {
        "direct_confrontation": "direct confrontation",
        "private_discussion": "private discussion",
        "withdrawal": "withdrawal",
        "cooperative_resolution": "cooperative resolution",
    }[top_behavior]

    return {
        "mode": "baseline",
        "predicted_behavior": readable,
        "behavior_scores": prediction.scores,
        "explanation_factors": prediction.explanation_factors,
        "note": "This is an engineering baseline for pipeline validation, not a validated psychological predictor.",
    }
