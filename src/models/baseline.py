from dataclasses import dataclass
from typing import Dict


@dataclass
class BaselinePrediction:
    scores: Dict[str, float]
    explanation_factors: Dict[str, str]


class RuleBasedBaseline:
    """A temporary baseline used only to validate the API and feature design.

    This is deliberately not presented as a psychological model.
    It will be replaced by a trained ML model once the dataset exists.
    """

    def predict(self, profile: Dict[str, float], scenario: str) -> BaselinePrediction:
        text = scenario.lower()
        conflict_words = ["argument", "fight", "insult", "betray", "stole", "unfair", "criticize", "criticism", "blame", "cancel"]
        conflict_signal = min(1.0, sum(w in text for w in conflict_words) / 3.0)

        anger = profile["anger"] / 100
        impulsiveness = profile["impulsiveness"] / 100
        patience = profile["patience"] / 100
        assertiveness = profile["assertiveness"] / 100
        avoidance = profile["conflict_avoidance"] / 100
        empathy = profile["empathy"] / 100
        self_control = profile["self_control"] / 100
        anxiety = profile["anxiety"] / 100

        confrontation = (
            0.35 * anger +
            0.20 * impulsiveness +
            0.15 * assertiveness +
            0.15 * conflict_signal +
            0.10 * (1 - patience) +
            0.05 * (1 - avoidance)
        )
        private_discussion = (
            0.20 * anger +
            0.20 * empathy +
            0.20 * assertiveness +
            0.15 * self_control +
            0.15 * conflict_signal +
            0.10 * patience
        )
        withdrawal = (
            0.25 * avoidance +
            0.20 * anxiety +
            0.15 * (1 - assertiveness) +
            0.15 * (1 - confidence(profile)) +
            0.15 * (1 - self_control) +
            0.10 * (1 - patience)
        )
        cooperation = (
            0.30 * empathy +
            0.20 * patience +
            0.20 * self_control +
            0.15 * confidence(profile) +
            0.15 * (1 - anger)
        )

        raw = {
            "direct_confrontation": confrontation,
            "private_discussion": private_discussion,
            "withdrawal": withdrawal,
            "cooperative_resolution": cooperation,
        }
        total = sum(raw.values()) or 1.0
        scores = {k: round(v / total, 4) for k, v in raw.items()}

        factors = {
            "anger": "High anger increases pressure to address the situation.",
            "impulsiveness": "Higher impulsiveness makes an immediate reaction more likely.",
            "patience": "Higher patience tends to reduce immediate escalation.",
            "empathy": "Higher empathy can moderate an aggressive reaction.",
            "scenario": "The baseline detects broad conflict-related language in the scenario.",
        }
        return BaselinePrediction(scores=scores, explanation_factors=factors)


def confidence(profile: Dict[str, float]) -> float:
    return profile["confidence"] / 100
