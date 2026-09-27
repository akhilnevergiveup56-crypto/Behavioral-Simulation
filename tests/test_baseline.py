from src.models.baseline import RuleBasedBaseline


def test_prediction_scores_sum_to_one():
    profile = {
        "anger": 80, "anxiety": 50, "fear": 40, "sadness": 30,
        "joy": 50, "empathy": 70, "patience": 20, "confidence": 40,
        "impulsiveness": 80, "assertiveness": 40, "conflict_avoidance": 60,
        "risk_tolerance": 50, "self_control": 30, "need_for_approval": 50,
        "trust_tendency": 50, "adaptability": 50,
    }
    result = RuleBasedBaseline().predict(profile, "My friend betrayed me and then blamed me.")
    assert abs(sum(result.scores.values()) - 1.0) < 1e-6
