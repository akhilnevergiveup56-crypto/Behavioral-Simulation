SCENARIO_TAXONOMY = {
    "relationship": {
        "repeated_disrespect",
        "trust_issue",
        "dishonesty",
        "relationship_uncertainty",
        "boundary_conflict",
        "friendship_conflict",
        "emotional_support",
        "financial_conflict",
    },

    "family": {
        "career_disagreement",
        "sibling_conflict",
        "family_expectation",
        "family_mistake",
        "financial_family_conflict",
    },

    "academic": {
        "public_criticism",
        "academic_failure",
        "academic_dishonesty",
        "academic_competition",
        "grading_dispute",
    },

    "workplace": {
        "credit_conflict",
        "teamwork_conflict",
        "manager_feedback",
        "workplace_embarrassment",
        "hiring_fairness",
        "workplace_competition",
    },

    "social": {
        "public_disrespect",
        "social_rejection",
        "social_embarrassment",
        "stranger_conflict",
        "social_pressure",
    },

    "ethical": {
        "moral_dilemma",
        "dishonesty",
        "loyalty_vs_fairness",
        "responsibility_conflict",
    },

    "emergency": {
        "helping_stranger",
        "crisis_response",
        "urgent_decision",
    },

    "personal_decision": {
        "major_life_choice",
        "financial_decision",
        "risk_decision",
        "future_uncertainty",
    },
}


def validate_scenario_category(
    category: str,
    subcategory: str,
) -> bool:

    if category not in SCENARIO_TAXONOMY:
        return False

    return subcategory in SCENARIO_TAXONOMY[category]