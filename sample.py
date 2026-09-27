example = {
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

    "scenario": """
    You have been friends with Alex for five years. Recently, Alex has
    started cancelling plans at the last minute. Today you rearranged
    your schedule for an important event together. Two hours before
    the event, Alex cancels again and says something more interesting
    came up. When you explain that you are upset because this has
    happened several times, Alex tells you that you are taking things
    too seriously and should just relax.
    """,

    "behavioral_targets": {
        "direct_confrontation": 0.72,
        "private_discussion": 0.81,
        "withdrawal": 0.43,
        "cooperative_resolution": 0.64,
        "delayed_response": 0.38,
        "support_seeking": 0.21,
    },
}


print(example)
print ("lauda")
