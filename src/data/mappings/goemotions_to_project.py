
# Our project's 8 emotion dimensions
PROJECT_EMOTIONS = [
    "anger",
    "anxiety",
    "fear",
    "sadness",
    "joy",
    "empathy",
    "patience",
    "frustration",
]

# Mapping from GoEmotions labels to project dimensions.
# Weights are heuristic alignment scores, NOT measured intensity.
GOEMOTIONS_TO_PROJECT = {
    "anger": {"anger": 1.0},
    "annoyance": {"anger": 0.75, "frustration": 1.0},
    "disapproval": {"anger": 0.50, "frustration": 0.50},

    "nervousness": {"anxiety": 1.0, "fear": 0.40},
    "fear": {"fear": 1.0},

    "sadness": {"sadness": 1.0},
    "grief": {"sadness": 1.0},
    "disappointment": {
        "sadness": 0.70,
        "frustration": 0.60,
    },
    "remorse": {"sadness": 0.50},

    "joy": {"joy": 1.0},
    "amusement": {"joy": 0.70},
    "excitement": {"joy": 0.70},
    "gratitude": {"joy": 0.50},
    "love": {"joy": 0.50},
    "relief": {"joy": 0.50},

    "caring": {"empathy": 1.0},
}


def map_goemotions_to_project(source_emotions):
    """
    Convert GoEmotions labels into 8 alignment scores
    on a 0-100 scale.
    """
    scores = {emotion: 0.0 for emotion in PROJECT_EMOTIONS}

    for source_emotion in source_emotions:
        mapped = GOEMOTIONS_TO_PROJECT.get(source_emotion, {})

        for project_emotion, weight in mapped.items():
            scores[project_emotion] = max(
                scores[project_emotion],
                weight * 100.0,
            )

    return scores


def get_project_emotion_mask(source_emotions):
    """
    Return 1 for project dimensions supported by
    a source-label mapping, and 0 otherwise.
    """
    mask = {emotion: 0.0 for emotion in PROJECT_EMOTIONS}

    for source_emotion in source_emotions:
        mapped = GOEMOTIONS_TO_PROJECT.get(source_emotion, {})

        for project_emotion in mapped:
            mask[project_emotion] = 1.0

    return mask
