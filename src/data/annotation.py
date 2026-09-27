BEHAVIORAL_TARGETS = [
    "direct_confrontation",
    "private_discussion",
    "withdrawal",
    "cooperative_resolution",
    "delayed_response",
    "support_seeking",
]


def validate_annotation(annotation: dict) -> bool:

    required_fields = [
        "example_id",
        "annotator_id",
        "behavioral_targets",
        "rationale",
        "response",
        "confidence",
    ]

    for field in required_fields:
        if field not in annotation:
            raise ValueError(
                f"Missing annotation field: {field}"
            )

    targets = annotation["behavioral_targets"]

    for target in BEHAVIORAL_TARGETS:

        if target not in targets:
            raise ValueError(
                f"Missing behavioral target: {target}"
            )

        value = targets[target]

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{target} must be numeric."
            )

        if not 0 <= value <= 100:
            raise ValueError(
                f"{target} must be between 0 and 100."
            )

    confidence = annotation["confidence"]

    if not isinstance(confidence, int):
        raise ValueError(
            "confidence must be an integer."
        )

    if not 1 <= confidence <= 5:
        raise ValueError(
            "confidence must be between 1 and 5."
        )

    if not isinstance(annotation["rationale"], str):
        raise ValueError(
            "rationale must be a string."
        )

    if not annotation["rationale"].strip():
        raise ValueError(
            "rationale cannot be empty."
        )

    if not isinstance(annotation["response"], str):
        raise ValueError(
            "response must be a string."
        )

    if not annotation["response"].strip():
        raise ValueError(
            "response cannot be empty."
        )

    return True