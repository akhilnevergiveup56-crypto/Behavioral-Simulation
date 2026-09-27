from src.data.taxonomy import validate_scenario_category


EMOTIONS = [
    "anger",
    "anxiety",
    "fear",
    "sadness",
    "joy",
    "empathy",
    "patience",
    "frustration",
]

BEHAVIORAL = [
    "impulsiveness",
    "assertiveness",
    "conflict_avoidance",
    "risk_tolerance",
    "self_control",
    "need_for_approval",
    "trust_tendency",
    "adaptability",
]

COGNITIVE = [
    "deliberative_thinking",
    "cognitive_flexibility",
    "consequence_awareness",
]

CONTEXT = [
    "family_environment",
    "social_support",
    "trust_history",
    "conflict_exposure",
    "relationship_history",
    "authority_experience",
    "rejection_experience",
    "environmental_stress",
]

TARGETS = [
    "direct_confrontation",
    "private_discussion",
    "withdrawal",
    "cooperative_resolution",
    "delayed_response",
    "support_seeking",
]


def validate_range(value, minimum, maximum, field_name):
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be numeric."
        )

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{field_name} must be between "
            f"{minimum} and {maximum}, got {value}."
        )


def validate_features(
    data,
    feature_names,
    section_name,
):
    if section_name not in data:
        raise ValueError(
            f"Missing section: {section_name}"
        )

    section = data[section_name]

    for feature in feature_names:

        if feature not in section:
            raise ValueError(
                f"Missing feature: "
                f"{section_name}.{feature}"
            )

        validate_range(
            section[feature],
            0,
            100,
            f"{section_name}.{feature}",
        )


def validate_example(example):
    required_fields = [
        "id",
        "emotions",
        "behavioral",
        "cognitive",
        "context_story",
        "context",
        "scenario_category",
        "scenario_subcategory",
        "scenario",
        "behavioral_targets",
        "response",
    ]

    for field in required_fields:
        if field not in example:
            raise ValueError(
                f"Missing required field: {field}"
            )

    if not isinstance(example["scenario"], str):
        raise ValueError(
            "scenario must be a string."
        )

    if not example["scenario"].strip():
        raise ValueError(
            "scenario cannot be empty."
        )

    if not isinstance(example["response"], str):
        raise ValueError(
            "response must be a string."
        )

    if not example["response"].strip():
        raise ValueError(
            "response cannot be empty."
        )

    if not isinstance(example["context_story"], str):
        raise ValueError(
            "context_story must be a string."
        )

    validate_features(
        example,
        EMOTIONS,
        "emotions",
    )

    validate_features(
        example,
        BEHAVIORAL,
        "behavioral",
    )

    validate_features(
        example,
        COGNITIVE,
        "cognitive",
    )

    validate_features(
        example,
        CONTEXT,
        "context",
    )

    if "behavioral_targets" not in example:
        raise ValueError(
            "Missing behavioral_targets"
        )

    for target in TARGETS:

        if target not in example["behavioral_targets"]:
            raise ValueError(
                f"Missing target: {target}"
            )

        validate_range(
            example["behavioral_targets"][target],
            0,
            1,
            f"behavioral_targets.{target}",
        )

    if not validate_scenario_category(
        example["scenario_category"],
        example["scenario_subcategory"],
    ):
        raise ValueError(
            "Invalid scenario category/subcategory: "
            f"{example['scenario_category']} / "
            f"{example['scenario_subcategory']}"
        )

    return True


def validate_dataset(examples):
    ids = set()

    for index, example in enumerate(examples):

        try:
            validate_example(example)
        except ValueError as error:
            raise ValueError(
                f"Example {index} failed validation: {error}"
            ) from error

        example_id = example["id"]

        if example_id in ids:
            raise ValueError(
                f"Duplicate example id: {example_id}"
            )

        ids.add(example_id)

    return True