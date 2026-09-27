from collections import Counter
from statistics import mean

from src.data.jsonl_loader import load_jsonl


DATA_PATH = "data/raw/behavioral_data.jsonl"

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


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def summarize_100_scale(examples, section_name, features):
    print_section(section_name)

    for feature in features:
        values = [
            example[section_name.lower()][feature]
            for example in examples
        ]

        print(
            f"{feature:25} "
            f"min={min(values):5} "
            f"max={max(values):5} "
            f"mean={mean(values):6.2f}"
        )


def summarize_targets(examples):
    print_section("BEHAVIORAL TARGETS")

    for target in TARGETS:
        values = [
            example["behavioral_targets"][target]
            for example in examples
        ]

        print(
            f"{target:25} "
            f"min={min(values):.2f} "
            f"max={max(values):.2f} "
            f"mean={mean(values):.2f}"
        )


def check_ranges(examples):
    print_section("RANGE CHECKS")

    problems = []

    for i, example in enumerate(examples):

        for feature in EMOTIONS:
            value = example["emotions"][feature]

            if not 0 <= value <= 100:
                problems.append(
                    f"Example {i}: emotion '{feature}' = {value}"
                )

        for feature in BEHAVIORAL:
            value = example["behavioral"][feature]

            if not 0 <= value <= 100:
                problems.append(
                    f"Example {i}: behavioral '{feature}' = {value}"
                )

        for feature in COGNITIVE:
            value = example["cognitive"][feature]

            if not 0 <= value <= 100:
                problems.append(
                    f"Example {i}: cognitive '{feature}' = {value}"
                )

        for feature in CONTEXT:
            value = example["context"][feature]

            if not 0 <= value <= 100:
                problems.append(
                    f"Example {i}: context '{feature}' = {value}"
                )

        for target in TARGETS:
            value = example["behavioral_targets"][target]

            if not 0 <= value <= 1:
                problems.append(
                    f"Example {i}: target '{target}' = {value}"
                )

    if problems:
        print("PROBLEMS FOUND:")
        for problem in problems:
            print("-", problem)
    else:
        print("All values are inside their expected ranges.")


def check_duplicates(examples):
    print_section("DUPLICATE CHECK")

    scenarios = [
        example["scenario"].strip().lower()
        for example in examples
    ]

    counts = Counter(scenarios)

    duplicates = [
        (scenario, count)
        for scenario, count in counts.items()
        if count > 1
    ]

    if duplicates:
        print("Duplicate scenarios found:")

        for scenario, count in duplicates:
            print(f"\nCount: {count}")
            print(scenario[:200])

    else:
        print("No exact duplicate scenarios found.")


def main():
    examples = load_jsonl(DATA_PATH)

    print_section("DATASET OVERVIEW")

    print(f"Number of examples: {len(examples)}")

    summarize_100_scale(
        examples,
        "emotions",
        EMOTIONS,
    )

    summarize_100_scale(
        examples,
        "behavioral",
        BEHAVIORAL,
    )

    summarize_100_scale(
        examples,
        "cognitive",
        COGNITIVE,
    )

    summarize_100_scale(
        examples,
        "context",
        CONTEXT,
    )

    summarize_targets(examples)

    check_ranges(examples)

    check_duplicates(examples)


if __name__ == "__main__":
    main()