from collections import Counter
from statistics import mean, stdev

import numpy as np

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


def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def values(examples, section_name, feature):
    return [
        example[section_name][feature]
        for example in examples
    ]


def show_category_distribution(examples):
    section("SCENARIO CATEGORY DISTRIBUTION")

    categories = Counter(
        example["scenario_category"]
        for example in examples
    )

    for category, count in sorted(categories.items()):
        percentage = count / len(examples) * 100

        print(
            f"{category:20} "
            f"{count:3} examples "
            f"({percentage:5.1f}%)"
        )


def show_subcategory_distribution(examples):
    section("SUBCATEGORY DISTRIBUTION")

    subcategories = Counter(
        (
            example["scenario_category"],
            example["scenario_subcategory"],
        )
        for example in examples
    )

    for (category, subcategory), count in sorted(
        subcategories.items()
    ):
        print(
            f"{category:20} "
            f"{subcategory:28} "
            f"{count}"
        )


def show_feature_variation(examples):
    section("FEATURE VARIATION")

    groups = {
        "emotions": EMOTIONS,
        "behavioral": BEHAVIORAL,
        "cognitive": COGNITIVE,
        "context": CONTEXT,
    }

    for group_name, features in groups.items():

        print(f"\n[{group_name}]")

        for feature in features:

            feature_values = values(
                examples,
                group_name,
                feature,
            )

            print(
                f"{feature:25} "
                f"mean={mean(feature_values):6.2f} "
                f"std={stdev(feature_values):6.2f}"
            )


def show_target_distribution(examples):
    section("TARGET DISTRIBUTION")

    for target in TARGETS:

        target_values = [
            example["behavioral_targets"][target]
            for example in examples
        ]

        print(
            f"{target:25} "
            f"mean={mean(target_values):.3f} "
            f"std={stdev(target_values):.3f}"
        )


def show_target_correlations(examples):
    section("BEHAVIORAL TARGET CORRELATIONS")

    matrix = np.array([
        [
            example["behavioral_targets"][target]
            for target in TARGETS
        ]
        for example in examples
    ])

    correlation = np.corrcoef(matrix, rowvar=False)

    print("                 " + " ".join(
        f"{target[:8]:>9}"
        for target in TARGETS
    ))

    for i, target in enumerate(TARGETS):

        row = " ".join(
            f"{correlation[i, j]:9.2f}"
            for j in range(len(TARGETS))
        )

        print(
            f"{target[:12]:12} {row}"
        )

    print("\nStrong relationships (absolute correlation >= 0.60):")

    found = False

    for i in range(len(TARGETS)):
        for j in range(i + 1, len(TARGETS)):

            corr = correlation[i, j]

            if abs(corr) >= 0.60:

                print(
                    f"{TARGETS[i]} ↔️ "
                    f"{TARGETS[j]} = {corr:.2f}"
                )

                found = True

    if not found:
        print("None found.")


def show_scenario_lengths(examples):
    section("SCENARIO LENGTH")

    lengths = [
        len(example["scenario"].split())
        for example in examples
    ]

    print(f"Minimum words: {min(lengths)}")
    print(f"Maximum words: {max(lengths)}")
    print(f"Average words: {mean(lengths):.2f}")


def main():

    examples = load_jsonl(DATA_PATH)

    print(f"Total examples: {len(examples)}")

    show_category_distribution(examples)

    show_subcategory_distribution(examples)

    show_feature_variation(examples)

    show_target_distribution(examples)

    show_target_correlations(examples)

    show_scenario_lengths(examples)


if __name__ == "__main__":
    main()