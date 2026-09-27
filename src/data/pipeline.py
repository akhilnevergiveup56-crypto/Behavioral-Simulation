from src.data.jsonl_loader import load_jsonl
from src.data.split import split_dataset
from src.data.profile_encoder import encode_profile
from src.data.features import encode_targets
from src.data.preprocessor import TextPreprocessor


def build_training_data(
    path: str,
    max_features: int = 100,
):
    # 1. Load raw examples
    examples = load_jsonl(path)

    # 2. Split BEFORE fitting TF-IDF
    train_examples, val_examples, test_examples = split_dataset(
        examples
    )

    # 3. Create text preprocessor
    text_preprocessor = TextPreprocessor(
        max_features=max_features
    )

    # 4. Fit ONLY on training scenarios
    train_scenarios = [
        example["scenario"]
        for example in train_examples
    ]

    text_preprocessor.fit(train_scenarios)

    # 5. Encode profiles
    train_profiles = [
        encode_profile(example)
        for example in train_examples
    ]

    val_profiles = [
        encode_profile(example)
        for example in val_examples
    ]

    test_profiles = [
        encode_profile(example)
        for example in test_examples
    ]

    # 6. Stack profile vectors
    import torch

    X_train_profile = torch.stack(train_profiles)
    X_val_profile = torch.stack(val_profiles)
    X_test_profile = torch.stack(test_profiles)

    # 7. Encode scenarios using fitted TF-IDF
    train_scenarios_vector = text_preprocessor.transform(
        train_scenarios
    )

    val_scenarios_vector = text_preprocessor.transform(
        [
            example["scenario"]
            for example in val_examples
        ]
    )

    test_scenarios_vector = text_preprocessor.transform(
        [
            example["scenario"]
            for example in test_examples
        ]
    )

    # 8. Convert TF-IDF to tensors
    X_train_scenario = torch.tensor(
        train_scenarios_vector.toarray(),
        dtype=torch.float32
    )

    X_val_scenario = torch.tensor(
        val_scenarios_vector.toarray(),
        dtype=torch.float32
    )

    X_test_scenario = torch.tensor(
        test_scenarios_vector.toarray(),
        dtype=torch.float32
    )

    # 9. Encode targets
    y_train = torch.stack([
        encode_targets(example["behavioral_targets"])
        for example in train_examples
    ])

    y_val = torch.stack([
        encode_targets(example["behavioral_targets"])
        for example in val_examples
    ])

    y_test = torch.stack([
        encode_targets(example["behavioral_targets"])
        for example in test_examples
    ])

    # 10. Combine profile + scenario
    X_train = torch.cat(
        [X_train_profile, X_train_scenario],
        dim=1
    )

    X_val = torch.cat(
        [X_val_profile, X_val_scenario],
        dim=1
    )

    X_test = torch.cat(
        [X_test_profile, X_test_scenario],
        dim=1
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )