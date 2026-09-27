import random


def split_dataset(
    examples: list[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    examples = examples.copy()

    random.Random(seed).shuffle(examples)

    n = len(examples)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_examples = examples[:train_end]
    val_examples = examples[train_end:val_end]
    test_examples = examples[val_end:]

    return train_examples, val_examples, test_examples