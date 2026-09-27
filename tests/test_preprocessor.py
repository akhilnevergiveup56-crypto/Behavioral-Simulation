import pytest

from src.data.preprocessor import TextPreprocessor


def test_preprocessor():

    train_scenarios = [
        "My friend cancelled our plans.",
        "My manager criticized my work.",
        "My friend apologized after an argument.",
    ]

    validation_scenarios = [
        "My coworker criticized my project."
    ]

    preprocessor = TextPreprocessor(
        max_features=20
    )

    preprocessor.fit(train_scenarios)

    train_vectors = preprocessor.transform(
        train_scenarios
    )

    validation_vectors = preprocessor.transform(
        validation_scenarios
    )

    assert train_vectors.shape[0] == 3
    assert validation_vectors.shape[0] == 1
    assert train_vectors.shape[1] == validation_vectors.shape[1]


def test_transform_before_fit():

    preprocessor = TextPreprocessor()

    with pytest.raises(RuntimeError):
        preprocessor.transform(
            ["This should fail."]
        )