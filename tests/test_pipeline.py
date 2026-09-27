from src.data.pipeline import build_training_data


def test_training_pipeline():

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = build_training_data(
        "data/raw/behavioral_data.jsonl",
        max_features=20,
    )

    assert X_train.shape[0] == 28
    assert X_val.shape[0] == 6
    assert X_test.shape[0] == 6

    assert X_train.shape[1] == 27 + 20
    assert X_val.shape[1] == 27 + 20
    assert X_test.shape[1] == 27 + 20

    assert y_train.shape == (28, 6)
    assert y_val.shape == (6, 6)
    assert y_test.shape == (6,6)