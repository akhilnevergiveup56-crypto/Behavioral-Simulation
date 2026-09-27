from src.data.taxonomy import validate_scenario_category


def test_valid_category():

    assert validate_scenario_category(
        "workplace",
        "credit_conflict",
    )


def test_invalid_category():

    assert not validate_scenario_category(
        "work",
        "credit_conflict",
    )


def test_invalid_subcategory():

    assert not validate_scenario_category(
        "workplace",
        "childhood_trauma",
    )