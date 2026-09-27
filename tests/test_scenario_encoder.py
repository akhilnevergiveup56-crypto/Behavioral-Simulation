from src.data.scenario_encoder import ScenarioEncoder


def test_scenario_encoder():

    scenarios = [
        "My friend cancelled our plans again.",
        "My manager criticized my work in front of everyone.",
        "My friend apologized after an argument.",
    ]

    encoder = ScenarioEncoder(max_features=20)

    encoder.fit(scenarios)

    vectors = encoder.encode(scenarios)

    assert vectors.shape[0] == 3
    assert vectors.shape[1] <= 20
    assert vectors.nnz > 0