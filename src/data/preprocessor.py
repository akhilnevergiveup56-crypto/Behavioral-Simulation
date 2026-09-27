from src.data.scenario_encoder import ScenarioEncoder


class TextPreprocessor:
    def __init__(self, max_features=1000):
        self.encoder = ScenarioEncoder(
            max_features=max_features
        )

        self.is_fitted = False

    def fit(self, scenarios: list[str]):
        self.encoder.fit(scenarios)
        self.is_fitted = True

    def transform(self, scenarios: list[str]):
        if not self.is_fitted:
            raise RuntimeError(
                "TextPreprocessor must be fitted before transform()."
            )

        return self.encoder.encode(scenarios)

    def fit_transform(self, scenarios: list[str]):
        self.fit(scenarios)
        return self.transform(scenarios)