from sklearn.feature_extraction.text import TfidfVectorizer


class ScenarioEncoder:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True,
            stop_words="english"
        )

    def fit(self, scenarios: list[str]):
        self.vectorizer.fit(scenarios)

    def encode(self, scenarios: list[str]):
        return self.vectorizer.transform(scenarios)