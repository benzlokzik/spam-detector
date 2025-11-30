import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..core.base_model import ModelConfig, SpamModel
from ..core.datasets import load_fasttext_dataset


class SklearnSpamModel(SpamModel):
    def __init__(self, cfg: ModelConfig, positive_label: str = "spam") -> None:
        super().__init__(cfg)
        self.positive_label = positive_label
        self._vec: TfidfVectorizer | None = None
        self._clf: LogisticRegression | None = None

    def fit(self) -> None:
        texts, labels = load_fasttext_dataset(self.cfg.train_paths())
        if not texts:
            raise ValueError("No labeled lines found in training data.")
        y = [1 if label == self.positive_label else 0 for label in labels]
        if len(set(y)) < 2:
            raise ValueError("Training data must contain both classes")
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=2,
            max_features=100_000,
        )
        Xv = vec.fit_transform(texts)
        clf = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
        )
        clf.fit(Xv, y)
        joblib.dump((vec, clf), str(self.cfg.model_path))
        self._vec, self._clf = vec, clf

    def load(self) -> None:
        self._vec, self._clf = joblib.load(str(self.cfg.model_path))

    def predict_proba(self, text: str) -> float:
        if self._vec is None or self._clf is None:
            self.load()
        if self._vec is None or self._clf is None:
            raise RuntimeError("Model is not loaded")
        processed = text.lower().replace("\n", "\t").strip()
        Xv = self._vec.transform([processed])
        return float(self._clf.predict_proba(Xv)[0, 1])
