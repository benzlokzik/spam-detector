import joblib
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..core.base_model import ModelConfig, SpamModel
from ..core.datasets import load_dataset


class SklearnSpamModel(SpamModel):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self._vec: TfidfVectorizer | None = None
        self._clf: LogisticRegression | None = None

    def fit(self) -> None:
        logger.info("Training sklearn model...")
        texts, labels = load_dataset()
        if not texts:
            raise ValueError("No labeled lines found in training data.")
        y = [int(label) for label in labels]
        if len(set(y)) < 2:
            raise ValueError("Training data must contain both classes")
        logger.debug("Fitting TF-IDF vectorizer...")
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=2,
            max_features=100_000,
        )
        Xv = vec.fit_transform(texts)
        logger.debug("Training LogisticRegression...")
        clf = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
        )
        clf.fit(Xv, y)
        joblib.dump((vec, clf), str(self.cfg.model_path))
        self._vec, self._clf = vec, clf
        logger.info(f"Model saved to {self.cfg.model_path}")

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
