import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ..core.base_model import ModelConfig, SpamModel
from ..core.datasets import load_fasttext_dataset


class RagSpamModel(SpamModel):
    def __init__(
        self,
        cfg: ModelConfig,
        n_neighbors: int = 8,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 100_000,
        positive_label: str = "spam",
    ) -> None:
        super().__init__(cfg)
        self.n_neighbors = n_neighbors
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.positive_label = positive_label
        self._vec: TfidfVectorizer | None = None
        self._nn: NearestNeighbors | None = None
        self._labels: list[str] | None = None

    def fit(self) -> None:
        texts, labels = load_fasttext_dataset(self.cfg.train_paths())
        if not texts:
            raise ValueError("Dataset is empty")
        vec = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            lowercase=True,
            max_features=self.max_features,
        )
        matrix = vec.fit_transform(texts)
        neighbor_count = min(self.n_neighbors, len(texts))
        nn = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
        nn.fit(matrix)
        payload = {
            "vectorizer": vec,
            "neighbors": nn,
            "labels": labels,
        }
        joblib.dump(payload, str(self.cfg.model_path))
        self._vec = vec
        self._nn = nn
        self._labels = labels

    def load(self) -> None:
        data = joblib.load(str(self.cfg.model_path))
        self._vec = data["vectorizer"]
        self._nn = data["neighbors"]
        self._labels = list(data["labels"])

    def predict_proba(self, text: str) -> float:
        if self._vec is None or self._nn is None or self._labels is None:
            self.load()
        if self._vec is None or self._nn is None or self._labels is None:
            raise RuntimeError("Model is not loaded")
        vector = self._vec.transform([text.lower().replace("\n", " ").strip()])
        distances, indices = self._nn.kneighbors(vector, return_distance=True)
        spam_score = 0.0
        total = 0.0
        for dist, idx in zip(distances[0], indices[0], strict=False):
            weight = max(1.0 - float(dist), 0.0)
            total += weight
            if self._labels[int(idx)] == self.positive_label:
                spam_score += weight
        if total == 0.0:
            return 0.0
        return float(spam_score / total)
