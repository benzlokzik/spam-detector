import json
import pathlib

import chromadb
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from .. import log_config  # noqa: F401
from ..core.base_model import ModelConfig, SpamModel
from ..core.datasets import load_dataset


class VectorDbRagSpamModel(SpamModel):
    def __init__(
        self,
        cfg: ModelConfig,
        min_matches: int = 3,
        similarity_threshold: float = 0.7,
        top_k: int = 10,
        collection_name: str = "spam_examples",
        db_path: pathlib.Path | None = None,
    ) -> None:
        super().__init__(cfg)
        self.min_matches = min_matches
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.collection_name = collection_name
        if db_path is None:
            self.db_path = self.cfg.data_dir / "chroma_db"
        else:
            self.db_path = pathlib.Path(db_path)
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._encoder: SentenceTransformer | None = None

    def _get_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.db_path))
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_collection(name=self.collection_name)
            except Exception:
                self._collection = client.create_collection(name=self.collection_name)
        return self._collection

    def _device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            device = self._device()
            logger.info(f"Using device: {device}")
            self._encoder = SentenceTransformer(
                "cointegrated/LaBSE-en-ru",
                device=device,
            )
        return self._encoder

    def fit(self) -> None:
        texts, labels = load_dataset()
        spam_texts = [text for text, label in zip(texts, labels, strict=False) if label]
        if not spam_texts:
            msg = "No spam examples found in training data"
            raise ValueError(msg)
        client = self._get_client()
        try:
            collection = client.get_collection(name=self.collection_name)
            client.delete_collection(name=self.collection_name)
        except Exception:
            logger.debug(f"Collection '{self.collection_name}' not found, creating new")
        collection = client.create_collection(name=self.collection_name)
        encoder = self._get_encoder()
        logger.info(f"Generating embeddings for {len(spam_texts)} spam examples...")
        embeddings = encoder.encode(spam_texts, show_progress_bar=True, batch_size=32)
        logger.info("Storing embeddings in ChromaDB...")
        batch_size = 5000  # ChromaDB max batch size is 5461
        total = len(spam_texts)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            collection.add(
                embeddings=embeddings[start_idx:end_idx].tolist(),
                documents=spam_texts[start_idx:end_idx],
                ids=[f"spam_{i}" for i in range(start_idx, end_idx)],
            )
            logger.debug(
                f"Added batch {start_idx // batch_size + 1}/"
                f"{(total + batch_size - 1) // batch_size} ({start_idx} - {end_idx})",
            )
        self._collection = collection
        config_path = self.cfg.model_path.with_suffix(".json")
        config_data = {
            "collection_name": self.collection_name,
            "db_path": str(self.db_path),
            "min_matches": self.min_matches,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
        }
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
        logger.info(f"Model saved to {config_path}")

    def load(self) -> None:
        config_path = self.cfg.model_path.with_suffix(".json")
        if not config_path.exists():
            msg = f"Model config not found: {config_path}"
            raise FileNotFoundError(msg)
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        self.collection_name = config_data.get("collection_name", self.collection_name)
        if "db_path" in config_data:
            self.db_path = pathlib.Path(config_data["db_path"])
        self.min_matches = config_data.get("min_matches", self.min_matches)
        self.similarity_threshold = config_data.get(
            "similarity_threshold",
            self.similarity_threshold,
        )
        self.top_k = config_data.get("top_k", self.top_k)
        self._collection = self._get_collection()
        self._encoder = self._get_encoder()

    def predict_proba(self, text: str) -> float:
        if self._collection is None or self._encoder is None:
            self.load()
        if self._collection is None or self._encoder is None:
            msg = "Model is not loaded"
            raise RuntimeError(msg)
        query_embedding = self._encoder.encode([text], show_progress_bar=False)[0]
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.top_k,
        )
        if not results["distances"] or not results["distances"][0]:
            return 0.0
        distances = results["distances"][0]
        similarities = [1.0 - d for d in distances]
        matches_above_threshold = [
            s for s in similarities if s >= self.similarity_threshold
        ]
        matches_count = len(matches_above_threshold)
        if matches_count >= self.min_matches:
            avg_similarity = (
                sum(matches_above_threshold) / len(matches_above_threshold)
                if matches_above_threshold
                else 0.0
            )
            probability = min(
                1.0,
                (matches_count / self.min_matches) * 0.5 + avg_similarity * 0.5,
            )
            return float(probability)
        if matches_count > 0:
            avg_similarity = sum(matches_above_threshold) / len(matches_above_threshold)
            probability = (matches_count / self.min_matches) * avg_similarity
            return float(probability)
        return 0.0
