import importlib

from .config import get_model_backend, get_model_filename
from .core.base_model import ModelConfig, SpamModel  # re-export

__all__ = ["ModelConfig", "SpamModel", "get_spam_model"]


def get_spam_model() -> SpamModel:
    """Instantiate the backend selected by env (``MODEL_BACKEND``)."""
    backend = get_model_backend()
    model_filename = get_model_filename()
    cfg = ModelConfig(model_name=model_filename)
    if backend == "sklearn":
        SklearnSpamModel = importlib.import_module(
            "spam_detector.linear.sk_model",
        ).SklearnSpamModel
        return SklearnSpamModel(cfg)
    if backend == "bert":
        BertSpamModel = importlib.import_module(
            "spam_detector.transformers.bert_model",
        ).BertSpamModel
        return BertSpamModel(cfg)
    if backend == "rag":
        RagSpamModel = importlib.import_module(
            "spam_detector.rag.rag_model",
        ).RagSpamModel
        return RagSpamModel(cfg)
    if backend == "vectordb":
        VectorDbRagSpamModel = importlib.import_module(
            "spam_detector.vectordb.vectordb_rag_model",
        ).VectorDbRagSpamModel
        return VectorDbRagSpamModel(cfg)
    FastTextSpamModel = importlib.import_module(
        "spam_detector.fastspam.ft_model",
    ).FastTextSpamModel
    return FastTextSpamModel(cfg)
