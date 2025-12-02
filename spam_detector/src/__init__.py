# Factory: create a spam_model instance chosen by env
import importlib

from .config import get_model_backend, get_model_filename
from .core.base_model import ModelConfig, SpamModel  # re-export


def _create_spam_model() -> SpamModel:
    backend = get_model_backend()
    model_filename = get_model_filename()
    cfg = ModelConfig(model_name=model_filename)
    if backend == "sklearn":
        SklearnSpamModel = importlib.import_module(
            "spam_detector.src.linear.sk_model",
        ).SklearnSpamModel
        return SklearnSpamModel(cfg)
    if backend == "bert":
        BertSpamModel = importlib.import_module(
            "spam_detector.src.transformers.bert_model",
        ).BertSpamModel
        return BertSpamModel(cfg)
    if backend == "rag":
        RagSpamModel = importlib.import_module(
            "spam_detector.src.rag.rag_model",
        ).RagSpamModel
        return RagSpamModel(cfg)
    if backend == "vectordb":
        VectorDbRagSpamModel = (
            importlib.import_module(
                "spam_detector.src.vectordb.vectordb_rag_model",
            )
        ).VectorDbRagSpamModel
        return VectorDbRagSpamModel(cfg)
    FastTextSpamModel = importlib.import_module(
        "spam_detector.src.fastspam.ft_model",
    ).FastTextSpamModel
    return FastTextSpamModel(cfg)


spam_model: SpamModel = _create_spam_model()
