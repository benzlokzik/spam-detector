import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from loguru import logger

from spam_detector.src import log_config  # noqa: F401
from spam_detector.src.core.base_model import ModelConfig
from spam_detector.src.fastspam.ft_model import FastTextSpamModel
from spam_detector.src.linear.sk_model import SklearnSpamModel
from spam_detector.src.rag.rag_model import RagSpamModel
from spam_detector.src.transformers.bert_model import BertSpamModel
from spam_detector.src.vectordb.vectordb_rag_model import VectorDbRagSpamModel


def train_fasttext() -> None:
    cfg = ModelConfig()
    FastTextSpamModel(cfg).fit()


def train_sklearn() -> None:
    cfg = ModelConfig(model_name="sklearn_spam.bin")
    SklearnSpamModel(cfg).fit()


def train_rag() -> None:
    cfg = ModelConfig(model_name="rag_spam.bin")
    RagSpamModel(cfg).fit()


def train_bert() -> None:
    cfg = ModelConfig(model_name="bert_spam")
    BertSpamModel(cfg).fit()


def train_vectordb() -> None:
    cfg = ModelConfig(model_name="vectordb_spam")
    VectorDbRagSpamModel(cfg).fit()


TRAINERS = {
    "fasttext": train_fasttext,
    "sklearn": train_sklearn,
    "rag": train_rag,
    "bert": train_bert,
    "vectordb": train_vectordb,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spam detection models")
    parser.add_argument(
        "--model",
        choices=list(TRAINERS.keys()) + ["all"],
        required=True,
        help="Model to train",
    )
    args = parser.parse_args()

    if args.model == "all":
        for name, trainer in TRAINERS.items():
            logger.info(f"Training {name}...")
            try:
                trainer()
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
            logger.info(f"Finished {name}")
    else:
        logger.info(f"Training {args.model}...")
        TRAINERS[args.model]()
        logger.info(f"Finished {args.model}")


if __name__ == "__main__":
    main()
