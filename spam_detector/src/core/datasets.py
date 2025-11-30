import pathlib

import pandas as pd
from loguru import logger

from .. import log_config  # noqa: F401

HF_URL = "hf://datasets/benzlokzik/russian-spam-fork/processed_combined.parquet"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"
LOCAL_PARQUET = DATA_DIR / "processed_combined.parquet"


def load_hf_dataframe() -> pd.DataFrame:
    if LOCAL_PARQUET.exists():
        logger.debug(f"Loading cached dataset from {LOCAL_PARQUET}")
        return pd.read_parquet(LOCAL_PARQUET)
    logger.info(f"Downloading dataset from {HF_URL}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(HF_URL)
    df.to_parquet(LOCAL_PARQUET)
    logger.info(f"Dataset cached to {LOCAL_PARQUET}")
    return df


def load_dataset() -> tuple[list[str], list[bool]]:
    df = load_hf_dataframe()
    logger.debug(f"Loaded {len(df)} samples")
    return df["text"].tolist(), df["label"].tolist()


def get_fasttext_file() -> pathlib.Path:
    fasttext_path = DATA_DIR / "train.txt"
    if fasttext_path.exists():
        logger.debug(f"Using cached fasttext file {fasttext_path}")
        return fasttext_path
    logger.info("Creating fasttext training file...")
    df = load_hf_dataframe()
    with fasttext_path.open("w", encoding="utf-8") as f:
        for text, label in zip(df["text"], df["label"], strict=False):
            tag = "__label__spam" if label else "__label__ham"
            clean = text.replace("\n", " ").strip()
            f.write(f"{tag} {clean}\n")
    logger.info(f"Fasttext file saved to {fasttext_path}")
    return fasttext_path
