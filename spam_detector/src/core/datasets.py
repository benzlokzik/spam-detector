import pathlib

import pandas as pd

HF_URL = "hf://datasets/benzlokzik/russian-spam-fork/processed_combined.parquet"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"
LOCAL_PARQUET = DATA_DIR / "processed_combined.parquet"


def load_hf_dataframe() -> pd.DataFrame:
    if LOCAL_PARQUET.exists():
        return pd.read_parquet(LOCAL_PARQUET)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(HF_URL)
    df.to_parquet(LOCAL_PARQUET)
    return df


def load_dataset() -> tuple[list[str], list[bool]]:
    df = load_hf_dataframe()
    return df["text"].tolist(), df["label"].tolist()


def get_fasttext_file() -> pathlib.Path:
    fasttext_path = DATA_DIR / "train.txt"
    if fasttext_path.exists():
        return fasttext_path
    df = load_hf_dataframe()
    with fasttext_path.open("w", encoding="utf-8") as f:
        for text, label in zip(df["text"], df["label"], strict=False):
            tag = "__label__spam" if label else "__label__ham"
            clean = text.replace("\n", " ").strip()
            f.write(f"{tag} {clean}\n")
    return fasttext_path
