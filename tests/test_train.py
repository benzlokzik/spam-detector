import pathlib

import pandas as pd
import pytest

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "spam_detector" / "data"
LOCAL_PARQUET = DATA_DIR / "processed_combined.parquet"
HF_URL = "hf://datasets/benzlokzik/russian-spam-fork/processed_combined.parquet"


def load_hf_dataframe() -> pd.DataFrame:
    if LOCAL_PARQUET.exists():
        return pd.read_parquet(LOCAL_PARQUET)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(HF_URL)
    df.to_parquet(LOCAL_PARQUET)
    return df


TEST_LIMIT = 200000
TEST_DIR = pathlib.Path(__file__).resolve().parent / "test_data"
MODELS_DIR = TEST_DIR / "models"


@pytest.fixture(scope="module")
def test_dataset():
    df = load_hf_dataframe()
    ham = df[df["label"] == False].head(TEST_LIMIT // 2)
    spam = df[df["label"] == True].head(TEST_LIMIT // 2)
    df = pd.concat([ham, spam]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df["text"].tolist(), df["label"].tolist()


@pytest.fixture(scope="module")
def test_fasttext_file():
    TEST_DIR.mkdir(exist_ok=True)
    fasttext_path = TEST_DIR / "train_test.txt"
    if not fasttext_path.exists():
        df = load_hf_dataframe()
        ham = df[df["label"] == False].head(TEST_LIMIT // 2)
        spam = df[df["label"] == True].head(TEST_LIMIT // 2)
        df = (
            pd.concat([ham, spam])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        with fasttext_path.open("w", encoding="utf-8") as f:
            for text, label in zip(df["text"], df["label"], strict=False):
                tag = "__label__spam" if label else "__label__ham"
                clean = text.replace("\n", " ").strip()
                f.write(f"{tag} {clean}\n")
    return fasttext_path


class TestFastText:
    def test_fit(self, test_fasttext_file):
        import fasttext

        model = fasttext.train_supervised(
            input=str(test_fasttext_file),
            dim=64,
            lr=0.5,
            epoch=5,
            wordNgrams=2,
            minn=2,
            maxn=4,
            loss="ova",
        )
        assert model is not None
        labels, probs = model.predict("тест", k=2)
        assert len(labels) >= 1

        save_path = MODELS_DIR / "fasttext"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_model(str(save_path / "model.bin"))


class TestSklearn:
    def test_fit(self, test_dataset):
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        texts, labels = test_dataset
        y = [int(label) for label in labels]

        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=2,
            max_features=100_000,
        )
        Xv = vec.fit_transform(texts)
        clf = LogisticRegression(
            solver="liblinear", max_iter=1000, class_weight="balanced"
        )
        clf.fit(Xv, y)

        proba = clf.predict_proba(vec.transform(["тест"]))[0, 1]
        assert 0.0 <= proba <= 1.0

        save_path = MODELS_DIR / "sklearn"
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(vec, save_path / "vectorizer.joblib")
        joblib.dump(clf, save_path / "classifier.joblib")


class TestRag:
    def test_fit(self, test_dataset):
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors

        texts, labels = test_dataset

        vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            max_features=100_000,
        )
        matrix = vec.fit_transform(texts)
        nn = NearestNeighbors(n_neighbors=min(8, len(texts)), metric="cosine")
        nn.fit(matrix)

        distances, indices = nn.kneighbors(
            vec.transform(["тест"]), return_distance=True
        )
        assert len(indices[0]) == 8

        save_path = MODELS_DIR / "rag"
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(vec, save_path / "vectorizer.joblib")
        joblib.dump(nn, save_path / "neighbors.joblib")
        joblib.dump({"texts": texts, "labels": labels}, save_path / "corpus.joblib")


class TestBert:
    def test_fit(self, test_dataset):
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        class TextDataset(Dataset):
            def __init__(self, texts, labels):
                self.texts = texts
                self.labels = labels

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx], self.labels[idx]

        texts, labels = test_dataset
        texts = texts[:500]
        labels = labels[:500]
        targets = [int(label) for label in labels]

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        model_name = "cointegrated/rubert-tiny2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        model.to(device)

        def collate(batch):
            batch_texts = [item[0] for item in batch]
            batch_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded["labels"] = batch_labels
            return {k: v.to(device) for k, v in encoded.items()}

        dataset = TextDataset(texts, targets)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        model.train()
        for batch in loader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break

        assert loss.item() >= 0

        save_path = MODELS_DIR / "bert"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


class TestVectorDb:
    def test_fit(self, test_dataset):
        import json
        import torch
        import chromadb
        from sentence_transformers import SentenceTransformer

        texts, labels = test_dataset
        spam_texts = [
            text
            for text, label in zip(texts[:100], labels[:100], strict=False)
            if label
        ]

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        save_path = MODELS_DIR / "vectordb"
        save_path.mkdir(parents=True, exist_ok=True)
        db_path = save_path / "chroma_db"
        client = chromadb.PersistentClient(path=str(db_path))
        try:
            client.delete_collection(name="test_spam")
        except Exception:
            pass
        collection = client.create_collection(name="test_spam")

        encoder_name = "cointegrated/LaBSE-en-ru"
        encoder = SentenceTransformer(encoder_name, device=device)
        embeddings = encoder.encode(spam_texts, show_progress_bar=False, batch_size=32)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=spam_texts,
            ids=[f"spam_{i}" for i in range(len(spam_texts))],
        )

        assert collection.count() == len(spam_texts)

        with (save_path / "config.json").open("w") as f:
            json.dump({"encoder": encoder_name, "collection": "test_spam"}, f)
