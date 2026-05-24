# Spam Detector

Comparative analysis of text classification methods: fastText, BERT, RAG approach, and scikit-learn models on a single corpus.

## Models

| Model            | Approach                     | Key Features                         |
| ---------------- | ---------------------------- | ------------------------------------ |
| **FastText**     | Shallow neural network       | Fast training, subword embeddings    |
| **Scikit-learn** | TF-IDF + Logistic Regression | Lightweight, interpretable           |
| **BERT**         | Transformer                  | High accuracy, contextual embeddings |
| **RAG**          | TF-IDF + kNN retrieval       | No training required, explainable    |
| **VectorDB RAG** | ChromaDB + LaBSE             | Semantic search, multilingual        |

All five model classes subclass `SpamModel` (`spam_detector.core.base_model`) and expose `predict_proba(text) -> float` (in `[0, 1]`) and `load()`. Two torch-free variants are recommended for embedding into other apps: **sklearn** (`spam_detector.linear.sk_model.SklearnSpamModel`, scikit-learn + joblib, pure-Python, works on 3.12/3.13) and **fastText** (`spam_detector.fastspam.ft_model.FastTextSpamModel`, the `[fasttext]` extra pins numpy<2.0). Both detect Russian-language spam.

## Installation

Local development (editable, all backends):

```bash
uv sync --all-extras
# or per-extra:
uv sync --extra sklearn
```

Consumer install from git (pin the release tag), lightweight extras:

```bash
uv add "spam-detector[sklearn] @ git+https://github.com/benzlokzik/spam-detector@v0.2.0"
uv add "spam-detector[fasttext] @ git+https://github.com/benzlokzik/spam-detector@v0.2.0"
```

Other extras: `[transformers]` (BERT), `[chromadb]` (VectorDB RAG).

## Usage

Weights are not shipped in the wheel; download them from Hugging Face and pass the local path via `ModelConfig`:

```python
from huggingface_hub import hf_hub_download
from spam_detector.core.base_model import ModelConfig
from spam_detector.fastspam.ft_model import FastTextSpamModel

path = hf_hub_download("benzlokzik/spam-detector-fasttext", "antispam.bin")
model = FastTextSpamModel(ModelConfig(model_name=path))
model.load()
print(model.predict_proba("..."))  # float in [0, 1]
```

The sklearn path is identical — swap in `SklearnSpamModel` and its `.joblib` weights:

```python
from spam_detector.linear.sk_model import SklearnSpamModel

model = SklearnSpamModel(ModelConfig(model_name=path))
model.load()
```

BERT loads directly from a Hugging Face repo id (or a local directory) — pass it as `pretrained`; `transformers` downloads and caches it:

```python
from spam_detector.core.base_model import ModelConfig
from spam_detector.transformers.bert_model import BertSpamModel, BertTrainingConfig

model = BertSpamModel(
    ModelConfig(),
    BertTrainingConfig(pretrained="benzlokzik/spam-detector-bert"),
)
model.load()
print(model.predict_proba("..."))  # float in [0, 1]
```

Env-driven alternative: `get_spam_model()` selects the backend from the `MODEL_BACKEND` environment variable. (Importing the package no longer instantiates a model.)

### Model weights

Weights ship via Hugging Face and are passed through `ModelConfig(model_name=<local path>)`:

- **fastText** — repo `benzlokzik/spam-detector-fasttext`, file `antispam.bin` (4.43 MB).
- **sklearn** — repo `benzlokzik/spam-detector-sklearn`, a `.joblib` file (pending upload; until then the local `spam_detector/data/sklearn_spam.bin` works).
- **BERT** — repo `benzlokzik/spam-detector-bert` (rubert-tiny2), loaded via `BertTrainingConfig(pretrained=...)`.

See `experiments/` for training and evaluation scripts.

## Testing

```bash
uv run pytest
```

## Hugging Face

Published models and datasets are available in the [spam-detection collection](https://huggingface.co/collections/benzlokzik/spam-detection).
