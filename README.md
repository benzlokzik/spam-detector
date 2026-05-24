# Spam Detector

Comparative analysis of Russian-language spam detection methods — fastText, scikit-learn, BERT, and two retrieval approaches (kNN-RAG and a ChromaDB vector store) — trained and evaluated on a single corpus, behind one common interface.

Every model subclasses `SpamModel` (`spam_detector.core.base_model`) and exposes the same three methods:

- `fit()` — train from the corpus and persist weights.
- `load()` — load persisted weights.
- `predict_proba(text) -> float` — spam probability in `[0, 1]` (higher = more spam).

All models are trained for **Russian** text.

## Models

| Model            | Class (import path)                                          | Extra            | Approach                     | Notes                                  |
| ---------------- | ------------------------------------------------------------ | ---------------- | ---------------------------- | -------------------------------------- |
| **fastText**     | `spam_detector.fastspam.ft_model.FastTextSpamModel`          | `[fasttext]`     | Shallow neural network       | Fast, torch-free, subword embeddings   |
| **scikit-learn** | `spam_detector.linear.sk_model.SklearnSpamModel`             | `[sklearn]`      | TF-IDF (char n-grams) + LogReg | Lightweight, pure-Python, interpretable |
| **BERT**         | `spam_detector.transformers.bert_model.BertSpamModel`        | `[transformers]` | Transformer (rubert-tiny2)   | Highest accuracy, pulls torch          |
| **RAG (kNN)**    | `spam_detector.rag.rag_model.RagSpamModel`                   | `[sklearn]`      | TF-IDF + nearest neighbours  | No model training, explainable         |
| **VectorDB RAG** | `spam_detector.vectordb.vectordb_rag_model.VectorDbRagSpamModel` | `[chromadb]` | ChromaDB + LaBSE embeddings  | Semantic search, pulls torch           |

The two **torch-free** variants — **fastText** and **scikit-learn** — are the ones recommended for embedding into other applications.

## Installation

The package builds with [hatchling](https://hatch.pypa.io/) and requires Python `>=3.12`. The base install is intentionally light (`loguru`, `python-dotenv`, `pandas`, `huggingface-hub`, `tqdm`); each model's heavy dependencies live in an optional **extra**, so you only install what you use.

Add it to another project from git, picking the extra you need (pin a release tag):

```bash
uv add "spam-detector[sklearn]  @ git+https://github.com/benzlokzik/spam-detector@v0.2.0"
uv add "spam-detector[fasttext] @ git+https://github.com/benzlokzik/spam-detector@v0.2.0"
uv add "spam-detector[transformers] @ git+https://github.com/benzlokzik/spam-detector@v0.2.0"
uv add "spam-detector[chromadb]  @ git+https://github.com/benzlokzik/spam-detector@v0.2.0"
```

| Extra            | Pulls in                                                    |
| ---------------- | ----------------------------------------------------------- |
| `[sklearn]`      | `scikit-learn`, `joblib` (also covers the RAG-kNN model)    |
| `[fasttext]`     | `fasttext-wheel`, `numpy<2.0`                               |
| `[transformers]` | `torch`, `transformers`, `tokenizers`, `sentence-transformers` |
| `[chromadb]`     | `chromadb`, `sentence-transformers` (→ `torch`)             |

Importing the package pulls **no** backend — `import spam_detector` and `from spam_detector.linear.sk_model import SklearnSpamModel` work with only `[sklearn]` installed; torch is never imported unless you use a torch-backed model.

### Local development

```bash
uv sync --all-extras          # every backend
uv sync --extra sklearn       # a single backend
uv run pytest                 # see Testing below
```

## Usage

Trained weights are **not** shipped in the wheel — download them from Hugging Face (see [Model weights](#model-weights)) and point the model at them via `ModelConfig`.

`ModelConfig(model_name=<path>)` controls where single-file models read their weights; `model_path` resolves to `<project_root>/<data_subdir>/<model_name>`, so pass an absolute path when consuming the package from elsewhere.

### fastText

```python
from huggingface_hub import hf_hub_download
from spam_detector.core.base_model import ModelConfig
from spam_detector.fastspam.ft_model import FastTextSpamModel

path = hf_hub_download("benzlokzik/spam-detector-fasttext", "antispam.bin")
model = FastTextSpamModel(ModelConfig(model_name=path))
model.load()
print(model.predict_proba("Срочно деньги! Кредит без процентов, пиши в ЛС"))  # ~0.99
```

### scikit-learn

```python
from huggingface_hub import hf_hub_download
from spam_detector.core.base_model import ModelConfig
from spam_detector.linear.sk_model import SklearnSpamModel

path = hf_hub_download("benzlokzik/spam-detector-sklearn", "sklearn_spam.joblib")
model = SklearnSpamModel(ModelConfig(model_name=path))
model.load()
print(model.predict_proba("..."))  # float in [0, 1]
```

### BERT

BERT is a multi-file model directory, so it loads straight from a Hugging Face repo id (or a local directory) via `pretrained` — `transformers` downloads and caches it:

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

### Environment-driven selection

`get_spam_model()` builds a backend from the `MODEL_BACKEND` env var (`fasttext` | `sklearn` | `bert` | `rag` | `vectordb`, default `fasttext`); `MODEL_FILENAME` sets the weights filename. These can be placed in a `.env` file.

```python
from spam_detector import get_spam_model  # importing the package instantiates nothing

model = get_spam_model()  # MODEL_BACKEND decides the class
model.load()
```

## Model weights

Weights ship via Hugging Face. The repos and the dataset:

- **fastText** — [benzlokzik/spam-detector-fasttext](https://huggingface.co/benzlokzik/spam-detector-fasttext), file `antispam.bin` (4.43 MB).
- **BERT** — [benzlokzik/spam-detector-bert](https://huggingface.co/benzlokzik/spam-detector-bert), rubert-tiny2, loaded via `BertTrainingConfig(pretrained=...)`.
- **scikit-learn** — [benzlokzik/spam-detector-sklearn](https://huggingface.co/benzlokzik/spam-detector-sklearn), a `.joblib` file *(pending upload; until then the local `spam_detector/data/sklearn_spam.bin` works)*.
- **Dataset** — [benzlokzik/russian-spam-fork](https://huggingface.co/datasets/benzlokzik/russian-spam-fork) (`processed_combined.parquet`), downloaded on demand by the training scripts.

All published artifacts are grouped in the [spam-detection collection](https://huggingface.co/collections/benzlokzik/spam-detection).

## Training

Training pulls the dataset from Hugging Face automatically and writes weights under `spam_detector/data/`. See the scripts in `experiments/`, or train programmatically:

```python
from spam_detector.core.base_model import ModelConfig
from spam_detector.linear.sk_model import SklearnSpamModel

SklearnSpamModel(ModelConfig(model_name="sklearn_spam.bin")).fit()
```

## Testing

```bash
uv run pytest                                   # collects all suites
uv run --extra sklearn --extra fasttext pytest tests/test_linear_model.py tests/test_fasttext_model.py tests/test_factory.py
uv run --extra transformers pytest tests/test_bert_model.py
uv run --extra chromadb    pytest tests/test_vectordb_rag_model.py
```

The model unit tests are network-free (the dataset and heavy `from_pretrained` calls are mocked). `tests/test_train.py` is a heavier end-to-end suite that downloads the corpus and trains real models.

## License

[GNU AGPL-3.0](LICENSE) — a network-use copyleft licence: anyone offering this software over a network must make the corresponding source available.
