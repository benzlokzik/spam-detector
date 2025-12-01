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

## Installation

```bash
uv sync # select required groups to work with
```

## Usage

See `experiments/` directory for training and evaluation scripts.

## Testing

```bash
uv run pytest
```

## Hugging Face

Published models and datasets are available in the [spam-detection collection](https://huggingface.co/collections/benzlokzik/spam-detection).
