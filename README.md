# Spam Detector

Comparative analysis of text classification methods: fastText, BERT, RAG approach, and scikit-learn models on a single corpus.

## Models

- **FastText** - FastText-based spam detection
- **Scikit-learn** - TF-IDF + Logistic Regression
- **BERT** - Transformer-based classification
- **RAG** - Retrieval-Augmented Generation with TF-IDF + kNN
- **VectorDB RAG** - RAG using ChromaDB and LaBSE embeddings

## Installation

```bash
uv sync # select required groups to work with
```

## Usage

See `experiments/` directory for training and evaluation scripts.

## Hugging Face

Published models and datasets are available in the [spam-detection collection](https://huggingface.co/collections/benzlokzik/spam-detection).
