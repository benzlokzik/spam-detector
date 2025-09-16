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
pip install uv
uv sync
```

## Usage

See `experiments/` directory for training and evaluation scripts.
