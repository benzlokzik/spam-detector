import json
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spam_detector.src.core.base_model import ModelConfig
from spam_detector.src.vectordb.vectordb_rag_model import VectorDbRagSpamModel


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


@pytest.fixture
def model_config(temp_dir):
    """Create a ModelConfig for testing."""
    return ModelConfig(
        project_root=temp_dir,
        data_subdir="data",
        model_name="test_model.bin",
    )


@pytest.fixture
def mock_encoder():
    """Create a mock SentenceTransformer encoder."""
    encoder = MagicMock()
    encoder.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return encoder


@pytest.fixture
def mock_collection():
    """Create a mock ChromaDB collection."""
    collection = MagicMock()
    collection.count.return_value = 0
    collection.query.return_value = {
        "distances": [[0.1, 0.2, 0.3]],
        "documents": [["spam1", "spam2", "spam3"]],
        "ids": [["spam_0", "spam_1", "spam_2"]],
    }
    return collection


@pytest.fixture
def mock_client(mock_collection):
    """Create a mock ChromaDB client."""
    client = MagicMock()
    client.get_collection.return_value = mock_collection
    client.create_collection.return_value = mock_collection
    client.delete_collection.return_value = None
    return client


class TestVectorDbRagSpamModelInit:
    """Test VectorDbRagSpamModel initialization."""

    def test_init_defaults(self, model_config):
        """Test initialization with default parameters."""
        model = VectorDbRagSpamModel(model_config)
        assert model.min_matches == 3
        assert model.similarity_threshold == 0.7
        assert model.top_k == 10
        assert model.collection_name == "spam_examples"
        assert model.db_path == model_config.data_dir / "chroma_db"
        assert model._client is None
        assert model._collection is None
        assert model._encoder is None

    def test_init_custom_params(self, model_config, temp_dir):
        """Test initialization with custom parameters."""
        custom_db_path = temp_dir / "custom_db"
        model = VectorDbRagSpamModel(
            model_config,
            min_matches=5,
            similarity_threshold=0.8,
            top_k=20,
            collection_name="custom_collection",
            db_path=custom_db_path,
        )
        assert model.min_matches == 5
        assert model.similarity_threshold == 0.8
        assert model.top_k == 20
        assert model.collection_name == "custom_collection"
        assert model.db_path == custom_db_path


class TestVectorDbRagSpamModelDevice:
    """Test device detection methods."""

    @patch("spam_detector.src.vectordb.vectordb_rag_model.torch")
    def test_device_cuda(self, mock_torch, model_config):
        """Test device detection when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        model = VectorDbRagSpamModel(model_config)
        assert model._device() == "cuda"

    @patch("spam_detector.src.vectordb.vectordb_rag_model.torch")
    def test_device_mps(self, mock_torch, model_config):
        """Test device detection when MPS is available."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        model = VectorDbRagSpamModel(model_config)
        assert model._device() == "mps"

    @patch("spam_detector.src.vectordb.vectordb_rag_model.torch")
    def test_device_cpu(self, mock_torch, model_config):
        """Test device detection when only CPU is available."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        model = VectorDbRagSpamModel(model_config)
        assert model._device() == "cpu"


class TestVectorDbRagSpamModelFit:
    """Test fit() method."""

    @patch("spam_detector.src.vectordb.vectordb_rag_model.load_dataset")
    @patch("spam_detector.src.vectordb.vectordb_rag_model.chromadb")
    @patch("spam_detector.src.vectordb.vectordb_rag_model.SentenceTransformer")
    def test_fit_success(
        self,
        mock_sentence_transformer,
        mock_chromadb,
        mock_load_dataset,
        model_config,
        mock_encoder,
        mock_client,
        mock_collection,
        temp_dir,
    ):
        """Test successful fit with spam examples."""
        # Setup mocks
        mock_load_dataset.return_value = (
            ["text1", "text2", "text3"],
            [True, False, True],  # 2 spam, 1 ham
        )
        mock_sentence_transformer.return_value = mock_encoder
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = mock_collection

        # Create model and fit
        model = VectorDbRagSpamModel(model_config)
        model.fit()

        # Verify encoder was created
        mock_sentence_transformer.assert_called_once()
        assert "cointegrated/LaBSE-en-ru" in str(mock_sentence_transformer.call_args)

        # Verify embeddings were generated
        mock_encoder.encode.assert_called_once()
        call_args = mock_encoder.encode.call_args
        assert call_args[0][0] == ["text1", "text3"]  # Only spam texts
        assert call_args[1]["show_progress_bar"] is True
        assert call_args[1]["batch_size"] == 32

        # Verify collection was created
        mock_client.create_collection.assert_called_once_with(name="spam_examples")

        # Verify config was saved
        config_path = model_config.model_path.with_suffix(".json")
        assert config_path.exists()
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        assert config_data["collection_name"] == "spam_examples"
        assert config_data["min_matches"] == 3
        assert config_data["similarity_threshold"] == 0.7
        assert config_data["top_k"] == 10

    @patch("spam_detector.src.vectordb.vectordb_rag_model.load_dataset")
    def test_fit_no_spam_examples(self, mock_load_dataset, model_config):
        """Test fit raises ValueError when no spam examples found."""
        mock_load_dataset.return_value = (
            ["text1", "text2"],
            [False, False],  # No spam
        )

        model = VectorDbRagSpamModel(model_config)
        with pytest.raises(ValueError, match="No spam examples found"):
            model.fit()

    @patch("spam_detector.src.vectordb.vectordb_rag_model.load_dataset")
    @patch("spam_detector.src.vectordb.vectordb_rag_model.chromadb")
    @patch("spam_detector.src.vectordb.vectordb_rag_model.SentenceTransformer")
    def test_fit_batch_processing(
        self,
        mock_sentence_transformer,
        mock_chromadb,
        mock_load_dataset,
        model_config,
        mock_encoder,
        mock_client,
        mock_collection,
    ):
        """Test that fit processes large datasets in batches."""
        # Create a large dataset (6000 spam examples to test batching)
        num_spam = 6000
        spam_texts = [f"spam_text_{i}" for i in range(num_spam)]
        all_texts = spam_texts + ["ham_text"]
        all_labels = [True] * num_spam + [False]

        mock_load_dataset.return_value = (all_texts, all_labels)
        mock_sentence_transformer.return_value = mock_encoder

        # Create embeddings array
        embeddings = np.random.rand(num_spam, 384).astype(np.float32)
        mock_encoder.encode.return_value = embeddings

        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = mock_collection

        model = VectorDbRagSpamModel(model_config)
        model.fit()

        # Verify collection.add was called multiple times (batches of 5000)
        assert (
            mock_collection.add.call_count == 2
        )  # 6000 items = 2 batches (5000 + 1000)

        # Verify first batch
        first_call = mock_collection.add.call_args_list[0]
        assert len(first_call[1]["documents"]) == 5000
        assert len(first_call[1]["ids"]) == 5000
        assert len(first_call[1]["embeddings"]) == 5000

        # Verify second batch
        second_call = mock_collection.add.call_args_list[1]
        assert len(second_call[1]["documents"]) == 1000
        assert len(second_call[1]["ids"]) == 1000
        assert len(second_call[1]["embeddings"]) == 1000

    @patch("spam_detector.src.vectordb.vectordb_rag_model.load_dataset")
    @patch("spam_detector.src.vectordb.vectordb_rag_model.chromadb")
    @patch("spam_detector.src.vectordb.vectordb_rag_model.SentenceTransformer")
    def test_fit_existing_collection_deleted(
        self,
        mock_sentence_transformer,
        mock_chromadb,
        mock_load_dataset,
        model_config,
        mock_encoder,
        mock_client,
        mock_collection,
    ):
        """Test that existing collection is deleted before creating new one."""
        mock_load_dataset.return_value = (["spam1"], [True])
        mock_sentence_transformer.return_value = mock_encoder
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_collection.return_value = mock_collection  # Collection exists
        mock_client.create_collection.return_value = mock_collection

        model = VectorDbRagSpamModel(model_config)
        model.fit()

        # Verify collection was deleted and recreated
        mock_client.delete_collection.assert_called_once_with(name="spam_examples")
        mock_client.create_collection.assert_called_once_with(name="spam_examples")


class TestVectorDbRagSpamModelLoad:
    """Test load() method."""

    def test_load_success(
        self, model_config, temp_dir, mock_encoder, mock_collection, mock_client
    ):
        """Test successful load from config file."""
        # Create config file
        config_path = model_config.model_path.with_suffix(".json")
        config_data = {
            "collection_name": "test_collection",
            "db_path": str(temp_dir / "custom_db"),
            "min_matches": 5,
            "similarity_threshold": 0.8,
            "top_k": 15,
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        with (
            patch(
                "spam_detector.src.vectordb.vectordb_rag_model.chromadb"
            ) as mock_chromadb,
            patch(
                "spam_detector.src.vectordb.vectordb_rag_model.SentenceTransformer"
            ) as mock_sentence_transformer,
        ):
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.return_value = mock_collection
            mock_sentence_transformer.return_value = mock_encoder

            model = VectorDbRagSpamModel(model_config)
            model.load()

            # Verify config values were loaded
            assert model.collection_name == "test_collection"
            assert model.db_path == pathlib.Path(temp_dir / "custom_db")
            assert model.min_matches == 5
            assert model.similarity_threshold == 0.8
            assert model.top_k == 15

            # Verify collection and encoder were initialized
            assert model._collection == mock_collection
            assert model._encoder == mock_encoder

    def test_load_file_not_found(self, model_config):
        """Test load raises FileNotFoundError when config doesn't exist."""
        model = VectorDbRagSpamModel(model_config)
        with pytest.raises(FileNotFoundError, match="Model config not found"):
            model.load()

    def test_load_partial_config(
        self, model_config, temp_dir, mock_encoder, mock_collection, mock_client
    ):
        """Test load with partial config (missing optional fields)."""
        # Create config file with only required fields
        config_path = model_config.model_path.with_suffix(".json")
        config_data = {
            "collection_name": "test_collection",
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        with (
            patch(
                "spam_detector.src.vectordb.vectordb_rag_model.chromadb"
            ) as mock_chromadb,
            patch(
                "spam_detector.src.vectordb.vectordb_rag_model.SentenceTransformer"
            ) as mock_sentence_transformer,
        ):
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.return_value = mock_collection
            mock_sentence_transformer.return_value = mock_encoder

            model = VectorDbRagSpamModel(model_config)
            model.load()

            # Verify defaults are used for missing fields
            assert model.collection_name == "test_collection"
            assert model.min_matches == 3  # Default
            assert model.similarity_threshold == 0.7  # Default
            assert model.top_k == 10  # Default


class TestVectorDbRagSpamModelPredictProba:
    """Test predict_proba() method."""

    def test_predict_proba_success_above_threshold(
        self, model_config, mock_encoder, mock_collection
    ):
        """Test predict_proba with matches above threshold."""
        # Setup collection to return good matches
        mock_collection.query.return_value = {
            "distances": [[0.1, 0.15, 0.2, 0.25]],  # Similarities: 0.9, 0.85, 0.8, 0.75
            "documents": [["spam1", "spam2", "spam3", "spam4"]],
            "ids": [["spam_0", "spam_1", "spam_2", "spam_3"]],
        }

        model = VectorDbRagSpamModel(
            model_config, min_matches=3, similarity_threshold=0.7
        )
        model._collection = mock_collection
        model._encoder = mock_encoder

        result = model.predict_proba("test text")

        # Verify encoder was called
        mock_encoder.encode.assert_called_once_with(
            ["test text"], show_progress_bar=False
        )

        # Verify query was made
        mock_collection.query.assert_called_once()
        query_args = mock_collection.query.call_args
        assert query_args[1]["n_results"] == 10

        # Should have 4 matches above threshold (0.7), which is >= min_matches (3)
        # Probability calculation: (4/3) * 0.5 + avg_similarity * 0.5
        # avg_similarity = (0.9 + 0.85 + 0.8 + 0.75) / 4 = 0.825
        # probability = min(1.0, (4/3) * 0.5 + 0.825 * 0.5) = min(1.0, 0.666 + 0.4125) = 1.0
        assert 0.0 <= result <= 1.0
        assert result > 0.5  # Should be high probability

    def test_predict_proba_below_min_matches(
        self, model_config, mock_encoder, mock_collection
    ):
        """Test predict_proba with matches below min_matches threshold."""
        # Setup collection to return only 2 matches above threshold
        mock_collection.query.return_value = {
            "distances": [[0.1, 0.15, 0.5, 0.6]],  # Similarities: 0.9, 0.85, 0.5, 0.4
            "documents": [["spam1", "spam2", "spam3", "spam4"]],
            "ids": [["spam_0", "spam_1", "spam_2", "spam_3"]],
        }

        model = VectorDbRagSpamModel(
            model_config, min_matches=3, similarity_threshold=0.7
        )
        model._collection = mock_collection
        model._encoder = mock_encoder

        result = model.predict_proba("test text")

        # Should have 2 matches above threshold (0.7), which is < min_matches (3)
        # Probability calculation: (2/3) * avg_similarity
        # avg_similarity = (0.9 + 0.85) / 2 = 0.875
        # probability = (2/3) * 0.875 = 0.583
        assert 0.0 <= result <= 1.0
        assert result < 0.7  # Should be lower probability

    def test_predict_proba_no_matches(
        self, model_config, mock_encoder, mock_collection
    ):
        """Test predict_proba with no matches above threshold."""
        # Setup collection to return matches below threshold
        mock_collection.query.return_value = {
            "distances": [
                [0.4, 0.5, 0.6]
            ],  # Similarities: 0.6, 0.5, 0.4 (all below 0.7)
            "documents": [["spam1", "spam2", "spam3"]],
            "ids": [["spam_0", "spam_1", "spam_2"]],
        }

        model = VectorDbRagSpamModel(model_config, similarity_threshold=0.7)
        model._collection = mock_collection
        model._encoder = mock_encoder

        result = model.predict_proba("test text")

        assert result == 0.0

    def test_predict_proba_empty_results(
        self, model_config, mock_encoder, mock_collection
    ):
        """Test predict_proba with empty query results."""
        mock_collection.query.return_value = {
            "distances": [],
            "documents": [],
            "ids": [],
        }

        model = VectorDbRagSpamModel(model_config)
        model._collection = mock_collection
        model._encoder = mock_encoder

        result = model.predict_proba("test text")

        assert result == 0.0

    def test_predict_proba_no_distances(
        self, model_config, mock_encoder, mock_collection
    ):
        """Test predict_proba when distances list is empty."""
        mock_collection.query.return_value = {
            "distances": [[]],
            "documents": [[]],
            "ids": [[]],
        }

        model = VectorDbRagSpamModel(model_config)
        model._collection = mock_collection
        model._encoder = mock_encoder

        result = model.predict_proba("test text")

        assert result == 0.0

    def test_predict_proba_auto_load(
        self, model_config, mock_encoder, mock_collection, temp_dir
    ):
        """Test predict_proba automatically loads model if not loaded."""
        # Create config file
        config_path = model_config.model_path.with_suffix(".json")
        config_data = {
            "collection_name": "test_collection",
            "db_path": str(temp_dir / "custom_db"),
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        mock_collection.query.return_value = {
            "distances": [[0.1, 0.2]],
            "documents": [["spam1", "spam2"]],
            "ids": [["spam_0", "spam_1"]],
        }

        with (
            patch(
                "spam_detector.src.vectordb.vectordb_rag_model.chromadb"
            ) as mock_chromadb,
            patch(
                "spam_detector.src.vectordb.vectordb_rag_model.SentenceTransformer"
            ) as mock_sentence_transformer,
        ):
            mock_chromadb.PersistentClient.return_value = MagicMock()
            mock_chromadb.PersistentClient.return_value.get_collection.return_value = (
                mock_collection
            )
            mock_sentence_transformer.return_value = mock_encoder

            model = VectorDbRagSpamModel(model_config)
            # Model not loaded yet
            assert model._collection is None

            result = model.predict_proba("test text")

            # Model should be loaded now
            assert model._collection is not None
            assert model._encoder is not None
            assert 0.0 <= result <= 1.0

    def test_predict_proba_raises_when_not_loaded(
        self, model_config, mock_encoder, mock_collection
    ):
        """Test predict_proba raises RuntimeError when load fails."""
        model = VectorDbRagSpamModel(model_config)
        model._collection = None
        model._encoder = None

        # Mock load to not set collection/encoder
        with patch.object(model, "load", return_value=None):
            with pytest.raises(RuntimeError, match="Model is not loaded"):
                model.predict_proba("test text")


class TestVectorDbRagSpamModelHelpers:
    """Test helper methods."""

    @patch("spam_detector.src.vectordb.vectordb_rag_model.chromadb")
    def test_get_client(self, mock_chromadb, model_config, temp_dir):
        """Test _get_client creates and caches client."""
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        model = VectorDbRagSpamModel(model_config)
        client1 = model._get_client()
        client2 = model._get_client()

        # Should return same client instance
        assert client1 is client2
        assert client1 is mock_client
        mock_chromadb.PersistentClient.assert_called_once()

    @patch("spam_detector.src.vectordb.vectordb_rag_model.chromadb")
    def test_get_collection_existing(self, mock_chromadb, model_config):
        """Test _get_collection retrieves existing collection."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        model = VectorDbRagSpamModel(model_config)
        collection = model._get_collection()

        assert collection is mock_collection
        mock_client.get_collection.assert_called_once_with(name="spam_examples")
        mock_client.create_collection.assert_not_called()

    @patch("spam_detector.src.vectordb.vectordb_rag_model.chromadb")
    def test_get_collection_new(self, mock_chromadb, model_config):
        """Test _get_collection creates new collection if not exists."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        model = VectorDbRagSpamModel(model_config)
        collection = model._get_collection()

        assert collection is mock_collection
        mock_client.create_collection.assert_called_once_with(name="spam_examples")

    @patch("spam_detector.src.vectordb.vectordb_rag_model.SentenceTransformer")
    def test_get_encoder(self, mock_sentence_transformer, model_config):
        """Test _get_encoder creates and caches encoder."""
        mock_encoder = MagicMock()
        mock_sentence_transformer.return_value = mock_encoder

        with patch.object(VectorDbRagSpamModel, "_device", return_value="cpu"):
            model = VectorDbRagSpamModel(model_config)
            encoder1 = model._get_encoder()
            encoder2 = model._get_encoder()

            # Should return same encoder instance
            assert encoder1 is encoder2
            assert encoder1 is mock_encoder
            mock_sentence_transformer.assert_called_once()
