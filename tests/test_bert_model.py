from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402

from spam_detector.core.base_model import ModelConfig  # noqa: E402
from spam_detector.transformers.bert_model import (  # noqa: E402
    BertSpamModel,
    BertTrainingConfig,
)

REPO_ID = "benzlokzik/spam-detector-bert"


@patch("spam_detector.transformers.bert_model.AutoModelForSequenceClassification")
@patch("spam_detector.transformers.bert_model.AutoTokenizer")
class TestBertSpamModel:
    def test_load_passes_pretrained_repo_id_through(self, mock_tokenizer, mock_model):
        model = BertSpamModel(ModelConfig(), BertTrainingConfig(pretrained=REPO_ID))
        model.load()
        mock_tokenizer.from_pretrained.assert_called_once_with(REPO_ID)
        mock_model.from_pretrained.assert_called_once_with(REPO_ID)

    def test_load_falls_back_to_local_dir(self, mock_tokenizer, mock_model, tmp_path):
        cfg = ModelConfig(project_root=tmp_path, data_subdir=".", model_name="bert.bin")
        BertSpamModel(cfg).load()
        used = mock_model.from_pretrained.call_args[0][0]
        assert used != REPO_ID
        assert used.endswith("bert")

    def test_predict_proba_returns_spam_probability(self, mock_tokenizer, mock_model):
        tokenizer = mock_tokenizer.from_pretrained.return_value
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        net = mock_model.from_pretrained.return_value
        net.return_value = MagicMock(logits=torch.tensor([[0.1, 2.5]]))

        model = BertSpamModel(ModelConfig(), BertTrainingConfig(pretrained=REPO_ID))
        model.load()
        score = model.predict_proba("купить дешево кредит")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.5
