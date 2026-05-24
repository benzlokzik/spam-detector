import pytest

pytest.importorskip("sklearn")
pytest.importorskip("fasttext")

from spam_detector import get_spam_model  # noqa: E402
from spam_detector.fastspam.ft_model import FastTextSpamModel  # noqa: E402
from spam_detector.linear.sk_model import SklearnSpamModel  # noqa: E402


class TestGetSpamModel:
    def test_sklearn_backend(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "sklearn")
        monkeypatch.setenv("MODEL_FILENAME", "m.joblib")
        assert isinstance(get_spam_model(), SklearnSpamModel)

    def test_fasttext_backend(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "fasttext")
        assert isinstance(get_spam_model(), FastTextSpamModel)

    def test_default_backend_is_fasttext(self, monkeypatch):
        monkeypatch.delenv("MODEL_BACKEND", raising=False)
        assert isinstance(get_spam_model(), FastTextSpamModel)
