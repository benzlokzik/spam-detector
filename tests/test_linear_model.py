from unittest.mock import patch

import pytest

pytest.importorskip("sklearn")

from spam_detector.core.base_model import ModelConfig  # noqa: E402
from spam_detector.linear.sk_model import SklearnSpamModel  # noqa: E402

SPAM = [
    "купить дешево кредит займ деньги срочно",
    "срочно деньги кредит займ онлайн дешево",
    "займ деньги срочно кредит без процентов",
    "дешево купить кредит займ деньги сейчас",
    "кредит займ деньги срочно онлайн заявка",
    "деньги срочно займ кредит дешево купить",
]
HAM = [
    "привет как дела хорошая погода сегодня",
    "сегодня хорошая погода пойдем гулять парк",
    "люблю читать книги вечером дома спокойно",
    "погода хорошая друзья пойдем гулять вместе",
    "вечером смотрел фильм дома было спокойно",
    "друзья встретились вечером парк хорошая погода",
]


@pytest.fixture
def trained_config(tmp_path):
    cfg = ModelConfig(project_root=tmp_path, data_subdir=".", model_name="sk.joblib")
    with patch(
        "spam_detector.linear.sk_model.load_dataset",
        return_value=(SPAM + HAM, [True] * len(SPAM) + [False] * len(HAM)),
    ):
        SklearnSpamModel(cfg).fit()
    return cfg


class TestSklearnSpamModel:
    def test_fit_persists_model(self, trained_config):
        assert trained_config.model_path.exists()

    def test_predict_proba_returns_probability(self, trained_config):
        model = SklearnSpamModel(trained_config)
        model.load()
        score = model.predict_proba("кредит займ деньги срочно")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_proba_auto_loads(self, trained_config):
        model = SklearnSpamModel(trained_config)
        assert isinstance(model.predict_proba("привет погода"), float)

    def test_spam_scores_higher_than_ham(self, trained_config):
        model = SklearnSpamModel(trained_config)
        model.load()
        spam = model.predict_proba("срочно кредит займ деньги дешево купить")
        ham = model.predict_proba("привет хорошая погода друзья вечером парк")
        assert spam > ham

    def test_fit_rejects_single_class(self, tmp_path):
        cfg = ModelConfig(project_root=tmp_path, data_subdir=".", model_name="x.joblib")
        with (
            patch(
                "spam_detector.linear.sk_model.load_dataset",
                return_value=(["spam one", "spam two"], [True, True]),
            ),
            pytest.raises(ValueError, match="both classes"),
        ):
            SklearnSpamModel(cfg).fit()
