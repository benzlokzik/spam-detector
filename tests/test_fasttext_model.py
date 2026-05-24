from unittest.mock import patch

import pytest

pytest.importorskip("fasttext")

from spam_detector.core.base_model import ModelConfig  # noqa: E402
from spam_detector.fastspam.ft_model import FastTextSpamModel  # noqa: E402

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
    train_file = tmp_path / "train.txt"
    lines: list[str] = []
    for _ in range(6):
        lines += [f"__label__spam {t}" for t in SPAM]
        lines += [f"__label__ham {t}" for t in HAM]
    train_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cfg = ModelConfig(project_root=tmp_path, data_subdir=".", model_name="ft.bin")
    model = FastTextSpamModel(
        cfg,
        dim=16,
        epoch=25,
        wordNgrams=1,
        minn=2,
        maxn=4,
        loss="ova",
        quantize=False,
    )
    with patch(
        "spam_detector.fastspam.ft_model.get_fasttext_file",
        return_value=train_file,
    ):
        model.fit()
    return cfg


class TestFastTextSpamModel:
    def test_fit_persists_model(self, trained_config):
        assert trained_config.model_path.exists()

    def test_predict_proba_range(self, trained_config):
        model = FastTextSpamModel(trained_config, quantize=False)
        model.load()
        score = model.predict_proba("срочно кредит займ деньги дешево")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_spam_scores_higher_than_ham(self, trained_config):
        model = FastTextSpamModel(trained_config, quantize=False)
        model.load()
        spam = model.predict_proba("кредит займ деньги срочно дешево купить")
        ham = model.predict_proba("привет хорошая погода друзья вечером парк")
        assert spam > ham
