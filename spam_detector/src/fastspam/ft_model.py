import pathlib
from tempfile import NamedTemporaryFile

import fasttext

from ..core.base_model import ModelConfig, SpamModel
from ..core.datasets import concatenate_fasttext_files, ensure_fasttext_files


class FastTextSpamModel(SpamModel):
    def __init__(
        self,
        cfg: ModelConfig,
        dim=64,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        minn=2,
        maxn=4,
        loss="ova",
        quantize=True,
        qnorm=True,
        retrain=True,
        cutoff=100000,
    ) -> None:
        super().__init__(cfg)
        self.params = {
            "dim": dim,
            "lr": lr,
            "epoch": epoch,
            "wordNgrams": wordNgrams,
            "minn": minn,
            "maxn": maxn,
            "loss": loss,
        }
        self.quantize = quantize
        self.qnorm = qnorm
        self.retrain = retrain
        self.cutoff = cutoff
        self._m: fasttext.FastText._FastText | None = None

    def fit(self) -> None:
        paths = ensure_fasttext_files(self.cfg.train_paths())
        with NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
            tmp_path = pathlib.Path(tmp.name)
        concatenate_fasttext_files(paths, tmp_path)
        model = fasttext.train_supervised(input=str(tmp_path), **self.params)
        if self.quantize:
            model.quantize(
                input=str(tmp_path),
                qnorm=self.qnorm,
                retrain=self.retrain,
                cutoff=self.cutoff,
            )
        model.save_model(str(self.cfg.model_path))
        tmp_path.unlink(missing_ok=True)
        self._m = model

    def load(self) -> None:
        self._m = fasttext.load_model(str(self.cfg.model_path))

    def predict_proba(self, text: str) -> float:
        if self._m is None:
            self.load()
        if self._m is None:
            raise RuntimeError("Model is not loaded")
        processed = text.lower().replace("\n", "\t").strip()
        labels, probs = self._m.predict(processed, k=2)
        return max(
            (p for l, p in zip(labels, probs, strict=False) if l == "__label__spam"),
            default=0.0,
        )


if __name__ == "__main__":
    cfg = ModelConfig()
    model = FastTextSpamModel(cfg)
    model.fit()
    print(
        model.predict_proba("Срочно работа в Москве! З/п 150 000 руб! Писать на в лс"),
    )
    print(
        model.predict_proba(
            "блять как же заебали кустовые эйдоры сука ненавижу их всех",
        ),
    )
    print(
        model.predict_proba(
            "qq",
        ),
    )
