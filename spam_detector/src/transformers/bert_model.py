import json
from dataclasses import dataclass

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ..core.base_model import ModelConfig, SpamModel
from ..core.datasets import load_dataset


@dataclass
class BertTrainingConfig:
    model_name: str = "cointegrated/rubert-tiny2"
    max_length: int = 128
    batch_size: int = 32
    epochs: int = 1
    lr: float = 2e-5
    warmup_ratio: float = 0.1


class _TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]


class BertSpamModel(SpamModel):
    def __init__(
        self,
        cfg: ModelConfig,
        train_cfg: BertTrainingConfig | None = None,
    ) -> None:
        super().__init__(cfg)
        self.train_cfg = train_cfg or BertTrainingConfig()
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSequenceClassification | None = None

    @property
    def _model_dir(self) -> str:
        path = self.cfg.model_path
        if path.suffix:
            return str(path.with_suffix(""))
        return str(path)

    def _device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self) -> None:
        logger.info(f"Training BERT model ({self.train_cfg.model_name})...")
        texts, labels = load_dataset()
        targets = [int(label) for label in labels]
        dataset = _TextDataset(texts, targets)
        device = self._device()
        logger.info(f"Using device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(self.train_cfg.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.train_cfg.model_name,
            num_labels=2,
        )
        model.to(device)

        def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
            batch_texts = [item[0] for item in batch]
            batch_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.train_cfg.max_length,
                return_tensors="pt",
            )
            encoded["labels"] = batch_labels
            return {k: v.to(device) for k, v in encoded.items()}

        loader = DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.train_cfg.lr)
        total_steps = max(len(loader) * self.train_cfg.epochs, 1)
        warmup_steps = int(total_steps * self.train_cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        model.train()
        for epoch in range(self.train_cfg.epochs):
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.train_cfg.epochs}")
            for batch in pbar:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        model.save_pretrained(self._model_dir)
        tokenizer.save_pretrained(self._model_dir)
        meta = {"model_name": self.train_cfg.model_name}
        meta_path = self.cfg.model_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        self._model = model
        self._tokenizer = tokenizer
        logger.info(f"Model saved to {self._model_dir}")

    def load(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_dir)
        model.to(self._device())
        self._tokenizer = tokenizer
        self._model = model

    def predict_proba(self, text: str) -> float:
        if self._tokenizer is None or self._model is None:
            self.load()
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model is not loaded")
        device = self._device()
        self._model.eval()
        with torch.no_grad():
            encoded = self._tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=self.train_cfg.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]
            return float(probs[1].item())
