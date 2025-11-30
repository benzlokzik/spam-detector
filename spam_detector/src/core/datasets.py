import pathlib
from collections.abc import Iterable, Iterator


def _iter_fasttext_lines(path: pathlib.Path) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as source:
        for raw in source:
            line = raw.strip()
            if not line.startswith("__label__"):
                continue
            parts = line.split(maxsplit=1)
            label = parts[0].replace("__label__", "", 1)
            text = parts[1] if len(parts) > 1 else ""
            yield label, text


def ensure_fasttext_files(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    resolved: list[pathlib.Path] = []
    for path in paths:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Training file not found: {p}")
        if not any(_iter_fasttext_lines(p)):
            raise ValueError(f"Training data in {p} has no fastText labels")
        resolved.append(p)
    if not resolved:
        raise FileNotFoundError("No training files provided")
    return resolved


def concatenate_fasttext_files(
    paths: Iterable[pathlib.Path],
    destination: pathlib.Path,
) -> None:
    dest = pathlib.Path(destination)
    with dest.open("w", encoding="utf-8") as target:
        for path in ensure_fasttext_files(paths):
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    target.write(line)


def load_fasttext_dataset(paths: Iterable[pathlib.Path]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for path in ensure_fasttext_files(paths):
        for label, text in _iter_fasttext_lines(path):
            labels.append(label)
            texts.append(text)
    return texts, labels


def dump_fasttext_dataset(
    path: pathlib.Path,
    texts: Iterable[str],
    labels: Iterable[str],
) -> None:
    dest = pathlib.Path(path)
    with dest.open("w", encoding="utf-8") as target:
        for text, label in zip(texts, labels, strict=False):
            clean_label = str(label).strip()
            clean_text = text.replace("\n", " ").strip()
            target.write(f"__label__{clean_label} {clean_text}\n")
