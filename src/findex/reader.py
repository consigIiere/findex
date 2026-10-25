import os
from collections.abc import Iterator
from pathlib import Path


def read_corpus(corpus_dir: str | Path) -> Iterator[tuple[str, str]]:
    """Рекурсивно або послідовно читає всі txt-файли з теки."""
    path = Path(corpus_dir)
    if not path.is_dir():
        return

    for root, _, files in os.walk(path):
        for file in sorted(files):
            if file.endswith(".txt"):
                file_path = Path(root) / file
                try:
                    content = file_path.read_text(encoding="utf-8")
                    yield file, content
                except UnicodeDecodeError:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    yield file, content
