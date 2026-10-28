import json
import os
from collections.abc import Iterator
from pathlib import Path


def iter_documents(source: str | Path) -> Iterator[tuple[str, str]]:
    """
    Читає документи потоково (не тримаючи всі в пам'яті):
    - якщо source — це .jsonl файл, читає його пострічково;
    - якщо source — це тека, рекурсивно читає .txt та .jsonl файли в ній.
    """
    path = Path(source)
    if path.is_file():
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    doc_id = str(record.get("url") or record.get("id") or "")
                    title = str(record.get("title") or "")
                    text = str(record.get("text") or "")
                    content = f"{title} {text}".strip() if title else text
                    if doc_id:
                        yield doc_id, content
        elif path.suffix.lower() == ".txt":
            content = path.read_text(encoding="utf-8", errors="ignore")
            yield path.name, content
        return

    if not path.is_dir():
        return

    for root, _, files in os.walk(path):
        for file in sorted(files):
            file_path = Path(root) / file
            if file.endswith(".txt"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    yield file, content
                except UnicodeDecodeError:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    yield file, content
            elif file.endswith(".jsonl"):
                yield from iter_documents(file_path)


def read_corpus(corpus_dir: str | Path) -> Iterator[tuple[str, str]]:
    """Зворотна сумісність із попередніми лабораторними."""
    return iter_documents(corpus_dir)