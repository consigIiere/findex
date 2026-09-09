from pathlib import Path
from typing import Iterator, Tuple


def iter_documents(corpus_path: str | Path) -> Iterator[Tuple[str, str]]:

    path = Path(corpus_path)

    if not path.exists():
        raise FileNotFoundError("Корпус не знайдено за шляхом:", path)

    for file_path in path.rglob("*.txt"):
        if file_path.is_file():
            with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                doc_id = str(file_path.relative_to(path))
                yield doc_id, content