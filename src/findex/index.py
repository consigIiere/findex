import pickle
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from findex.models import Posting
from findex.reader import read_corpus
from findex.tokenizer import tokenize


class InvertedIndex(Mapping[str, list[Posting]]):
    """Повнотекстовий інвертований індекс корпусу документів."""

    def __init__(self) -> None:
        self.postings: dict[str, list[Posting]] = defaultdict(list)
        self.docs: dict[str, int] = {}
        self.doc_lengths: dict[str, int] = {}

    def add_document(self, doc_id: str, text: str) -> None:
        """Додає документ до індексу з підрахунком частоти і позицій слів."""
        tokens = list(tokenize(text))
        self.doc_lengths[doc_id] = len(tokens)
        self.docs[doc_id] = len(tokens)

        positions_map: dict[str, list[int]] = defaultdict(list)
        for pos, term in enumerate(tokens):
            positions_map[term].append(pos)

        for term, positions in positions_map.items():
            self.postings[term].append(
                Posting(doc_id=doc_id, freq=len(positions), positions=positions)
            )

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_postings(self, term: str) -> list[Posting]:
        return self.postings.get(term.lower(), [])

    def __getitem__(self, term: str) -> list[Posting]:
        term_norm = term.lower()
        if term_norm not in self.postings:
            raise KeyError(term)
        return self.postings[term_norm]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, term: object) -> bool:
        if not isinstance(term, str):
            return False
        return term.lower() in self.postings

    def save(self, filepath: str | Path) -> None:
        """Серіалізує індекс у бінарний файл за допомогою pickle."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "postings": dict(self.postings),
            "docs": self.docs,
            "doc_lengths": self.doc_lengths,
        }
        with path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str | Path) -> "InvertedIndex":
        """Відновлює індекс із бінарного файлу."""
        path = Path(filepath)
        with path.open("rb") as f:
            data: dict[str, Any] = pickle.load(f)

        idx = cls()
        idx.postings = defaultdict(list, data.get("postings", {}))
        idx.docs = data.get("docs", {})
        idx.doc_lengths = data.get("doc_lengths", {})
        return idx

    @classmethod
    def build(cls, corpus_dir: str | Path) -> "InvertedIndex":
        """Будує індекс за шляхом до теки документів."""
        idx = cls()
        for doc_id, text in read_corpus(corpus_dir):
            idx.add_document(doc_id, text)
        return idx
