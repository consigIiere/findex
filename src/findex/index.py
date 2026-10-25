from collections.abc import Mapping
from contextlib import contextmanager
from functools import cached_property
import math
import os
import pickle
from typing import Generator, Iterator

from findex.models import DocMeta, Posting
from findex.reader import iter_documents
from findex.tokenizer import tokenize


class InvertedIndex(Mapping):
    """Інвертований індекс з підтримкою інтерфейсу Mapping (dunder-методи)."""

    def __init__(self) -> None:
        self._postings: dict[str, list[Posting]] = {}
        self._docs: dict[str, DocMeta] = {}


    def __len__(self) -> int:
        """Повертає кількість унікальних термів в індексі: len(index)."""
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        """Ітерація по термах індексу: for term in index."""
        return iter(self._postings)

    def __getitem__(self, term: str) -> list[Posting]:
        """Отримання списку входжень: index[term]. Викидає KeyError, якщо терм відсутній."""
        normalized = term.casefold()
        if normalized not in self._postings:
            raise KeyError(f"Терм '{term}' відсутній в індексі")
        return self._postings[normalized]

    def __contains__(self, term: object) -> bool:
        """Перевірка наявності слова: 'word' in index."""
        if not isinstance(term, str):
            return False
        return term.casefold() in self._postings

    def __repr__(self) -> str:
        return f"InvertedIndex(num_docs={self.num_docs}, num_terms={len(self)})"


    @property
    def num_docs(self) -> int:
        """Загальна кількість опрацьованих документів."""
        return len(self._docs)

    @property
    def docs(self) -> dict[str, DocMeta]:
        return self._docs

    @property
    def postings(self) -> dict[str, list[Posting]]:
        return self._postings

    @cached_property
    def avg_doc_length(self) -> float:
        """
        Середня довжина документів у корпусі.
        Значення кешується автоматично завдяки @cached_property.
        """
        if not self._docs:
            return 0.0
        total_len = sum(doc.length for doc in self._docs.values())
        return total_len / len(self._docs)

    def df(self, term: str) -> int:
        """Document Frequency: кількість документів, у яких є зазначений терм."""
        postings = self._postings.get(term.casefold())
        return len(postings) if postings else 0

    def idf(self, term: str) -> float:
        """Обчислення Inverse Document Frequency за сгладженою формулою."""
        doc_freq = self.df(term)
        if doc_freq == 0:
            return 0.0
        return math.log(1.0 + (self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def get_postings(self, term: str) -> list[Posting]:
        """Безпечне отримання списку постингів (повертає [] якщо терм не знайдено)."""
        return self._postings.get(term.casefold(), [])

    def add_document(self, doc_id: str, text: str) -> None:
        """Індексує документ, обраховуючи точні координати та частоти слів."""
        tokens = list(tokenize(text))
        self._docs[doc_id] = DocMeta(doc_id=doc_id, length=len(tokens))

        term_positions: dict[str, list[int]] = {}
        for pos, token in enumerate(tokens):
            term_positions.setdefault(token, []).append(pos)

        for term, positions in term_positions.items():
            posting = Posting(doc_id=doc_id, freq=len(positions), positions=positions)
            self._postings.setdefault(term, []).append(posting)

    @classmethod
    def build(cls, corpus_path: str) -> "InvertedIndex":
        """Фабричний метод побудови індексу з файлів заданого корпусу."""
        instance = cls()
        for doc_id, text in iter_documents(corpus_path):
            instance.add_document(doc_id, text)
        return instance


    def save(self, filepath: str) -> None:
        """Зберігає структури індексу у бінарний файл за допомогою pickle."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump((self._docs, self._postings), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> "InvertedIndex":
        """Фабричний метод завантаження індексу з бінарного файлу."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл індексу не знайдено: {filepath}")
        instance = cls()
        with open(filepath, "rb") as f:
            instance._docs, instance._postings = pickle.load(f)
        return instance



@contextmanager
def open_index(path: str) -> Generator[InvertedIndex, None, None]:
    """
    Контекстний менеджер для безпечного завантаження та вивільнення ресурсів індексу:
    with open_index("index.bin") as idx:
        ...
    Гарантовано очищує пам'ять при виході з блоку або виникненні винятку.
    """
    index = InvertedIndex.load(path)
    try:
        yield index
    finally:
        index._postings.clear()
        index._docs.clear()

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу")
    parser.add_argument("corpus_path", help="Шлях до теки з документами")
    parser.add_argument("-o", "--output", default="index.bin", help="Шлях до вихідного бінарного файлу")

    args = parser.parse_args()

    print(f"Побудова індексу для корпусу '{args.corpus_path}'...")
    start_time = time.perf_counter()
    index = InvertedIndex.build(args.corpus_path)
    elapsed = time.perf_counter() - start_time

    print(f"Індекс успішно побудовано за {elapsed:.2f} с")
    print(f"Документів: {index.num_docs} | Унікальних термінів: {len(index)}")

    index.save(args.output)
    print(f"Індекс збережено у файл '{args.output}'")