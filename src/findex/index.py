import argparse
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

from findex.models import DocMeta, Posting
from findex.reader import iter_documents
from findex.tokenizer import tokenize


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, list[Posting]] = defaultdict(list)
        self.docs: dict[str, DocMeta] = {}

    def add_document(self, doc_id: str, text: str) -> None:
        """Індексує один документ та оновлює списки входжень (postings)."""
        doc_tokens = list(tokenize(text))
        doc_length = len(doc_tokens)

        self.docs[doc_id] = DocMeta(doc_id=doc_id, length=doc_length)

        term_counts = Counter(doc_tokens)

        for term, tf in term_counts.items():
            self.postings[term].append(Posting(doc_id=doc_id, term_frequency=tf))

    @classmethod
    def build(cls, corpus_path: str | Path) -> "InvertedIndex":
        """Будує індекс за один прохід по корпусу за допомогою потокового генератора."""
        index = cls()
        for doc_id, text in iter_documents(corpus_path):
            index.add_document(doc_id, text)
        return index

    def get_postings(self, term: str) -> list[Posting]:
        """Повертає список входжень для заданого терміна."""
        normalized_term = term.casefold()
        return self.postings.get(normalized_term, [])

    def save(self, output_path: str | Path) -> None:
        """Зберігає індекс у бінарний файл за допомогою pickle."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, input_path: str | Path) -> "InvertedIndex":
        """
        Завантажує індекс із файлу pickle.

        """
        path = Path(input_path)
        with open(path, "rb") as f:
            index = pickle.load(f)
        if not isinstance(index, cls):
            raise TypeError("Завантажений файл не є валідним об'єктом InvertedIndex")
        return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Побудова та збереження Inverted Index")
    parser.add_argument("corpus_path", type=str, help="Шлях до директорії корпусу (наприклад, data)")
    parser.add_argument("-o", "--output", type=str, default="index.bin", help="Шлях для збереження індексу")
    args = parser.parse_args()

    print(f"Побудова індексу для корпусу '{args.corpus_path}'...")
    start_time = time.perf_counter()
    index = InvertedIndex.build(args.corpus_path)
    build_time = time.perf_counter() - start_time

    print(f"Індекс успішно побудовано за {build_time:.2f} с")
    print(f"Документів: {len(index.docs)} | Унікальних термінів: {len(index.postings)}")

    index.save(args.output)
    print(f"Індекс збережено у файл '{args.output}'")