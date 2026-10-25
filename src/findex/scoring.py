import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from findex.index import InvertedIndex


class BaseScorer(ABC):
    """Базовий клас для функцій ранжування документів."""

    @abstractmethod
    def score_document(
        self, index: InvertedIndex, doc_id: str, query_terms: Sequence[str]
    ) -> float:
        """Обчислює релевантність документа до списку термів запиту."""


class BM25Scorer(BaseScorer):
    """Ранжування за алгоритмом Okapi BM25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1: float = k1
        self.b: float = b

    def idf(self, index: InvertedIndex, term: str) -> float:
        """Обчислення Inverse Document Frequency для BM25."""
        num_docs = index.num_docs
        postings = index.get_postings(term)
        doc_freq = len(postings)
        if doc_freq == 0:
            return 0.0
        # Стандартна формула BM25 IDF
        return math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def score_document(
        self, index: InvertedIndex, doc_id: str, query_terms: Sequence[str]
    ) -> float:
        total_score: float = 0.0
        doc_len: int = index.doc_lengths.get(doc_id, 0)
        avg_len: float = index.avg_doc_length

        if avg_len == 0.0:
            return 0.0

        len_norm: float = 1.0 - self.b + self.b * (doc_len / avg_len)

        for term in query_terms:
            term_norm = term.lower()
            postings = index.get_postings(term_norm)
            doc_posting = next((p for p in postings if p.doc_id == doc_id), None)
            if doc_posting is None:
                continue

            tf: int = doc_posting.freq
            term_idf: float = self.idf(index, term_norm)
            numerator: float = tf * (self.k1 + 1.0)
            denominator: float = tf + self.k1 * len_norm

            total_score += term_idf * (numerator / denominator)

        return total_score


class TfidfScorer(BaseScorer):
    """Ранжування за класичною метрикою TF-IDF."""

    def idf(self, index: InvertedIndex, term: str) -> float:
        num_docs = index.num_docs
        postings = index.get_postings(term)
        doc_freq = len(postings)
        if doc_freq == 0 or num_docs == 0:
            return 0.0
        return math.log((num_docs + 1.0) / (doc_freq + 1.0)) + 1.0

    def score_document(
        self, index: InvertedIndex, doc_id: str, query_terms: Sequence[str]
    ) -> float:
        total_score: float = 0.0
        doc_len: int = index.doc_lengths.get(doc_id, 0)
        if doc_len == 0:
            return 0.0

        for term in query_terms:
            term_norm = term.lower()
            postings = index.get_postings(term_norm)
            doc_posting = next((p for p in postings if p.doc_id == doc_id), None)
            if doc_posting is None:
                continue

            tf: float = doc_posting.freq / doc_len
            term_idf: float = self.idf(index, term_norm)
            total_score += tf * term_idf

        return total_score
