import math
from typing import Protocol, runtime_checkable

from findex.index import InvertedIndex


@runtime_checkable
class Scorer(Protocol):
    """Протокол для взаємозамінних алгоритмів підрахунку балів релевантності."""

    def score_term(self, term: str, tf: int, doc_id: str, index: InvertedIndex) -> float:
        ...


class TfidfScorer:
    """Класичний ранжувальник TF-IDF: логарифмічна вага TF, помножена на IDF."""

    def score_term(self, term: str, tf: int, doc_id: str, index: InvertedIndex) -> float:
        if tf <= 0:
            return 0.0
        tf_weight = 1.0 + math.log(tf)
        return tf_weight * index.idf(term)


class BM25Scorer:
    """
    Ранжувальник Okapi BM25 із параметрами насичення частоти k1=1.5
    та штрафу за довжину документа b=0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score_term(self, term: str, tf: int, doc_id: str, index: InvertedIndex) -> float:
        if tf <= 0:
            return 0.0

        doc_len = index.docs[doc_id].length if doc_id in index.docs else index.avg_doc_length
        avg_len = index.avg_doc_length

        if avg_len == 0:
            len_norm = 1.0
        else:
            len_norm = 1.0 - self.b + self.b * (doc_len / avg_len)

        numerator = tf * (self.k1 + 1.0)
        denominator = tf + self.k1 * len_norm
        tf_component = numerator / denominator

        return index.idf(term) * tf_component