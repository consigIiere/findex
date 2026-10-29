# pyright: basic
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from findex.index import InvertedIndex
from findex.search import SearchResult
from findex.tokenizer import tokenize


@dataclass(slots=True)
class VectorizedIndex:
    """Компактне представлення індексу на базі масивів NumPy int32."""

    doc_ids: list[str]
    doc_to_idx: dict[str, int]
    doc_lengths: npt.NDArray[np.int32]
    avg_doc_length: float
    # term -> масив форми (M, 2) з dtype=int32: колонка 0 = doc_idx, колонка 1 = freq
    postings_np: dict[str, npt.NDArray[np.int32]]

    @classmethod
    def from_inverted_index(cls, index: InvertedIndex) -> "VectorizedIndex":
        """Конвертує стандартний InvertedIndex у масиви NumPy int32."""
        doc_ids = sorted(index.doc_lengths.keys())
        doc_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        lengths_arr = np.array([index.doc_lengths[d] for d in doc_ids], dtype=np.int32)
        avg_len = float(lengths_arr.mean()) if len(lengths_arr) > 0 else 0.0

        postings_map = getattr(index, "postings", None) or getattr(index, "index", {})
        postings_np: dict[str, npt.NDArray[np.int32]] = {}

        for term, plist in postings_map.items():
            pairs = [(doc_to_idx[p.doc_id], p.freq) for p in plist if p.doc_id in doc_to_idx]
            if pairs:
                postings_np[term] = np.array(pairs, dtype=np.int32)
            else:
                postings_np[term] = np.empty((0, 2), dtype=np.int32)

        return cls(
            doc_ids=doc_ids,
            doc_to_idx=doc_to_idx,
            doc_lengths=lengths_arr,
            avg_doc_length=avg_len,
            postings_np=postings_np,
        )


def top_k_argpartition(
    scores: npt.NDArray[np.float64], doc_ids: list[str], top_k: int
) -> list[SearchResult]:
    """Вибирає top-k документів за допомогою np.argpartition за O(N)."""
    nonzero_mask = scores > 0.0
    valid_indices = np.flatnonzero(nonzero_mask)
    if valid_indices.size == 0:
        return []

    valid_scores = scores[valid_indices]
    k = min(top_k, valid_indices.size)

    if valid_indices.size > k:
        part_idx = np.argpartition(-valid_scores, k - 1)[:k]
        top_doc_indices = valid_indices[part_idx]
    else:
        top_doc_indices = valid_indices

    sorted_order = sorted(
        top_doc_indices.tolist(),
        key=lambda idx: (-float(scores[idx]), doc_ids[idx]),
    )

    return [
        SearchResult(doc_id=doc_ids[idx], score=float(scores[idx]), snippet="")
        for idx in sorted_order[:k]
    ]


class NumpyBM25Scorer:
    """Векторизований BM25 на базі NumPy ufunc та broadcasting."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score_query(
        self, v_index: VectorizedIndex, query: str, top_k: int = 10
    ) -> list[SearchResult]:
        n_docs = len(v_index.doc_ids)
        if n_docs == 0:
            return []

        terms = list(tokenize(query))
        scores = np.zeros(n_docs, dtype=np.float64)
        avgdl = v_index.avg_doc_length if v_index.avg_doc_length > 0 else 1.0

        matched_Counts = np.zeros(n_docs, dtype=np.int32)
        unique_terms = list(dict.fromkeys(terms))
        for term in unique_terms:
            arr = v_index.postings_np.get(term)
            if arr is None or arr.shape[0] == 0:
                return []
            matched_Counts[arr[:, 0]] += 1

            doc_idx = arr[:, 0]
            tf = arr[:, 1].astype(np.float64)
            df = float(arr.shape[0])

            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            dl = v_index.doc_lengths[doc_idx].astype(np.float64)

            denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
            term_scores = idf * (tf * (self.k1 + 1.0)) / denom
            scores[doc_idx] += term_scores

        scores = np.where(matched_Counts == len(unique_terms), scores, 0.0)
        return top_k_argpartition(scores, v_index.doc_ids, top_k)


class NumpyTFIDFScorer:
    """Векторизований TF-IDF на базі NumPy ufunc."""

    def score_query(
        self, v_index: VectorizedIndex, query: str, top_k: int = 10
    ) -> list[SearchResult]:
        n_docs = len(v_index.doc_ids)
        if n_docs == 0:
            return []

        terms = list(tokenize(query))
        scores = np.zeros(n_docs, dtype=np.float64)

        for term in terms:
            arr = v_index.postings_np.get(term)
            if arr is None or arr.shape[0] == 0:
                continue

            doc_idx = arr[:, 0]
            tf = arr[:, 1].astype(np.float64)
            df = float(arr.shape[0])

            idf = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            term_scores = (1.0 + np.log(tf)) * idf
            scores[doc_idx] += term_scores

        return top_k_argpartition(scores, v_index.doc_ids, top_k)
