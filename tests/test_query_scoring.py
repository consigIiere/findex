from pathlib import Path

import pytest

from findex.index import InvertedIndex
from findex.query import And, Not, Or, Phrase, QueryParser, Term
from findex.scoring import BM25Scorer, TfidfScorer
from findex.search import ranked_search


@pytest.mark.parametrize(
    ("query_str", "expected_type"),
    [
        ("python", Term),
        ('"quick brown"', Phrase),
        ("python AND rust", And),
        ("cat OR dog", Or),
        ("NOT bird", Not),
    ],
)
def test_query_parser_ast_nodes(query_str: str, expected_type: type[object]) -> None:
    parser = QueryParser(query_str)
    ast = parser.parse()
    assert isinstance(ast, expected_type)


def test_query_parser_nested_parentheses() -> None:
    parser = QueryParser("love AND (romeo OR juliet) NOT death")
    ast = parser.parse()
    assert isinstance(ast, And)


def test_scoring_bm25_vs_tfidf(sample_index: InvertedIndex, sample_corpus: Path) -> None:
    res_bm25 = ranked_search(
        sample_index, "python", BM25Scorer(), top_k=2, corpus_dir=str(sample_corpus)
    )
    res_tfidf = ranked_search(
        sample_index, "python", TfidfScorer(), top_k=2, corpus_dir=str(sample_corpus)
    )

    assert len(res_bm25) > 0
    assert len(res_tfidf) > 0
    assert res_bm25[0].score > 0
    assert res_tfidf[0].score > 0


def test_phrase_search(sample_index: InvertedIndex, sample_corpus: Path) -> None:
    results = ranked_search(
        sample_index, '"inverted index"', BM25Scorer(), top_k=5, corpus_dir=str(sample_corpus)
    )
    assert len(results) == 1
    assert results[0].doc_id == "doc_2.txt"


def test_not_query_search(sample_index: InvertedIndex, sample_corpus: Path) -> None:
    results = ranked_search(
        sample_index, "python NOT inverted", BM25Scorer(), top_k=5, corpus_dir=str(sample_corpus)
    )
    doc_ids = [r.doc_id for r in results]
    assert "doc_1.txt" in doc_ids
    assert "doc_2.txt" not in doc_ids
