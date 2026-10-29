# pyright: basic
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from findex.fast_scoring import NumpyBM25Scorer, NumpyTFIDFScorer, VectorizedIndex
from findex.index import InvertedIndex, build_index
from findex.scoring import BM25Scorer
from findex.search import ranked_search
from findex.semantic import SemanticIndex, reciprocal_rank_fusion
from findex.web.app import app, app_state, get_corpus_map, get_index


def test_numpy_bm25_order_matches_lab3() -> None:
    """Перевіряє, що порядок результатів NumPy BM25 збігається з класичним скорером з Лаби 3."""
    idx = InvertedIndex()
    idx.add_document("doc1.txt", "python search engine python fast indexing")
    idx.add_document("doc2.txt", "python tutorial basics for beginners")
    idx.add_document("doc3.txt", "rust systems programming and memory safety")
    idx.add_document("doc4.txt", "advanced python search ranking with bm25 and python numpy")

    classic_scorer = BM25Scorer(k1=1.5, b=0.75)
    classic_hits = ranked_search(idx, "python search", scorer=classic_scorer, top_k=10)

    v_idx = VectorizedIndex.from_inverted_index(idx)
    numpy_scorer = NumpyBM25Scorer(k1=1.5, b=0.75)
    numpy_hits = numpy_scorer.score_query(v_idx, "python search", top_k=10)

    assert [h.doc_id for h in numpy_hits] == [h.doc_id for h in classic_hits]
    for n_hit, c_hit in zip(numpy_hits, classic_hits, strict=True):
        assert abs(n_hit.score - c_hit.score) < 1e-5


def test_numpy_tfidf_and_argpartition() -> None:
    """Перевіряє векторизований TF-IDF та вибір top-k через np.argpartition."""
    idx = InvertedIndex()
    for i in range(15):
        idx.add_document(f"d_{i}.txt", "common word " + ("target " * (i + 1)))

    v_idx = VectorizedIndex.from_inverted_index(idx)
    scorer = NumpyTFIDFScorer()
    top_3 = scorer.score_query(v_idx, "target", top_k=3)

    assert len(top_3) == 3
    assert top_3[0].doc_id == "d_14.txt"
    assert top_3[1].doc_id == "d_13.txt"
    assert top_3[2].doc_id == "d_12.txt"


def test_semantic_and_hybrid_rrf_paraphrased(tmp_path: Path) -> None:
    """Перевіряє семантичний пошук (.npy) та гібридне злиття RRF на перефразованих запитах."""
    docs = {
        "gil_doc.txt": "threads and multiprocessing concurrency in cpython interpreter",
        "web_doc.txt": "fastapi http server endpoint with uvicorn deployment",
        "math_doc.txt": "matrix multiplication and cosine distance with numpy arrays",
    }
    sem_idx = SemanticIndex.build_from_dict(docs)
    npy_file = tmp_path / "emb.npy"
    sem_idx.save(npy_file)

    loaded_sem = SemanticIndex.load(npy_file)
    # Перефразований запит українською мовою без жодного спільного англійського слова:
    sem_hits = loaded_sem.search("потоки та процеси", top_k=2)
    assert len(sem_hits) > 0
    assert sem_hits[0].doc_id == "gil_doc.txt"

    idx = InvertedIndex()
    for name, body in docs.items():
        idx.add_document(name, body)

    lex_hits = ranked_search(idx, "uvicorn", scorer=BM25Scorer(), top_k=3)
    hybrid_hits = reciprocal_rank_fusion(lex_hits, sem_hits, top_k=3)
    assert len(hybrid_hits) >= 2
    returned_ids = {h.doc_id for h in hybrid_hits}
    assert "web_doc.txt" in returned_ids
    assert "gil_doc.txt" in returned_ids


def test_web_api_search_modes() -> None:
    """Перевіряє параметр mode={lexical,semantic,hybrid} у веб-API."""
    docs = {
        "a.txt": "python concurrency gil threads",
        "b.txt": "fastapi web server http",
    }
    idx = InvertedIndex()
    for k, v in docs.items():
        idx.add_document(k, v)

    app.dependency_overrides[get_index] = lambda: idx
    app.dependency_overrides[get_corpus_map] = lambda: docs
    app_state.index = idx
    app_state.semantic_index = SemanticIndex.build_from_dict(docs)

    with TestClient(app) as client:
        for mode in ("lexical", "semantic", "hybrid"):
            res = client.get("/search", params={"q": "python", "mode": mode})
            assert res.status_code == 200
            assert len(res.json()["results"]) >= 1

    app.dependency_overrides.clear()
    app_state.index = None
    app_state.semantic_index = None


def test_benchmark_search_and_build_index(benchmark: Any, tmp_path: Path) -> None:
    """Регресійний тест pytest-benchmark для пошуку та індексації."""
    for i in range(20):
        (tmp_path / f"doc_{i}.txt").write_text(
            f"python search engine benchmark document number {i} with tokens",
            encoding="utf-8",
        )
    idx = build_index(tmp_path)
    v_idx = VectorizedIndex.from_inverted_index(idx)
    scorer = NumpyBM25Scorer()

    res = benchmark(scorer.score_query, v_idx, "python benchmark", 5)
    assert len(res) == 5
