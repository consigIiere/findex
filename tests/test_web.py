# pyright: basic
import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from findex.index import InvertedIndex
from findex.web.app import app, app_state, get_corpus_map, get_index


@pytest.fixture()
def mock_index() -> InvertedIndex:
    idx = InvertedIndex()
    idx.add_document("doc1.txt", "python fastapi web search engine with bm25 ranking")
    idx.add_document("doc2.txt", "uvicorn async event loop and threadpool execution")
    idx.add_document("doc3.txt", "python concurrency gil threads and multiprocessing")
    return idx


@pytest.fixture()
def client(mock_index: InvertedIndex) -> TestClient:
    app.dependency_overrides[get_index] = lambda: mock_index
    app.dependency_overrides[get_corpus_map] = lambda: {
        "doc1.txt": "python fastapi web search engine with bm25 ranking",
        "doc2.txt": "uvicorn async event loop and threadpool execution",
        "doc3.txt": "python concurrency gil threads and multiprocessing",
    }
    app_state.index = mock_index
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app_state.index = None


def test_web_search_success(client: TestClient) -> None:
    """Перевіряє успішний пошук з валідацією та пагінацією."""
    response = client.get("/search", params={"q": "python", "k": 2, "scorer": "bm25", "page": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "python"
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["doc_id"] in {"doc1.txt", "doc3.txt"}


def test_web_search_validation_422(client: TestClient) -> None:
    """Перевіряє код 422 при порожньому запиті або невалідному параметрі."""
    res_empty = client.get("/search", params={"q": ""})
    assert res_empty.status_code == 422

    res_bad_scorer = client.get("/search", params={"q": "python", "scorer": "invalid"})
    assert res_bad_scorer.status_code == 422


def test_web_docs_found_and_404(client: TestClient) -> None:
    """Перевіряє отримання документа (200) та 404 для невідомого doc_id."""
    ok_res = client.get("/docs/doc1.txt")
    assert ok_res.status_code == 200
    assert ok_res.json()["doc_id"] == "doc1.txt"

    not_found_res = client.get("/docs/unknown_file.txt")
    assert not_found_res.status_code == 404


def test_web_health_503_when_index_not_loaded() -> None:
    """Перевіряє, що /health повертає 503, коли індекс не завантажений."""
    app.dependency_overrides.clear()
    app_state.index = None
    with TestClient(app, raise_server_exceptions=False) as raw_client:
        res = raw_client.get("/health")
        assert res.status_code == 503


def test_web_stats_and_html_ui(client: TestClient) -> None:
    """Перевіряє маршрут /stats та головну HTML-сторінку Jinja2."""
    stats_res = client.get("/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["num_docs"] == 3

    ui_res = client.get("/", params={"q": "python"})
    assert ui_res.status_code == 200
    assert "findex Web Search" in ui_res.text


def test_unexpected_exception_returns_500_without_stacktrace(client: TestClient) -> None:
    """Перевіряє, що неочікуваний виняток повертає 500 з request_id без витоку стека."""

    def broken_index() -> InvertedIndex:
        raise RuntimeError("Secret internal failure")

    app.dependency_overrides[get_index] = broken_index
    res = client.get("/stats")
    assert res.status_code == 500
    body = res.json()
    assert body["detail"] == "Internal Server Error"
    assert "request_id" in body
    assert "Secret internal failure" not in res.text


def test_20_concurrent_requests_def_vs_async(mock_index: InvertedIndex) -> None:
    """Вимірює 20 одночасних запитів на /search."""
    app.dependency_overrides[get_index] = lambda: mock_index

    async def run_batch() -> float:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            t0 = time.perf_counter()
            tasks = [ac.get("/search", params={"q": "python"}) for _ in range(20)]
            responses = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - t0
            assert all(r.status_code == 200 for r in responses)
            return elapsed

    duration = asyncio.run(run_batch())
    assert duration >= 0.0
    app.dependency_overrides.clear()
