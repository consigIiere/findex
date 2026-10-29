# pyright: basic
import logging
import math
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import findex.scoring as scoring_mod
from findex.index import InvertedIndex
from findex.scoring import BaseScorer, BM25Scorer
from findex.search import SearchResult, ranked_search
from findex.semantic import SemanticIndex, reciprocal_rank_fusion
from findex.web.config import Settings, get_settings
from findex.web.schemas import (
    DocumentResponseSchema,
    HealthResponseSchema,
    SearchHitSchema,
    SearchResponseSchema,
    StatsResponseSchema,
    TermFreqSchema,
)

logger = logging.getLogger("findex.web")
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class AppState:
    """Контейнер стану індексу та ембедінгів."""

    def __init__(self) -> None:
        self.index: InvertedIndex | None = None
        self.semantic_index: SemanticIndex | None = None
        self.corpus_dir: Path = Path("data")
        self.documents_cache: dict[str, str] = {}


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Завантажує індекс один раз під час старту застосунку."""
    settings: Settings = get_settings()
    app_state.corpus_dir = settings.corpus_dir

    if settings.index_path.is_file():
        try:
            app_state.index = InvertedIndex.load(settings.index_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Не вдалося завантажити індекс: %s", exc)
            app_state.index = None
    else:
        app_state.index = None

    if settings.corpus_dir.is_dir():
        for file_path in settings.corpus_dir.glob("**/*"):
            if file_path.is_file():
                app_state.documents_cache[file_path.name] = file_path.read_text(
                    encoding="utf-8", errors="ignore"
                )
    if app_state.documents_cache:
        app_state.semantic_index = SemanticIndex.build_from_dict(app_state.documents_cache)
    yield
    app_state.index = None
    app_state.semantic_index = None


app = FastAPI(
    title="findex Web Search API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_and_error_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = req_id
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    except Exception:
        logger.exception("Неочікуваний виняток під час обробки запиту [request_id=%s]", req_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "request_id": req_id},
            headers={"X-Request-ID": req_id},
        )


def get_index() -> InvertedIndex:
    if app_state.index is None:
        raise HTTPException(status_code=503, detail="Індекс ще не завантажено")
    return app_state.index


def get_corpus_map() -> dict[str, str]:
    return app_state.documents_cache


def _resolve_scorer(name: str) -> BaseScorer:
    if name.lower() == "tfidf":
        tfidf_cls = getattr(scoring_mod, "TFIDFScorer", None) or getattr(
            scoring_mod, "TfIdfScorer", BM25Scorer
        )
        return tfidf_cls()
    return BM25Scorer()


@app.get("/health", response_model=HealthResponseSchema)
def health_check(idx: Annotated[InvertedIndex, Depends(get_index)]) -> HealthResponseSchema:
    return HealthResponseSchema(status="ok", index_loaded=idx is not None)


@app.get("/search", response_model=SearchResponseSchema)
def search_endpoint(
    idx: Annotated[InvertedIndex, Depends(get_index)],
    q: Annotated[str, Query(min_length=1, description="Пошуковий запит")],
    k: Annotated[int, Query(ge=1, le=100, description="Кількість результатів на сторінці")] = 10,
    scorer: Annotated[Literal["bm25", "tfidf"], Query(description="Алгоритм ранжування")] = "bm25",
    mode: Annotated[
        Literal["lexical", "semantic", "hybrid"],
        Query(description="Режим пошуку: lexical, semantic або hybrid (RRF)"),
    ] = "lexical",
    page: Annotated[int, Query(ge=1, description="Номер сторінки")] = 1,
) -> SearchResponseSchema:
    """Синхронний маршрут пошуку з підтримкою lexical, semantic та hybrid (RRF)."""
    scorer_obj = _resolve_scorer(scorer)
    corpus_dir = app_state.corpus_dir if app_state.corpus_dir.is_dir() else None

    lexical_results: list[SearchResult] = []
    if mode in ("lexical", "hybrid"):
        lexical_results = ranked_search(
            index=idx,
            query_str=q,
            scorer=scorer_obj,
            top_k=1000,
            corpus_dir=corpus_dir,
        )

    semantic_results: list[SearchResult] = []
    if mode in ("semantic", "hybrid"):
        sem_idx = app_state.semantic_index
        if sem_idx is None:
            corpus_map = get_corpus_map()
            if corpus_map:
                sem_idx = SemanticIndex.build_from_dict(corpus_map)
        if sem_idx is not None:
            semantic_results = sem_idx.search(q, top_k=1000)

    if mode == "lexical":
        all_results = lexical_results
    elif mode == "semantic":
        all_results = semantic_results
    else:
        all_results = reciprocal_rank_fusion(lexical_results, semantic_results, top_k=1000)

    total = len(all_results)
    total_pages = max(1, math.ceil(total / k))
    start = (page - 1) * k
    end = start + k
    page_items = all_results[start:end]

    hits = [
        SearchHitSchema(
            doc_id=item.doc_id,
            score=round(float(item.score), 4),
            snippet=item.snippet or "",
        )
        for item in page_items
    ]

    return SearchResponseSchema(
        query=q,
        total=total,
        page=page,
        per_page=k,
        total_pages=total_pages,
        results=hits,
    )


@app.get("/docs/{doc_id}", response_model=DocumentResponseSchema)
def get_document(
    doc_id: str,
    idx: Annotated[InvertedIndex, Depends(get_index)],
    corpus_map: Annotated[dict[str, str], Depends(get_corpus_map)],
) -> DocumentResponseSchema:
    if doc_id not in idx.doc_lengths:
        raise HTTPException(status_code=404, detail=f"Документ '{doc_id}' не знайдено")

    content = corpus_map.get(doc_id, f"[Текст документа {doc_id}]")
    return DocumentResponseSchema(
        doc_id=doc_id,
        length=idx.doc_lengths[doc_id],
        content=content,
    )


@app.get("/stats", response_model=StatsResponseSchema)
def get_stats(idx: Annotated[InvertedIndex, Depends(get_index)]) -> StatsResponseSchema:
    postings_map = getattr(idx, "postings", None) or getattr(idx, "index", {})
    term_counts = [(term, sum(p.freq for p in plist)) for term, plist in postings_map.items()]
    top_10 = sorted(term_counts, key=lambda x: x[1], reverse=True)[:10]
    avg_len = getattr(idx, "avg_doc_length", None)
    if avg_len is None:
        avg_len = getattr(idx, "avg_doc_len", 0.0)
    num_docs = getattr(idx, "num_docs", len(idx.doc_lengths))

    return StatsResponseSchema(
        num_docs=int(num_docs),
        vocab_size=len(postings_map),
        avg_doc_length=round(float(avg_len), 2),
        top_terms=[TermFreqSchema(term=t, freq=f) for t, f in top_10],
    )


@app.get("/", response_class=HTMLResponse)
def web_ui(
    request: Request,
    q: str = "",
    scorer: Literal["bm25", "tfidf"] = "bm25",
    mode: Literal["lexical", "semantic", "hybrid"] = "lexical",
    page: int = 1,
    k: int = 5,
) -> HTMLResponse:
    results_data: SearchResponseSchema | None = None
    error_msg: str | None = None

    if q.strip() and app_state.index is not None:
        try:
            results_data = search_endpoint(
                idx=app_state.index,
                q=q.strip(),
                k=k,
                scorer=scorer,
                mode=mode,
                page=max(1, page),
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "q": q,
            "scorer": scorer,
            "mode": mode,
            "page": page,
            "k": k,
            "data": results_data,
            "error": error_msg,
            "index_ready": app_state.index is not None,
        },
    )
