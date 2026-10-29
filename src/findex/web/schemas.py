from typing import Literal

from pydantic import BaseModel, Field


class SearchQueryParams(BaseModel):
    """Вхідна модель валідації параметрів пошуку."""

    q: str = Field(..., min_length=1, description="Пошуковий запит")
    k: int = Field(default=10, ge=1, le=100, description="Розмір сторінки")
    scorer: Literal["bm25", "tfidf"] = Field(default="bm25", description="Модель ранжування")
    page: int = Field(default=1, ge=1, description="Номер сторінки")


class SearchHitSchema(BaseModel):
    """Окремий результат пошуку на виході API."""

    doc_id: str
    score: float
    snippet: str


class SearchResponseSchema(BaseModel):
    """Відповідь маршруту /search з пагінацією."""

    query: str
    total: int
    page: int
    per_page: int
    total_pages: int
    results: list[SearchHitSchema]


class DocumentResponseSchema(BaseModel):
    """Відповідь маршруту /docs/{doc_id}."""

    doc_id: str
    length: int
    content: str


class TermFreqSchema(BaseModel):
    term: str
    freq: int


class StatsResponseSchema(BaseModel):
    """Відповідь маршруту /stats."""

    num_docs: int
    vocab_size: int
    avg_doc_length: float
    top_terms: list[TermFreqSchema]


class HealthResponseSchema(BaseModel):
    status: str
    index_loaded: bool
