from dataclasses import dataclass, field


@dataclass(slots=True)
class Posting:
    """Запис у списку інвертованого індексу."""

    doc_id: str
    freq: int = 1
    positions: list[int] = field(default_factory=lambda: list[int]())


@dataclass(slots=True)
class SearchResult:
    """Результат ранжованого пошуку."""

    doc_id: str
    score: float
    snippet: str = ""
