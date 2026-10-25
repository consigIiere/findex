from dataclasses import dataclass, field


@dataclass(slots=True)
class DocMeta:
    """Метадані документа: ідентифікатор та загальна довжина у токенах."""
    doc_id: str
    length: int


@dataclass(slots=True)
class Posting:
    """Входження терма в документ зі збереженням індексів позицій слів."""
    doc_id: str
    freq: int
    positions: list[int] = field(default_factory=list)


@dataclass(order=True, slots=True)
class SearchResult:
    """
    Результат ранжованого пошуку.
    """
    score: float
    doc_id: str = field(compare=True)
    snippet: str = field(default="", compare=False)