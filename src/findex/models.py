from dataclasses import dataclass


@dataclass(slots=True)
class DocMeta:
    doc_id: str
    length: int


@dataclass(slots=True)
class Posting:
    doc_id: str
    term_frequency: int