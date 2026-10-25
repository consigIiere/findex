from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

from findex.reader import read_corpus
from findex.tokenizer import tokenize


class CorpusStats(TypedDict):
    """Типізована структура результатів аналізу корпусу."""

    name: str
    num_docs: int
    num_tokens: int
    num_unique_tokens: int
    avg_tokens_per_doc: float
    top_tokens: list[tuple[str, int]]


def compute_tokens_stats(name: str, doc_tokens: list[list[str]]) -> CorpusStats:
    """Обчислює зведену статистику за списком токенізованих документів."""
    num_docs: int = len(doc_tokens)
    vocab: Counter[str] = Counter()
    total_tokens: int = 0

    for tokens in doc_tokens:
        total_tokens += len(tokens)
        vocab.update(tokens)

    num_unique_tokens: int = len(vocab)
    avg_tokens: float = total_tokens / num_docs if num_docs > 0 else 0.0
    top_tokens: list[tuple[str, int]] = vocab.most_common(10)

    return CorpusStats(
        name=name,
        num_docs=num_docs,
        num_tokens=total_tokens,
        num_unique_tokens=num_unique_tokens,
        avg_tokens_per_doc=round(avg_tokens, 2),
        top_tokens=top_tokens,
    )


def analyze_corpus(corpus_dir: str | Path) -> CorpusStats:
    """Зчитує корпус та рахує загальну статистику слів."""
    all_tokens: list[list[str]] = []
    for _doc_id, text in read_corpus(corpus_dir):
        doc_tokens: list[str] = list(tokenize(text))
        all_tokens.append(doc_tokens)

    return compute_tokens_stats(name=str(corpus_dir), doc_tokens=all_tokens)


def get_token_stream(corpus_dir: str | Path) -> Iterator[str]:
    """Генератор потоку токенів з усіх документів теки."""
    for _doc_id, text in read_corpus(corpus_dir):
        yield from tokenize(text)
