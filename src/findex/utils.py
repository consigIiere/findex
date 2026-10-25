import functools
import re
import sys
import time
from collections.abc import Callable, Sequence
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор для заміру часу виконання функції з виводом у stderr."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        sys.stderr.write(f"[TIMED] {func.__name__} виконано за {elapsed:.5f} с\n")
        return result

    return wrapper


def extract_snippet(text: str, query: str | Sequence[str], window: int = 40) -> str:
    """Витягує фрагмент тексту навколо першого входження терма із підсвічуванням."""
    if isinstance(query, str):
        words = re.findall(r"\b\w+\b", query.lower())
    else:
        words = [w.lower() for w in query if w]

    if not words:
        return text[: window * 2].strip() + ("..." if len(text) > window * 2 else "")

    pattern = re.compile(rf"\b({'|'.join(re.escape(w) for w in words)})\b", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return text[: window * 2].strip() + ("..." if len(text) > window * 2 else "")

    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)

    snippet = text[start:end].replace("\n", " ").strip()
    highlighted = pattern.sub(r"[\1]", snippet)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{highlighted}{suffix}"
