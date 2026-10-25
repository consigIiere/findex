from functools import wraps
import re
import time
from typing import Any, Callable


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Декоратор для заміру часу виконання функцій (search, build, load)
    зі збереженням метаданих через functools.wraps.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"[TIMED] {func.__name__} виконано за {elapsed:.5f} с")
        return result

    return wrapper


def extract_snippet(text: str, query_terms: list[str], window: int = 80) -> str:
    """
    Формує сніпет довжиною близько 80 символів навколо першого знайденого терма
    з підсвіткою слів квадратними дужками [терм].
    """
    lower_text = text.casefold()
    first_match_idx = -1

    for term in query_terms:
        idx = lower_text.find(term.casefold())
        if idx != -1 and (first_match_idx == -1 or idx < first_match_idx):
            first_match_idx = idx

    if first_match_idx == -1:
        snippet_text = text[:window].strip()
        return f"{snippet_text}..." if len(text) > window else snippet_text

    start = max(0, first_match_idx - window // 2)
    end = min(len(text), first_match_idx + window // 2)

    snippet_slice = text[start:end].replace("\n", " ").strip()

    for term in query_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        snippet_slice = pattern.sub(lambda m: f"[{m.group(0)}]", snippet_slice)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet_slice}{suffix}"