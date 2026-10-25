import re
import unicodedata
from collections.abc import Iterator

TOKEN_RE = re.compile(r"[\w]+(?:['’\-][\w]+)*")


def tokenize(text: str) -> Iterator[str]:
    """Розбиває текст на токени в нижньому регістрі з NFC-нормалізацією."""
    normalized = unicodedata.normalize("NFC", text)
    for match in TOKEN_RE.finditer(normalized):
        yield match.group(0).lower()
