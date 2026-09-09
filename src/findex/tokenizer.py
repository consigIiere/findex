import re
import unicodedata
from typing import Iterator

TOKEN_PATTERN = re.compile(r"\b\w+(?:['’-]\w+)*\b")

def tokenize(text: str) -> Iterator[str]:

    normalized_text = unicodedata.normalize("NFC", text).casefold()

    for match in TOKEN_PATTERN.finditer(normalized_text):
        yield match.group(0)