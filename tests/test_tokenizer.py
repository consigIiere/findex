import pytest

from findex.tokenizer import tokenize


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("Python 3.12 is awesome", ["python", "3", "12", "is", "awesome"]),
        ("state-of-the-art model", ["state-of-the-art", "model"]),
        ("don't stop believing", ["don't", "stop", "believing"]),
        ("Київ — столиця України", ["київ", "столиця", "україни"]),
        ("Schön, café, naïve", ["schön", "café", "naïve"]),
        ("   ", []),
        ("!@#$%^&*()", []),
        ("word123 and 456word", ["word123", "and", "456word"]),
        ("it's user's test-case", ["it's", "user's", "test-case"]),
    ],
)
def test_tokenize_parametrized(input_text: str, expected: list[str]) -> None:
    assert list(tokenize(input_text)) == expected


def test_tokenize_unicode_nfc_normalization() -> None:
    decomposed = "киі\u0308в"
    tokens = list(tokenize(decomposed))
    assert tokens == ["київ"]
