from findex.tokenizer import tokenize


def test_basic_tokenization():
    text = "Hello world! This is findex."
    tokens = list(tokenize(text))
    assert tokens == ["hello", "world", "this", "is", "findex"]


def test_apostrophes_and_hyphens():
    text = "Don't touch state-of-the-art м'ясо та з’єднати."
    tokens = list(tokenize(text))
    assert tokens == ["don't", "touch", "state-of-the-art", "м'ясо", "та", "з’єднати"]


def test_numbers_and_punctuation():
    text = "Python 3.12 was released in 2023, cost: $100."
    tokens = list(tokenize(text))
    assert tokens == ["python", "3", "12", "was", "released", "in", "2023", "cost", "100"]