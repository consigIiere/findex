import pytest

from findex.index import InvertedIndex


def test_index_add_and_contains() -> None:
    index = InvertedIndex()
    index.add_document("doc1.txt", "quick brown fox jumps")

    assert "quick" in index
    assert "fox" in index
    assert "cat" not in index
    assert len(index) == 4


def test_index_getitem() -> None:
    index = InvertedIndex()
    index.add_document("doc1.txt", "hello world")

    postings = index["hello"]
    assert len(postings) == 1
    assert postings[0].doc_id == "doc1.txt"
    assert postings[0].freq == 1
    assert postings[0].positions == [0]


def test_index_missing_key() -> None:
    index = InvertedIndex()
    with pytest.raises(KeyError):
        _ = index["non_existent_token"]


def test_index_avg_doc_length() -> None:
    index = InvertedIndex()
    index.add_document("doc1.txt", "one two three")
    index.add_document("doc2.txt", "four five")

    assert index.avg_doc_length == 2.5
