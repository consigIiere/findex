import pytest
from findex.index import InvertedIndex
from findex.search import intersect_postings, intersect_sets, search_boolean


def test_intersect_postings():
    list1 = ["doc1", "doc3", "doc5", "doc7"]
    list2 = ["doc2", "doc3", "doc6", "doc7"]
    assert intersect_postings(list1, list2) == ["doc3", "doc7"]
    assert intersect_sets(list1, list2) == ["doc3", "doc7"]


def test_intersect_empty():
    assert intersect_postings([], ["doc1"]) == []
    assert intersect_postings(["doc1"], []) == []


def test_inverted_index_and_search():
    index = InvertedIndex()
    index.add_document("doc1.txt", "quick brown fox jumps")
    index.add_document("doc2.txt", "lazy brown dog jumps")
    index.add_document("doc3.txt", "quick blue bird")

    assert search_boolean(index, "fox") == ["doc1.txt"]

    assert search_boolean(index, "brown AND jumps", use_merge=True) == ["doc1.txt", "doc2.txt"]
    assert search_boolean(index, "quick AND dog", use_merge=True) == []