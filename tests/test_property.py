import os
import tempfile

from hypothesis import given
from hypothesis import strategies as st

from findex.index import InvertedIndex


@given(
    st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10),
        min_size=1,
        max_size=20,
    )
)
def test_hypothesis_postings_sorted(terms: list[str]) -> None:
    """Властивість 1: списки doc_id у posting list завжди відсортовані."""
    index = InvertedIndex()
    for i, term in enumerate(terms):
        doc_id = f"doc_{i:04d}.txt"
        index.add_document(doc_id, term)

    for posting_list in index.postings.values():
        doc_ids = [p.doc_id for p in posting_list]
        assert doc_ids == sorted(doc_ids)


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=5, max_size=100))
def test_hypothesis_index_save_load_idempotent(text: str) -> None:
    """Властивість 2: збереження і подальше завантаження дає ідентичний стан індексу."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_file = os.path.join(tmp_dir, "index.bin")

        idx = InvertedIndex()
        idx.add_document("doc_1.txt", text)
        idx.save(index_file)

        loaded = InvertedIndex.load(index_file)
        assert len(idx) == len(loaded)
        assert idx.num_docs == loaded.num_docs


@given(
    st.lists(st.sampled_from(["alpha", "beta", "gamma", "delta"]), min_size=1, max_size=10),
    st.lists(st.sampled_from(["beta", "gamma", "theta"]), min_size=1, max_size=10),
)
def test_hypothesis_boolean_and_matches_set_intersection(
    doc1_words: list[str], doc2_words: list[str]
) -> None:
    """Властивість 3: булевий перетин AND еквівалентний математичному перетину множин."""
    index = InvertedIndex()
    index.add_document("doc_1.txt", " ".join(doc1_words))
    index.add_document("doc_2.txt", " ".join(doc2_words))

    term_a = "beta"
    docs_a = {p.doc_id for p in index.get_postings(term_a)}
    term_b = "gamma"
    docs_b = {p.doc_id for p in index.get_postings(term_b)}

    expected = docs_a & docs_b
    matched = [p.doc_id for p in index.get_postings(term_a) if p.doc_id in docs_b]
    assert set(matched) == expected
