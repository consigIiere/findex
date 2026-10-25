from pathlib import Path

import pytest

from findex.index import InvertedIndex


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Створює тимчасовий міні-корпус з 3 документів для швидких тестів."""
    doc1 = tmp_path / "doc_1.txt"
    doc1.write_text("Python is a high-level programming language.", encoding="utf-8")

    doc2 = tmp_path / "doc_2.txt"
    doc2.write_text("Search engine indexing with Python and inverted index.", encoding="utf-8")

    doc3 = tmp_path / "doc_3.txt"
    doc3.write_text("Unit testing with pytest and hypothesis framework.", encoding="utf-8")

    return tmp_path


@pytest.fixture
def sample_index(sample_corpus: Path) -> InvertedIndex:
    """Повертає вже зібраний тестовий індекс."""
    return InvertedIndex.build(str(sample_corpus))
