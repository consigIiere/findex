from pathlib import Path

from findex.index import build_index, build_partial, merge_indices
from findex.parallel import ExecutorType, run_parallel_indexing


def test_merge_equivalence(tmp_path: Path) -> None:
    """Перевіряє, що розбиття та наступний merge дають 100% ідентичний результат до послідовної побудови."""
    docs = {
        "doc1.txt": "python state of the art search engine",
        "doc2.txt": "inverted index with boolean queries and ranking",
        "doc3.txt": "multiprocessing gil threads concurrency",
        "doc4.txt": "benchmarks and speedup measurements",
    }
    for name, content in docs.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    all_paths = sorted(tmp_path.glob("*.txt"))

    reference_idx = build_index(tmp_path)

    part1 = build_partial(all_paths[:2])
    part2 = build_partial(all_paths[2:])
    merged_idx = merge_indices([part1, part2])

    assert merged_idx.num_docs == reference_idx.num_docs
    assert merged_idx.doc_lengths == reference_idx.doc_lengths
    assert set(merged_idx.postings.keys()) == set(reference_idx.postings.keys())

    for term, ref_postings in reference_idx.postings.items():
        mrg_postings = merged_idx.postings[term]
        assert len(ref_postings) == len(mrg_postings)
        for r_p, m_p in zip(ref_postings, mrg_postings, strict=True):
            assert r_p.doc_id == m_p.doc_id
            assert r_p.freq == m_p.freq
            assert r_p.positions == m_p.positions


def test_parallel_executors_equivalence(tmp_path: Path) -> None:
    """Перевіряє, що запуски через threads та processes повертають однаковий індекс."""
    for i in range(10):
        (tmp_path / f"file_{i}.txt").write_text(f"content word_{i} sample text", encoding="utf-8")

    files = sorted(tmp_path.glob("*.txt"))
    ref = run_parallel_indexing(files, workers=1, executor_type=ExecutorType.SERIAL)
    thr = run_parallel_indexing(files, workers=2, executor_type=ExecutorType.THREADS)
    prc = run_parallel_indexing(files, workers=2, executor_type=ExecutorType.PROCESSES)

    assert set(ref.postings.keys()) == set(thr.postings.keys()) == set(prc.postings.keys())
    assert ref.num_docs == thr.num_docs == prc.num_docs
