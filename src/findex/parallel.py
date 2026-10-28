import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum
from pathlib import Path

from findex.index import InvertedIndex, build_partial, merge_indices


class ExecutorType(str, Enum):
    SERIAL = "serial"
    THREADS = "threads"
    PROCESSES = "processes"


def split_into_chunks(paths: list[Path], num_chunks: int) -> list[list[Path]]:
    """Розбиває список файлів на неперетинні частини однакового розміру."""
    if not paths or num_chunks <= 1:
        return [paths]

    chunk_size = math.ceil(len(paths) / num_chunks)
    chunks = [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]
    return [c for c in chunks if c]


def run_parallel_indexing(
    file_paths: list[Path],
    workers: int = 1,
    executor_type: ExecutorType = ExecutorType.PROCESSES,
) -> InvertedIndex:
    """Виконує паралельну індексацію заданим пулом виконавців."""
    if workers <= 1 or executor_type == ExecutorType.SERIAL:
        return build_partial(file_paths)

    chunks = split_into_chunks(file_paths, workers)

    if executor_type == ExecutorType.THREADS:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            partial_indices = list(executor.map(build_partial, chunks))
        return merge_indices(partial_indices)

    if executor_type == ExecutorType.PROCESSES:
        # Безпечний контекст spawn для сумісності з будь-якою платформою
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            partial_indices = list(executor.map(build_partial, chunks))
        return merge_indices(partial_indices)

    raise ValueError(f"Невідомий тип виконавця: {executor_type}")
