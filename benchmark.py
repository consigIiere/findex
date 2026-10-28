# type: ignore
import os
import resource
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

from findex.parallel import ExecutorType, run_parallel_indexing


def generate_benchmark_corpus(data_dir: Path, num_docs: int = 1200) -> list[Path]:
    """Створює корпус документів для заміру навантаження."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    base_text = (
        "Concurrency and parallel processing in Python can be bounded by the Global Interpreter Lock. "
        "When handling CPU-bound tasks such as full text indexing, tokenization, and AST query evaluation, "
        "multiprocessing allows scaling across multiple physical CPU cores. "
    ) * 30

    for i in range(num_docs):
        p = data_dir / f"doc_{i:04d}.txt"
        if not p.exists():
            p.write_text(f"{base_text} Document unique token_{i % 50} end.", encoding="utf-8")
        paths.append(p)
    return sorted(paths)


def get_peak_rss_mb() -> float:
    """Повертає пікове споживання пам'яті (RSS) у мегабайтах."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage) / 1024.0


def measure_execution(
    paths: list[Path], workers: int, executor: ExecutorType
) -> tuple[float, float, float]:
    """Вимірює (wall_time, cpu_time, peak_rss)."""
    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    _ = run_parallel_indexing(paths, workers=workers, executor_type=executor)

    wall_time = time.perf_counter() - start_wall
    cpu_time = time.process_time() - start_cpu
    rss = get_peak_rss_mb()
    return wall_time, cpu_time, rss


def main() -> None:
    corpus_dir = Path("data/bench_corpus")
    paths = generate_benchmark_corpus(corpus_dir, num_docs=1200)

    cpu_count: int = os.cpu_count() or 4
    worker_configs: list[int] = sorted({1, 2, 4, 8, cpu_count})

    print(f"=== Початок бенчмарку (CPU: {cpu_count} cores, Документів: {len(paths)}) ===")

    results: dict[str, dict[int, float]] = {"processes": {}, "threads": {}}

    print("\n| Виконавець | Воркери | Wall Time (s) | CPU Time (s) | Пік RSS (MB) | Прискорення |")
    print("|---|---|---|---|---|---|")

    base_walls: list[float] = []
    for _ in range(3):
        w, _, _ = measure_execution(paths, 1, ExecutorType.SERIAL)
        base_walls.append(w)
    base_time: float = statistics.median(base_walls)

    executors_list: list[tuple[str, ExecutorType]] = [
        ("processes", ExecutorType.PROCESSES),
        ("threads", ExecutorType.THREADS),
    ]

    for exec_name, exec_type in executors_list:
        for w_count in worker_configs:
            walls: list[float] = []
            cpus: list[float] = []
            rsses: list[float] = []
            for _ in range(3):
                w, c, r = measure_execution(paths, w_count, exec_type)
                walls.append(w)
                cpus.append(c)
                rsses.append(r)

            med_wall: float = statistics.median(walls)
            med_cpu: float = statistics.median(cpus)
            med_rss: float = statistics.median(rsses)
            speedup: float = base_time / med_wall if med_wall > 0 else 0.0

            results[exec_name][w_count] = speedup
            print(
                f"| {exec_name:<9} | {w_count:<7} | {med_wall:>13.3f} | "
                f"{med_cpu:>12.3f} | {med_rss:>12.1f} | {speedup:>10.2f}x |"
            )

    plt.figure(figsize=(9, 5))
    x_vals: list[int] = worker_configs
    proc_y: list[float] = [results["processes"][w] for w in x_vals]
    thr_y: list[float] = [results["threads"][w] for w in x_vals]

    plt.plot(
        x_vals,
        proc_y,
        marker="o",
        label="Processes (ProcessPoolExecutor)",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        x_vals,
        thr_y,
        marker="s",
        label="Threads (ThreadPoolExecutor - GIL)",
        color="red",
        linewidth=2,
    )
    plt.plot(
        x_vals,
        x_vals,
        "--",
        label="Ідеальне лінійне прискорення",
        color="gray",
        alpha=0.7,
    )

    plt.title(f"Прискорення індексації vs Кількість воркерів ({cpu_count} cores)")
    plt.xlabel("Кількість воркерів (Workers)")
    plt.ylabel("Прискорення (Speedup = T1 / Tn)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("speedup_graph.png", dpi=300)
    print("\nГрафік збережено у speedup_graph.png")


if __name__ == "__main__":
    main()
