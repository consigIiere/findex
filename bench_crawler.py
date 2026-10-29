import asyncio
import json
import resource
import time
from pathlib import Path

from findex.crawler import CrawlStats, crawl

START_URL = "https://books.toscrape.com/index.html"
MAX_PAGES = 200


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


async def run_single_benchmark(concurrency: int) -> dict[str, float | int]:
    stats = CrawlStats()
    out_jsonl = Path("pages.jsonl")
    log_path = Path(f"crawl_c{concurrency}.log")

    t0 = time.perf_counter()
    with out_jsonl.open("w", encoding="utf-8") as f:  # noqa: ASYNC230
        async for doc in crawl(
            start_urls=[START_URL],
            max_pages=MAX_PAGES,
            concurrency=concurrency,
            per_host_limit=concurrency,
            min_delay=0.01,
            log_path=log_path,
            stats=stats,
        ):
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    wall_time = time.perf_counter() - t0
    peak_rss = get_peak_rss_mb()

    return {
        "concurrency": concurrency,
        "pages": stats.pages_collected,
        "wall_time": wall_time,
        "pages_per_sec": stats.pages_collected / wall_time,
        "errors": stats.errors,
        "peak_rss_mb": peak_rss,
    }


async def main() -> None:
    results: list[dict[str, float | int]] = []
    for c in (1, 5, 20):
        print(f"Running crawl benchmark with concurrency={c}...")
        res = await run_single_benchmark(c)
        results.append(res)
        print(
            f"  Done: {res['wall_time']:.2f}s, {res['pages_per_sec']:.2f} pages/s, "
            f"errors={res['errors']}, Peak RSS={res['peak_rss_mb']:.1f} MB"
        )

    print("\n--- Таблиця для README.md ---\n")
    print("| Concurrency | Pages | Wall Time (s) | Pages / sec | Errors | Peak RSS (MB) |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['concurrency']} | {r['pages']} | {r['wall_time']:.2f} | "
            f"{r['pages_per_sec']:.2f} | {r['errors']} | {r['peak_rss_mb']:.1f} |"
        )


if __name__ == "__main__":
    asyncio.run(main())
