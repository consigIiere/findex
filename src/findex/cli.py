import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from findex.crawler import CrawlStats, crawl
from findex.index import InvertedIndex
from findex.logging_conf import setup_logging
from findex.models import SearchResult
from findex.scoring import BM25Scorer, TfidfScorer
from findex.search import ranked_search

app = typer.Typer(
    name="findex",
    help="CLI повнотекстового пошукового рушія з булевим фільтром, ранжуванням та асинхронним краулером.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Рівень деталізації логів (-v: INFO, -vv: DEBUG).",
        ),
    ] = 0,
) -> None:
    """Глобальна конфігурація логування."""
    setup_logging(verbose)


@app.command(name="crawl")
def crawl_command(
    start_url: Annotated[
        str,
        typer.Argument(
            help="Початковий URL для обходу.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Шлях до вихідного JSONL файлу.",
        ),
    ] = Path("pages.jsonl"),
    max_pages: Annotated[
        int,
        typer.Option(
            "--max-pages",
            "-m",
            min=1,
            help="Максимальна кількість сторінок для збору.",
        ),
    ] = 200,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            "-c",
            min=1,
            help="Загальний ліміт одночасних запитів.",
        ),
    ] = 5,
    per_host: Annotated[
        int,
        typer.Option(
            "--per-host",
            min=1,
            help="Ліміт одночасних запитів на один хост.",
        ),
    ] = 5,
    min_delay: Annotated[
        float,
        typer.Option(
            "--delay",
            min=0.0,
            help="Мінімальна пауза між запитами до одного хоста (сек).",
        ),
    ] = 0.05,
    log_path: Annotated[
        Path,
        typer.Option(
            "--log",
            help="Файл логу обходу URL.",
        ),
    ] = Path("crawl.log"),
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Запустити цикл подій asyncio у режимі debug.",
        ),
    ] = False,
) -> None:
    """Асинхронно обійти сайт і потоково записати зібрані сторінки у JSONL."""
    stats = CrawlStats()

    def build_status_table() -> Table:
        table = Table(title="Стан асинхронного краулера (findex crawl)")
        table.add_column("Сторінок зібрано", justify="right", style="green")
        table.add_column("Сторінок/сек", justify="right", style="cyan")
        table.add_column("Чекають відповіді (in-flight)", justify="right", style="yellow")
        table.add_column("Помилок", justify="right", style="red")
        table.add_column("Глибина черги", justify="right", style="magenta")

        table.add_row(
            f"{stats.pages_collected}/{max_pages}",
            f"{stats.pages_per_sec:.2f}",
            str(stats.in_flight),
            str(stats.errors),
            str(stats.queue_depth),
        )
        return table

    async def _run_crawl() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with (
            output.open("w", encoding="utf-8") as out_file,
            Live(build_status_table(), console=console, refresh_per_second=8) as live,
        ):
            async for doc in crawl(
                start_urls=[start_url],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host_limit=per_host,
                min_delay=min_delay,
                log_path=log_path,
                stats=stats,
            ):
                out_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
                out_file.flush()
                live.update(build_status_table())

    asyncio.run(_run_crawl(), debug=debug)

    console.print(
        f"[green]✓[/green] Обхід завершено за [bold]{stats.elapsed:.2f} с[/bold] "
        f"([cyan]{stats.pages_collected}[/cyan] сторінок, "
        f"[cyan]{stats.pages_per_sec:.2f}[/cyan] стор/с, "
        f"помилок: [red]{stats.errors}[/red]). Збережено у [bold]{output}[/bold]"
    )


@app.command(name="index")
def index_command(
    corpus_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help="Шлях до теки з документами (.txt) або файлу (.jsonl).",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Шлях до бінарного файлу збереження індексу.",
        ),
    ] = Path("index.bin"),
) -> None:
    """Побудувати інвертований індекс корпусу (поза циклом подій) та зберегти його у файл."""
    with console.status("[bold green]Індексація документів..."):
        index: InvertedIndex = InvertedIndex.build(corpus_path)
        index.save(output)

    console.print(
        f"[green]✓[/green] Індекс успішно збережено до [bold]{output}[/bold] "
        f"([cyan]{index.num_docs}[/cyan] документів, [cyan]{len(index)}[/cyan] унікальних термів)"
    )


@app.command(name="search")
def search_command(
    index_path: Annotated[
        Path,
        typer.Argument(
            help="Шлях до збереженого файлу індексу.",
        ),
    ],
    query: Annotated[
        str,
        typer.Argument(
            help="Пошуковий запит (наприклад, 'python AND inverted NOT test').",
        ),
    ],
    top_k: Annotated[
        int,
        typer.Option(
            "-k",
            "--top-k",
            min=1,
            max=100,
            help="Максимальна кількість документів у видачі.",
        ),
    ] = 5,
    scorer_type: Annotated[
        str,
        typer.Option(
            "--scorer",
            "-s",
            help="Алгоритм ранжування: bm25 або tfidf.",
        ),
    ] = "bm25",
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Вивести результат у машиночитаному форматі JSON.",
        ),
    ] = False,
    corpus_dir: Annotated[
        Path | None,
        typer.Option(
            "--corpus",
            "-c",
            help="Шлях до первинного корпусу для генерації текстових сніпетів.",
        ),
    ] = None,
) -> None:
    """Виконати пошук за булевим виразом та ранжуванням."""
    if not index_path.exists():
        err_console.print(f"[bold red]Помилка:[/bold red] Файл індексу '{index_path}' не знайдено.")
        raise typer.Exit(code=2)

    index: InvertedIndex = InvertedIndex.load(index_path)
    scorer = BM25Scorer() if scorer_type.lower() == "bm25" else TfidfScorer()

    results: list[SearchResult] = ranked_search(
        index=index,
        query_str=query,
        scorer=scorer,
        top_k=top_k,
        corpus_dir=corpus_dir,
    )

    if as_json:
        data: list[dict[str, object]] = [
            {
                "rank": i + 1,
                "doc_id": r.doc_id,
                "score": round(r.score, 4),
                "snippet": r.snippet if r.snippet else "",
            }
            for i, r in enumerate(results)
        ]
        sys.stdout.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return

    if not results:
        console.print(f"[yellow]За запитом '{query}' нічого не знайдено.[/yellow]")
        return

    table = Table(title=f"Результати пошуку: [bold]{query}[/bold]", show_lines=True)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Документ", style="magenta")
    table.add_column("Оцінка", justify="right", style="green")
    table.add_column("Фрагмент", style="white")

    for i, r in enumerate(results, start=1):
        snippet_text: str = r.snippet if r.snippet else "-"
        table.add_row(str(i), r.doc_id, f"{r.score:.4f}", snippet_text)

    console.print(table)


@app.command(name="stats")
def stats_command(
    index_path: Annotated[
        Path,
        typer.Argument(
            help="Шлях до збереженого бінарного індексу.",
        ),
    ],
) -> None:
    """Показати загальну статистику побудованого індексу."""
    if not index_path.exists():
        err_console.print(f"[bold red]Помилка:[/bold red] Файл '{index_path}' не існує.")
        raise typer.Exit(code=1)

    index: InvertedIndex = InvertedIndex.load(index_path)

    table = Table(title="Статистика індексу")
    table.add_column("Параметр", style="cyan")
    table.add_column("Значення", style="magenta")

    table.add_row("Кількість документів", str(index.num_docs))
    table.add_row("Кількість унікальних термів", str(len(index)))
    table.add_row("Середня довжина документа", f"{index.avg_doc_length:.2f}")

    console.print(table)