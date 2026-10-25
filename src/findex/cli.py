import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from findex.index import InvertedIndex
from findex.logging_conf import setup_logging
from findex.models import SearchResult
from findex.scoring import BM25Scorer, TfidfScorer
from findex.search import ranked_search

app = typer.Typer(
    name="findex",
    help="CLI повнотекстового пошукового рушія з булевим фільтром та ранжуванням.",
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


@app.command(name="index")
def index_command(
    corpus_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Шлях до теки з документами (.txt).",
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
    """Побудувати інвертований індекс корпусу та зберегти його у файл."""
    with console.status("[bold green]Індексація документів..."):
        index: InvertedIndex = InvertedIndex.build(corpus_dir)
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
