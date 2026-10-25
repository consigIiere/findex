import json
from pathlib import Path
from typing import cast

from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "findex" in result.stdout


def test_cli_index_and_stats(sample_corpus: Path, tmp_path: Path) -> None:
    index_path = str(tmp_path / "test_idx.bin")

    # Побудова індексу через CLI
    res_index = runner.invoke(app, ["index", str(sample_corpus), "-o", index_path])
    assert res_index.exit_code == 0
    assert "успішно збережено" in res_index.stdout

    # Перевірка команди stats
    res_stats = runner.invoke(app, ["stats", index_path])
    assert res_stats.exit_code == 0
    assert "Кількість документів" in res_stats.stdout


def test_cli_search_json(sample_corpus: Path, tmp_path: Path) -> None:
    index_path = str(tmp_path / "test_idx.bin")
    runner.invoke(app, ["index", str(sample_corpus), "-o", index_path])

    result = runner.invoke(app, ["search", index_path, "python", "--json"])
    assert result.exit_code == 0
    data = cast(list[object], json.loads(result.stdout))
    assert len(data) > 0


def test_cli_search_missing_index() -> None:
    result = runner.invoke(app, ["search", "non_existent_path.bin", "query"])
    assert result.exit_code == 2


def test_cli_verbose_flags(sample_corpus: Path, tmp_path: Path) -> None:
    index_path = str(tmp_path / "test_idx.bin")
    runner.invoke(app, ["index", str(sample_corpus), "-o", index_path])

    res_v = runner.invoke(app, ["-v", "search", index_path, "python"])
    assert res_v.exit_code == 0

    res_vv = runner.invoke(app, ["-vv", "search", index_path, "python"])
    assert res_vv.exit_code == 0


def test_cli_stats_invalid_path() -> None:
    result = runner.invoke(app, ["stats", "invalid_missing.bin"])
    assert result.exit_code == 1
