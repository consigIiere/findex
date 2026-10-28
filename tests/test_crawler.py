import asyncio
from pathlib import Path

import httpx

from findex.crawler import (
    CrawlStats,
    RobotsCache,
    canonicalize_url,
    crawl,
    fetch,
    parse_html,
)


def test_canonicalize_url() -> None:
    # Нижній регістр схеми і хоста, відкидання фрагмента #
    assert (
        canonicalize_url("HTTP://Example.COM/Path/To/Page/#section")
        == "http://example.com/Path/To/Page"
    )
    # Сортування параметрів query
    assert (
        canonicalize_url("https://example.com/search?b=2&a=1")
        == "https://example.com/search?a=1&b=2"
    )
    # Відносне посилання з base_url
    assert (
        canonicalize_url("../about#team", base_url="https://example.com/blog/post")
        == "https://example.com/about"
    )
    # Непідтримувані схеми відкидаються
    assert canonicalize_url("mailto:user@example.com") is None
    assert canonicalize_url("javascript:void(0)") is None


def test_parse_html() -> None:
    html = b"""
    <html>
      <head>
        <title>Test Page</title>
        <style>body { color: red; }</style>
      </head>
      <body>
        <script>console.log("ignore me");</script>
        <h1>Hello World</h1>
        <p>Some body text.</p>
        <a href="/page2#frag">Page 2</a>
        <a href="https://example.com/page3/">Page 3</a>
      </body>
    </html>
    """
    title, text, links = parse_html(html, base_url="https://example.com/page1")
    assert title == "Test Page"
    assert "Hello World" in text
    assert "Some body text." in text
    assert "ignore me" not in text
    assert links == [
        "https://example.com/page2",
        "https://example.com/page3",
    ]


def test_fetch_retries_on_429_and_503() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, content=b"rate limited")
        if attempts == 2:
            return httpx.Response(503, content=b"server busy")
        return httpx.Response(200, headers={"Content-Type": "text/html"}, content=b"ok")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            status, body, elapsed, ctype = await fetch(
                client, "https://example.com/test", base_backoff=0.01
            )
            assert status == 200
            assert body == b"ok"
            assert elapsed >= 0.0
            assert "text/html" in ctype
            assert attempts == 3

    asyncio.run(_run())


def test_fetch_does_not_retry_on_404() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, content=b"not found")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            status, _, _, _ = await fetch(
                client, "https://example.com/missing", base_backoff=0.01
            )
            assert status == 404
            assert attempts == 1

    asyncio.run(_run())


def test_robots_cache_fetches_once_per_host() -> None:
    robots_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_calls
        if request.url.path == "/robots.txt":
            robots_calls += 1
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /private\n",
            )
        return httpx.Response(200, text="ok")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        cache = RobotsCache(user_agent="findex-bot/1.0")
        async with httpx.AsyncClient(transport=transport) as client:
            assert await cache.can_fetch(client, "https://example.com/public") is True
            assert await cache.can_fetch(client, "https://example.com/private/secret") is False
            # Перевіряємо, що robots.txt запитувався лише 1 раз завдяки кешу
            assert robots_calls == 1

    asyncio.run(_run())


def test_crawl_respects_max_pages_and_robots(tmp_path: Path) -> None:
    pages = {
        "/robots.txt": "User-agent: *\nDisallow: /secret\n",
        "/": '<html><title>Home</title><body>Home <a href="/p1">1</a> <a href="/p2">2</a> <a href="/secret">S</a></body></html>',
        "/p1": '<html><title>P1</title><body>Page 1 <a href="/p2#section">2</a> <a href="/p3">3</a></body></html>',
        "/p2": '<html><title>P2</title><body>Page 2 <a href="/p3">3</a></body></html>',
        "/p3": "<html><title>P3</title><body>Page 3</body></html>",
        "/secret": "<html><title>Secret</title><body>Forbidden</body></html>",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in pages:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                content=pages[path].encode("utf-8"),
            )
        return httpx.Response(404, content=b"Not found")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        log_file = tmp_path / "crawl.log"
        stats = CrawlStats()
        collected: list[dict[str, str]] = []

        async with httpx.AsyncClient(transport=transport) as client:
            async for doc in crawl(
                start_urls=["https://example.com/"],
                max_pages=3,
                concurrency=2,
                per_host_limit=2,
                min_delay=0.0,
                client=client,
                log_path=log_file,
                stats=stats,
                base_backoff=0.01,
            ):
                collected.append(doc)

        assert len(collected) == 3
        assert stats.pages_collected == 3
        urls = {d["url"] for d in collected}
        assert "https://example.com/secret" not in urls
        assert log_file.exists()

    asyncio.run(_run())