import asyncio
import logging
import random
import time
import urllib.robotparser
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx


def canonicalize_url(url: str, base_url: str = "") -> str | None:
    """
    Канонізує адресу: схема і хост у нижньому регістрі, фрагмент після # відкидається,
    слеші і порядок query приводяться до одного вигляду.
    """
    if base_url:
        url = urljoin(base_url, url)

    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        return None

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if not netloc:
        return None

    path = parsed.path or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    query_params = sorted(parse_qsl(parsed.query, keep_blank_values=True))
    query = urlencode(query_params)

    return urlunparse((scheme, netloc, path, "", query, ""))


class RobotsCache:
    """Читає robots.txt окремо для кожного хоста і кешує відповідь."""

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._cache: dict[str, urllib.robotparser.RobotFileParser | bool] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def can_fetch(self, client: httpx.AsyncClient, url: str) -> bool:
        parsed = urlparse(url)
        base_host = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"

        if base_host not in self._locks:
            self._locks[base_host] = asyncio.Lock()

        async with self._locks[base_host]:
            if base_host not in self._cache:
                robots_url = f"{base_host}/robots.txt"
                try:
                    async with asyncio.timeout(5.0):
                        resp = await client.get(
                            robots_url,
                            headers={"User-Agent": self.user_agent},
                            follow_redirects=True,
                        )
                    if resp.status_code == 200:
                        rp = urllib.robotparser.RobotFileParser()
                        rp.parse(resp.text.splitlines())
                        self._cache[base_host] = rp
                    elif resp.status_code in (401, 403):
                        self._cache[base_host] = False
                    else:
                        self._cache[base_host] = True
                except Exception:  # noqa: BLE001
                    self._cache[base_host] = True

            cached = self._cache[base_host]
            if isinstance(cached, bool):
                return cached
            return cached.can_fetch(self.user_agent, url)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    timeout_sec: float = 10.0,
    max_retries: int = 3,
    base_backoff: float = 0.5,
) -> tuple[int, bytes, float, str]:
    """
    Завантажує сторінку з asyncio.timeout.
    Записує статус, байти й час. Повтор — лише для 429, 5xx і таймауту (щонайбільше тричі)
    з exponential backoff і jitter. Враховує Retry-After. Інші 4xx не повторює.
    """
    for attempt in range(max_retries + 1):
        start_time = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_sec):
                resp = await client.get(url, follow_redirects=True)
                body = resp.content
            elapsed = time.perf_counter() - start_time
            status = resp.status_code

            if (status == 429 or 500 <= status < 600) and attempt < max_retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = float(retry_after)
                else:
                    delay = (base_backoff * (2**attempt)) + random.uniform(0, 0.3)
                await asyncio.sleep(delay)
                continue

            content_type = resp.headers.get("Content-Type", "")
            return status, body, elapsed, content_type

        except (TimeoutError, httpx.TransportError, httpx.TimeoutException):
            elapsed = time.perf_counter() - start_time
            if attempt < max_retries:
                delay = (base_backoff * (2**attempt)) + random.uniform(0, 0.3)
                await asyncio.sleep(delay)
                continue
            return 0, b"", elapsed, ""

    return 0, b"", 0.0, ""


class _PageHTMLParser(HTMLParser):
    """Вбудований парсер для витягування title, тексту (без script/style) і посилань."""

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.in_title = False
        self.skip_depth = 0
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript"):
            self.skip_depth += 1
        elif tag_lower == "title":
            self.in_title = True
        elif tag_lower == "a":
            for attr_name, attr_val in attrs:
                if attr_name.lower() == "href" and attr_val:
                    canon = canonicalize_url(attr_val, base_url=self.base_url)
                    if canon:
                        self.links.append(canon)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript") and self.skip_depth > 0:
            self.skip_depth -= 1
        elif tag_lower == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.skip_depth > 0:
            return
        cleaned = data.strip()
        if not cleaned:
            return
        if self.in_title:
            self.title_parts.append(cleaned)
        else:
            self.text_parts.append(cleaned)


def parse_html(html_bytes: bytes, base_url: str) -> tuple[str, str, list[str]]:
    """Витягує заголовок, чистий текст і канонізовані посилання через html.parser."""
    html_str = html_bytes.decode("utf-8", errors="ignore")
    parser = _PageHTMLParser(base_url=base_url)
    parser.feed(html_str)
    title = " ".join(parser.title_parts)
    text = " ".join(parser.text_parts)
    return title, text, parser.links


class HostLimiter:
    """Окремий семафор на хост + мінімальна пауза між запитами до того самого хоста."""

    def __init__(self, per_host_limit: int = 2, min_delay: float = 0.1):
        self.per_host_limit = per_host_limit
        self.min_delay = min_delay
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    async def acquire(self, host: str) -> asyncio.Semaphore:
        if host not in self._semaphores:
            self._semaphores[host] = asyncio.Semaphore(self.per_host_limit)
            self._locks[host] = asyncio.Lock()

        sem = self._semaphores[host]
        await sem.acquire()

        if self.min_delay > 0:
            async with self._locks[host]:
                now = time.perf_counter()
                last = self._last_request.get(host, 0.0)
                wait_time = self.min_delay - (now - last)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self._last_request[host] = time.perf_counter()

        return sem


@dataclass
class CrawlStats:
    pages_collected: int = 0
    in_flight: int = 0
    errors: int = 0
    queue_depth: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return max(time.perf_counter() - self.start_time, 1e-6)

    @property
    def pages_per_sec(self) -> float:
        return self.pages_collected / self.elapsed


async def crawl(
    start_urls: list[str],
    max_pages: int = 200,
    concurrency: int = 5,
    per_host_limit: int = 2,
    min_delay: float = 0.1,
    user_agent: str = "findex-bot/1.0 (+https://github.com/oleksiiboiko/findex; contact: oleksii@example.com)",
    same_host_only: bool = True,
    client: httpx.AsyncClient | None = None,
    log_path: Path | str = "crawl.log",
    stats: CrawlStats | None = None,
    timeout_sec: float = 10.0,
    base_backoff: float = 0.5,
) -> AsyncIterator[dict[str, str]]:
    """
    Асинхронний генератор. Черга адрес, воркери всередині TaskGroup,
    один семафор на всі запити і окремий на хост, мінімальна пауза між запитами.
    На max_pages воркери скасовуються, клієнт закривається.
    """
    if stats is None:
        stats = CrawlStats()

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    results_queue: asyncio.Queue[dict[str, str] | None] = asyncio.Queue()
    seen: set[str] = set()
    allowed_hosts: set[str] = set()

    for u in start_urls:
        canon = canonicalize_url(u)
        if canon and canon not in seen:
            seen.add(canon)
            queue.put_nowait(canon)
            allowed_hosts.add(urlparse(canon).netloc.lower())

    stats.queue_depth = queue.qsize()

    global_sem = asyncio.Semaphore(concurrency)
    host_limiter = HostLimiter(per_host_limit=per_host_limit, min_delay=min_delay)
    robots_cache = RobotsCache(user_agent=user_agent)

    crawl_logger = logging.getLogger(f"findex.crawl.{id(stats)}")
    crawl_logger.setLevel(logging.INFO)
    crawl_logger.propagate = False
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
    crawl_logger.addHandler(file_handler)

    owns_client = client is None
    active_client: httpx.AsyncClient = (
        httpx.AsyncClient(headers={"User-Agent": user_agent}, follow_redirects=True)
        if client is None
        else client
    )

    stop_event = asyncio.Event()
    worker_tasks: list[asyncio.Task[None]] = []

    async def worker() -> None:
        while not stop_event.is_set():
            try:
                url = await queue.get()
            except asyncio.CancelledError:
                break

            if url is None:
                queue.task_done()
                break

            stats.queue_depth = queue.qsize()
            try:
                if stop_event.is_set():
                    continue

                parsed = urlparse(url)
                host = parsed.netloc.lower()

                allowed = await robots_cache.can_fetch(active_client, url)
                if not allowed:
                    crawl_logger.info(f"BLOCKED_ROBOTS\t{url}\t0B\t0.000s")
                    continue

                async with global_sem:
                    host_sem = await host_limiter.acquire(host)
                    stats.in_flight += 1
                    try:
                        if stop_event.is_set():
                            continue
                        status, body, elapsed, content_type = await fetch(
                            active_client, url, timeout_sec=timeout_sec, base_backoff=base_backoff
                        )
                    finally:
                        stats.in_flight -= 1
                        host_sem.release()

                if status == 200:
                    crawl_logger.info(f"{status}\t{url}\t{len(body)}B\t{elapsed:.3f}s")

                    if stop_event.is_set() or stats.pages_collected >= max_pages:
                        stop_event.set()
                        continue

                    if "text/html" in content_type.lower() or not content_type:
                        title, text, links = parse_html(body, base_url=url)
                    else:
                        continue

                    stats.pages_collected += 1
                    doc = {
                        "id": url,
                        "url": url,
                        "title": title,
                        "text": text,
                    }
                    await results_queue.put(doc)

                    if stats.pages_collected >= max_pages:
                        stop_event.set()
                        for t in worker_tasks:
                            if t is not asyncio.current_task():
                                t.cancel()
                        break

                    for link in links:
                        if link not in seen:
                            link_host = urlparse(link).netloc.lower()
                            if not same_host_only or link_host in allowed_hosts:
                                seen.add(link)
                                queue.put_nowait(link)
                                stats.queue_depth = queue.qsize()
                else:
                    stats.errors += 1
                    crawl_logger.info(f"ERR_{status}\t{url}\t{len(body)}B\t{elapsed:.3f}s")

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                stats.errors += 1
                crawl_logger.info(f"EXCEPTION\t{url}\t{exc}")
            finally:
                queue.task_done()
                stats.queue_depth = queue.qsize()

    async def runner() -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                for _ in range(concurrency):
                    t = tg.create_task(worker())
                    worker_tasks.append(t)

                queue_join_task = asyncio.create_task(queue.join())
                stop_wait_task = asyncio.create_task(stop_event.wait())

                _, pending = await asyncio.wait(
                    [queue_join_task, stop_wait_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for p in pending:
                    p.cancel()

                stop_event.set()
                for t in worker_tasks:
                    t.cancel()
        finally:
            await results_queue.put(None)

    runner_task = asyncio.create_task(runner())

    try:
        while True:
            item = await results_queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        if not runner_task.done():
            runner_task.cancel()
            try:
                await runner_task
            except asyncio.CancelledError:
                pass
        if owns_client:
            await active_client.aclose()
        file_handler.close()
        crawl_logger.removeHandler(file_handler)
