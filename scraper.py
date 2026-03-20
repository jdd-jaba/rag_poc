"""
Scraper for https://www.googlecloudevents.com/next-vegas

Crawls all internal pages asynchronously and saves each page as a JSON
file under raw_pages/. Already-scraped URLs are tracked in scraped.log so
the run can be safely interrupted and resumed.

Usage:
    python scraper.py
"""

import asyncio
import json
import os
import re
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.googlecloudevents.com/next-vegas"
SESSION_LIBRARY_TAB = "sessions"
SESSION_LIBRARY_DATE = "all"
SESSION_LIBRARY_MAX_PAGES = 500
_MORE_INFO_URL_RE = re.compile(r'"moreInfoUrl"\s*:\s*"(https?:\\/\\/[^"]+)"')
OUTPUT_DIR = "raw_pages"
LOG_FILE = "scraped.log"
CONCURRENCY = 10
DELAY = 0.1  # seconds between requests per worker


_BASE_PARSED = urlparse(BASE_URL)
_BASE_NETLOC = _BASE_PARSED.netloc
_BASE_PATH_PREFIX = _BASE_PARSED.path.rstrip("/")


def url_to_slug(url: str) -> str:
    """Convert a URL to a safe filename slug.

    Example:
        https://www.googlecloudevents.com/next-vegas -> next-vegas.json
        https://www.googlecloudevents.com/next-vegas/speakers -> next-vegas__speakers.json
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", path)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug + ".json"


def is_internal_doc_url(url: str) -> bool:
    """Return True if the URL is an internal page under the same site and path prefix."""
    parsed = urlparse(url)
    netloc = parsed.netloc or _BASE_NETLOC
    return (
        netloc == _BASE_NETLOC
        and parsed.path.startswith(_BASE_PATH_PREFIX)
        and not parsed.fragment
        and not parsed.path.endswith((".pdf", ".zip", ".png", ".jpg", ".svg", ".css", ".js"))
    )


def extract_session_urls_from_library_html(html: str) -> set[str]:
    """Parse session detail URLs embedded in session-library HTML (GoogleAgendaBuilder JSON)."""
    out: set[str] = set()
    for m in _MORE_INFO_URL_RE.finditer(html):
        raw = m.group(1).replace("\\/", "/").split("#")[0]
        if not is_internal_doc_url(raw):
            continue
        parsed = urlparse(raw)
        normalized = parsed._replace(query="", fragment="").geturl()
        out.add(normalized)
    return out


async def discover_session_urls_from_library(
    session: aiohttp.ClientSession,
) -> list[str]:
    """Walk session-library ?page=N until a page adds no new session URLs."""
    seen: set[str] = set()
    page = 1
    while page <= SESSION_LIBRARY_MAX_PAGES:
        lib_url = (
            f"{BASE_URL}/session-library"
            f"?tab={SESSION_LIBRARY_TAB}&date={SESSION_LIBRARY_DATE}&page={page}"
        )
        html = await fetch(session, lib_url)
        if html is None:
            break
        found = extract_session_urls_from_library_html(html)
        if not found:
            break
        new = found - seen
        if not new:
            break
        seen |= new
        page += 1
    return sorted(seen)


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract all internal doc links from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].split("#")[0]  # drop fragment
        if not href:
            continue
        full_url = urljoin(base_url, href)
        if is_internal_doc_url(full_url):
            # Normalize: strip query strings
            parsed = urlparse(full_url)
            normalized = parsed._replace(query="", fragment="").geturl()
            links.append(normalized)
    return links


def extract_text(html: str) -> tuple[str, str]:
    """Extract page title and main content text from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", role="main")
        or soup.find("div", id="main")
        or soup.find("div", class_="content")
        or soup.body
    )

    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    return title, text


def load_scraped_log() -> set[str]:
    """Load already-scraped URLs from the log file."""
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_to_log(url: str) -> None:
    """Append a completed URL to the log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(url + "\n")


def save_page(url: str, title: str, text: str) -> None:
    """Save a scraped page as a JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = url_to_slug(url)
    filepath = os.path.join(OUTPUT_DIR, slug)
    data = {"url": url, "title": title, "text": text}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def fetch(session: aiohttp.ClientSession, url: str) -> str | None:
    """Fetch a URL and return HTML content, or None on error."""
    try:
        await asyncio.sleep(DELAY)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                return await resp.text()
            return None
    except Exception:
        return None


async def worker(
    session: aiohttp.ClientSession,
    queue: asyncio.Queue,
    visited: set[str],
    scraped: set[str],
    pbar: tqdm,
    lock: asyncio.Lock,
) -> None:
    """Pull URLs from the queue, scrape them, and enqueue newly found links."""
    while True:
        url = await queue.get()
        try:
            html = await fetch(session, url)
            if html is None:
                continue

            title, text = extract_text(html)
            save_page(url, title, text)
            append_to_log(url)

            async with lock:
                scraped.add(url)
                pbar.update(1)
                pbar.set_postfix({"queued": queue.qsize(), "done": len(scraped)})

            new_links = extract_links(html, url)
            async with lock:
                for link in new_links:
                    if link not in visited:
                        visited.add(link)
                        pbar.total = len(visited)
                        pbar.refresh()
                        await queue.put(link)
        finally:
            queue.task_done()


async def run() -> None:
    already_scraped = load_scraped_log()
    print(f"Resuming: {len(already_scraped)} pages already scraped.")

    visited: set[str] = set(already_scraped)
    scraped: set[str] = set(already_scraped)
    queue: asyncio.Queue = asyncio.Queue()
    lock = asyncio.Lock()

    # Seed the queue with the start URL if not already done
    start_url = BASE_URL
    if start_url not in visited:
        visited.add(start_url)
        await queue.put(start_url)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; rag-scraper/1.0)"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        session_urls = await discover_session_urls_from_library(session)
        queued_sessions = 0
        for u in session_urls:
            if u not in visited:
                visited.add(u)
                await queue.put(u)
                queued_sessions += 1
        print(
            f"Session library: {len(session_urls)} unique session URLs "
            f"({queued_sessions} newly queued)."
        )

        if queue.empty():
            print("Nothing new to scrape. All pages already done.")
            return

        with tqdm(total=len(visited), desc="Scraping", unit="page") as pbar:
            pbar.update(len(already_scraped))

            workers = [
                asyncio.create_task(
                    worker(session, queue, visited, scraped, pbar, lock)
                )
                for _ in range(CONCURRENCY)
            ]
            await queue.join()
            for w in workers:
                w.cancel()

    total = len(scraped) - len(already_scraped)
    print(f"\nDone. Scraped {total} new pages. Total: {len(scraped)}.")
    print(f"Files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(run())
