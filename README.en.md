# Python Docs RAG

A local Retrieval-Augmented Generation (RAG) system built on top of the official [Python 3 documentation](https://docs.python.org/3/). Everything runs on your machine — no cloud APIs, no data sent anywhere.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 — Scrape                                               │
│                                                                 │
│  docs.python.org  ──►  BeautifulSoup  ──►  raw_pages/*.json    │
│       (~1,000 pages)     (clean text)       (url, title, text) │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2 — Ingest                                               │
│                                                                 │
│  raw_pages/*.json  ──►  Chunker  ──►  Embedder  ──►  ChromaDB  │
│                        (500 chars)  (MiniLM-L6)   (local disk) │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3 — Query                                                │
│                                                                 │
│  question ──► LLM rephrases ──► ChromaDB (per variant)         │
│               (3 variants)      (top-5 each, deduplicated)     │
│                                       │                        │
│                                       ▼                        │
│                            llama3.2 (streamed) ──► answer      │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Tool | Details |
|---|---|---|
| Web scraper | `aiohttp` + `BeautifulSoup4` | Async, 10 concurrent workers, resumable |
| Text chunker | `langchain-text-splitters` | `RecursiveCharacterTextSplitter`, 500 chars, 50 overlap |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2`, runs fully offline (~90 MB) |
| Vector store | `ChromaDB` | Persistent local storage in `./chroma_db/` |
| LLM | `Ollama` + `llama3.2` | Runs locally, no API key needed |
| RAG chain | `LangChain` LCEL | Multi-query retrieval + streamed answer |

## Prerequisites

1. **Python 3.11+**

2. **Ollama** — install from [ollama.com](https://ollama.com), then pull the model:
   ```bash
   ollama pull llama3.2
   ```

3. **Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Step 1 — Scrape the docs

Crawls all pages under `docs.python.org/3/` and saves them as JSON files.
Safe to interrupt and resume — already-scraped pages are skipped automatically.

```bash
python scraper.py
```

Output: `raw_pages/*.json` (~1,000 files) and `scraped.log`
Time: ~2–3 minutes

### Step 2 — Ingest into ChromaDB

Chunks all scraped pages, embeds them, and stores everything in a local vector database.
Re-running is safe — skips if the collection already exists.

```bash
python ingest.py
```

Output: `chroma_db/` directory
Time: ~1–2 minutes on Apple Silicon

### Step 3 — Ask questions

```bash
python query.py "How does asyncio work?"
python query.py "What is the difference between a list and a tuple?"
python query.py "How do I use context managers?"
```

Example output:
```
Question: How does asyncio work?
------------------------------------------------------------

Generating query variants...

[Query variants (4 total)]
  original: How does asyncio work?
  variant 1: What is Python's asyncio event loop?
  variant 2: How are coroutines scheduled in Python?
  variant 3: What does async/await do under the hood?

[Retrieved chunks for original query]
  score 0.1823 | asyncio — Asynchronous I/O — Python 3.14.3 doc
  score 0.2104 | asyncio-task — Coroutines and Tasks — Python 3...
  score 0.2341 | asyncio-eventloop — Event Loop — Python 3.14.3
  score 0.2789 | library/concurrent.futures — Python 3.14.3 doc
  score 0.3102 | whatsnew/3.11 — What's New In Python 3.11

[Total unique chunks passed to LLM: 14]

------------------------------------------------------------
Answer:

asyncio is a library for writing concurrent code using the async/await
syntax. It uses an event loop to manage and schedule coroutines...

Sources:
  - https://docs.python.org/3/library/asyncio.html
  - https://docs.python.org/3/library/asyncio-task.html
```

> **Score note:** Lower = more relevant (cosine distance). Under `0.2` is a strong match, above `0.4` is a weak match — useful for spotting when a question is outside the scope of the docs.

## How Multi-Query Retrieval Works

A single question like `"How does asyncio work?"` only matches chunks using similar wording. Relevant content might be phrased differently across the docs (e.g. "event loop scheduling", "coroutine execution model"). 

`query.py` solves this by asking the LLM to generate 3 rephrasings of your question before searching. It then retrieves top-5 chunks per variant, deduplicates, and passes all unique chunks to the LLM as context — giving a broader and more accurate answer.

## Project Structure

```
rag_poc/
├── scraper.py          # Phase 1: async web crawler
├── ingest.py           # Phase 2: chunk, embed, store
├── query.py            # Phase 3: multi-query retrieval + streamed answer
├── requirements.txt    # Python dependencies
├── .gitignore
│
│   # Generated — not committed to git
├── raw_pages/          # scraped JSON files (one per page)
├── scraped.log         # tracks completed URLs for resumability
└── chroma_db/          # ChromaDB vector store
```

> `raw_pages/`, `scraped.log`, and `chroma_db/` are excluded from git.
> Run the three steps above to rebuild them from scratch.
