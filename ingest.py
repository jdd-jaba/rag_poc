"""
Ingest scraped Google Cloud Next site into ChromaDB.

Reads all raw_pages/*.json files, splits text into chunks, embeds them with
a local sentence-transformers model, and stores everything in a persistent
ChromaDB collection at ./chroma_db/.

Run AFTER scraper.py has finished:
    python ingest.py

Re-running is safe: if the collection already contains documents it will
print the current count and exit without re-ingesting.
"""

import glob
import json
import logging
import os

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

RAW_PAGES_DIR = "raw_pages"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "google_cloud_next"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_TEXT_LENGTH = 100
BATCH_SIZE = 256  # how many chunks to upsert to ChromaDB at once


def load_pages(directory: str) -> list[dict]:
    """Load all JSON files from the raw_pages directory."""
    files = glob.glob(os.path.join(directory, "*.json"))
    pages = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if data.get("text") and len(data["text"]) >= MIN_TEXT_LENGTH:
                    pages.append(data)
            except json.JSONDecodeError:
                pass
    return pages


def chunk_pages(pages: list[dict]) -> tuple[list[str], list[dict]]:
    """Split page texts into chunks, returning texts and their metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_texts: list[str] = []
    all_metadata: list[dict] = []

    for page in tqdm(pages, desc="Chunking pages"):
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadata.append({
                "source_url": page.get("url", ""),
                "title": page.get("title", ""),
                "chunk_index": i,
            })

    return all_texts, all_metadata


def embed_and_store(
    texts: list[str],
    metadatas: list[dict],
    collection: chromadb.Collection,
    model: SentenceTransformer,
) -> None:
    """Embed chunks in batches and upsert into ChromaDB."""
    total = len(texts)
    for start in tqdm(range(0, total, BATCH_SIZE), desc="Embedding & storing"):
        end = min(start + BATCH_SIZE, total)
        batch_texts = texts[start:end]
        batch_meta = metadatas[start:end]

        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        ids = [f"chunk_{start + i}" for i in range(len(batch_texts))]

        collection.upsert(
            ids=ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
        )


def main() -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0:
        print(f"Collection '{COLLECTION_NAME}' already has {existing:,} chunks.")
        print("Delete ./chroma_db/ and re-run to rebuild from scratch.")
        return

    print(f"Loading pages from {RAW_PAGES_DIR}/...")
    pages = load_pages(RAW_PAGES_DIR)
    print(f"Loaded {len(pages)} pages (skipped pages with < {MIN_TEXT_LENGTH} chars).")

    print("Chunking text...")
    texts, metadatas = chunk_pages(pages)
    print(f"Created {len(texts):,} chunks from {len(pages)} pages.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (downloads once ~90 MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding and storing chunks in ChromaDB...")
    embed_and_store(texts, metadatas, collection, model)

    final_count = collection.count()
    print(f"\nDone. {final_count:,} chunks stored in {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
