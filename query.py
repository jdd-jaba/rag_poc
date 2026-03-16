"""
Query the Google Cloud Next RAG system.

Uses LangChain LCEL to wire together:
  MultiQueryRetriever → prompt → ChatOllama (llama3.2, streamed) → answer

MultiQueryRetriever rephrases the question into several variants, retrieves
chunks for each, deduplicates, then passes all unique chunks as context.
This improves recall for conceptual questions where wording varies.

Usage:
    python query.py "When is Google Cloud Next 2026?"
    python query.py "Who are the keynote speakers?"

Run AFTER ingest.py has finished building the ChromaDB collection.
"""

import logging
import sys

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "google_cloud_next"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful assistant for the Google Cloud Next 2026 conference.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Keep your answer concise and accurate."""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def format_docs(docs: list[Document]) -> str:
    """Combine retrieved chunks into a single context string."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source_url", "")
        parts.append(f"{doc.page_content}\n[source: {source}]")
    return "\n\n---\n\n".join(parts)


def get_sources(docs: list[Document]) -> list[str]:
    """Return deduplicated source URLs from retrieved docs."""
    seen: set[str] = set()
    sources = []
    for doc in docs:
        url = doc.metadata.get("source_url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    return sources


MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates search queries."),
    ("human", (
        "Generate {n} different rephrasings of the following question to improve "
        "document retrieval. Each rephrasing should approach the topic from a "
        "slightly different angle. Output ONLY the questions, one per line, "
        "with no numbering or extra text.\n\nQuestion: {question}"
    )),
])


def generate_query_variants(question: str, llm: ChatOllama, n: int = 3) -> list[str]:
    """Use the LLM to rephrase the question into multiple search variants."""
    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question, "n": n})
    variants = [q.strip() for q in result.strip().splitlines() if q.strip()]
    # Always include the original question
    all_queries = [question] + variants
    return all_queries


def multi_query_retrieve(
    question: str,
    vectorstore: Chroma,
    llm: ChatOllama,
) -> tuple[list[Document], list[tuple[Document, float]]]:
    """Generate query variants, retrieve for each, and deduplicate results.

    Returns:
        unique_docs: deduplicated docs for context
        scored_hits: list of (doc, score) for the first (original) query,
                     used to display relevance scores to the user
    """
    queries = generate_query_variants(question, llm)

    print(f"\n[Query variants ({len(queries)} total)]")
    for i, q in enumerate(queries):
        label = "original" if i == 0 else f"variant {i}"
        print(f"  {label}: {q}")

    seen_content: set[str] = set()
    unique_docs: list[Document] = []
    scored_hits: list[tuple[Document, float]] = []

    for i, query in enumerate(queries):
        # Use similarity_search_with_score to get cosine distances
        hits = vectorstore.similarity_search_with_score(query, k=TOP_K)

        if i == 0:
            # Log scores for the original question
            print(f"\n[Retrieved chunks for original query]")
            for doc, score in hits:
                title = doc.metadata.get("title", "untitled")[:55]
                print(f"  score {score:.4f} | {title}")
            scored_hits = hits

        for doc, _ in hits:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)

    print(f"\n[Total unique chunks passed to LLM: {len(unique_docs)}]")
    return unique_docs, scored_hits


def build_components(chroma_dir: str, collection_name: str):
    """Build and return the vectorstore and LLM."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        client=chromadb.PersistentClient(path=chroma_dir),
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    return vectorstore, llm


def ask(question: str, vectorstore, llm) -> None:
    """Run a single question through the RAG pipeline and print the answer."""
    print(f"\nQuestion: {question}")
    print("-" * 60)

    print("\nGenerating query variants...")
    docs, _ = multi_query_retrieve(question, vectorstore, llm)
    sources = get_sources(docs)
    context = format_docs(docs)

    prompt_value = PROMPT_TEMPLATE.invoke({"context": context, "question": question})

    print("\n" + "-" * 60)
    print("Answer:\n")
    for chunk in llm.stream(prompt_value):
        print(chunk.content, end="", flush=True)
    print()

    if sources:
        print("\nSources:")
        for url in sources:
            print(f"  - {url}")


def main() -> None:
    try:
        vectorstore, llm = build_components(CHROMA_DIR, COLLECTION_NAME)
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        print("Make sure you have run ingest.py first.")
        sys.exit(1)

    # If a question was passed as a CLI argument, answer it and drop into the loop
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]), vectorstore, llm)

    print("\nEnter your question (or 'exit' / Ctrl+C to quit):")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        ask(question, vectorstore, llm)


if __name__ == "__main__":
    main()
