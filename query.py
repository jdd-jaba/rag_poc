"""
Query the Python docs RAG system.

Uses LangChain LCEL to wire together:
  ChromaDB retriever → prompt → ChatOllama (llama3.2) → answer

Usage:
    python query.py "How does asyncio work?"
    python query.py "What is the difference between a list and a tuple?"

Run AFTER ingest.py has finished building the ChromaDB collection.
"""

import sys

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "python_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful Python documentation assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Keep your answer concise and accurate. Include code examples when helpful."""

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


def build_chain(chroma_dir: str, collection_name: str):
    """Build and return the LangChain LCEL retrieval chain."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        client=chromadb.PersistentClient(path=chroma_dir),
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def get_sources(retriever, question: str) -> list[str]:
    """Return deduplicated source URLs for the retrieved chunks."""
    docs = retriever.invoke(question)
    seen = set()
    sources = []
    for doc in docs:
        url = doc.metadata.get("source_url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append(url)
    return sources


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    print(f"\nQuestion: {question}")
    print("-" * 60)

    try:
        chain, retriever = build_chain(CHROMA_DIR, COLLECTION_NAME)
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        print("Make sure you have run ingest.py first.")
        sys.exit(1)

    print("Answer:\n")
    answer = chain.invoke(question)
    print(answer)

    sources = get_sources(retriever, question)
    if sources:
        print("\nSources:")
        for url in sources:
            print(f"  - {url}")


if __name__ == "__main__":
    main()
