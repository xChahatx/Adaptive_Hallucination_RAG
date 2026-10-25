import chromadb
from pathlib import Path
from src.config import DB_DIR, EMBED_MODEL


def get_client():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DB_DIR))


def get_collection(client, name: str = "documents"):
    return client.get_or_create_collection(
        name=name,
        metadata={"embedding_model": EMBED_MODEL},
    )


def upsert_chunks(collection, chunks: list[dict]):
    """
    chunks: [{chunk_id, doc_id, page, text, n_tokens, embedding}]
    """
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"doc_id": c["doc_id"], "page": c["page"], "n_tokens": c["n_tokens"]}
            for c in chunks
        ],
    )


def query(collection, query_embedding: list[float], top_k: int = 8):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )


if __name__ == "__main__":
    from src.config import RAW_DIR, CHUNK_TOKENS, CHUNK_OVERLAP, TOP_K
    from src.parser import parse_folder
    from src.chunker import chunk_pages
    from src.embedder import embed_chunks, embed_text

    pages = parse_folder(RAW_DIR)
    chunks = chunk_pages(pages, CHUNK_TOKENS, CHUNK_OVERLAP)

    print(f"embedding all {len(chunks)} chunks (this will take a few minutes)...")
    chunks = embed_chunks(chunks)

    client = get_client()
    collection = get_collection(client)
    upsert_chunks(collection, chunks)
    print(f"\nstored {collection.count()} chunks in ChromaDB at {DB_DIR}")

    # quick sanity query
    test_q = "What is the scheme of examination for B.Tech first year?"
    q_emb = embed_text(test_q)
    results = query(collection, q_emb, top_k=TOP_K)

    print(f"\n--- test query: '{test_q}' ---")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        print(f"\n[{i+1}] dist={dist:.4f} doc={meta['doc_id']} page={meta['page']}")
        print(doc[:200])