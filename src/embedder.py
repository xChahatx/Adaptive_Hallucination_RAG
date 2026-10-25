import ollama
from src.config import EMBED_MODEL


def embed_text(text: str) -> list[float]:
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    chunks: [{chunk_id, doc_id, page, text, n_tokens}]
    Returns the same list with an added "embedding" key.
    Mutates and returns for convenience.
    """
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embed_text(chunk["text"])
        if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
            print(f"  embedded {i + 1}/{len(chunks)}")
    return chunks


if __name__ == "__main__":
    from src.config import RAW_DIR, CHUNK_TOKENS, CHUNK_OVERLAP
    from src.parser import parse_folder
    from src.chunker import chunk_pages

    pages = parse_folder(RAW_DIR)
    chunks = chunk_pages(pages, CHUNK_TOKENS, CHUNK_OVERLAP)

    # Sanity check on a handful first, not all 511 — fail fast if something's wrong
    sample = chunks[:5]
    embed_chunks(sample)

    print(f"\nembedded {len(sample)} sample chunks")
    print("embedding dim:", len(sample[0]["embedding"]))
    print("first 5 values:", sample[0]["embedding"][:5])