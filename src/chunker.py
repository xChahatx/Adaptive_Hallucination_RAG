import re
from pathlib import Path


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter — good enough for chunk boundaries.
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_pages(pages: list[dict], chunk_tokens: int, overlap_tokens: int) -> list[dict]:
    """
    pages: [{doc_id, page, text}]
    Returns: [{chunk_id, doc_id, page, text, n_tokens}]
    Chunks within a page first; if a page's sentences don't fill a chunk,
    they still form their own chunk (keeps page attribution honest —
    we don't merge across pages).
    """
    chunks = []
    chunk_counter = 0

    for page in pages:
        sentences = _split_sentences(page["text"])
        if not sentences:
            continue

        current: list[str] = []
        current_tokens = 0

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_tokens = _word_count(sent)

            if current_tokens + sent_tokens > chunk_tokens and current:
                chunk_text = " ".join(current)
                chunks.append({
                    "chunk_id": f"{page['doc_id']}_p{page['page']}_c{chunk_counter}",
                    "doc_id": page["doc_id"],
                    "page": page["page"],
                    "text": chunk_text,
                    "n_tokens": current_tokens,
                })
                chunk_counter += 1

                # build overlap: walk back from the end of `current`
                # until we've accumulated ~overlap_tokens
                overlap_sents = []
                overlap_count = 0
                for s in reversed(current):
                    st = _word_count(s)
                    if overlap_count + st > overlap_tokens:
                        break
                    overlap_sents.insert(0, s)
                    overlap_count += st
                current = overlap_sents
                current_tokens = overlap_count
                continue  # re-evaluate this sentence against the reset window

            current.append(sent)
            current_tokens += sent_tokens
            i += 1

        if current:
            chunk_text = " ".join(current)
            chunks.append({
                "chunk_id": f"{page['doc_id']}_p{page['page']}_c{chunk_counter}",
                "doc_id": page["doc_id"],
                "page": page["page"],
                "text": chunk_text,
                "n_tokens": current_tokens,
            })
            chunk_counter += 1

    return chunks


if __name__ == "__main__":
    from src.config import RAW_DIR, CHUNK_TOKENS, CHUNK_OVERLAP
    from src.parser import parse_folder

    pages = parse_folder(RAW_DIR)
    chunks = chunk_pages(pages, CHUNK_TOKENS, CHUNK_OVERLAP)

    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print("\n--- sample chunk ---")
    print(chunks[0]["chunk_id"], "| tokens:", chunks[0]["n_tokens"])
    print(chunks[0]["text"][:400])
    print("\n--- token size distribution ---")
    sizes = [c["n_tokens"] for c in chunks]
    print("min:", min(sizes), "max:", max(sizes), "avg:", sum(sizes)//len(sizes))