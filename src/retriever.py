from sentence_transformers import CrossEncoder
from src.config import RERANK_MODEL, TOP_K, RERANK_KEEP
from src.embedder import embed_text
from src.store import get_client, get_collection, query as store_query

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def retrieve(question: str, top_k: int = TOP_K, keep: int = RERANK_KEEP) -> list[dict]:
    """
    Returns reranked chunks: [{chunk_id, doc_id, page, text, score}]
    score is the cross-encoder relevance score (higher = better).
    """
    client = get_client()
    collection = get_collection(client)

    q_emb = embed_text(question)
    raw = store_query(collection, q_emb, top_k=top_k)

    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    ids = raw["ids"][0]

    if not docs:
        return []

    reranker = get_reranker()
    pairs = [[question, d] for d in docs]
    scores = reranker.predict(pairs)

    candidates = [
        {
            "chunk_id": ids[i],
            "doc_id": metas[i]["doc_id"],
            "page": metas[i]["page"],
            "text": docs[i],
            "score": float(scores[i]),
        }
        for i in range(len(docs))
    ]
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:keep]


if __name__ == "__main__":
    questions = [
        "What is the scheme of examination for B.Tech first year?",
        "What is the minimum attendance required to appear in examinations?",
        "What is the capital of France?",  # should retrieve nothing relevant — sanity check
    ]

    for q in questions:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
        results = retrieve(q)
        for i, r in enumerate(results):
            print(f"\n[{i+1}] score={r['score']:.4f} doc={r['doc_id']} page={r['page']}")
            print(r["text"][:200])