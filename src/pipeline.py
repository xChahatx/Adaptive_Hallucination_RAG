from src.config import SUFFICIENCY_THRESHOLD, SUPPORT_THRESHOLD, TOP_K, MAX_CORRECTIONS
from src.retriever import retrieve
from src.generator import generate_answer
from src.verifier import verify_answer
from src.corrector import build_corrective_query


def dedupe_chunks(chunks: list[dict]) -> list[dict]:
    """Drop chunks with identical text (the corpus repeats some sections on several pages)."""
    seen, out = set(), []
    for c in chunks:
        key = " ".join(c["text"].split())
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def check_sufficiency(chunks: list[dict], threshold: float = SUFFICIENCY_THRESHOLD) -> bool:
    """Gate before generation: is the top reranked chunk even relevant?"""
    if not chunks:
        return False
    return chunks[0]["score"] >= threshold


def answer_question(question: str, prune: bool = True) -> dict:
    """
    Full cycle: retrieve -> sufficiency check -> generate -> verify ->
    correct (up to MAX_CORRECTIONS) -> decide accept / corrected_accept /
    partial_accept / model_abstain / abstain.
    prune=True keeps supported claims when some are unsupported;
    prune=False abstains on the whole answer (strict mode).
    """
    trace = {"question": question, "attempts": [], "decision": None, "reason": None}

    chunks = dedupe_chunks(retrieve(question, top_k=TOP_K))

    if not check_sufficiency(chunks):
        trace["decision"] = "abstain"
        trace["reason"] = "No sufficiently relevant evidence found in the document collection for this question."
        trace["final_answer"] = "I don't have enough evidence in the provided documents to answer this reliably."
        return trace

    answer = generate_answer(question, chunks)
    verdict = verify_answer(answer, chunks, threshold=SUPPORT_THRESHOLD)
    trace["attempts"].append({
        "chunks_used": [c["chunk_id"] for c in chunks],
        "answer": answer,
        "verdict": verdict,
    })

    corrections_used = 0
    while (not verdict["all_supported"] and not verdict["has_hedge"]
           and corrections_used < MAX_CORRECTIONS):
        unsupported = [c for c in verdict["claims"] if not c["supported"]]
        corrective_q = build_corrective_query(question, unsupported)

        # keep the original evidence, add new chunks, dedupe by text
        new_chunks = retrieve(corrective_q, top_k=TOP_K + 4)
        merged = dedupe_chunks(chunks + new_chunks)
        if len(merged) == len(chunks):
            break  # no new evidence; regenerating would repeat the same answer
        chunks = merged

        answer = generate_answer(question, chunks)  # still answer the ORIGINAL question
        verdict = verify_answer(answer, chunks, threshold=SUPPORT_THRESHOLD)

        trace["attempts"].append({
            "chunks_used": [c["chunk_id"] for c in chunks],
            "answer": answer,
            "verdict": verdict,
        })
        corrections_used += 1

    # Final decision
    if verdict["has_hedge"] or not verdict["claims"]:
        trace["decision"] = "model_abstain"
        trace["reason"] = "Generator stated the evidence does not contain the answer."
        trace["final_answer"] = "I don't have enough evidence in the provided documents to answer this reliably."
        trace["final_verdict"] = verdict
        return trace

    if verdict["all_supported"] and corrections_used == 0:
        trace["decision"] = "accept"
        trace["reason"] = "Answer is fully supported by retrieved evidence."
    elif verdict["all_supported"] and corrections_used > 0:
        trace["decision"] = "corrected_accept"
        trace["reason"] = f"Initial answer had unsupported claims; corrected after {corrections_used} retrieval retry(ies)."
    else:
        supported = [c["claim"] for c in verdict["claims"] if c["supported"]]
        if prune and supported:
            trace["decision"] = "partial_accept"
            trace["reason"] = (f"Kept {len(supported)} supported claim(s); removed "
                               f"{verdict['unsupported_count']} unsupported claim(s).")
            trace["final_answer"] = " ".join(supported)
            trace["final_verdict"] = verdict
            return trace
        trace["decision"] = "abstain"
        trace["reason"] = "Answer still contains unsupported claims after correction; abstaining to avoid a hallucinated response."
        trace["final_answer"] = "I don't have sufficient evidence in the provided documents to answer this reliably."
        trace["final_verdict"] = verdict
        return trace

    trace["final_answer"] = answer
    trace["final_verdict"] = verdict
    return trace


if __name__ == "__main__":
    questions = [
        "What is the duration of the B.Tech program?",
        "What is the capital of France?",
        "What is the minimum attendance required to appear in examinations?",
    ]

    for q in questions:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        print(f"top score: {retrieve(q, top_k=TOP_K)[0]['score']:.3f}")
        result = answer_question(q)
        print(f"DECISION: {result['decision'].upper()}")
        print(f"REASON: {result['reason']}")
        print(f"FINAL: {result['final_answer']}\n")

        for i, attempt in enumerate(result["attempts"]):
            print(f"--- attempt {i+1} ---")
            print(f"ANSWER: {attempt['answer']}\n")
            for c in attempt["verdict"]["claims"]:
                status = "✓" if c["supported"] else "✗"
                print(f"  [{status}] score={c['best_score']:.3f} page={c['best_page']} "
                      f"missing={c['missing']} entail={c['entail']}: {c['claim']}")