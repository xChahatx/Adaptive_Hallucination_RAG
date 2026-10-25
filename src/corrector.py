from src.config import MAX_CORRECTIONS, SUPPORT_THRESHOLD, TOP_K
from src.retriever import retrieve
from src.generator import generate_answer
from src.verifier import verify_answer


def build_corrective_query(original_question: str, unsupported_claims: list[dict]) -> str:
    """Rewrite the query to target the specific unsupported claims."""
    unsupported_texts = " ".join(c["claim"] for c in unsupported_claims)
    return f"{original_question} {unsupported_texts}"


def answer_with_correction(question: str) -> dict:
    """
    Runs the full cycle: retrieve -> generate -> verify -> (correct once if needed).
    Returns a trace dict with the final answer, verdict, and what happened.
    """
    chunks = retrieve(question, top_k=TOP_K)
    answer = generate_answer(question, chunks)
    verdict = verify_answer(answer, chunks, threshold=SUPPORT_THRESHOLD)

    trace = {
        "question": question,
        "attempts": [{
            "chunks_used": [c["chunk_id"] for c in chunks],
            "answer": answer,
            "verdict": verdict,
        }],
        "corrected": False,
    }

    corrections_used = 0
    while not verdict["all_supported"] and corrections_used < MAX_CORRECTIONS:
        unsupported = [c for c in verdict["claims"] if not c["supported"]]
        corrective_q = build_corrective_query(question, unsupported)

        chunks = retrieve(corrective_q, top_k=TOP_K + 4)  # widen the net
        answer = generate_answer(question, chunks)  # still answer the ORIGINAL question
        verdict = verify_answer(answer, chunks, threshold=SUPPORT_THRESHOLD)

        trace["attempts"].append({
            "chunks_used": [c["chunk_id"] for c in chunks],
            "answer": answer,
            "verdict": verdict,
        })
        trace["corrected"] = True
        corrections_used += 1

    trace["final_answer"] = answer
    trace["final_verdict"] = verdict
    return trace


if __name__ == "__main__":
    questions = [
        "What is the minimum attendance required to appear in examinations?",
        "What is the capital of France?",
    ]

    for q in questions:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
        trace = answer_with_correction(q)

        print(f"attempts: {len(trace['attempts'])} | corrected: {trace['corrected']}")
        for i, attempt in enumerate(trace["attempts"]):
            v = attempt["verdict"]
            print(f"\n--- attempt {i+1} ---")
            print(f"claims: {len(v['claims'])}, unsupported: {v['unsupported_count']}")
            print(f"answer: {attempt['answer'][:200]}")

        print(f"\nFINAL VERDICT: all_supported={trace['final_verdict']['all_supported']}")