import ollama
from src.config import LLM_MODEL

ABSTAIN_PHRASE = "The provided documents do not contain enough information to answer this question."

SYSTEM_PROMPT = f"""You are a document question-answering assistant.
Rules:
1. Answer ONLY using the provided evidence. Do not use outside knowledge.
2. Write short, plain, self-contained sentences. Each sentence must state one fact and must make sense on its own (repeat the subject instead of using "it" or "this").
3. Do NOT mention evidence numbers, page numbers, or sources. Do NOT write phrases like "According to the evidence". Do NOT add a references list, notes, or bullet points.
4. Copy numbers, names, and dates exactly as they appear in the evidence. Do not compute, infer, or reword them.
5. Answer only the question asked. Do not add unrelated facts.
6. Do not add conclusions or inferences (such as "which is also...") that the evidence does not state. Copy the evidence wording instead.
7. If the evidence does not contain enough information to answer, reply with exactly this sentence and nothing else: "{ABSTAIN_PHRASE}"

"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    evidence_block = "\n\n".join(c["text"] for c in chunks)
    return f"""Evidence:
{evidence_block}

Question: {question}

Answer in plain sentences using only the evidence above."""


def generate_answer(question: str, chunks: list[dict]) -> str:
    prompt = build_prompt(question, chunks)
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0, "seed": 42},
    )
    return resp["message"]["content"].strip()


if __name__ == "__main__":
    from src.retriever import retrieve

    questions = [
        "What is the minimum attendance required to appear in examinations?",
        "What is the capital of France?",
        "What is the duration of the B.Tech program?",
    ]

    for q in questions:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
        chunks = retrieve(q)
        answer = generate_answer(q, chunks)
        print(f"\nAnswer:\n{answer}")