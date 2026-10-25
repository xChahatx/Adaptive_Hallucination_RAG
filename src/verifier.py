import re
import spacy
from sentence_transformers import SentenceTransformer, util
_nlp = None
_embedder = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def extract_claims(answer: str) -> list[str]:
    """
    Extract candidate factual claims: sentences containing named entities,
    numbers, or dates. Skips hedging/meta sentences about the evidence itself.
    """
    nlp = get_nlp()
    doc = nlp(answer)
    claims = []
    evidence_talk_words = {"information", "evidence", "mention", "specify",
                            "specified", "find", "found", "contain", "provided",
                            "source", "data", "detail", "details"}
    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        text = _strip_citation_scaffold(text)
        if len(text.split()) < 4:
            continue
        tokens = {t.lower_ for t in sent}
        has_neg = any(t.dep_ == "neg" for t in sent) or bool(tokens & {"no", "none", "nothing", "unable", "cannot"})
        is_hedge = has_neg and bool(tokens & evidence_talk_words)
        if is_hedge:
            continue
        has_entity = any(ent.label_ in ("PERSON", "ORG", "GPE", "DATE", "CARDINAL",
                                        "PERCENT", "ORDINAL", "QUANTITY")
                                        for ent in sent.ents)
        has_number = bool(re.search(r"\d", text))
        if has_entity or has_number:
            claims.append(text)
    return claims

_SEP = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*)"
_NUMLIST = rf"\d+(?:{_SEP}\d+)*"

def _strip_citation_scaffold(claim: str) -> str:
    """Remove citation scaffolding so page/evidence numbers aren't treated as facts."""
    claim = re.sub(r"\[[^\]]*\]", "", claim)
    claim = re.sub(rf"\(\s*(?:evidence|pages?)\s+{_NUMLIST}[^)]*\)", "", claim, flags=re.I)
    claim = re.sub(rf"\b(?:evidence|pages?)\s+{_NUMLIST}", "", claim, flags=re.I)
    claim = re.sub(r"\s+", " ", claim)
    return claim.strip(" ,;")

def _clean_for_split(text: str) -> str:
    """Normalize PDF list numbering so spaCy splits clauses properly."""
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"(?<=[.:;])\s+\d{1,2}\.\s+(?=[A-Z(])", " ", text)
    return text

def _prep_evidence(chunks):
    nlp, units, seen = get_nlp(), [], set()
    for c in chunks:
        for s in nlp(_clean_for_split(c["text"])).sents:
            t = s.text.strip()
            if t and t not in seen:
                seen.add(t)
                units.append((t, c["page"]))
    if not units:
        return [], None
    embs = get_embedder().encode([u[0] for u in units], convert_to_tensor=True)
    return units, embs

def _key_tokens(text):
    doc = get_nlp()(text)
    nums = set(re.findall(r"\d+(?:\.\d+)?", text))
    ents = {e.text.lower() for e in doc.ents if e.label_ in ("PERSON", "ORG", "GPE")}
    return nums | ents

_WORD_NUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
             "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
             "first": "1", "second": "2", "third": "3", "fourth": "4",
             "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8"}

def _norm_numbers(text: str) -> str:
    return re.sub(r"\b(" + "|".join(_WORD_NUM) + r")\b",
                  lambda m: _WORD_NUM[m.group(1).lower()], text, flags=re.I)

def _in_context(k: str, context: str) -> bool:
    if re.fullmatch(r"\d+(?:\.\d+)?", k):
        return re.search(rf"(?<![\d.]){re.escape(k)}(?!\d|\.\d)", context) is not None
    return re.search(rf"\b{re.escape(k)}\b", context) is not None

NLI_THRESHOLD = 0.5
_nli = None

def get_nli():
    global _nli
    if _nli is None:
        from sentence_transformers import CrossEncoder
        _nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli

def entailment_prob(premise: str, hypothesis: str) -> float:
    import numpy as np
    logits = np.array(get_nli().predict([(premise, hypothesis)])[0])
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return float(probs[1])  # label order: contradiction, entailment, neutral

def verify_claim(claim, units, embs, threshold):
    if not units:
        return {"claim": claim, "supported": False, "best_score": -1.0,
                "best_page": None, "missing": [], "entail": 0.0}
    cleaned = _strip_citation_scaffold(claim) or claim
    c_emb = get_embedder().encode(cleaned, convert_to_tensor=True)
    scores = util.cos_sim(c_emb, embs)[0]
    idx = int(scores.argmax())
    best_page = units[idx][1]
    top = scores.topk(min(3, len(units))).indices.tolist()
    premise = " ".join(units[i][0] for i in top)
    context = _norm_numbers(premise).lower()
    missing = [k for k in _key_tokens(cleaned) if not _in_context(k, context)]
    score = scores[idx].item()
    entail = entailment_prob(premise, cleaned)
    return {"claim": claim,
            "supported": score >= threshold and not missing and entail >= NLI_THRESHOLD,
            "best_score": round(score, 4), "best_page": best_page,
            "missing": missing, "entail": round(entail, 3)}

def has_hedge(answer: str) -> bool:
    """True if any sentence says the evidence lacks the answer."""
    evidence_words = {"information", "evidence", "mention", "specify",
                      "specified", "find", "found", "contain", "provided",
                      "source", "data", "detail", "details"}
    for sent in get_nlp()(answer).sents:
        tokens = {t.lower_ for t in sent}
        neg = any(t.dep_ == "neg" for t in sent) or bool(tokens & {"no", "none", "nothing", "unable", "cannot"})
        if neg and tokens & evidence_words:
            return True
    return False


def verify_answer(answer: str, evidence_chunks: list[dict], threshold: float = 0.5) -> dict:
    hedge = has_hedge(answer)
    claims = extract_claims(answer)
    if not claims:
        # No checkable claims: likely a pure abstention/hedge answer
        return {"claims": [], "all_supported": True, "unsupported_count": 0, "has_hedge": hedge}

    units, embs = _prep_evidence(evidence_chunks)
    results = [verify_claim(c, units, embs, threshold) for c in claims]
    unsupported = [r for r in results if not r["supported"]]

    return {
        "claims": results,
        "all_supported": len(unsupported) == 0,
        "unsupported_count": len(unsupported),
        "has_hedge": hedge,
    }


if __name__ == "__main__":
    from src.retriever import retrieve
    from src.generator import generate_answer
    from src.config import RERANK_MODEL, TOP_K, RERANK_KEEP

    questions = [
        "What is the minimum attendance required to appear in examinations?",
        "What is the capital of France?",
    ]

    for q in questions:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
        chunks = retrieve(q)
        answer = generate_answer(q, chunks)
        print(f"\nAnswer:\n{answer}\n")

        verdict = verify_answer(answer, chunks)
        print(f"--- Verification ---")
        print(f"claims found: {len(verdict['claims'])}, unsupported: {verdict['unsupported_count']}")
        for c in verdict["claims"]:
            status = "✓ SUPPORTED" if c["supported"] else "✗ UNSUPPORTED"
            print(f"  [{status}] score={c['best_score']:.3f} page={c['best_page']} missing={c['missing']}: {c['claim'][:100]}")