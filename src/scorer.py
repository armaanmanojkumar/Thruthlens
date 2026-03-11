from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_similarity(answer: str, gold_answer: str) -> float:
    if not answer or not gold_answer:
        return 0.0

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([answer, gold_answer])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return float(score)


def lexical_overlap(answer: str, gold_answer: str) -> float:
    if not answer or not gold_answer:
        return 0.0
    return float(SequenceMatcher(None, answer, gold_answer).ratio())


def citation_coverage(answer: str, evidence: list[str]) -> float:
    if not answer.strip() or not evidence:
        return 0.0

    answer_lower = answer.lower()
    matched = 0

    for ev in evidence:
        tokens = [token for token in ev.lower().split() if len(token) > 4]
        if any(token in answer_lower for token in tokens):
            matched += 1

    return matched / len(evidence)


def support_score(verdict: str) -> float:
    mapping = {
        "supported": 1.0,
        "partially_supported": 0.6,
        "unsupported": 0.2,
        "contradicted": 0.0,
    }
    return mapping.get(verdict, 0.0)