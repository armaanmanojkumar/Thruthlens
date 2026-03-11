from src.data_loader import load_benchmark, build_source_chunks
from src.retriever import Retriever
from src.generator import AnswerGenerator
from src.verifier import Verifier
from src.scorer import semantic_similarity, lexical_overlap, citation_coverage, support_score
from src.storage import init_db, save_run, load_runs

df = load_benchmark()
sources = build_source_chunks(df)

retriever = Retriever()
retriever.fit(sources)

generator = AnswerGenerator()
verifier = Verifier()

init_db()

row = df.iloc[0]
query = row["question"]
gold_answer = row["gold_answer"]
topic = row["topic"]
difficulty = row["difficulty"]

results = retriever.retrieve(query)
context_chunks = results["chunk_text"].tolist()

answer = generator.generate(query, context_chunks)
verification = verifier.verify(query, answer, context_chunks)

record = {
    "question": query,
    "topic": topic,
    "difficulty": difficulty,
    "retrieved_context": "\n\n".join(context_chunks),
    "answer": answer,
    "verdict": verification.verdict,
    "confidence": verification.confidence,
    "reason": verification.reason,
    "evidence": " | ".join(verification.evidence),
    "semantic_similarity": semantic_similarity(answer, gold_answer),
    "lexical_overlap": lexical_overlap(answer, gold_answer),
    "citation_coverage": citation_coverage(answer, verification.evidence),
    "support_score": support_score(verification.verdict),
}

save_run(record)

runs = load_runs()
print(runs.head())