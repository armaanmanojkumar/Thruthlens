from src.data_loader import load_benchmark, build_source_chunks
from src.retriever import Retriever
from src.generator import AnswerGenerator
from src.verifier import Verifier
from src.scorer import semantic_similarity, lexical_overlap, citation_coverage, support_score

df = load_benchmark()
sources = build_source_chunks(df)

retriever = Retriever()
retriever.fit(sources)

generator = AnswerGenerator()
verifier = Verifier()

# Use a row directly from the dataset so the gold answer always exists
row = df.iloc[0]
query = row["question"]
gold_answer = row["gold_answer"]

results = retriever.retrieve(query)
context_chunks = results["chunk_text"].tolist()

answer = generator.generate(query, context_chunks)
verification = verifier.verify(query, answer, context_chunks)

print("Question:")
print(query)

print("\nGenerated answer:")
print(answer)

print("\nGold answer:")
print(gold_answer)

print("\nScores:")
print("Semantic similarity:", semantic_similarity(answer, gold_answer))
print("Lexical overlap:", lexical_overlap(answer, gold_answer))
print("Citation coverage:", citation_coverage(answer, verification.evidence))
print("Support score:", support_score(verification.verdict))