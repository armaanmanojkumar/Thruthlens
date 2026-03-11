from src.data_loader import load_benchmark, build_source_chunks
from src.retriever import Retriever
from src.generator import AnswerGenerator
from src.verifier import Verifier

df = load_benchmark()
sources = build_source_chunks(df)

retriever = Retriever()
retriever.fit(sources)

generator = AnswerGenerator()
verifier = Verifier()

query = "What is overfitting?"
results = retriever.retrieve(query)
context_chunks = results["chunk_text"].tolist()

answer = generator.generate(query, context_chunks)
verification = verifier.verify(query, answer, context_chunks)

print("Question:", query)
print("\nAnswer:")
print(answer)
print("\nVerification:")
print(verification.model_dump())