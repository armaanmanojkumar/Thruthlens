from src.data_loader import load_benchmark, build_source_chunks
from src.retriever import Retriever
from src.generator import AnswerGenerator

df = load_benchmark()
sources = build_source_chunks(df)

retriever = Retriever()
retriever.fit(sources)

generator = AnswerGenerator()

query = "What is overfitting?"
results = retriever.retrieve(query)
context_chunks = results["chunk_text"].tolist()

answer = generator.generate(query, context_chunks)

print("Question:", query)
print("\nRetrieved context:")
for i, chunk in enumerate(context_chunks, start=1):
    print(f"{i}. {chunk}")

print("\nGenerated answer:")
print(answer)