from src.data_loader import load_benchmark, build_source_chunks
from src.retriever import Retriever

df = load_benchmark()
sources = build_source_chunks(df)

retriever = Retriever()
retriever.fit(sources)

query = "What is overfitting?"
results = retriever.retrieve(query)

print("Retrieved results:")
print(results)