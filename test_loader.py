from src.data_loader import load_benchmark, build_source_chunks

df = load_benchmark()
print("Benchmark data:")
print(df.head())

chunks = build_source_chunks(df)
print("\nSource chunks:")
print(chunks.head())