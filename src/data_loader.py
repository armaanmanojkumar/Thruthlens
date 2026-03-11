from pathlib import Path
import pandas as pd


def load_benchmark(csv_path: str = "data/benchmark.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {csv_path}")
    return pd.read_csv(path)


def build_source_chunks(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "source_id": row["id"],
                "topic": row["topic"],
                "chunk_text": str(row["trusted_context"]),
            }
        )
    return pd.DataFrame(records)