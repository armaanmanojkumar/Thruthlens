import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = "results/runs.db"


def init_db():
    Path("results").mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            topic TEXT,
            difficulty TEXT,
            retrieved_context TEXT,
            answer TEXT,
            verdict TEXT,
            confidence REAL,
            reason TEXT,
            evidence TEXT,
            semantic_similarity REAL,
            lexical_overlap REAL,
            citation_coverage REAL,
            support_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_run(record: dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO runs (
            question,
            topic,
            difficulty,
            retrieved_context,
            answer,
            verdict,
            confidence,
            reason,
            evidence,
            semantic_similarity,
            lexical_overlap,
            citation_coverage,
            support_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["question"],
        record["topic"],
        record["difficulty"],
        record["retrieved_context"],
        record["answer"],
        record["verdict"],
        record["confidence"],
        record["reason"],
        record["evidence"],
        record["semantic_similarity"],
        record["lexical_overlap"],
        record["citation_coverage"],
        record["support_score"],
    ))

    conn.commit()
    conn.close()


def load_runs() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY timestamp DESC", conn)
    conn.close()
    return df