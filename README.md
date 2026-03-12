# TruthLens

TruthLens is a Python project for evaluating LLM factual reliability using retrieval-backed verification, structured verdicts, and benchmark metrics.

## Features

- Benchmark dataset with trusted context and gold answers
- Local semantic retrieval using SentenceTransformers + FAISS
- Grounded answer generation baseline
- Structured verification with verdict, confidence, reason, and evidence
- Metrics including semantic similarity, lexical overlap, citation coverage, and support score
- SQLite run logging
- Streamlit dashboard for saved runs and charts

## Project Structure

```text
truthlens/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── src/
│   ├── data_loader.py
│   ├── retriever.py
│   ├── generator.py
│   ├── verifier.py
│   ├── scorer.py
│   ├── storage.py
│   └── utils.py
├── data/
│   ├── benchmark.csv
│   └── sources/
├── results/
│   └── runs.db
└── notebooks/
    └── analysis.ipynb
    