import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.source_df = None
        self.embeddings = None

    def _embed_texts(self, texts):
        vectors = self.model.encode(texts)
        return np.array(vectors).astype("float32")

    def fit(self, source_df: pd.DataFrame):
        self.source_df = source_df.reset_index(drop=True)
        texts = self.source_df["chunk_text"].tolist()

        self.embeddings = self._embed_texts(texts)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 3):
        if self.index is None:
            raise ValueError("Retriever has not been fitted yet.")

        query_vector = self._embed_texts([query])
        distances, indices = self.index.search(query_vector, top_k)

        rows = self.source_df.iloc[indices[0]].copy()
        rows["distance"] = distances[0]

        cosine_scores = cosine_similarity(query_vector, self.embeddings[indices[0]])[0]
        rows["cosine_similarity"] = cosine_scores

        return rows.reset_index(drop=True)