class AnswerGenerator:
    def generate(self, question: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return "Insufficient evidence in provided sources."

        top_chunk = context_chunks[0].strip()

        if not top_chunk:
            return "Insufficient evidence in provided sources."

        return top_chunk