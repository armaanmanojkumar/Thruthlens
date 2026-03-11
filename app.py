import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_benchmark, build_source_chunks
from src.retriever import Retriever
from src.generator import AnswerGenerator
from src.verifier import Verifier
from src.scorer import semantic_similarity, lexical_overlap, citation_coverage, support_score
from src.storage import init_db, save_run, load_runs

st.set_page_config(page_title="TruthLens", layout="wide")
st.title("TruthLens — Hallucination Detection and Reliability Benchmark")

init_db()


@st.cache_resource
def setup_system():
    df = load_benchmark()
    sources = build_source_chunks(df)

    retriever = Retriever()
    retriever.fit(sources)

    generator = AnswerGenerator()
    verifier = Verifier()

    return df, retriever, generator, verifier


df, retriever, generator, verifier = setup_system()

tab1, tab2 = st.tabs(["Run Verification", "Dashboard"])

with tab1:
    st.subheader("Ask a question")
    question = st.text_input("Enter your question")

    if st.button("Run") and question:
        results = retriever.retrieve(question)
        context_chunks = results["chunk_text"].tolist()

        answer = generator.generate(question, context_chunks)
        verification = verifier.verify(question, answer, context_chunks)

        gold_match = df[df["question"] == question]

        if not gold_match.empty:
            gold_answer = gold_match["gold_answer"].iloc[0]
            topic = gold_match["topic"].iloc[0]
            difficulty = gold_match["difficulty"].iloc[0]
        else:
            gold_answer = ""
            topic = "unknown"
            difficulty = "unknown"

        sem_score = semantic_similarity(answer, gold_answer)
        lex_score = lexical_overlap(answer, gold_answer)
        cite_score = citation_coverage(answer, verification.evidence)
        supp_score = support_score(verification.verdict)

        record = {
            "question": question,
            "topic": topic,
            "difficulty": difficulty,
            "retrieved_context": "\n\n".join(context_chunks),
            "answer": answer,
            "verdict": verification.verdict,
            "confidence": verification.confidence,
            "reason": verification.reason,
            "evidence": " | ".join(verification.evidence),
            "semantic_similarity": sem_score,
            "lexical_overlap": lex_score,
            "citation_coverage": cite_score,
            "support_score": supp_score,
        }

        save_run(record)

        st.markdown("### Generated Answer")
        st.write(answer)

        st.markdown("### Verification Result")
        st.json(verification.model_dump())

        st.markdown("### Retrieved Context")
        st.dataframe(results, use_container_width=True)

        st.markdown("### Scores")
        score_df = pd.DataFrame([{
            "semantic_similarity": sem_score,
            "lexical_overlap": lex_score,
            "citation_coverage": cite_score,
            "support_score": supp_score,
        }])
        st.dataframe(score_df, use_container_width=True)

with tab2:
    st.subheader("Benchmark Dashboard")

    runs_df = load_runs()

    if runs_df.empty:
        st.info("No runs saved yet.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        hallucination_rate = runs_df["verdict"].isin(["unsupported", "contradicted"]).mean()

        col1.metric("Total Runs", len(runs_df))
        col2.metric("Hallucination Rate", f"{hallucination_rate:.1%}")
        col3.metric("Avg Support Score", f"{runs_df['support_score'].mean():.2f}")
        col4.metric("Avg Citation Coverage", f"{runs_df['citation_coverage'].mean():.2f}")

        verdict_counts = runs_df["verdict"].value_counts().reset_index()
        verdict_counts.columns = ["verdict", "count"]
        fig1 = px.bar(verdict_counts, x="verdict", y="count", title="Verdict Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        if "topic" in runs_df.columns:
            topic_scores = runs_df.groupby("topic", as_index=False)["support_score"].mean()
            fig2 = px.bar(topic_scores, x="topic", y="support_score", title="Average Support Score by Topic")
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.scatter(
            runs_df,
            x="confidence",
            y="semantic_similarity",
            color="verdict",
            title="Confidence vs Semantic Similarity"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Saved Runs")
        st.dataframe(runs_df, use_container_width=True)