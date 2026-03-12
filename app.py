import sys
import os
sys.path.append(os.path.abspath("src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from data_loader import load_benchmark, build_source_chunks
from retriever import Retriever
from generator import AnswerGenerator
from verifier import Verifier
from scorer import semantic_similarity, lexical_overlap, citation_coverage, support_score
from storage import init_db, save_run, load_runs

# -----------------------------
# Page config
# -----------------------------

st.set_page_config(
    page_title="TruthLens",
    page_icon="🔎",
    layout="wide"
)

# -----------------------------
# Custom Dark Theme Styling
# -----------------------------

st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.stApp {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: #ffffff;
}

.metric-box {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title Section
# -----------------------------

st.title("🔎 TruthLens")
st.subheader("LLM Hallucination Detection & Reliability Benchmark")

st.markdown(
"""
TruthLens evaluates whether generated answers are **supported by evidence**.

Pipeline:

1️⃣ Retrieve relevant evidence  
2️⃣ Generate answer  
3️⃣ Verify factual support  
4️⃣ Score reliability  
5️⃣ Track hallucination metrics
"""
)

init_db()

# -----------------------------
# Setup system
# -----------------------------

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

# -----------------------------
# Tabs
# -----------------------------

tab1, tab2 = st.tabs(["🧠 Run Verification", "📊 Dashboard"])

# =====================================================
# TAB 1
# =====================================================

with tab1:

    st.markdown("### Ask a Question")

    question = st.text_input("Enter your question")

    if st.button("Run Analysis") and question:

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

        # -----------------------------
        # Answer
        # -----------------------------

        st.markdown("## Generated Answer")
        st.write(answer)

        # -----------------------------
        # Verdict
        # -----------------------------

        verdict = verification.verdict

        if verdict == "supported":
            st.success(f"✅ Verdict: {verdict}")

        elif verdict == "partially_supported":
            st.warning(f"⚠️ Verdict: {verdict}")

        else:
            st.error(f"❌ Verdict: {verdict}")

        st.write("Confidence:", verification.confidence)
        st.write("Reason:", verification.reason)

        # -----------------------------
        # Metrics
        # -----------------------------

        st.markdown("### Reliability Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Semantic Similarity", f"{sem_score:.2f}")
        col2.metric("Lexical Overlap", f"{lex_score:.2f}")
        col3.metric("Citation Coverage", f"{cite_score:.2f}")
        col4.metric("Support Score", f"{supp_score:.2f}")

        # -----------------------------
        # Evidence
        # -----------------------------

        st.markdown("### Retrieved Evidence")

        st.dataframe(results, use_container_width=True)


# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.markdown("### Benchmark Dashboard")

    runs_df = load_runs()

    if runs_df.empty:

        st.info("No runs saved yet.")

    else:

        hallucination_rate = runs_df["verdict"].isin(
            ["unsupported", "contradicted"]
        ).mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Runs", len(runs_df))
        col2.metric("Hallucination Rate", f"{hallucination_rate:.1%}")
        col3.metric("Avg Support Score", f"{runs_df['support_score'].mean():.2f}")
        col4.metric("Avg Citation Coverage", f"{runs_df['citation_coverage'].mean():.2f}")

        # -----------------------------
        # Verdict Chart
        # -----------------------------

        verdict_counts = runs_df["verdict"].value_counts().reset_index()
        verdict_counts.columns = ["verdict", "count"]

        fig1 = px.bar(
            verdict_counts,
            x="verdict",
            y="count",
            color="verdict",
            title="Answer Verification Distribution",
            template="plotly_dark"
        )

        st.plotly_chart(fig1, use_container_width=True)

        # -----------------------------
        # Topic Chart
        # -----------------------------

        if "topic" in runs_df.columns:

            topic_scores = runs_df.groupby("topic", as_index=False)["support_score"].mean()

            fig2 = px.bar(
                topic_scores,
                x="topic",
                y="support_score",
                title="Average Support Score by Topic",
                template="plotly_dark"
            )

            st.plotly_chart(fig2, use_container_width=True)

        # -----------------------------
        # Confidence vs Similarity
        # -----------------------------

        fig3 = px.scatter(
            runs_df,
            x="confidence",
            y="semantic_similarity",
            color="verdict",
            title="Confidence vs Semantic Similarity",
            template="plotly_dark"
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Saved Runs")

        st.dataframe(runs_df, use_container_width=True)