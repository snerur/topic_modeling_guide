"""Turftopic modeling with multiple model types."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional


TURFTOPIC_MODELS = {
    "Semantic Signal Separation (S3)": "S3",
    "Gaussian Mixture Model (GMM)": "GMM",
    "Clustering Topic Model (CTM)": "ClusteringTopicModel",
    "KeyNMF": "KeyNMF",
    "Autoencoding Topics (FASTopic)": "FASTopic",
}

# Models that support n_components="auto"
AUTO_TOPIC_MODELS = {"GMM", "KeyNMF", "ClusteringTopicModel"}


def get_model_descriptions() -> Dict[str, str]:
    """Return descriptions of available Turftopic models."""
    return {
        "Semantic Signal Separation (S3)": "Discovers latent semantic signals using ICA on embeddings",
        "Gaussian Mixture Model (GMM)": "Soft clustering using Gaussian mixtures in embedding space",
        "Clustering Topic Model (CTM)": "Clustering-based approach with contextual embeddings",
        "KeyNMF": "Keyword extraction + NMF for interpretable topics",
        "Autoencoding Topics (FASTopic)": "Fast autoencoder-based topic model",
    }


def run_turftopic(texts: List[str], model_type: str = "S3",
                   n_topics: int = 10, encoder_model: str = "all-MiniLM-L6-v2",
                   embeddings: Optional[np.ndarray] = None,
                   auto_topics: bool = False) -> Dict[str, Any]:
    """Run Turftopic model and return comprehensive results."""
    import turftopic

    # Determine n_components: "auto" for supported models, else int
    n_comp = "auto" if (auto_topics and model_type in AUTO_TOPIC_MODELS) else n_topics

    with st.spinner(f"Fitting Turftopic {model_type} model..."):
        if model_type == "S3":
            model = turftopic.SemanticSignalSeparation(
                n_components=n_topics,  # S3 doesn't support auto
                encoder=encoder_model,
            )
        elif model_type == "GMM":
            model = turftopic.GMM(
                n_components=n_comp,
                encoder=encoder_model,
            )
        elif model_type == "ClusteringTopicModel":
            model = turftopic.ClusteringTopicModel(
                n_reduce_to=n_topics if not auto_topics else None,
                encoder=encoder_model,
            )
        elif model_type == "KeyNMF":
            model = turftopic.KeyNMF(
                n_components=n_comp,
                encoder=encoder_model,
            )
        elif model_type == "FASTopic":
            model = turftopic.FASTopic(
                n_components=n_topics,  # FASTopic doesn't support auto
                encoder=encoder_model,
            )
        else:
            model = turftopic.KeyNMF(
                n_components=n_comp,
                encoder=encoder_model,
            )

        doc_topic_matrix = model.fit_transform(texts)

    # Extract topics
    n_actual = doc_topic_matrix.shape[1]
    if auto_topics:
        st.info(f"Auto-detected **{n_actual}** topics")

    topics = {}
    topic_word_weights = {}

    try:
        vocab = model.get_vocab()
        components = model.components_
        for i in range(n_actual):
            top_indices = components[i].argsort()[-15:][::-1]
            words = [vocab[idx] for idx in top_indices]
            weights = [float(components[i][idx]) for idx in top_indices]
            topics[f"Topic {i+1}"] = words
            topic_word_weights[f"Topic {i+1}"] = list(zip(words, weights))
    except Exception:
        for i in range(n_actual):
            topics[f"Topic {i+1}"] = [f"word_{j}" for j in range(10)]
            topic_word_weights[f"Topic {i+1}"] = [(f"word_{j}", 1.0 / (j + 1)) for j in range(10)]

    # Document-topic DataFrame
    doc_topic_df = pd.DataFrame(
        doc_topic_matrix,
        columns=[f"Topic {i+1}" for i in range(n_actual)]
    )
    doc_topic_df["dominant_topic"] = doc_topic_df.idxmax(axis=1)
    doc_topic_df["dominant_topic_prob"] = doc_topic_df.iloc[:, :n_actual].max(axis=1)
    doc_topic_df["text_preview"] = [t[:100] + "..." if len(t) > 100 else t for t in texts]

    topic_corr = np.corrcoef(doc_topic_matrix.T) if doc_topic_matrix.shape[1] > 1 else None
    topic_sizes = doc_topic_df["dominant_topic"].value_counts().sort_index()

    return {
        "model": model,
        "model_type": model_type,
        "doc_topic_matrix": doc_topic_matrix,
        "topics": topics,
        "topic_word_weights": topic_word_weights,
        "doc_topic_df": doc_topic_df,
        "topic_correlations": topic_corr,
        "topic_sizes": topic_sizes,
        "n_topics": n_actual,
        "texts": texts,
    }


def label_topics_with_llm(results: Dict[str, Any],
                           openai_api_key: str,
                           model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Use OpenAI to name and describe topics via turftopic's OpenAIAnalyzer."""
    import os
    os.environ["OPENAI_API_KEY"] = openai_api_key

    try:
        from turftopic.analyzers.openai import OpenAIAnalyzer
        analyzer = OpenAIAnalyzer(model_name=model_name)
        model = results["model"]
        analysis = model.analyze_topics(analyzer, use_documents=True)

        results["llm_topic_names"] = analysis.topic_names
        results["llm_topic_descriptions"] = analysis.topic_descriptions
        if analysis.document_summaries:
            results["llm_document_summaries"] = analysis.document_summaries
    except Exception as e:
        st.warning(f"Turftopic LLM labeling failed: {e}")

    return results


def plot_topic_words(results: Dict[str, Any], n_words: int = 10) -> go.Figure:
    """Plot top words per topic."""
    from plotly.subplots import make_subplots

    n_topics = results["n_topics"]
    cols = min(3, n_topics)
    rows = (n_topics + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Topic {i+1}" for i in range(n_topics)])

    for idx in range(n_topics):
        topic_name = f"Topic {idx + 1}"
        word_weights = results["topic_word_weights"].get(topic_name, [])[:n_words]
        if not word_weights:
            continue
        words = [w for w, _ in word_weights][::-1]
        weights = [w for _, w in word_weights][::-1]

        row = idx // cols + 1
        col = idx % cols + 1

        fig.add_trace(
            go.Bar(y=words, x=weights, orientation="h", name=topic_name,
                   marker_color=px.colors.qualitative.Set2[idx % len(px.colors.qualitative.Set2)]),
            row=row, col=col
        )

    fig.update_layout(height=300 * rows, showlegend=False,
                      title_text=f"Turftopic ({results['model_type']}): Top Words per Topic")
    return fig


def plot_topic_distribution(results: Dict[str, Any]) -> go.Figure:
    """Plot topic size distribution."""
    sizes = results["topic_sizes"]
    fig = px.bar(
        x=sizes.index, y=sizes.values,
        labels={"x": "Topic", "y": "Number of Documents"},
        title=f"Turftopic ({results['model_type']}): Document Distribution",
        color=sizes.index,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_topic_correlations(results: Dict[str, Any]) -> Optional[go.Figure]:
    """Plot topic correlation heatmap."""
    corr = results.get("topic_correlations")
    if corr is None:
        return None

    n = results["n_topics"]
    labels = [f"Topic {i+1}" for i in range(n)]

    fig = go.Figure(data=go.Heatmap(
        z=corr, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title=f"Turftopic ({results['model_type']}): Topic Correlations",
                      height=500)
    return fig


def plot_doc_topic_heatmap(results: Dict[str, Any], max_docs: int = 50) -> go.Figure:
    """Plot document-topic probability heatmap."""
    matrix = results["doc_topic_matrix"][:max_docs]
    n = results["n_topics"]
    labels = [f"Topic {i+1}" for i in range(n)]
    doc_labels = [f"Doc {i+1}" for i in range(matrix.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=labels, y=doc_labels,
        colorscale="YlOrRd",
    ))
    fig.update_layout(title=f"Document-Topic Matrix (first {max_docs} docs)",
                      height=max(400, max_docs * 15))
    return fig


def plot_topic_evolution(results: Dict[str, Any], dates: pd.Series,
                         freq: str = "Y") -> Optional[go.Figure]:
    """Plot topic evolution over time."""
    if dates is None or dates.empty:
        return None

    doc_topic_df = results["doc_topic_df"].copy()
    doc_topic_df["date"] = dates.values[:len(doc_topic_df)]
    doc_topic_df["period"] = doc_topic_df["date"].dt.to_period(freq).astype(str)

    topic_cols = [f"Topic {i+1}" for i in range(results["n_topics"])]
    evolution = doc_topic_df.groupby("period")[topic_cols].mean()

    fig = go.Figure()
    for col in topic_cols:
        fig.add_trace(go.Scatter(
            x=evolution.index, y=evolution[col],
            mode="lines+markers", name=col
        ))

    fig.update_layout(
        title=f"Turftopic ({results['model_type']}): Topic Evolution",
        xaxis_title="Time Period", yaxis_title="Average Topic Proportion",
        height=500
    )
    return fig
