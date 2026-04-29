"""BunkaTopics modeling with visualization."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional


def run_bunkatopics(texts: List[str], n_topics: int = 10,
                     embedding_model: str = "all-MiniLM-L6-v2",
                     min_count_terms: int = 2) -> Dict[str, Any]:
    """Run BunkaTopics and return comprehensive results."""
    from sentence_transformers import SentenceTransformer

    with st.spinner("Fitting BunkaTopics model..."):
        # Import with langchain compatibility handling
        try:
            from bunkatopics import Bunka
        except ImportError as e:
            if "pydantic_v1" in str(e) or "langchain" in str(e):
                st.error(
                    "BunkaTopics has a dependency conflict with langchain/pydantic. "
                    "Fix: `pip install --upgrade langchain-core langchain-community pydantic` "
                    "or `pip install bunkatopics --upgrade`"
                )
                raise
            raise

        # Pass the SentenceTransformer model object to the Bunka constructor
        emb_model = SentenceTransformer(embedding_model)
        bunka = Bunka(embedding_model=emb_model)
        bunka.fit(texts)

        # Get topics
        topics_df = bunka.get_topics(n_clusters=n_topics, name_length=5)

    # Extract topic information
    topics = {}
    topic_word_weights = {}

    if topics_df is not None and not topics_df.empty:
        for _, row in topics_df.iterrows():
            topic_id = row.get("topic_id", row.name)
            topic_label = f"Topic {topic_id}"

            # Get terms - try multiple column names used across versions
            terms = None
            for col_name in ["topic_name", "name", "topic_label", "terms"]:
                if col_name in row.index:
                    terms = row[col_name]
                    break

            if terms is None:
                terms = str(topic_id)

            if isinstance(terms, str):
                # Split by common delimiters
                for sep in [" | ", "|", ", ", "; "]:
                    if sep in terms:
                        word_list = [w.strip() for w in terms.split(sep) if w.strip()]
                        break
                else:
                    word_list = [terms.strip()] if terms.strip() else []
            elif isinstance(terms, list):
                word_list = terms
            else:
                word_list = [str(terms)]

            topics[topic_label] = word_list[:15]
            topic_word_weights[topic_label] = [
                (w, 1.0 / (i + 1)) for i, w in enumerate(word_list[:15])
            ]

    # Get document-topic information from bunka.docs (list of Document objects)
    doc_records = []
    for doc in bunka.docs:
        doc_records.append({
            "content": getattr(doc, "content", ""),
            "topic_id": getattr(doc, "topic_id", None),
            "doc_id": getattr(doc, "doc_id", None),
        })
    docs_df = pd.DataFrame(doc_records)

    # Build document-topic matrix approximation
    topic_col = "topic_id" if "topic_id" in docs_df.columns and docs_df["topic_id"].notna().any() else None

    doc_topic_data = {}
    if topic_col:
        unique_topics = sorted(docs_df[topic_col].dropna().unique())
        for t in unique_topics:
            col_name = f"Topic {t}"
            doc_topic_data[col_name] = (docs_df[topic_col] == t).astype(float)

    doc_topic_df = pd.DataFrame(doc_topic_data)
    if topic_col:
        doc_topic_df["dominant_topic"] = docs_df[topic_col].values
    else:
        doc_topic_df["dominant_topic"] = "unknown"
    doc_topic_df["text_preview"] = [
        t[:100] + "..." if len(t) > 100 else t for t in texts[:len(doc_topic_df)]
    ]

    # Topic sizes
    if topic_col:
        topic_sizes = docs_df[topic_col].value_counts().sort_index()
    else:
        topic_sizes = pd.Series(dtype=int)

    return {
        "model": bunka,
        "topics": topics,
        "topic_word_weights": topic_word_weights,
        "topics_df": topics_df,
        "docs_df": docs_df,
        "doc_topic_df": doc_topic_df,
        "topic_sizes": topic_sizes,
        "n_topics": len(topics),
        "texts": texts,
    }


def label_topics_with_llm(results: Dict[str, Any],
                           openai_api_key: str,
                           model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Use OpenAI via langchain to generate clean topic names for BunkaTopics."""
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model_name, api_key=openai_api_key)
        bunka = results["model"]
        cleaned_df = bunka.get_clean_topic_name(llm=llm, use_doc=True)

        # Re-extract topics with cleaned names
        llm_names = []
        llm_descriptions = []
        if cleaned_df is not None and not cleaned_df.empty:
            for _, row in cleaned_df.iterrows():
                name = None
                for col in ["topic_name", "name", "topic_label"]:
                    if col in row.index and pd.notna(row[col]):
                        name = str(row[col])
                        break
                llm_names.append(name or "Unknown")
                # Use topic words as description
                topic_id = row.get("topic_id", row.name)
                topic_key = f"Topic {topic_id}"
                words = results["topics"].get(topic_key, [])
                llm_descriptions.append(", ".join(words[:8]) if words else name or "")

        results["llm_topic_names"] = llm_names
        results["llm_topic_descriptions"] = llm_descriptions
        results["topics_df"] = cleaned_df
    except Exception as e:
        st.warning(f"BunkaTopics LLM labeling failed: {e}")

    return results


def plot_bunka_map(results: Dict[str, Any]) -> Optional[Any]:
    """Get BunkaTopics interactive topic map."""
    try:
        fig = results["model"].visualize_topics()
        return fig
    except Exception as e:
        st.warning(f"BunkaTopics map visualization failed: {e}")
        return None


def plot_bunka_docs(results: Dict[str, Any]) -> Optional[Any]:
    """Get BunkaTopics document visualization using topic repartition or dimensions."""
    try:
        model = results["model"]
        if hasattr(model, "get_topic_repartition"):
            fig = model.get_topic_repartition()
            return fig
        elif hasattr(model, "visualize_dimensions"):
            fig = model.visualize_dimensions()
            return fig
        else:
            st.info("Document visualization not available in this BunkaTopics version.")
            return None
    except Exception as e:
        st.warning(f"BunkaTopics document visualization failed: {e}")
        return None


def plot_topic_words(results: Dict[str, Any], n_words: int = 10) -> go.Figure:
    """Plot top words per topic."""
    from plotly.subplots import make_subplots

    n_topics = results["n_topics"]
    if n_topics == 0:
        return go.Figure().update_layout(title="No topics found")

    cols = min(3, n_topics)
    rows = (n_topics + cols - 1) // cols

    topic_names = list(results["topic_word_weights"].keys())
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=topic_names[:n_topics])

    for idx, topic_name in enumerate(topic_names[:n_topics]):
        word_weights = results["topic_word_weights"][topic_name][:n_words]
        if not word_weights:
            continue
        words = [w for w, _ in word_weights][::-1]
        weights = [w for _, w in word_weights][::-1]

        row = idx // cols + 1
        col = idx % cols + 1

        fig.add_trace(
            go.Bar(y=words, x=weights, orientation="h", name=topic_name,
                   marker_color=px.colors.qualitative.Pastel[idx % len(px.colors.qualitative.Pastel)]),
            row=row, col=col
        )

    fig.update_layout(height=300 * rows, showlegend=False,
                      title_text="BunkaTopics: Top Words per Topic")
    return fig


def plot_topic_distribution(results: Dict[str, Any]) -> go.Figure:
    """Plot topic size distribution."""
    sizes = results["topic_sizes"]
    if sizes.empty:
        return go.Figure().update_layout(title="No topic distribution data")

    fig = px.bar(
        x=[str(x) for x in sizes.index], y=sizes.values,
        labels={"x": "Topic", "y": "Number of Documents"},
        title="BunkaTopics: Document Distribution",
        color=[str(x) for x in sizes.index],
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(showlegend=False)
    return fig
