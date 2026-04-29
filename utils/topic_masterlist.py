"""Cross-model topic masterlist: deduplicate and merge topics from all models."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional


def _collect_topics(all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Gather all topics from every model into a single DataFrame.

    Returns a DataFrame with columns:
        model, topic_id, words (list), word_str, llm_name, llm_description
    """
    rows = []
    for model_name, results in all_results.items():
        if not isinstance(results, dict):
            continue

        # Get word-based topics (works for LDA, Turftopic, STM, BunkaTopics)
        topics = results.get("topics", {})
        topic_word_weights = results.get("topic_word_weights", {})
        llm_names = results.get("llm_topic_names", [])
        llm_descs = results.get("llm_topic_descriptions", [])

        # Use topic_words if available (BERTopic), else topics dict
        word_source = topics
        if not isinstance(word_source, dict):
            word_source = results.get("topic_words", {})
        if not isinstance(word_source, dict):
            word_source = {}

        for idx, (topic_key, words) in enumerate(word_source.items()):
            if not isinstance(words, list):
                words = [str(words)]
            rows.append({
                "model": model_name,
                "topic_id": topic_key,
                "words": words[:10],
                "word_str": " ".join(str(w) for w in words[:10]),
                "llm_name": llm_names[idx] if idx < len(llm_names) else "",
                "llm_description": llm_descs[idx] if idx < len(llm_descs) else "",
            })

    return pd.DataFrame(rows)


def _compute_topic_similarity(topic_df: pd.DataFrame,
                               embedding_model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Compute pairwise cosine similarity between topics using their word embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)
    word_strs = topic_df["word_str"].tolist()
    embeddings = model.encode(word_strs, show_progress_bar=False)

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    similarity = normalized @ normalized.T
    return similarity


def _cluster_topics(similarity: np.ndarray, threshold: float = 0.65) -> List[int]:
    """Greedy agglomerative clustering: merge topics with similarity > threshold.

    Returns a list of cluster IDs (one per topic row).
    """
    n = similarity.shape[0]
    cluster_ids = list(range(n))

    # Build pairs sorted by similarity (descending)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((similarity[i, j], i, j))
    pairs.sort(reverse=True)

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for sim, i, j in pairs:
        if sim < threshold:
            break
        union(i, j)

    # Relabel clusters to 0..k-1
    root_to_id = {}
    for i in range(n):
        root = find(i)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        cluster_ids[i] = root_to_id[root]

    return cluster_ids


def build_masterlist(all_results: Dict[str, Dict[str, Any]],
                     similarity_threshold: float = 0.65,
                     embedding_model: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Build a deduplicated master topic list from all model results.

    Returns dict with:
        masterlist_df: DataFrame of master topics
        topic_df: per-model topics with cluster assignment
        similarity: pairwise similarity matrix
    """
    topic_df = _collect_topics(all_results)
    if topic_df.empty:
        return {"masterlist_df": pd.DataFrame(), "topic_df": topic_df, "similarity": None}

    with st.spinner("Computing topic similarity for masterlist..."):
        similarity = _compute_topic_similarity(topic_df, embedding_model)
        cluster_ids = _cluster_topics(similarity, threshold=similarity_threshold)
        topic_df["cluster"] = cluster_ids

    # Build master topics: one entry per cluster
    master_rows = []
    for cid in sorted(set(cluster_ids)):
        cluster_topics = topic_df[topic_df["cluster"] == cid]
        # Merge words across models, preserving order, removing dupes
        all_words = []
        seen = set()
        for words in cluster_topics["words"]:
            for w in words:
                wl = str(w).lower()
                if wl not in seen:
                    seen.add(wl)
                    all_words.append(str(w))

        source_models = sorted(cluster_topics["model"].unique())
        source_topic_ids = cluster_topics["topic_id"].tolist()

        # Prefer LLM name if available
        llm_name = ""
        llm_desc = ""
        for _, row in cluster_topics.iterrows():
            if row["llm_name"]:
                llm_name = row["llm_name"]
                llm_desc = row["llm_description"]
                break

        master_rows.append({
            "Master Topic": f"MT-{cid + 1}",
            "Top Words": ", ".join(all_words[:12]),
            "LLM Label": llm_name,
            "LLM Description": llm_desc,
            "Models": ", ".join(source_models),
            "Model Count": len(source_models),
            "Source Topics": "; ".join(
                f"{r['model']}:{r['topic_id']}" for _, r in cluster_topics.iterrows()
            ),
        })

    masterlist_df = pd.DataFrame(master_rows)
    return {
        "masterlist_df": masterlist_df,
        "topic_df": topic_df,
        "similarity": similarity,
    }


def plot_topic_similarity_heatmap(masterlist_result: Dict[str, Any]) -> Optional[go.Figure]:
    """Plot pairwise topic similarity heatmap across all models."""
    sim = masterlist_result.get("similarity")
    topic_df = masterlist_result.get("topic_df")
    if sim is None or topic_df is None or topic_df.empty:
        return None

    labels = [f"{row['model']}:{row['topic_id']}" for _, row in topic_df.iterrows()]

    fig = go.Figure(data=go.Heatmap(
        z=sim, x=labels, y=labels,
        colorscale="Viridis", zmin=0, zmax=1,
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Similarity: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Cross-Model Topic Similarity",
        height=max(500, len(labels) * 20),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def plot_cluster_network(masterlist_result: Dict[str, Any],
                          threshold: float = 0.5) -> Optional[go.Figure]:
    """Plot topics as a network graph colored by cluster."""
    sim = masterlist_result.get("similarity")
    topic_df = masterlist_result.get("topic_df")
    if sim is None or topic_df is None or topic_df.empty:
        return None

    n = len(topic_df)
    clusters = topic_df["cluster"].values

    # Layout: circular within each cluster
    import math
    cluster_centers = {}
    unique_clusters = sorted(set(clusters))
    for i, cid in enumerate(unique_clusters):
        angle = 2 * math.pi * i / max(len(unique_clusters), 1)
        cluster_centers[cid] = (3 * math.cos(angle), 3 * math.sin(angle))

    x_pos = np.zeros(n)
    y_pos = np.zeros(n)
    cluster_counts = {}
    for i in range(n):
        cid = clusters[i]
        cx, cy = cluster_centers[cid]
        count = cluster_counts.get(cid, 0)
        angle = 2 * math.pi * count / max(sum(1 for c in clusters if c == cid), 1)
        x_pos[i] = cx + 0.8 * math.cos(angle)
        y_pos[i] = cy + 0.8 * math.sin(angle)
        cluster_counts[cid] = count + 1

    # Edges
    edge_x, edge_y = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > threshold:
                edge_x.extend([x_pos[i], x_pos[j], None])
                edge_y.extend([y_pos[i], y_pos[j], None])

    fig = go.Figure()
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color="rgba(150,150,150,0.4)"),
            hoverinfo="none",
        ))

    labels = [f"{row['model']}:{row['topic_id']}" for _, row in topic_df.iterrows()]
    hover_text = [
        f"{row['model']}:{row['topic_id']}<br>{row['word_str'][:60]}"
        for _, row in topic_df.iterrows()
    ]

    fig.add_trace(go.Scatter(
        x=x_pos.tolist(), y=y_pos.tolist(),
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=8),
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=14,
            color=clusters.tolist(),
            colorscale="Set1",
            showscale=False,
            line=dict(width=1, color="white"),
        ),
    ))

    fig.update_layout(
        title="Cross-Model Topic Clusters",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )
    return fig
