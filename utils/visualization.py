"""Shared visualization utilities for topic modeling."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional


def plot_embedding_space(embeddings: np.ndarray, labels: Optional[np.ndarray] = None,
                          title: str = "Document Embedding Space") -> go.Figure:
    """Plot 2D UMAP projection of document embeddings."""
    from umap import UMAP

    if embeddings.shape[1] > 2:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(embeddings)
    else:
        coords = embeddings

    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    if labels is not None:
        df["label"] = [str(l) for l in labels]
        fig = px.scatter(df, x="x", y="y", color="label",
                         title=title, opacity=0.6,
                         color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        fig = px.scatter(df, x="x", y="y", title=title, opacity=0.6)

    fig.update_layout(height=600, width=800)
    fig.update_traces(marker=dict(size=5))
    return fig


def plot_topic_comparison(results_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Compare topic distributions across multiple models."""
    fig = go.Figure()

    for model_name, results in results_dict.items():
        if not isinstance(results, dict):
            continue
        if "topic_sizes" in results:
            sizes = results["topic_sizes"]
            if isinstance(sizes, pd.Series):
                fig.add_trace(go.Bar(
                    name=model_name,
                    x=[str(x) for x in sizes.index],
                    y=sizes.values,
                ))

    fig.update_layout(
        title="Topic Distribution Comparison Across Models",
        xaxis_title="Topic",
        yaxis_title="Number of Documents",
        barmode="group",
        height=500,
    )
    return fig


def plot_topic_similarity_network(corr_matrix: np.ndarray, threshold: float = 0.3,
                                    title: str = "Topic Similarity Network") -> go.Figure:
    """Plot topic correlations as a network graph."""
    n = corr_matrix.shape[0]
    labels = [f"Topic {i+1}" for i in range(n)]

    # Create circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_nodes = np.cos(angles)
    y_nodes = np.sin(angles)

    # Edges
    edge_x, edge_y = [], []
    edge_weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > threshold:
                edge_x.extend([x_nodes[i], x_nodes[j], None])
                edge_y.extend([y_nodes[i], y_nodes[j], None])
                edge_weights.append(abs(corr_matrix[i, j]))

    fig = go.Figure()

    # Add edges
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1, color="rgba(150,150,150,0.5)"),
            hoverinfo="none"
        ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_nodes.tolist(), y=y_nodes.tolist(), mode="markers+text",
        text=labels, textposition="top center",
        marker=dict(size=20, color=list(range(n)), colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Topic")),
        hoverinfo="text"
    ))

    fig.update_layout(
        title=title, showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500, width=600,
    )
    return fig


def create_summary_table(results: Dict[str, Any], model_name: str) -> pd.DataFrame:
    """Create a summary table for topics."""
    rows = []
    topics = results.get("topics", {})
    topic_sizes = results.get("topic_sizes", pd.Series())

    # Handle case where topics is a list instead of dict (can happen with R bridge)
    if isinstance(topics, list):
        topics = {f"Topic {i+1}": (t if isinstance(t, list) else [str(t)])
                  for i, t in enumerate(topics)}
    elif not isinstance(topics, dict):
        topics = {}

    for topic_name, words in topics.items():
        if not isinstance(words, list):
            words = [str(words)]
        size = topic_sizes.get(topic_name, 0) if isinstance(topic_sizes, pd.Series) else 0
        rows.append({
            "Model": model_name,
            "Topic": topic_name,
            "Top Words": ", ".join(str(w) for w in words[:8]),
            "Documents": size,
        })

    return pd.DataFrame(rows)
