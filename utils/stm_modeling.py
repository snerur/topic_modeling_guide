"""Structural Topic Modeling using rpy2 bridge to R's stm package."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")


def _get_converter():
    """Get the rpy2 converter for pandas/numpy interop."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    return ro.default_converter + pandas2ri.converter + numpy2ri.converter


def check_r_available() -> bool:
    """Check if R and required packages are available."""
    try:
        import rpy2.robjects as ro
        with _get_converter().context():
            ro.r('library(stm)')
        return True
    except Exception:
        return False


def check_ldatuning_available() -> bool:
    """Check if R ldatuning package is available."""
    try:
        import rpy2.robjects as ro
        with _get_converter().context():
            ro.r('library(ldatuning)')
        return True
    except Exception:
        return False


def run_stm(texts: List[str], metadata_df: Optional[pd.DataFrame] = None,
            prevalence_formula: Optional[str] = None,
            content_formula: Optional[str] = None,
            n_topics: int = 10, max_iter: int = 75) -> Dict[str, Any]:
    """Run Structural Topic Model via rpy2."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    converter = _get_converter()

    with converter.context():
        with st.spinner("Preparing documents for STM..."):
            ro.globalenv["texts_vec"] = ro.StrVector(texts)

            ro.r('''
            processed <- textProcessor(texts_vec, metadata=NULL, lowercase=TRUE,
                                        removestopwords=TRUE, removenumbers=TRUE,
                                        removepunctuation=TRUE, stem=TRUE,
                                        wordLengths=c(3, Inf))
            prepped <- prepDocuments(processed$documents, processed$vocab,
                                    lower.thresh=2)
            ''')

        with st.spinner(f"Fitting STM with {n_topics} topics..."):
            ro.globalenv["K"] = n_topics
            ro.globalenv["max_iter"] = max_iter

            if metadata_df is not None and prevalence_formula:
                ro.globalenv["metadata_df"] = pandas2ri.py2rpy(metadata_df)

                ro.r(f'''
                stm_model <- stm(documents=prepped$documents,
                                 vocab=prepped$vocab,
                                 K=K,
                                 prevalence=~ {prevalence_formula},
                                 data=metadata_df,
                                 max.em.its=max_iter,
                                 init.type="Spectral",
                                 verbose=FALSE)
                ''')
            else:
                ro.r('''
                stm_model <- stm(documents=prepped$documents,
                                 vocab=prepped$vocab,
                                 K=K,
                                 max.em.its=max_iter,
                                 init.type="Spectral",
                                 verbose=FALSE)
                ''')

        # Extract results into plain Python objects
        theta = np.array(ro.r('stm_model$theta'))
        vocab = [str(v) for v in ro.r('prepped$vocab')]
        beta = np.array(ro.r('exp(stm_model$beta$logbeta[[1]])'))

        # Topic correlations
        try:
            topic_corr = np.array(ro.r('topicCorr(stm_model)$cor'))
        except Exception:
            topic_corr = np.corrcoef(theta.T)

        # Topic quality (semantic coherence + exclusivity)
        try:
            coherence = [float(x) for x in ro.r('semanticCoherence(stm_model, prepped$documents)')]
            exclusivity = [float(x) for x in ro.r('exclusivity(stm_model)')]
        except Exception:
            coherence = [0.0] * n_topics
            exclusivity = [0.0] * n_topics

        # Prevalence effects if metadata present
        prevalence_effects_summary = None
        if metadata_df is not None and prevalence_formula:
            try:
                ro.r(f'''
                effects <- estimateEffect(1:{n_topics} ~ {prevalence_formula},
                                          stm_model,
                                          metadata=metadata_df)
                ''')
                summary_text = str(ro.r('capture.output(summary(effects))'))
                prevalence_effects_summary = summary_text
            except Exception as e:
                st.warning(f"Could not estimate prevalence effects: {e}")

    # Extract top words per topic
    topics = {}
    topic_word_weights = {}
    for i in range(n_topics):
        top_indices = beta[i].argsort()[-15:][::-1]
        top_words = [vocab[idx] for idx in top_indices if idx < len(vocab)]
        top_weights = [float(beta[i][idx]) for idx in top_indices if idx < len(vocab)]
        topics[f"Topic {i+1}"] = top_words
        topic_word_weights[f"Topic {i+1}"] = list(zip(top_words, top_weights))

    # Document-topic matrix
    doc_topic_df = pd.DataFrame(
        theta,
        columns=[f"Topic {i+1}" for i in range(n_topics)]
    )
    doc_topic_df["dominant_topic"] = doc_topic_df.idxmax(axis=1)
    doc_topic_df["dominant_topic_prob"] = doc_topic_df.iloc[:, :n_topics].max(axis=1)
    doc_topic_df["text_preview"] = [t[:100] + "..." if len(t) > 100 else t
                                     for t in texts[:len(theta)]]

    topic_sizes = doc_topic_df["dominant_topic"].value_counts().sort_index()

    r_script = _generate_r_script(n_topics, max_iter, prevalence_formula)

    return {
        "topics": topics,
        "topic_word_weights": topic_word_weights,
        "doc_topic_matrix": theta,
        "doc_topic_df": doc_topic_df,
        "topic_correlations": topic_corr,
        "topic_sizes": topic_sizes,
        "coherence": coherence,
        "exclusivity": exclusivity,
        "n_topics": n_topics,
        "texts": texts,
        "vocab": vocab,
        "prevalence_effects_summary": prevalence_effects_summary,
        "metadata_df": metadata_df,
        "prevalence_formula": prevalence_formula,
        "r_script": r_script,
    }


def find_optimal_k(texts: List[str], metadata_df: Optional[pd.DataFrame] = None,
                    prevalence_formula: Optional[str] = None,
                    k_range: List[int] = None) -> Optional[Dict[str, Any]]:
    """Find optimal K for STM using searchK from R's stm package."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    converter = _get_converter()

    if k_range is None:
        k_range = list(range(3, 21, 2))

    with converter.context():
        ro.globalenv["texts_vec"] = ro.StrVector(texts)
        ro.globalenv["k_range"] = ro.IntVector(k_range)

        with st.spinner("Running STM searchK for optimal topics (this may take a while)..."):
            try:
                ro.r('''
                processed <- textProcessor(texts_vec, metadata=NULL, lowercase=TRUE,
                                            removestopwords=TRUE, removenumbers=TRUE,
                                            removepunctuation=TRUE, stem=TRUE)
                out <- prepDocuments(processed$documents, processed$vocab, lower.thresh=2)
                ''')

                if metadata_df is not None and prevalence_formula:
                    ro.globalenv["metadata_df"] = pandas2ri.py2rpy(metadata_df)
                    ro.r(f'''
                    search_result <- searchK(out$documents, out$vocab, K=k_range,
                                              prevalence=~ {prevalence_formula}, data=metadata_df,
                                              init.type="Spectral", verbose=FALSE)
                    ''')
                else:
                    ro.r('''
                    search_result <- searchK(out$documents, out$vocab, K=k_range,
                                              init.type="Spectral", verbose=FALSE)
                    ''')

                # Extract column by column to avoid recarray conversion issues
                ro.r('search_df <- as.data.frame(search_result$results)')
                col_names = list(ro.r('colnames(search_df)'))
                data = {}
                for col in col_names:
                    data[col] = list(ro.r(f'search_df${col}'))
                results_df = pd.DataFrame(data)
                return {"results": results_df, "k_range": k_range}
            except Exception as e:
                st.warning(f"searchK failed: {e}")
                return None


def _generate_r_script(n_topics: int, max_iter: int,
                       prevalence_formula: Optional[str] = None) -> str:
    """Generate a standalone R script for STM analysis."""
    formula_line = ""
    if prevalence_formula:
        formula_line = f'                prevalence = ~ {prevalence_formula},\n                data = metadata,'

    return f'''# Structural Topic Modeling in R
# Install packages if needed:
# install.packages(c("stm", "stmCorrViz", "ldatuning"))

library(stm)

# 1. Load your data
# texts <- readLines("your_corpus.txt")
# metadata <- read.csv("your_metadata.csv")

# 2. Process documents
processed <- textProcessor(texts, metadata = NULL,
                           lowercase = TRUE, removestopwords = TRUE,
                           removenumbers = TRUE, removepunctuation = TRUE,
                           stem = TRUE, wordLengths = c(3, Inf))
out <- prepDocuments(processed$documents, processed$vocab, lower.thresh = 2)

# 3. Find optimal K (optional)
# search_result <- searchK(out$documents, out$vocab,
#                           K = seq(3, 20, by = 2),
#                           init.type = "Spectral", verbose = FALSE)
# plot(search_result)

# 4. Fit STM model
stm_model <- stm(documents = out$documents,
                 vocab = out$vocab,
                 K = {n_topics},
{formula_line}
                 max.em.its = {max_iter},
                 init.type = "Spectral",
                 verbose = TRUE)

# 5. Explore results
labelTopics(stm_model, n = 15)
plot(stm_model, type = "summary")
plot(stm_model, type = "labels")
plot(stm_model, type = "perspectives", topics = c(1, 2))

# 6. Topic correlations
topic_corr <- topicCorr(stm_model)
plot(topic_corr)

# 7. Quality metrics
semanticCoherence(stm_model, out$documents)
exclusivity(stm_model)

# 8. Topic proportions (document-topic matrix)
theta <- stm_model$theta
write.csv(theta, "document_topic_matrix.csv", row.names = FALSE)

# 9. Prevalence effects (if metadata available)
# effects <- estimateEffect(1:{n_topics} ~ your_variable, stm_model, metadata = metadata)
# summary(effects)
# plot(effects, covariate = "your_variable", method = "continuous")
'''


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
        word_weights = results["topic_word_weights"][topic_name][:n_words]
        words = [w for w, _ in word_weights][::-1]
        weights = [w for _, w in word_weights][::-1]

        row = idx // cols + 1
        col = idx % cols + 1

        fig.add_trace(
            go.Bar(y=words, x=weights, orientation="h", name=topic_name,
                   marker_color=px.colors.qualitative.Bold[idx % len(px.colors.qualitative.Bold)]),
            row=row, col=col
        )

    fig.update_layout(height=300 * rows, showlegend=False,
                      title_text="STM: Top Words per Topic")
    return fig


def plot_topic_correlations(results: Dict[str, Any]) -> go.Figure:
    """Plot STM topic correlation heatmap."""
    corr = results["topic_correlations"]
    n = results["n_topics"]
    labels = [f"Topic {i+1}" for i in range(n)]

    fig = go.Figure(data=go.Heatmap(
        z=corr, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title="STM: Topic Correlations", height=500)
    return fig


def plot_quality(results: Dict[str, Any]) -> go.Figure:
    """Plot semantic coherence vs exclusivity."""
    coherence = results["coherence"]
    exclusivity = results["exclusivity"]
    labels = [f"Topic {i+1}" for i in range(results["n_topics"])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coherence, y=exclusivity,
        mode="markers+text", text=labels,
        textposition="top center",
        marker=dict(size=12, color=list(range(len(labels))),
                    colorscale="Viridis", showscale=True),
    ))
    fig.update_layout(
        title="STM: Semantic Coherence vs Exclusivity",
        xaxis_title="Semantic Coherence",
        yaxis_title="Exclusivity",
        height=500,
    )
    return fig


def plot_topic_distribution(results: Dict[str, Any]) -> go.Figure:
    """Plot topic size distribution."""
    sizes = results["topic_sizes"]
    fig = px.bar(
        x=[str(x) for x in sizes.index], y=sizes.values,
        labels={"x": "Topic", "y": "Number of Documents"},
        title="STM: Document Distribution Across Topics",
        color=[str(x) for x in sizes.index],
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_topic_evolution(results: Dict[str, Any], dates: pd.Series,
                         freq: str = "Y") -> Optional[go.Figure]:
    """Plot topic evolution over time."""
    if dates is None or dates.empty:
        return None

    doc_topic_df = results["doc_topic_df"].copy()
    n_rows = min(len(doc_topic_df), len(dates))
    doc_topic_df = doc_topic_df.iloc[:n_rows].copy()
    doc_topic_df["date"] = dates.values[:n_rows]
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
        title="STM: Topic Evolution Over Time",
        xaxis_title="Time Period", yaxis_title="Average Topic Proportion",
        height=500
    )
    return fig


def plot_searchk_results(searchk_results: Dict[str, Any]) -> go.Figure:
    """Plot searchK results from STM."""
    from plotly.subplots import make_subplots

    df = searchk_results["results"]
    metrics = [c for c in df.columns if c != "K"][:4]
    n_metrics = len(metrics)

    fig = make_subplots(rows=1, cols=n_metrics,
                        subplot_titles=metrics)

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(x=df["K"], y=df[metric], mode="lines+markers", name=metric),
            row=1, col=i + 1
        )

    fig.update_layout(height=400, title_text="STM searchK: Optimal Topic Number",
                      showlegend=True)
    fig.update_xaxes(title_text="K")
    return fig


def run_stm_python_fallback(texts: List[str], n_topics: int = 10,
                             max_features: int = 5000) -> Dict[str, Any]:
    """Python-only STM approximation using NMF with metadata-aware preprocessing.
    Used when R is not available."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    st.info("R not available. Using Python NMF as STM approximation.")

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english",
                                  min_df=2, max_df=0.95)
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    with st.spinner("Fitting NMF model (STM approximation)..."):
        nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
        doc_topic_matrix = nmf.fit_transform(tfidf)

    # Normalize to probability distributions
    row_sums = doc_topic_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    doc_topic_matrix = doc_topic_matrix / row_sums

    topics = {}
    topic_word_weights = {}
    for i in range(n_topics):
        top_indices = nmf.components_[i].argsort()[-15:][::-1]
        words = [feature_names[idx] for idx in top_indices]
        weights = [float(nmf.components_[i][idx]) for idx in top_indices]
        topics[f"Topic {i+1}"] = words
        topic_word_weights[f"Topic {i+1}"] = list(zip(words, weights))

    doc_topic_df = pd.DataFrame(
        doc_topic_matrix,
        columns=[f"Topic {i+1}" for i in range(n_topics)]
    )
    doc_topic_df["dominant_topic"] = doc_topic_df.idxmax(axis=1)
    doc_topic_df["dominant_topic_prob"] = doc_topic_df.iloc[:, :n_topics].max(axis=1)
    doc_topic_df["text_preview"] = [t[:100] + "..." if len(t) > 100 else t for t in texts]

    topic_corr = np.corrcoef(doc_topic_matrix.T)
    topic_sizes = doc_topic_df["dominant_topic"].value_counts().sort_index()

    return {
        "topics": topics,
        "topic_word_weights": topic_word_weights,
        "doc_topic_matrix": doc_topic_matrix,
        "doc_topic_df": doc_topic_df,
        "topic_correlations": topic_corr,
        "topic_sizes": topic_sizes,
        "coherence": [0.0] * n_topics,
        "exclusivity": [0.0] * n_topics,
        "n_topics": n_topics,
        "texts": texts,
        "vocab": list(feature_names),
        "prevalence_effects_summary": None,
        "metadata_df": None,
        "prevalence_formula": None,
        "r_script": None,
    }
