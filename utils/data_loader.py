"""Data loading utilities for CSV files and text folders."""

import os
import re
import pandas as pd
import streamlit as st
from typing import List, Optional, Tuple


def load_csv(uploaded_file) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


def load_folder(folder_path: str) -> pd.DataFrame:
    """Load all text files from a folder into a DataFrame."""
    documents = []
    filenames = []
    supported_extensions = {".txt", ".md", ".text", ".doc"}

    if not os.path.isdir(folder_path):
        st.error(f"Directory not found: {folder_path}")
        return pd.DataFrame()

    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported_extensions:
            fpath = os.path.join(folder_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                if text:
                    documents.append(text)
                    filenames.append(fname)
            except Exception:
                continue

    if not documents:
        st.error("No text files found in the specified folder.")
        return pd.DataFrame()

    return pd.DataFrame({"filename": filenames, "text": documents})


def concatenate_columns(df: pd.DataFrame, columns: List[str], separator: str = " ") -> pd.Series:
    """Concatenate multiple text columns into a single series."""
    combined = df[columns[0]].astype(str).fillna("")
    for col in columns[1:]:
        combined = combined + separator + df[col].astype(str).fillna("")
    return combined.str.strip()


def preprocess_texts(texts: pd.Series, min_length: int = 10) -> Tuple[pd.Series, pd.Index]:
    """Basic preprocessing: remove empty/short docs, return texts and valid indices."""
    texts = texts.fillna("").astype(str).str.strip()
    mask = texts.str.len() >= min_length
    return texts[mask], mask


def advanced_preprocess(texts: List[str], lowercase: bool = True,
                        remove_punctuation: bool = True, remove_numbers: bool = True,
                        remove_stopwords: bool = True, lemmatize: bool = True,
                        content_words_only: bool = False,
                        custom_stopwords: Optional[List[str]] = None) -> List[str]:
    """Advanced text preprocessing pipeline.

    Args:
        texts: List of raw document strings.
        lowercase: Convert to lowercase.
        remove_punctuation: Strip punctuation characters.
        remove_numbers: Strip numeric tokens.
        remove_stopwords: Remove English stopwords.
        lemmatize: Apply lemmatization (requires spacy).
        content_words_only: Keep only nouns, verbs, adjectives, adverbs.
        custom_stopwords: Additional stopwords to remove.

    Returns:
        List of preprocessed document strings.
    """
    # Build stopword set
    stop_words = set()
    if remove_stopwords:
        try:
            from nltk.corpus import stopwords as nltk_stopwords
            import nltk
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            stop_words = set(nltk_stopwords.words("english"))
        except ImportError:
            # Fallback minimal stopwords
            stop_words = {
                "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
                "for", "of", "with", "by", "from", "is", "was", "are", "were",
                "be", "been", "being", "have", "has", "had", "do", "does", "did",
                "will", "would", "could", "should", "may", "might", "shall",
                "can", "it", "its", "this", "that", "these", "those", "i", "me",
                "my", "we", "our", "you", "your", "he", "she", "they", "them",
                "his", "her", "not", "no", "just", "also", "very", "really",
                "so", "too", "if", "then", "than", "as", "all", "any", "each",
                "some", "such", "more", "most", "other", "about", "up", "out",
                "into", "over", "after", "before", "between", "through",
                "during", "only", "own", "same", "what", "which", "who",
                "whom", "how", "when", "where", "why", "there", "here",
            }
    if custom_stopwords:
        stop_words.update(w.lower() for w in custom_stopwords)

    # Content word POS tags (nouns, verbs, adjectives, adverbs)
    content_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

    # Try spacy for lemmatization / POS filtering
    nlp = None
    if lemmatize or content_words_only:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception:
            st.warning("spacy en_core_web_sm not available. "
                       "Install with: `python -m spacy download en_core_web_sm`. "
                       "Falling back to basic preprocessing.")

    processed = []
    for text in texts:
        if lowercase:
            text = text.lower()
        if remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)
        if remove_numbers:
            text = re.sub(r"\b\d+\b", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if nlp is not None:
            doc = nlp(text)
            tokens = []
            for token in doc:
                if token.is_space:
                    continue
                if content_words_only and token.pos_ not in content_pos:
                    continue
                word = token.lemma_ if lemmatize else token.text
                if remove_stopwords and word.lower() in stop_words:
                    continue
                if len(word) < 2:
                    continue
                tokens.append(word)
            text = " ".join(tokens)
        else:
            # Basic token-level filtering without spacy
            tokens = text.split()
            tokens = [t for t in tokens if len(t) >= 2]
            if remove_stopwords:
                tokens = [t for t in tokens if t.lower() not in stop_words]
            text = " ".join(tokens)

        processed.append(text)

    return processed


def chunk_texts(texts: List[str], max_words: int = 500,
                overlap_words: int = 50) -> Tuple[List[str], List[int]]:
    """Split long documents into overlapping chunks.

    Returns:
        chunks: list of text chunks
        doc_indices: parallel list mapping each chunk back to its source document index
    """
    chunks = []
    doc_indices = []

    for doc_idx, text in enumerate(texts):
        words = text.split()
        if len(words) <= max_words:
            chunks.append(text)
            doc_indices.append(doc_idx)
        else:
            start = 0
            while start < len(words):
                end = start + max_words
                chunk = " ".join(words[start:end])
                chunks.append(chunk)
                doc_indices.append(doc_idx)
                start = end - overlap_words
                if start >= len(words):
                    break

    return chunks, doc_indices


def aggregate_chunk_results(chunk_topic_matrix, doc_indices: List[int],
                            n_docs: int):
    """Aggregate chunk-level topic distributions back to document level.

    Averages the topic probability vectors of all chunks belonging to the same
    document.
    """
    import numpy as np

    n_topics = chunk_topic_matrix.shape[1]
    doc_topic_matrix = np.zeros((n_docs, n_topics))
    doc_counts = np.zeros(n_docs)

    for chunk_idx, doc_idx in enumerate(doc_indices):
        doc_topic_matrix[doc_idx] += chunk_topic_matrix[chunk_idx]
        doc_counts[doc_idx] += 1

    # Average
    doc_counts[doc_counts == 0] = 1
    doc_topic_matrix = doc_topic_matrix / doc_counts[:, np.newaxis]

    return doc_topic_matrix


def get_text_columns(df: pd.DataFrame) -> List[str]:
    """Identify likely text columns (object dtype with reasonable text content)."""
    text_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > 20:
                text_cols.append(col)
    if not text_cols:
        text_cols = [col for col in df.columns if df[col].dtype == "object"]
    return text_cols


def get_metadata_columns(df: pd.DataFrame, text_columns: List[str]) -> List[str]:
    """Get columns that could serve as metadata (non-text columns + date columns)."""
    all_cols = list(df.columns)
    metadata_cols = [c for c in all_cols if c not in text_columns]
    return metadata_cols


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Try to detect a datetime column for topic evolution analysis."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                return col
        except Exception:
            continue
    return None
