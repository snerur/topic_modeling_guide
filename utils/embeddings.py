"""Embedding model wrappers for topic modeling."""

import numpy as np
import streamlit as st
from typing import List, Optional


def get_embedding_model(model_name: str, api_key: Optional[str] = None):
    """Get embedding model instance for BERTopic/Turftopic/Bunka."""
    if model_name == "SBERT (all-MiniLM-L6-v2)":
        return _get_sbert_model()
    elif model_name == "SBERT (all-mpnet-base-v2)":
        return _get_sbert_model("all-mpnet-base-v2")
    elif model_name == "OpenAI (text-embedding-3-small)":
        return _get_openai_model(api_key, "text-embedding-3-small")
    elif model_name == "OpenAI (text-embedding-3-large)":
        return _get_openai_model(api_key, "text-embedding-3-large")
    elif model_name == "Voyage (voyage-3)":
        return _get_voyage_model(api_key, "voyage-3")
    elif model_name == "Voyage (voyage-3-lite)":
        return _get_voyage_model(api_key, "voyage-3-lite")
    elif model_name == "Gemini (gemini-embedding-001)":
        return _get_gemini_model(api_key)
    else:
        return _get_sbert_model()


def compute_embeddings(texts: List[str], model_name: str, api_key: Optional[str] = None,
                       batch_size: int = 100) -> np.ndarray:
    """Compute embeddings for a list of texts. Returns numpy array."""
    if "SBERT" in model_name:
        return _compute_sbert_embeddings(texts, model_name)
    elif "OpenAI" in model_name:
        return _compute_openai_embeddings(texts, api_key, model_name, batch_size)
    elif "Voyage" in model_name:
        return _compute_voyage_embeddings(texts, api_key, model_name, batch_size)
    elif "Gemini" in model_name:
        return _compute_gemini_embeddings(texts, api_key, batch_size)
    else:
        return _compute_sbert_embeddings(texts, model_name)


def _get_sbert_model(model_id: str = "all-MiniLM-L6-v2"):
    """Get sentence-transformers model for BERTopic."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id)


def _compute_sbert_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model_id = "all-MiniLM-L6-v2"
    if "mpnet" in model_name:
        model_id = "all-mpnet-base-v2"
    model = SentenceTransformer(model_id)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return np.array(embeddings)


def _get_openai_model(api_key: str, model_id: str = "text-embedding-3-small"):
    """Get OpenAI embedding model wrapper for BERTopic."""
    from bertopic.backend import OpenAIBackend
    import openai
    client = openai.OpenAI(api_key=api_key)
    return OpenAIBackend(client, model_id)


def _compute_openai_embeddings(texts: List[str], api_key: str, model_name: str,
                                batch_size: int = 100) -> np.ndarray:
    import openai
    client = openai.OpenAI(api_key=api_key)
    model_id = "text-embedding-3-small"
    if "large" in model_name:
        model_id = "text-embedding-3-large"

    all_embeddings = []
    progress = st.progress(0)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model_id)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        progress.progress(min((i + batch_size) / len(texts), 1.0))
    progress.empty()
    return np.array(all_embeddings)


def _get_voyage_model(api_key: str, model_id: str = "voyage-3"):
    """Get Voyage embedding model - returns None, use compute_embeddings instead."""
    return None


def _compute_voyage_embeddings(texts: List[str], api_key: str, model_name: str,
                                batch_size: int = 100) -> np.ndarray:
    import voyageai
    client = voyageai.Client(api_key=api_key)
    model_id = "voyage-3"
    if "lite" in model_name:
        model_id = "voyage-3-lite"

    all_embeddings = []
    progress = st.progress(0)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = client.embed(batch, model=model_id, input_type="document")
        all_embeddings.extend(result.embeddings)
        progress.progress(min((i + batch_size) / len(texts), 1.0))
    progress.empty()
    return np.array(all_embeddings)


def _get_gemini_model(api_key: str):
    """Get Gemini embedding model - returns None, use compute_embeddings instead."""
    return None


def _compute_gemini_embeddings(texts: List[str], api_key: str,
                                batch_size: int = 100) -> np.ndarray:
    from google import genai

    client = genai.Client(api_key=api_key)

    all_embeddings = []
    progress = st.progress(0)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
        )
        for embedding in result.embeddings:
            all_embeddings.append(embedding.values)
        progress.progress(min((i + batch_size) / len(texts), 1.0))
    progress.empty()
    return np.array(all_embeddings)


def get_available_models() -> List[str]:
    """Return list of available embedding models."""
    return [
        "SBERT (all-MiniLM-L6-v2)",
        "SBERT (all-mpnet-base-v2)",
        "OpenAI (text-embedding-3-small)",
        "OpenAI (text-embedding-3-large)",
        "Voyage (voyage-3)",
        "Voyage (voyage-3-lite)",
        "Gemini (gemini-embedding-001)",
    ]


def requires_api_key(model_name: str) -> bool:
    """Check if the model requires an API key."""
    return any(provider in model_name for provider in ["OpenAI", "Voyage", "Gemini"])


def get_api_key_label(model_name: str) -> str:
    """Get the appropriate API key label for a model."""
    if "OpenAI" in model_name:
        return "OpenAI API Key"
    elif "Voyage" in model_name:
        return "Voyage API Key"
    elif "Gemini" in model_name:
        return "Google/Gemini API Key"
    return ""
