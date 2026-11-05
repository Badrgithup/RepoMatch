"""Phase 4 — Content-Based Recommendation Engine (Similar Projects & Query Profile modes)."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.features.metadata_features import (
    build_feature_bundle,
    transform_categorical_query,
)
from src.features.text_features import transform_text_query
from src.utils.config import DEFAULT_FEATURE_WEIGHTS
from src.utils.helpers import parse_topic_list


def recommend(query_vector, feature_matrix, top_k: int = 10) -> np.ndarray:
    """Return feature-matrix indices ordered by cosine similarity to a query vector."""
    scores = cosine_similarity(query_vector, feature_matrix).ravel()
    return scores.argsort()[::-1][:top_k]


def _normalize_weights(feature_weights: dict[str, float] | None) -> dict[str, float]:
    """Normalize feature group weights so they sum to 1.0."""
    raw = dict(DEFAULT_FEATURE_WEIGHTS)
    if feature_weights:
        for k in ("text", "category", "metadata"):
            if k in feature_weights:
                raw[k] = max(0.0, float(feature_weights[k]))
    total = sum(raw.values())
    if total <= 0:
        return {"text": 1.0, "category": 0.0, "metadata": 0.0}
    return {k: v / total for k, v in raw.items()}


def resolve_repository_index(data: pd.DataFrame, repo_id: int | str) -> int:
    """Locate the positional index of a repository by positional index, full_name, repo_name, or URL."""
    if isinstance(repo_id, (int, np.integer)):
        idx = int(repo_id)
        if 0 <= idx < len(data):
            return idx
        raise KeyError(f"Repository index {idx} is out of bounds for dataset of size {len(data)}.")

    query = str(repo_id).strip()
    if not query:
        raise ValueError("Repository identifier cannot be empty.")

    query_lower = query.lower()
    for col in ("full_name", "name", "repo_url", "url", "repo_name"):
        if col in data.columns:
            matches = np.flatnonzero(data[col].fillna("").astype(str).str.lower().to_numpy() == query_lower)
            if matches.size > 0:
                return int(matches[0])

    # Partial match on full_name or repo_url tail if exact match not found
    for col in ("full_name", "repo_url", "url"):
        if col in data.columns:
            matches = np.flatnonzero(
                data[col].fillna("").astype(str).str.lower().str.endswith("/" + query_lower).to_numpy()
            )
            if matches.size > 0:
                return int(matches[0])

    raise KeyError(f"Repository '{repo_id}' was not found in the dataset.")


def recommend_similar(
    repo_id: int | str,
    data: pd.DataFrame,
    feature_bundle: dict[str, Any] | None = None,
    top_k: int = 10,
    feature_weights: dict[str, float] | None = None,
    candidate_indices: np.ndarray | list[int] | None = None,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    """Mode 1: Recommend repositories most similar to an existing reference repository."""
    if data.empty:
        raise ValueError("Cannot generate recommendations from an empty dataset.")

    bundle = feature_bundle if feature_bundle is not None else build_feature_bundle(data, save=False)
    ref_pos = resolve_repository_index(data, repo_id)
    weights = _normalize_weights(feature_weights)

    if candidate_indices is None:
        cand_pos = np.arange(len(data), dtype=int)
    else:
        cand_pos = np.asarray(candidate_indices, dtype=int)

    # Exclude the selected reference repository itself
    cand_pos = cand_pos[cand_pos != ref_pos]
    if cand_pos.size == 0:
        return data.iloc[0:0].copy()

    text_mat = bundle["text_matrix"]
    cat_mat = bundle["category_matrix"]
    meta_mat = bundle["metadata_matrix"]

    text_sim = cosine_similarity(text_mat[ref_pos], text_mat[cand_pos]).ravel()
    cat_sim = cosine_similarity(cat_mat[ref_pos], cat_mat[cand_pos]).ravel()
    meta_sim = cosine_similarity(meta_mat[ref_pos : ref_pos + 1], meta_mat[cand_pos]).ravel()

    combined_sim = (
        weights["text"] * text_sim
        + weights["category"] * cat_sim
        + weights["metadata"] * meta_sim
    )

    # Filter by minimum similarity threshold
    valid_mask = combined_sim >= float(min_similarity)
    cand_pos = cand_pos[valid_mask]
    text_sim = text_sim[valid_mask]
    cat_sim = cat_sim[valid_mask]
    meta_sim = meta_sim[valid_mask]
    combined_sim = combined_sim[valid_mask]

    if cand_pos.size == 0:
        return data.iloc[0:0].copy()

    order = np.argsort(combined_sim)[::-1][: max(1, int(top_k))]
    selected_pos = cand_pos[order]

    results = data.iloc[selected_pos].copy().reset_index(drop=True)
    results["similarity_score"] = np.round(combined_sim[order], 4)
    results["text_similarity"] = np.round(text_sim[order], 4)
    results["category_similarity"] = np.round(cat_sim[order], 4)
    results["metadata_similarity"] = np.round(meta_sim[order], 4)

    ref_row = data.iloc[ref_pos]
    ref_lang = str(ref_row.get("language", "Unknown"))
    ref_topics = set(parse_topic_list(ref_row.get("topics", "")))

    shared_langs: list[bool] = []
    shared_topics_list: list[str] = []
    shared_topics_count: list[int] = []

    for _, row in results.iterrows():
        cand_lang = str(row.get("language", "Unknown"))
        shared_langs.append(cand_lang.lower() == ref_lang.lower() and ref_lang.lower() != "unknown")
        cand_topics = set(parse_topic_list(row.get("topics", "")))
        overlap = sorted(ref_topics & cand_topics)
        shared_topics_list.append(", ".join(overlap))
        shared_topics_count.append(len(overlap))

    results["shared_language"] = shared_langs
    results["shared_topics"] = shared_topics_list
    results["shared_topics_count"] = shared_topics_count
    return results


def recommend_from_profile(
    description: str = "",
    topics: str | list[str] | None = None,
    language: str = "",
    data: pd.DataFrame | None = None,
    feature_bundle: dict[str, Any] | None = None,
    top_k: int = 10,
    feature_weights: dict[str, float] | None = None,
    candidate_indices: np.ndarray | list[int] | None = None,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    """Mode 2: Recommend repositories matching a free-text description, topics, and language profile.

    Uses pre-fitted TF-IDF and One-Hot encoders without refitting on user queries.
    """
    if data is None or data.empty:
        raise ValueError("A non-empty repository DataFrame is required.")

    bundle = feature_bundle if feature_bundle is not None else build_feature_bundle(data, save=False)
    weights = _normalize_weights(feature_weights)

    if candidate_indices is None:
        cand_pos = np.arange(len(data), dtype=int)
    else:
        cand_pos = np.asarray(candidate_indices, dtype=int)

    if cand_pos.size == 0:
        return data.iloc[0:0].copy()

    # Build query vectors using already-fitted transformers
    query_text_vec = transform_text_query(
        vectorizer=bundle["tfidf_vectorizer"],
        description=description,
        topics=topics,
        language=language,
    )
    text_sim = cosine_similarity(query_text_vec, bundle["text_matrix"][cand_pos]).ravel()

    if language and language.strip().lower() not in {"", "any", "unknown"}:
        query_cat_vec = transform_categorical_query(
            encoder=bundle["category_encoder"],
            language=language.strip(),
            category=language.strip(),
        )
        cat_sim = cosine_similarity(query_cat_vec, bundle["category_matrix"][cand_pos]).ravel()
    else:
        cat_sim = np.zeros(cand_pos.size, dtype=float)

    # For profile queries without a specific reference repo's star count, use normalized popularity/recency as metadata affinity
    if "popularity_score" in data.columns:
        meta_sim = data.iloc[cand_pos]["popularity_score"].fillna(0.0).to_numpy(dtype=float)
    else:
        meta_mat = bundle["metadata_matrix"][cand_pos]
        meta_sim = np.mean(meta_mat, axis=1)

    # If no language was specified in the query profile, redistribute category weight to text
    effective_weights = dict(weights)
    if not language or language.strip().lower() in {"", "any", "unknown"}:
        effective_weights["text"] += effective_weights["category"]
        effective_weights["category"] = 0.0

    combined_sim = (
        effective_weights["text"] * text_sim
        + effective_weights["category"] * cat_sim
        + effective_weights["metadata"] * meta_sim
    )

    valid_mask = combined_sim >= float(min_similarity)
    cand_pos = cand_pos[valid_mask]
    text_sim = text_sim[valid_mask]
    cat_sim = cat_sim[valid_mask]
    meta_sim = meta_sim[valid_mask]
    combined_sim = combined_sim[valid_mask]

    if cand_pos.size == 0:
        return data.iloc[0:0].copy()

    order = np.argsort(combined_sim)[::-1][: max(1, int(top_k))]
    selected_pos = cand_pos[order]

    results = data.iloc[selected_pos].copy().reset_index(drop=True)
    results["similarity_score"] = np.round(combined_sim[order], 4)
    results["text_similarity"] = np.round(text_sim[order], 4)
    results["category_similarity"] = np.round(cat_sim[order], 4)
    results["metadata_similarity"] = np.round(meta_sim[order], 4)

    query_topics = set(parse_topic_list(topics))
    req_lang = (language or "").strip().lower()

    shared_langs: list[bool] = []
    shared_topics_list: list[str] = []
    shared_topics_count: list[int] = []

    for _, row in results.iterrows():
        cand_lang = str(row.get("language", "Unknown")).lower()
        shared_langs.append(bool(req_lang and req_lang not in {"any", "unknown"} and cand_lang == req_lang))
        cand_topics = set(parse_topic_list(row.get("topics", "")))
        overlap = sorted(query_topics & cand_topics)
        shared_topics_list.append(", ".join(overlap))
        shared_topics_count.append(len(overlap))

    results["shared_language"] = shared_langs
    results["shared_topics"] = shared_topics_list
    results["shared_topics_count"] = shared_topics_count
    return results
