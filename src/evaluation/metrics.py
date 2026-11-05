"""Phase 7.2 — Recommendation evaluation metrics (Precision@K, Recall@K, NDCG@K, Diversity, Coverage, Constraint Satisfaction)."""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def precision_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    """Calculate Precision@K: fraction of top-K recommended items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = list(recommended)[:k]
    if not top_k:
        return 0.0
    rel_set = set(relevant)
    hits = sum(1 for item in top_k if item in rel_set)
    return float(hits) / float(k)


def recall_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    """Calculate Recall@K: fraction of all relevant items retrieved in the top-K."""
    rel_set = set(relevant)
    if k <= 0 or not rel_set:
        return 0.0
    top_k = list(recommended)[:k]
    hits = sum(1 for item in top_k if item in rel_set)
    return float(hits) / float(len(rel_set))


def ndcg_at_k(recommended: Sequence, relevant: Iterable | dict, k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Supports binary relevance (when `relevant` is a set/list) or graded relevance
    (when `relevant` is a dict mapping item_id -> relevance grade).
    """
    if k <= 0:
        return 0.0
    top_k = list(recommended)[:k]
    if not top_k:
        return 0.0

    if isinstance(relevant, dict):
        rel_map = {key: float(val) for key, val in relevant.items() if float(val) > 0}
    else:
        rel_map = {key: 1.0 for key in set(relevant)}

    if not rel_map:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(top_k):
        gain = rel_map.get(item, 0.0)
        if gain > 0:
            dcg += gain / np.log2(idx + 2.0)

    ideal_gains = sorted(rel_map.values(), reverse=True)[:k]
    idcg = sum(gain / np.log2(idx + 2.0) for idx, gain in enumerate(ideal_gains))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def intra_list_diversity(
    recommended_indices: Sequence[int],
    feature_matrix,
) -> float:
    r"""Calculate Intra-List Diversity (ILD): average pairwise cosine dissimilarity $1 - \text{Cosine}(i, j)$."""
    idx_list = [int(i) for i in recommended_indices]
    n = len(idx_list)
    if n <= 1:
        return 0.0

    sub_matrix = feature_matrix[idx_list]
    sim_matrix = cosine_similarity(sub_matrix)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]
    if pairwise_sims.size == 0:
        return 0.0
    mean_sim = float(np.clip(np.mean(pairwise_sims), 0.0, 1.0))
    return float(1.0 - mean_sim)


def catalog_coverage(
    all_recommended_lists: Sequence[Sequence],
    total_catalog_size: int,
) -> float:
    """Calculate Catalog Coverage: fraction of unique catalog items recommended across all queries."""
    if total_catalog_size <= 0:
        return 0.0
    unique_recommended = {item for rec_list in all_recommended_lists for item in rec_list}
    return float(len(unique_recommended)) / float(total_catalog_size)


def constraint_satisfaction_rate(
    recommendations: pd.DataFrame,
    constraints: dict | None,
) -> float:
    """Calculate the fraction of recommended repositories that satisfy all hard constraints."""
    if recommendations.empty:
        return 1.0
    if not constraints:
        return 1.0

    satisfied_mask = pd.Series(True, index=recommendations.index)

    languages = constraints.get("languages")
    if languages and "language" in recommendations.columns:
        if isinstance(languages, str):
            lang_list = [l.strip().lower() for l in languages.split(",") if l.strip() and l.strip().lower() != "any"]
        else:
            lang_list = [str(l).strip().lower() for l in languages if str(l).strip() and str(l).strip().lower() != "any"]
        if lang_list:
            satisfied_mask &= recommendations["language"].fillna("").astype(str).str.lower().isin(lang_list)

    min_stars = float(constraints.get("min_stars", 0) or 0)
    if min_stars > 0 and "stars" in recommendations.columns:
        satisfied_mask &= pd.to_numeric(recommendations["stars"], errors="coerce").fillna(0) >= min_stars

    min_forks = float(constraints.get("min_forks", 0) or 0)
    if min_forks > 0 and "forks" in recommendations.columns:
        satisfied_mask &= pd.to_numeric(recommendations["forks"], errors="coerce").fillna(0) >= min_forks

    include_archived = constraints.get("include_archived", True)
    if not include_archived and "is_archived" in recommendations.columns:
        satisfied_mask &= ~recommendations["is_archived"].astype(bool)

    return float(satisfied_mask.mean())
