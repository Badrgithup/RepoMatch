"""Phase 6 — Cascade Hybrid Recommendation Engine (Knowledge-Based + Content-Based + Popularity & Explainability)."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.features.metadata_features import build_feature_bundle
from src.recommenders.content_based import (
    recommend_from_profile,
    recommend_similar,
    resolve_repository_index,
)
from src.recommenders.knowledge_based import (
    apply_hard_constraints,
    diagnose_constraint_conflicts,
    recommend_by_constraints,
    score_soft_preferences,
)
from src.utils.helpers import format_compact_number, min_max_normalize, parse_topic_list


def combine(
    content_scores: np.ndarray,
    metadata_scores: np.ndarray,
    content_weight: float = 0.7,
) -> np.ndarray:
    r"""Combine aligned content similarity and popularity/metadata scores ($\alpha S + \beta P$, $\alpha+\beta=1$)."""
    alpha = float(np.clip(content_weight, 0.0, 1.0))
    beta = 1.0 - alpha
    return alpha * np.asarray(content_scores, dtype=float) + beta * np.asarray(metadata_scores, dtype=float)


def compute_popularity_scores(df: pd.DataFrame, star_lambda: float = 0.7) -> np.ndarray:
    r"""Calculate $\text{Popularity}(r) = \lambda \text{Norm}(\log(1+\text{stars})) + (1-\lambda)\text{Norm}(\log(1+\text{forks}))$."""
    if df.empty:
        return np.array([], dtype=float)
    if "popularity_score" in df.columns and np.isclose(star_lambda, 0.7):
        return pd.to_numeric(df["popularity_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    stars = pd.to_numeric(df.get("stars", 0), errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    forks = pd.to_numeric(df.get("forks", 0), errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    norm_stars = min_max_normalize(np.log1p(stars))
    norm_forks = min_max_normalize(np.log1p(forks))
    lam = float(np.clip(star_lambda, 0.0, 1.0))
    return lam * norm_stars + (1.0 - lam) * norm_forks


def generate_explanation(
    row: pd.Series | dict[str, Any],
    reference_repo: pd.Series | dict[str, Any] | None = None,
    user_profile: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
) -> str:
    """Produce a clear, human-readable explanation for why a repository was recommended."""
    clauses: list[str] = []
    lang = str(row.get("language", "Unknown"))
    sim_score = float(row.get("similarity_score", row.get("final_score", 0.0)) or 0.0)
    shared_topics = parse_topic_list(row.get("shared_topics", row.get("matched_topics", "")))

    # 1. Language alignment
    if reference_repo is not None:
        ref_lang = str(reference_repo.get("language", "Unknown"))
        if lang.lower() == ref_lang.lower() and lang.lower() != "unknown":
            clauses.append(f"shares {lang} as its primary programming language")
        elif lang.lower() != "unknown":
            clauses.append(f"is built with {lang}")
    elif constraints and constraints.get("languages"):
        clauses.append(f"matches your {lang} language requirement")
    elif user_profile and user_profile.get("language") and str(user_profile.get("language")).lower() not in {"any", "unknown"}:
        clauses.append(f"matches your preferred language ({lang})")
    elif lang.lower() != "unknown":
        clauses.append(f"uses {lang}")

    # 2. Topic & semantic alignment
    if shared_topics:
        formatted_topics = ", ".join(shared_topics[:3])
        clauses.append(f"shares key topics ({formatted_topics})")
    elif sim_score >= 0.35:
        clauses.append(f"shows strong semantic description similarity ({int(round(sim_score * 100))}%)")
    elif sim_score > 0:
        clauses.append(f"aligns with the query profile ({int(round(sim_score * 100))}% content match)")

    # 3. Hard constraint satisfaction & popularity context
    # Hard constraints are already enforced in Stage 1 and shown next to the results,
    # so repeating them per row adds noise. Mention popularity only when it drove the ranking.
    stars = int(row.get("stars", 0) or 0)
    if not clauses and stars > 0:
        clauses.append(f"has strong community adoption ({format_compact_number(stars)} stars)")

    if not clauses:
        return "Recommended based on overall metadata and constraint alignment."

    if len(clauses) == 1:
        return f"Recommended because it {clauses[0]}."
    return "Recommended because it " + ", ".join(clauses[:-1]) + f", and {clauses[-1]}."


def hybrid_recommend(
    data: pd.DataFrame,
    feature_bundle: dict[str, Any] | None = None,
    repo_id: int | str | None = None,
    user_profile: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    top_k: int = 10,
    feature_weights: dict[str, float] | None = None,
    popularity_weight: float = 0.15,
    min_similarity: float = 0.0,
) -> dict[str, Any]:
    """Execute the 3-stage Cascade Hybrid Recommendation pipeline.

    Stage 1 (Knowledge-Based): Eliminate repositories that violate mandatory constraints.
    Stage 2 (Content-Based): Calculate multi-modal similarity against eligible repositories.
    Stage 3 (Hybrid Ranking & Explanations): Incorporate optional popularity/soft preferences and generate explanations.

    Returns a dictionary with:
      - `recommendations`: Ranked DataFrame of top_k repositories with scores & explanations
      - `eligible_count`: Number of repositories surviving Stage 1
      - `conflict_diagnosis`: Conflict resolution diagnostics if Stage 1 returned 0 repositories
    """
    if data.empty:
        return {
            "recommendations": data.copy(),
            "eligible_count": 0,
            "conflict_diagnosis": None,
        }

    bundle = feature_bundle if feature_bundle is not None else build_feature_bundle(data, save=False)
    cons = constraints or {}
    profile = user_profile or {}

    languages = cons.get("languages", profile.get("language"))
    min_stars = cons.get("min_stars", 0)
    min_forks = cons.get("min_forks", 0)
    include_archived = cons.get("include_archived", True)
    max_days = cons.get("max_days_since_update", None)
    preferred_topics = cons.get("preferred_topics", profile.get("topics"))

    # Stage 1: Knowledge-Based Hard Constraint Filtering
    indexed_df = data.copy()
    indexed_df["_pos_index"] = np.arange(len(data), dtype=int)

    eligible_df = apply_hard_constraints(
        indexed_df,
        languages=languages,
        min_stars=min_stars,
        min_forks=min_forks,
        include_archived=include_archived,
        max_days_since_update=max_days,
    )

    if eligible_df.empty:
        diagnosis = diagnose_constraint_conflicts(
            data,
            languages=languages,
            min_stars=min_stars,
            min_forks=min_forks,
            include_archived=include_archived,
            max_days_since_update=max_days,
        )
        return {
            "recommendations": eligible_df.drop(columns=["_pos_index"], errors="ignore"),
            "eligible_count": 0,
            "conflict_diagnosis": diagnosis,
        }

    candidate_indices = eligible_df["_pos_index"].to_numpy(dtype=int)
    reference_repo_row = None

    # Stage 2: Content-Based Similarity over Eligible Candidates
    has_profile_query = bool(
        str(profile.get("description", "")).strip()
        or parse_topic_list(profile.get("topics"))
        or parse_topic_list(preferred_topics)
    )

    if repo_id is not None and str(repo_id).strip() != "":
        ref_idx = resolve_repository_index(data, repo_id)
        reference_repo_row = data.iloc[ref_idx]
        # Fetch more than top_k from Stage 2 if Stage 3 popularity reranking is active
        pool_k = max(int(top_k) * 3, int(top_k)) if popularity_weight > 0 else int(top_k)
        cb_results = recommend_similar(
            repo_id=ref_idx,
            data=data,
            feature_bundle=bundle,
            top_k=pool_k,
            feature_weights=feature_weights,
            candidate_indices=candidate_indices,
            min_similarity=min_similarity,
        )
    elif has_profile_query:
        pool_k = max(int(top_k) * 3, int(top_k)) if popularity_weight > 0 else int(top_k)
        lang_for_cb = (
            languages[0]
            if isinstance(languages, list) and len(languages) == 1
            else (languages if isinstance(languages, str) else str(profile.get("language", "")))
        )
        cb_results = recommend_from_profile(
            description=str(profile.get("description", "")),
            topics=profile.get("topics") or preferred_topics,
            language=str(lang_for_cb or ""),
            data=data,
            feature_bundle=bundle,
            top_k=pool_k,
            feature_weights=feature_weights,
            candidate_indices=candidate_indices,
            min_similarity=min_similarity,
        )
    else:
        # Pure Knowledge-Based mode when neither reference repo nor text/topic query is given
        kb_results = recommend_by_constraints(
            data,
            languages=languages,
            min_stars=min_stars,
            min_forks=min_forks,
            include_archived=include_archived,
            preferred_topics=preferred_topics,
            max_days_since_update=max_days,
            top_k=top_k,
        )
        kb_results["similarity_score"] = kb_results.get("preference_score", 0.0)
        kb_results["text_similarity"] = kb_results.get("preference_score", 0.0)
        kb_results["category_similarity"] = 1.0 if languages else 0.0
        kb_results["metadata_similarity"] = compute_popularity_scores(kb_results)
        kb_results["final_score"] = kb_results["similarity_score"]
        kb_results["explanation"] = [
            generate_explanation(row, None, profile, cons) for _, row in kb_results.iterrows()
        ]
        return {
            "recommendations": kb_results,
            "eligible_count": len(eligible_df),
            "conflict_diagnosis": None,
        }

    if cb_results.empty:
        return {
            "recommendations": cb_results,
            "eligible_count": len(eligible_df),
            "conflict_diagnosis": None,
        }

    # Stage 3: Hybrid Ranking (Content Similarity + Soft Preferences & Popularity Component)
    pop_component = compute_popularity_scores(cb_results)
    soft_pref_component = score_soft_preferences(
        cb_results,
        preferred_topics=preferred_topics,
        prefer_popular=True,
        prefer_recent=True,
    ).to_numpy(dtype=float)

    aux_score = 0.6 * pop_component + 0.4 * soft_pref_component
    beta = float(np.clip(popularity_weight, 0.0, 1.0))
    alpha = 1.0 - beta

    cb_results["popularity_component"] = np.round(pop_component, 4)
    cb_results["final_score"] = np.round(
        combine(cb_results["similarity_score"].to_numpy(dtype=float), aux_score, content_weight=alpha),
        4,
    )

    cb_results = cb_results.sort_values("final_score", ascending=False).head(max(1, int(top_k))).reset_index(drop=True)
    cb_results["explanation"] = [
        generate_explanation(row, reference_repo_row, profile, cons)
        for _, row in cb_results.iterrows()
    ]

    return {
        "recommendations": cb_results,
        "eligible_count": len(eligible_df),
        "conflict_diagnosis": None,
    }
