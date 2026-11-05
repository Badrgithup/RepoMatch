"""Phase 5 — Knowledge-Based (Constraint-Based) Recommendation Engine with Conflict Resolution."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.utils.helpers import min_max_normalize, parse_topic_list


def filter_by_language(df: pd.DataFrame, languages: str | list[str] | None) -> pd.DataFrame:
    """Hard constraint: keep only repositories matching the specified programming language(s)."""
    if languages is None or "language" not in df.columns:
        return df.copy()

    if isinstance(languages, str):
        lang_list = [
            l.strip().lower()
            for l in languages.split(",")
            if l.strip() and l.strip().lower() != "any"
        ]
    else:
        lang_list = [
            str(l).strip().lower()
            for l in languages
            if str(l).strip() and str(l).strip().lower() != "any"
        ]

    if not lang_list:
        return df.copy()

    mask = df["language"].fillna("").astype(str).str.lower().isin(lang_list)
    return df[mask].copy()


def filter_by_min_stars(df: pd.DataFrame, min_stars: int | float = 0) -> pd.DataFrame:
    """Hard constraint: keep only repositories with at least `min_stars` stars."""
    if "stars" not in df.columns or min_stars is None or float(min_stars) <= 0:
        return df.copy()
    stars = pd.to_numeric(df["stars"], errors="coerce").fillna(0)
    return df[stars >= float(min_stars)].copy()


def filter_by_min_forks(df: pd.DataFrame, min_forks: int | float = 0) -> pd.DataFrame:
    """Hard constraint: keep only repositories with at least `min_forks` forks."""
    if "forks" not in df.columns or min_forks is None or float(min_forks) <= 0:
        return df.copy()
    forks = pd.to_numeric(df["forks"], errors="coerce").fillna(0)
    return df[forks >= float(min_forks)].copy()


def filter_archived(df: pd.DataFrame, include_archived: bool = False) -> pd.DataFrame:
    """Hard constraint: exclude archived/dormant repositories when `include_archived` is False."""
    if include_archived or "is_archived" not in df.columns:
        return df.copy()
    return df[~df["is_archived"].astype(bool)].copy()


def filter_by_max_days_since_update(
    df: pd.DataFrame,
    max_days: int | float | None = None,
) -> pd.DataFrame:
    """Hard constraint: keep only repositories updated within `max_days` of the snapshot."""
    if max_days is None or float(max_days) <= 0 or "days_since_last_update" not in df.columns:
        return df.copy()
    days = pd.to_numeric(df["days_since_last_update"], errors="coerce").fillna(0)
    return df[days <= float(max_days)].copy()


def apply_hard_constraints(
    df: pd.DataFrame,
    languages: str | list[str] | None = None,
    min_stars: int | float = 0,
    min_forks: int | float = 0,
    include_archived: bool = True,
    max_days_since_update: int | float | None = None,
) -> pd.DataFrame:
    """Apply all mandatory (hard) constraints sequentially."""
    filtered = filter_by_language(df, languages)
    filtered = filter_by_min_stars(filtered, min_stars)
    filtered = filter_by_min_forks(filtered, min_forks)
    filtered = filter_archived(filtered, include_archived=include_archived)
    filtered = filter_by_max_days_since_update(filtered, max_days=max_days_since_update)
    return filtered


def score_soft_preferences(
    df: pd.DataFrame,
    preferred_topics: str | list[str] | None = None,
    prefer_popular: bool = True,
    prefer_recent: bool = True,
) -> pd.Series:
    """Compute a soft-preference score in [0, 1] without eliminating candidates."""
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    req_topics = set(parse_topic_list(preferred_topics))

    # 1. Topic overlap ratio
    if req_topics and "topics" in df.columns:
        topic_scores = np.array(
            [
                len(req_topics & set(parse_topic_list(val))) / len(req_topics)
                for val in df["topics"]
            ],
            dtype=float,
        )
    else:
        topic_scores = np.zeros(n, dtype=float)

    # 2. Popularity preference
    if prefer_popular:
        if "popularity_score" in df.columns:
            pop_scores = pd.to_numeric(df["popularity_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        elif "stars" in df.columns:
            pop_scores = min_max_normalize(np.log1p(pd.to_numeric(df["stars"], errors="coerce").fillna(0.0).to_numpy()))
        else:
            pop_scores = np.zeros(n, dtype=float)
    else:
        pop_scores = np.zeros(n, dtype=float)

    # 3. Recent activity preference
    if prefer_recent and "recency_score" in df.columns:
        rec_scores = pd.to_numeric(df["recency_score"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    else:
        rec_scores = np.zeros(n, dtype=float)

    if req_topics:
        combined = 0.55 * topic_scores + 0.30 * pop_scores + 0.15 * rec_scores
    else:
        combined = 0.70 * pop_scores + 0.30 * rec_scores

    return pd.Series(np.round(combined, 4), index=df.index)


def diagnose_constraint_conflicts(
    df: pd.DataFrame,
    languages: str | list[str] | None = None,
    min_stars: int | float = 0,
    min_forks: int | float = 0,
    include_archived: bool = False,
    max_days_since_update: int | float | None = None,
) -> dict[str, Any]:
    """Identify which hard constraint(s) caused an empty result set and suggest controlled relaxations.

    Never silently violates hard constraints; instead explains the bottleneck and computes actionable
    relaxed thresholds from the dataset.
    """
    total = len(df)
    steps: list[dict[str, Any]] = []
    failing_constraints: list[str] = []
    suggestions: list[str] = []
    relaxed_params: dict[str, Any] = {
        "languages": languages,
        "min_stars": min_stars,
        "min_forks": min_forks,
        "include_archived": include_archived,
        "max_days_since_update": max_days_since_update,
    }

    # Evaluate individual and sequential selectivity
    lang_df = filter_by_language(df, languages)
    steps.append({"constraint": "language", "remaining": len(lang_df), "individual_matches": len(lang_df)})
    if len(lang_df) == 0 and languages:
        failing_constraints.append("language")
        available_langs = (
            df["language"].value_counts().head(5).index.tolist() if "language" in df.columns else []
        )
        suggestions.append(
            f"No repositories found for language '{languages}'. Try one of the top languages ({', '.join(available_langs)}) or set language to 'Any'."
        )
        relaxed_params["languages"] = None
        base_df = df
    else:
        base_df = lang_df

    stars_ind = len(filter_by_min_stars(df, min_stars))
    stars_df = filter_by_min_stars(base_df, min_stars)
    steps.append({"constraint": "min_stars", "remaining": len(stars_df), "individual_matches": stars_ind})
    if len(stars_df) == 0 and float(min_stars or 0) > 0:
        failing_constraints.append("min_stars")
        max_avail_stars = int(base_df["stars"].max()) if "stars" in base_df.columns and not base_df.empty else 0
        suggested_stars = int(base_df["stars"].quantile(0.75)) if "stars" in base_df.columns and not base_df.empty else 0
        suggestions.append(
            f"Minimum stars ({int(min_stars):,}) exceeds available repositories (max available is {max_avail_stars:,}). Relax minimum stars to <= {suggested_stars:,}."
        )
        relaxed_params["min_stars"] = suggested_stars
        base_df = base_df
    else:
        base_df = stars_df

    forks_ind = len(filter_by_min_forks(df, min_forks))
    forks_df = filter_by_min_forks(base_df, min_forks)
    steps.append({"constraint": "min_forks", "remaining": len(forks_df), "individual_matches": forks_ind})
    if len(forks_df) == 0 and float(min_forks or 0) > 0:
        failing_constraints.append("min_forks")
        max_avail_forks = int(base_df["forks"].max()) if "forks" in base_df.columns and not base_df.empty else 0
        suggested_forks = int(base_df["forks"].quantile(0.75)) if "forks" in base_df.columns and not base_df.empty else 0
        suggestions.append(
            f"Minimum forks ({int(min_forks):,}) is too restrictive (max available in subset is {max_avail_forks:,}). Relax minimum forks to <= {suggested_forks:,}."
        )
        relaxed_params["min_forks"] = suggested_forks
    else:
        base_df = forks_df

    arch_ind = len(filter_archived(df, include_archived=include_archived))
    arch_df = filter_archived(base_df, include_archived=include_archived)
    steps.append({"constraint": "active_only", "remaining": len(arch_df), "individual_matches": arch_ind})
    if len(arch_df) == 0 and not include_archived:
        failing_constraints.append("active_only")
        suggestions.append(
            "All matching repositories were last updated > 365 days before the snapshot. Enable 'Include archived / dormant repositories'."
        )
        relaxed_params["include_archived"] = True
    else:
        base_df = arch_df

    if max_days_since_update is not None and float(max_days_since_update) > 0:
        rec_ind = len(filter_by_max_days_since_update(df, max_days_since_update))
        rec_df = filter_by_max_days_since_update(base_df, max_days_since_update)
        steps.append(
            {
                "constraint": "max_days_since_update",
                "remaining": len(rec_df),
                "individual_matches": rec_ind,
            }
        )
        if len(rec_df) == 0:
            failing_constraints.append("max_days_since_update")
            min_days = float(base_df["days_since_last_update"].min()) if "days_since_last_update" in base_df.columns and not base_df.empty else 365.0
            suggestions.append(
                f"No repositories in this subset were updated within {int(max_days_since_update)} days (most recent update is {int(np.ceil(min_days))} days). Relax the recency filter."
            )
            relaxed_params["max_days_since_update"] = None

    return {
        "total_catalog": total,
        "has_conflict": len(failing_constraints) > 0,
        "failing_constraints": failing_constraints,
        "trace": steps,
        "suggestions": suggestions,
        "recommended_relaxation": relaxed_params,
    }


def recommend_by_constraints(
    df: pd.DataFrame,
    languages: str | list[str] | None = None,
    min_stars: int | float = 0,
    min_forks: int | float = 0,
    include_archived: bool = True,
    preferred_topics: str | list[str] | None = None,
    max_days_since_update: int | float | None = None,
    prefer_popular: bool = True,
    prefer_recent: bool = True,
    top_k: int = 10,
) -> pd.DataFrame:
    """Filter repositories using hard constraints and rank matching candidates using soft preferences."""
    filtered = apply_hard_constraints(
        df,
        languages=languages,
        min_stars=min_stars,
        min_forks=min_forks,
        include_archived=include_archived,
        max_days_since_update=max_days_since_update,
    )
    if filtered.empty:
        return filtered

    filtered["preference_score"] = score_soft_preferences(
        filtered,
        preferred_topics=preferred_topics,
        prefer_popular=prefer_popular,
        prefer_recent=prefer_recent,
    )

    req_topics = set(parse_topic_list(preferred_topics))
    if "topics" in filtered.columns and req_topics:
        filtered["matched_topics"] = [
            ", ".join(sorted(req_topics & set(parse_topic_list(t))))
            for t in filtered["topics"]
        ]
    else:
        filtered["matched_topics"] = ""

    sort_cols = ["preference_score"]
    if "stars" in filtered.columns:
        sort_cols.append("stars")
    filtered = filtered.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return filtered.head(max(1, int(top_k))).reset_index(drop=True)


def recommend(
    data: pd.DataFrame,
    language: str | list[str] | None = None,
    min_stars: int | float = 0,
    top_k: int = 10,
    min_forks: int | float = 0,
    include_archived: bool = True,
    preferred_topics: str | list[str] | None = None,
) -> pd.DataFrame:
    """Backwards-compatible Knowledge-Based recommendation entry point."""
    result = apply_hard_constraints(
        data,
        languages=language,
        min_stars=min_stars,
        min_forks=min_forks,
        include_archived=include_archived,
    )
    if result.empty:
        return result
    if preferred_topics:
        return recommend_by_constraints(
            data,
            languages=language,
            min_stars=min_stars,
            min_forks=min_forks,
            include_archived=include_archived,
            preferred_topics=preferred_topics,
            top_k=top_k,
        )
    if "stars" in result.columns:
        result = result.sort_values("stars", ascending=False)
    return result.head(max(1, int(top_k))).reset_index(drop=True)
