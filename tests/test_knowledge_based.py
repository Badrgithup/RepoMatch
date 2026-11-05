"""Phase 8.1 — Unit tests for the Knowledge-Based Recommendation Engine & Conflict Resolution."""

import pandas as pd

from src.recommenders.knowledge_based import (
    diagnose_constraint_conflicts,
    filter_archived,
    filter_by_language,
    filter_by_min_forks,
    filter_by_min_stars,
    recommend,
    recommend_by_constraints,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "full_name": ["org/py-ml", "org/py-old", "org/js-web"],
            "language": ["Python", "Python", "JavaScript"],
            "stars": [15000, 2000, 50000],
            "forks": [3000, 150, 8000],
            "is_archived": [False, True, False],
            "days_since_last_update": [10.0, 500.0, 5.0],
            "topics": ["machine-learning, deep-learning", "machine-learning", "frontend-ui"],
            "popularity_score": [0.8, 0.3, 0.95],
            "recency_score": [0.9, 0.1, 0.95],
        }
    )


def test_knowledge_recommender_filters_language_and_stars():
    data = pd.DataFrame({"language": ["Python", "JavaScript"], "stars": [20, 100]})
    result = recommend(data, language="python", min_stars=10)
    assert len(result) == 1
    assert result.iloc[0]["language"] == "Python"


def test_individual_hard_constraint_filters():
    df = _sample_df()
    assert len(filter_by_language(df, ["Python"])) == 2
    assert len(filter_by_min_stars(df, 10000)) == 2
    assert len(filter_by_min_forks(df, 5000)) == 1
    assert len(filter_archived(df, include_archived=False)) == 2
    assert len(filter_archived(df, include_archived=True)) == 3


def test_recommend_by_constraints_respects_hard_and_soft_preferences():
    df = _sample_df()
    res = recommend_by_constraints(
        df,
        languages=["Python"],
        min_stars=1000,
        min_forks=100,
        include_archived=False,
        preferred_topics=["machine-learning"],
    )
    assert len(res) == 1
    assert res.iloc[0]["full_name"] == "org/py-ml"


def test_conflict_resolution_identifies_restrictive_constraints():
    df = _sample_df()
    diag = diagnose_constraint_conflicts(
        df,
        languages=["Python"],
        min_stars=500_000,
        min_forks=100_000,
        include_archived=False,
    )
    assert diag["has_conflict"] is True
    assert "min_stars" in diag["failing_constraints"]
    assert len(diag["suggestions"]) >= 1
    assert diag["recommended_relaxation"]["min_stars"] < 500_000
