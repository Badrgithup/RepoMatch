"""Phase 8.1 — Unit tests for the Content-Based Recommendation Engine."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import clean_repositories
from src.features.metadata_features import build_feature_bundle
from src.recommenders.content_based import (
    recommend,
    recommend_from_profile,
    recommend_similar,
)


def _sample_catalog() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "repo_name": [
                "scikit-learn",
                "pytorch-vision",
                "vue-router",
                "react-dom",
                "godot-cpp",
                "empty-repo",
            ],
            "username": ["sklearn", "pytorch", "vuejs", "facebook", "godot", "anon"],
            "repo_url": [
                "https://github.com/sklearn/scikit-learn",
                "https://github.com/pytorch/pytorch-vision",
                "https://github.com/vuejs/vue-router",
                "https://github.com/facebook/react-dom",
                "https://github.com/godot/godot-cpp",
                "https://github.com/anon/empty-repo",
            ],
            "description": [
                "Machine learning and data science library in Python for classification and regression",
                "Deep learning and machine learning computer vision models in Python",
                "Official router for Vue.js frontend web framework in JavaScript",
                "Declarative UI frontend web library for building user interfaces in JavaScript",
                "C++ bindings for the Godot 3D and 2D game engine",
                "",
            ],
            "language": ["Python", "Python", "JavaScript", "JavaScript", "C++", "Unknown"],
            "item": ["Python", "Python", "JavaScript", "JavaScript", "CPP", "top-100-stars"],
            "stars": [46000, 19000, 18000, 170000, 4500, 1200],
            "forks": [21000, 5000, 3200, 34000, 800, 100],
            "issues": [120, 80, 40, 500, 25, 0],
            "last_commit": ["2021-07-12T00:00:00Z"] * 6,
        }
    )
    return clean_repositories(raw)


def test_content_recommender_returns_most_similar_first():
    query = np.array([[1.0, 0.0]])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert recommend(query, matrix, top_k=1).tolist() == [0]


def test_recommend_similar_excludes_reference_and_ranks_domain_peers_first():
    df = _sample_catalog()
    bundle = build_feature_bundle(df, save=False)

    recs = recommend_similar("sklearn/scikit-learn", df, feature_bundle=bundle, top_k=3)
    assert len(recs) == 3
    assert "sklearn/scikit-learn" not in recs["full_name"].tolist()
    assert recs.iloc[0]["full_name"] == "pytorch/pytorch-vision"
    assert recs.iloc[0]["shared_language"] == True  # noqa: E712


def test_recommend_similar_handles_repo_without_description_and_invalid_id():
    df = _sample_catalog()
    bundle = build_feature_bundle(df, save=False)

    recs = recommend_similar("anon/empty-repo", df, feature_bundle=bundle, top_k=2)
    assert len(recs) == 2

    with pytest.raises(KeyError):
        recommend_similar("nonexistent/repository", df, feature_bundle=bundle)


def test_recommend_from_profile_matches_javascript_web_and_cpp_game():
    df = _sample_catalog()
    bundle = build_feature_bundle(df, save=False)

    js_recs = recommend_from_profile(
        description="frontend user interface web framework",
        topics=["frontend-ui"],
        language="JavaScript",
        data=df,
        feature_bundle=bundle,
        top_k=2,
    )
    assert set(js_recs["language"].tolist()) == {"JavaScript"}

    cpp_recs = recommend_from_profile(
        description="3d game engine bindings",
        topics=["game-development"],
        language="C++",
        data=df,
        feature_bundle=bundle,
        top_k=1,
    )
    assert cpp_recs.iloc[0]["full_name"] == "godot/godot-cpp"
