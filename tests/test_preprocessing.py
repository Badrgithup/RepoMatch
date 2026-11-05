"""Phase 8.1 — Unit tests for data preprocessing, normalization, and feature engineering."""

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    clean_repositories,
    extract_repository_topics,
    normalize_text,
    standardize_language,
)
from src.features.metadata_features import (
    build_categorical_features,
    build_feature_bundle,
    build_numerical_features,
)
from src.features.text_features import build_text_features, combine_text_fields


def test_clean_repositories_strips_text_and_deduplicates_urls():
    data = pd.DataFrame({"name": [" A ", "A"], "url": ["u", "u"]})
    result = clean_repositories(data)
    assert result.loc[0, "name"] == "A"
    assert len(result) == 1


def test_clean_repositories_supports_github_dataset_columns():
    data = pd.DataFrame(
        {
            "repo_name": [" repo-one ", "repo-two", "repo-one-copy"],
            "username": ["user1", "user2", "user1"],
            "repo_url": [
                "https://example.com/one",
                "https://example.com/two",
                "https://example.com/one",
            ],
            "language": [None, " Matlab ", "VimL"],
            "item": ["Python", "top-100-stars", "Vim-script"],
            "stars": [100, 500, 100],
            "forks": [10, 50, 10],
            "issues": [1, 5, 1],
            "last_commit": [
                "2021-07-10T00:00:00Z",
                "2019-01-01T00:00:00Z",
                "2021-07-10T00:00:00Z",
            ],
            "description": [
                "Deep learning framework in Python!",
                None,
                "Duplicate row",
            ],
        }
    )
    result = clean_repositories(data)
    assert len(result) == 2
    assert result.loc[0, "repo_name"] == "repo-one"
    assert result.loc[0, "full_name"] == "user1/repo-one"
    # Language inferred from 'item' = 'Python' when language is None
    assert result.loc[0, "language"] == "Python"
    # Language standardized from 'Matlab' -> 'MATLAB'
    assert result.loc[1, "language"] == "MATLAB"
    # Missing description replaced with empty string, row preserved
    assert result.loc[1, "description"] == ""
    assert "log_stars" in result.columns
    assert "is_archived" in result.columns
    assert bool(result.loc[0, "is_archived"]) is False
    assert bool(result.loc[1, "is_archived"]) is True


def test_normalize_text_preserves_technical_tokens():
    raw = "Modern C++ and C# library for AI, R, Go, and .NET! Visit https://example.com :rocket:"
    cleaned = normalize_text(raw)
    for token in ("c++", "c#", "ai", "r", "go", ".net"):
        assert token in cleaned.split()
    assert "https" not in cleaned
    assert "rocket" not in cleaned


def test_standardize_language_handles_variants_and_unknown():
    assert standardize_language("Matlab") == "MATLAB"
    assert standardize_language("VimL") == "Vim script"
    assert standardize_language(None, "CPP") == "C++"
    assert standardize_language(None, "top-100-stars") == "Unknown"


def test_extract_repository_topics_identifies_domains():
    topics = extract_repository_topics(
        "yolov5",
        "Real-time object detection and computer vision in PyTorch",
        "Python",
    )
    assert "computer-vision" in topics
    assert "deep-learning" in topics


def test_topic_keywords_do_not_match_inside_other_words():
    html_topics = extract_repository_topics("reveal.js", "The HTML Presentation Framework", "JavaScript")
    assert "machine-learning" not in html_topics
    engine_topics = extract_repository_topics("godot", "A cross-platform engine", "C++")
    assert "web-framework" not in engine_topics
    assert "machine-learning" in extract_repository_topics("x", "ML toolkit for tabular data", "Python")


def test_tfidf_and_feature_bundle_dimensions():
    df = pd.DataFrame(
        {
            "repo_name": ["sklearn-toolkit", "react-ui", "cpp-engine"],
            "username": ["ml", "web", "game"],
            "repo_url": ["https://gh.com/1", "https://gh.com/2", "https://gh.com/3"],
            "description": [
                "Python machine learning algorithms",
                "Frontend UI components in JavaScript",
                "3D OpenGL game engine in C++",
            ],
            "language": ["Python", "JavaScript", "C++"],
            "item": ["Python", "JavaScript", "CPP"],
            "stars": [5000, 12000, 3000],
            "forks": [800, 1500, 400],
            "issues": [20, 50, 15],
            "last_commit": [
                "2021-07-01T00:00:00Z",
                "2021-07-05T00:00:00Z",
                "2021-06-20T00:00:00Z",
            ],
        }
    )
    clean_df = clean_repositories(df)
    texts = combine_text_fields(clean_df)
    tfidf_mat, vectorizer = build_text_features(texts)
    assert tfidf_mat.shape[0] == 3
    assert tfidf_mat.shape[1] > 0

    cat_mat, _ = build_categorical_features(clean_df)
    num_mat, _ = build_numerical_features(clean_df)
    assert cat_mat.shape[0] == 3
    assert num_mat.shape == (3, 4)

    bundle = build_feature_bundle(clean_df, save=False)
    assert bundle["text_matrix"].shape[0] == 3
    assert np.all((bundle["metadata_matrix"] >= 0.0) & (bundle["metadata_matrix"] <= 1.0))
