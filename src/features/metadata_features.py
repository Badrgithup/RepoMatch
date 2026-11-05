"""Phase 3.3, 3.4 & 3.5 — Categorical encoding, numerical scaling, and feature bundle persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, spmatrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.features.text_features import build_text_features, combine_text_fields
from src.utils.config import FEATURE_BUNDLE_PATH, TFIDF_VECTORIZER_PATH


def build_metadata_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return numeric metadata features with missing values filled (backwards-compatible helper)."""
    columns = [column for column in ("stars", "forks", "issues") if column in data.columns]
    if not columns:
        return pd.DataFrame(index=data.index)
    return data[columns].apply(pd.to_numeric, errors="coerce").fillna(0)


def _extract_categorical_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Extract standardized categorical columns (`language` and `primary_category`) for One-Hot Encoding."""
    lang = (
        data["language"].fillna("Unknown").astype(str).replace("", "Unknown")
        if "language" in data.columns
        else pd.Series(["Unknown"] * len(data), index=data.index)
    )
    if "item" in data.columns:
        cat = data["item"].fillna("general").astype(str).replace("", "general")
    elif "categories" in data.columns:
        cat = (
            data["categories"]
            .fillna("general")
            .astype(str)
            .apply(lambda s: s.split(",")[0].strip() if s.strip() else "general")
        )
    else:
        cat = pd.Series(["general"] * len(data), index=data.index)

    return pd.DataFrame({"language": lang, "category": cat}, index=data.index)


def build_categorical_features(
    data: pd.DataFrame,
    encoder: OneHotEncoder | None = None,
) -> tuple[spmatrix, OneHotEncoder]:
    """Encode categorical attributes (language, repository category) using One-Hot Encoding."""
    cat_df = _extract_categorical_frame(data)
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        matrix = encoder.fit_transform(cat_df)
    else:
        matrix = encoder.transform(cat_df)
    return matrix, encoder


def transform_categorical_query(
    encoder: OneHotEncoder,
    language: str = "Unknown",
    category: str = "general",
) -> spmatrix:
    """Encode a single user query's language and category with the fitted OneHotEncoder."""
    lang_val = language.strip() if language and language.strip().lower() != "any" else "Unknown"
    cat_val = category.strip() if category and category.strip() else (lang_val if lang_val != "Unknown" else "general")
    query_df = pd.DataFrame({"language": [lang_val], "category": [cat_val]})
    return encoder.transform(query_df)


def _extract_numerical_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Extract log-transformed numerical metadata features for scaling."""
    stars = pd.to_numeric(data.get("stars", 0), errors="coerce").fillna(0).clip(lower=0)
    forks = pd.to_numeric(data.get("forks", 0), errors="coerce").fillna(0).clip(lower=0)
    issues = pd.to_numeric(data.get("issues", 0), errors="coerce").fillna(0).clip(lower=0)

    log_stars = data["log_stars"] if "log_stars" in data.columns else np.log1p(stars)
    log_forks = data["log_forks"] if "log_forks" in data.columns else np.log1p(forks)
    log_issues = data["log_issues"] if "log_issues" in data.columns else np.log1p(issues)
    recency = data["recency_score"] if "recency_score" in data.columns else pd.Series(0.5, index=data.index)

    return pd.DataFrame(
        {
            "log_stars": pd.to_numeric(log_stars, errors="coerce").fillna(0.0),
            "log_forks": pd.to_numeric(log_forks, errors="coerce").fillna(0.0),
            "log_issues": pd.to_numeric(log_issues, errors="coerce").fillna(0.0),
            "recency_score": pd.to_numeric(recency, errors="coerce").fillna(0.5),
        },
        index=data.index,
    )


def build_numerical_features(
    data: pd.DataFrame,
    scaler: MinMaxScaler | None = None,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Scale log-transformed numerical metadata features into [0, 1]."""
    num_df = _extract_numerical_frame(data)
    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(num_df)
    else:
        scaled = scaler.transform(num_df)
    return scaled, scaler


def build_feature_bundle(
    data: pd.DataFrame,
    tfidf_params: dict[str, Any] | None = None,
    save: bool = False,
    tfidf_path: Path | None = None,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Build the complete three-group feature architecture (Text TF-IDF + Categorical OHE + Numerical Scaled)."""
    combined_texts = combine_text_fields(data)
    text_kwargs = tfidf_params or {}
    text_matrix, tfidf_vectorizer = build_text_features(combined_texts, **text_kwargs)
    category_matrix, category_encoder = build_categorical_features(data)
    metadata_matrix, metadata_scaler = build_numerical_features(data)

    bundle: dict[str, Any] = {
        "text_matrix": csr_matrix(text_matrix),
        "tfidf_vectorizer": tfidf_vectorizer,
        "category_matrix": csr_matrix(category_matrix),
        "category_encoder": category_encoder,
        "metadata_matrix": np.asarray(metadata_matrix, dtype=float),
        "metadata_scaler": metadata_scaler,
        "repo_names": data["full_name"].tolist() if "full_name" in data.columns else data.get("repo_name", pd.Series(dtype=str)).tolist(),
        "repo_urls": data["url"].tolist() if "url" in data.columns else data.get("repo_url", pd.Series(dtype=str)).tolist(),
    }

    if save:
        vec_target = Path(tfidf_path) if tfidf_path is not None else TFIDF_VECTORIZER_PATH
        bun_target = Path(bundle_path) if bundle_path is not None else FEATURE_BUNDLE_PATH
        vec_target.parent.mkdir(parents=True, exist_ok=True)
        bun_target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(tfidf_vectorizer, vec_target)
        joblib.dump(bundle, bun_target)

    return bundle


def load_or_build_feature_bundle(
    data: pd.DataFrame,
    bundle_path: Path | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Load persisted feature bundle from disk, or build and save it if missing/outdated."""
    target = Path(bundle_path) if bundle_path is not None else FEATURE_BUNDLE_PATH
    if not force_rebuild and target.exists() and target.stat().st_size > 100:
        bundle = joblib.load(target)
        if bundle.get("text_matrix") is not None and bundle["text_matrix"].shape[0] == len(data):
            return bundle
    return build_feature_bundle(data, save=True, bundle_path=target)
