"""Phase 3.1 & 3.2 — Text feature engineering and TF-IDF vectorization."""

from __future__ import annotations

from typing import Iterable
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.preprocessing import normalize_text
from src.utils.config import DEFAULT_TFIDF_PARAMS
from src.utils.helpers import parse_topic_list

# Token pattern that keeps single-letter languages (c, r) and symbols (c++, c#, .net, node.js)
TECHNICAL_TOKEN_PATTERN = r"(?u)(?:[a-zA-Z0-9_]+(?:\+\+|#|\.js|\.net)?|\.[a-zA-Z0-9_]+)"


def combine_text_fields(
    data: pd.DataFrame,
    desc_weight: int = 2,
    topics_weight: int = 2,
    name_weight: int = 1,
) -> pd.Series:
    """Combine repository name, normalized description, and topics into a weighted text document."""
    n = len(data)
    if "clean_description" in data.columns:
        desc_series = data["clean_description"].fillna("").astype(str)
    elif "description" in data.columns:
        desc_series = data["description"].fillna("").apply(normalize_text)
    else:
        desc_series = pd.Series([""] * n, index=data.index)

    if "topics" in data.columns:
        topics_series = (
            data["topics"]
            .fillna("")
            .astype(str)
            .str.replace(",", " ", regex=False)
            .str.replace("-", " ", regex=False)
        )
    else:
        topics_series = pd.Series([""] * n, index=data.index)

    if "repo_name" in data.columns:
        name_series = (
            data["repo_name"]
            .fillna("")
            .astype(str)
            .str.replace("-", " ", regex=False)
            .str.replace("_", " ", regex=False)
            .apply(normalize_text)
        )
    elif "name" in data.columns:
        name_series = data["name"].fillna("").astype(str).apply(normalize_text)
    else:
        name_series = pd.Series([""] * n, index=data.index)

    combined: list[str] = []
    for name_val, desc_val, topic_val in zip(name_series, desc_series, topics_series):
        parts: list[str] = []
        if name_val and name_weight > 0:
            parts.extend([name_val] * name_weight)
        if desc_val and desc_weight > 0:
            parts.extend([desc_val] * desc_weight)
        if topic_val and topics_weight > 0:
            parts.extend([topic_val] * topics_weight)
        doc = " ".join(parts).strip()
        combined.append(doc if doc else "repository")

    return pd.Series(combined, index=data.index, dtype=str)


def build_text_features(
    texts: Iterable[str],
    **kwargs,
) -> tuple[spmatrix, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer on repository texts and return (tfidf_matrix, vectorizer)."""
    text_list = [str(t) if str(t).strip() else "repository" for t in texts]
    params: dict[str, object] = dict(DEFAULT_TFIDF_PARAMS)
    params["token_pattern"] = TECHNICAL_TOKEN_PATTERN

    # Adapt document frequency thresholds automatically for small unit-test corpora
    if len(text_list) < 10:
        params.setdefault("min_df", 1)
        params["min_df"] = 1
        params["max_df"] = 1.0
        params["stop_words"] = None

    params.update(kwargs)
    vectorizer = TfidfVectorizer(**params)
    matrix = vectorizer.fit_transform(text_list)
    return matrix, vectorizer


def transform_text_query(
    vectorizer: TfidfVectorizer,
    description: str = "",
    topics: str | list[str] | None = None,
    language: str = "",
    desc_weight: int = 2,
    topics_weight: int = 2,
) -> spmatrix:
    """Transform a user query profile using the already-fitted TF-IDF vectorizer."""
    clean_desc = normalize_text(description)
    parsed_topics = parse_topic_list(topics)
    topic_text = " ".join(t.replace("-", " ") for t in parsed_topics)
    lang_text = normalize_text(language) if language and language.lower() not in {"any", "unknown"} else ""

    parts: list[str] = []
    if clean_desc:
        parts.extend([clean_desc] * max(1, desc_weight))
    if topic_text:
        parts.extend([topic_text] * max(1, topics_weight))
    if lang_text:
        parts.append(lang_text)

    query_doc = " ".join(parts).strip() or "repository"
    return vectorizer.transform([query_doc])
