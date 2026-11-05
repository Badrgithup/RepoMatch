"""Utilities for loading raw and processed repository datasets."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def load_repositories(path: str | Path | None = None) -> pd.DataFrame:
    """Load a repository CSV into a DataFrame. Defaults to RAW_DATA_PATH."""
    target = Path(path) if path is not None else RAW_DATA_PATH
    if not target.exists():
        raise FileNotFoundError(f"Repository dataset not found at: {target}")
    return pd.read_csv(target)


def load_clean_repositories(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned repository dataset, generating it from raw data if needed."""
    target = Path(path) if path is not None else PROCESSED_DATA_PATH
    if target.exists() and target.stat().st_size > 100:
        df = pd.read_csv(target)
        for col in ("description", "clean_description", "language", "topics", "categories"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        if "is_archived" in df.columns:
            df["is_archived"] = df["is_archived"].astype(bool)
        return df

    from src.data.preprocessing import clean_repositories

    raw_df = load_repositories(RAW_DATA_PATH)
    clean_df = clean_repositories(raw_df)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(target, index=False)
    return clean_df
