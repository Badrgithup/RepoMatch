"""Shared utility functions for formatting, parsing, and score normalization."""

from __future__ import annotations

import numpy as np


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to an inclusive range."""
    return max(lower, min(value, upper))


def format_compact_number(value: float | int) -> str:
    """Format large numbers compactly (e.g. 12500 -> '12.5k')."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "0"
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if abs(num) >= 1_000:
        return f"{num / 1_000:.1f}k"
    return f"{int(num)}"


def parse_topic_list(topics_value: object) -> list[str]:
    """Parse a comma/semicolon-separated string or iterable of topics into a clean list."""
    if topics_value is None:
        return []
    if isinstance(topics_value, (list, tuple, set)):
        return [str(t).strip().lower() for t in topics_value if str(t).strip()]
    text = str(topics_value).strip()
    if not text or text.lower() == "nan":
        return []
    sep = ";" if ";" in text else ","
    return [part.strip().lower() for part in text.split(sep) if part.strip()]


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize a 1D numpy array to [0, 1] safely."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if np.isclose(max_val, min_val):
        return np.zeros_like(arr, dtype=float)
    return (arr - min_val) / (max_val - min_val)
