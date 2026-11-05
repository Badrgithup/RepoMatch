"""Reusable repository result row (flat, GitHub-search style)."""

from __future__ import annotations

import html
from typing import Any
import pandas as pd
import streamlit as st

from src.utils.helpers import format_compact_number, parse_topic_list

_DESC_LIMIT = 240


def _pct(value: Any) -> int:
    try:
        return max(0, min(100, int(round(float(value) * 100))))
    except (TypeError, ValueError):
        return 0


def _short_reason(text: str) -> str:
    """Drop the repeated 'Recommended because it' lead-in so rows scan quickly."""
    text = (text or "").strip()
    for lead in ("Recommended because it ", "Recommended because "):
        if text.startswith(lead):
            text = text[len(lead):]
            break
    return (text[:1].upper() + text[1:]) if text else ""


def repository_row_html(row: dict[str, Any], rank: int | None = None) -> str:
    """Build the HTML for one result row. Kept free of indentation (Markdown-safe)."""
    full_name = str(row.get("full_name") or row.get("name") or row.get("repo_name") or "unknown")
    owner, _, name = full_name.partition("/")
    if not name:
        owner, name = "", full_name
    url = html.escape(str(row.get("url") or row.get("repo_url") or "#"))
    desc = str(row.get("description") or "").strip() or "No description in the dataset."
    if len(desc) > _DESC_LIMIT:
        desc = desc[:_DESC_LIMIT].rsplit(" ", 1)[0] + "..."
    language = str(row.get("language") or "Unknown")

    stars = int(row.get("stars", 0) or 0)
    forks = int(row.get("forks", 0) or 0)
    issues = int(row.get("issues", 0) or 0)

    score = row.get("final_score", row.get("similarity_score", row.get("preference_score")))
    topics = parse_topic_list(row.get("topics", ""))
    hits = set(parse_topic_list(row.get("shared_topics", row.get("matched_topics", ""))))
    ordered = [t for t in topics if t in hits] + [t for t in topics if t not in hits]
    tags = "".join(
        f'<span class="rm-tag{" hit" if t in hits else ""}">{html.escape(t)}</span>' for t in ordered[:4]
    )

    n = f'<span class="n">{rank}</span>' if rank is not None else ""
    own = f'<span class="own">{html.escape(owner)} / </span>' if owner else ""
    dormant = '<span class="rm-tag-dormant">Dormant</span>' if bool(row.get("is_archived", False)) else ""

    score_html = ""
    if score is not None:
        p = _pct(score)
        score_html = (
            f'<div class="rm-score"><div class="v">{p}<small>%</small></div>'
            f'<div class="rm-bar"><i style="width:{p}%"></i></div>'
            f'<div class="l">match score</div></div>'
        )

    why = _short_reason(str(row.get("explanation") or ""))
    why_html = f'<div class="rm-why"><b>Why</b> &nbsp;{html.escape(why)}</div>' if why else ""

    det_html = ""
    if all(row.get(k) is not None for k in ("text_similarity", "category_similarity", "metadata_similarity")):
        parts = "".join(
            f'<span>{label}</span><div class="rm-bar"><i style="width:{_pct(row[key])}%"></i></div>'
            f'<span>{_pct(row[key])}</span>'
            for label, key in (
                ("Text", "text_similarity"),
                ("Language", "category_similarity"),
                ("Metadata", "metadata_similarity"),
            )
        )
        det_html = f'<details class="rm-det"><summary>Score breakdown</summary><div class="rm-parts">{parts}</div></details>'

    return (
        f'<div class="rm-row"><div class="rm-main">'
        f'<div class="rm-title">{n}{own}<a href="{url}" target="_blank" rel="noopener noreferrer">{html.escape(name)}</a>{dormant}</div>'
        f'<div class="rm-desc">{html.escape(desc)}</div>'
        f'<div class="rm-tags">{tags}</div>'
        f'<div class="rm-meta"><span><b>{html.escape(language)}</b></span>'
        f'<span><b>{format_compact_number(stars)}</b> stars</span>'
        f'<span><b>{format_compact_number(forks)}</b> forks</span>'
        f'<span><b>{format_compact_number(issues)}</b> open issues</span></div>'
        f'{why_html}{det_html}</div>{score_html}</div>'
    )


def render_result_list(results: pd.DataFrame, start: int = 1) -> None:
    """Render a DataFrame of recommendations as one continuous list."""
    rows = [
        repository_row_html(r.to_dict(), rank=i)
        for i, (_, r) in enumerate(results.iterrows(), start=start)
    ]
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_repository_card(
    repository: dict[str, Any] | pd.Series,
    rank: int | None = None,
    show_breakdown: bool = True,
) -> None:
    """Backwards-compatible single-row renderer."""
    row = repository.to_dict() if isinstance(repository, pd.Series) else dict(repository)
    st.markdown(repository_row_html(row, rank), unsafe_allow_html=True)
