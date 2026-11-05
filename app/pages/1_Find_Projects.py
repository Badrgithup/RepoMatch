"""Find Projects: hard requirements first, soft preferences second."""

from __future__ import annotations

import html
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.repository_card import render_result_list
from app.components.styles import (
    apply_github_dark_style,
    get_clean_dataset,
    get_feature_bundle,
    render_sidebar_header,
)
from src.recommenders.hybrid import hybrid_recommend
from src.utils.helpers import format_compact_number, parse_topic_list

st.set_page_config(page_title="Find Projects - RepoMatch", page_icon="R", layout="wide")
apply_github_dark_style()
render_sidebar_header("Find Projects")

df = get_clean_dataset()
bundle = get_feature_bundle()
all_languages = sorted(df["language"].dropna().unique().tolist())
all_topics = sorted({t for v in df["topics"] for t in parse_topic_list(v)})

RECENCY = {"Any time": None, "Last 30 days": 30, "Last 90 days": 90, "Last 180 days": 180}
PAGE = 10
DEFAULTS = {
    "fp_languages": ["Python"],
    "fp_min_stars": 1000,
    "fp_min_forks": 100,
    "fp_include_archived": False,
    "fp_query_text": "machine learning and computer vision toolkit",
    "fp_topics": ["machine-learning", "computer-vision"],
    "fp_recency": "Any time",
    "fp_pop_weight": 0.15,
    "fp_sort_by": "Best match",
    "fp_shown": PAGE,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def _reset() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def _show_more() -> None:
    st.session_state["fp_shown"] += PAGE


st.title("Find projects")
st.markdown(
    '<p class="rm-lede">Hard requirements remove repositories. Preferences only change the order.</p>',
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1, 2.2], gap="large")

with left:
    with st.form("filters", border=True):
        st.markdown('<div class="rm-h">REQUIREMENTS (HARD)</div>', unsafe_allow_html=True)
        st.multiselect("Language", all_languages, key="fp_languages", placeholder="Any language")
        a, b = st.columns(2)
        a.number_input("Min stars", min_value=0, max_value=500_000, step=500, key="fp_min_stars")
        b.number_input("Min forks", min_value=0, max_value=300_000, step=100, key="fp_min_forks")
        st.checkbox("Include dormant repositories (no commit in 365+ days)", key="fp_include_archived")
        st.selectbox("Last commit", list(RECENCY), key="fp_recency")

        st.markdown('<hr class="rm-sep" style="margin:14px 0"/>', unsafe_allow_html=True)
        st.markdown('<div class="rm-h">PREFERENCES (RANKING ONLY)</div>', unsafe_allow_html=True)
        st.text_input("What are you building?", key="fp_query_text", placeholder="e.g. object detection toolkit")
        st.multiselect("Topics", all_topics, key="fp_topics", placeholder="Any topic")
        st.slider("Weight of popularity", 0.0, 0.5, step=0.05, key="fp_pop_weight",
                  help="0 ranks by content match only. Higher values favour repositories with more stars and forks.")

        go, rs = st.columns(2)
        go.form_submit_button("Search", type="primary", use_container_width=True,
                              on_click=lambda: st.session_state.update(fp_shown=PAGE))
        rs.form_submit_button("Reset", use_container_width=True, on_click=_reset)

ss = st.session_state
constraints = {
    "languages": ss["fp_languages"] or None,
    "min_stars": int(ss["fp_min_stars"]),
    "min_forks": int(ss["fp_min_forks"]),
    "include_archived": bool(ss["fp_include_archived"]),
    "preferred_topics": ss["fp_topics"],
    "max_days_since_update": RECENCY[ss["fp_recency"]],
}
profile = {
    "description": ss["fp_query_text"],
    "topics": ss["fp_topics"],
    "language": ss["fp_languages"][0] if len(ss["fp_languages"]) == 1 else "",
}

with right:
    out = hybrid_recommend(
        data=df, feature_bundle=bundle, user_profile=profile, constraints=constraints,
        top_k=100, popularity_weight=float(ss["fp_pop_weight"]),
    )
    recs, diag, eligible = out["recommendations"], out["conflict_diagnosis"], out["eligible_count"]

    active = [
        ", ".join(ss["fp_languages"]) or "Any language",
        f"{format_compact_number(ss['fp_min_stars'])}+ stars" if ss["fp_min_stars"] else None,
        f"{format_compact_number(ss['fp_min_forks'])}+ forks" if ss["fp_min_forks"] else None,
        None if ss["fp_include_archived"] else "Active only",
        None if ss["fp_recency"] == "Any time" else ss["fp_recency"].lower(),
    ]
    st.markdown(
        f'<div class="rm-muted">Requirements: <b style="color:#E6EDF3">'
        f'{html.escape(" · ".join(x for x in active if x))}</b></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if recs.empty and diag is not None:
        failing = ", ".join(c.replace("_", " ") for c in diag["failing_constraints"]) or "the combination"
        st.markdown(
            f'<div class="rm-notice"><b>No repository meets all requirements.</b><br/>'
            f"The limiting requirement is {html.escape(failing)}. Nothing was loosened for you.</div>",
            unsafe_allow_html=True,
        )
        for s in diag["suggestions"]:
            st.markdown(f"- {s}")
        relaxed = diag["recommended_relaxation"]

        def _apply_relaxation() -> None:
            if relaxed.get("languages") is None:
                ss["fp_languages"] = []
            ss["fp_min_stars"] = int(relaxed.get("min_stars", 0) or 0)
            ss["fp_min_forks"] = int(relaxed.get("min_forks", 0) or 0)
            ss["fp_include_archived"] = bool(relaxed.get("include_archived", True))
            ss["fp_recency"] = "Any time"

        st.button("Apply the suggested relaxation", type="primary", on_click=_apply_relaxation)
    elif recs.empty:
        st.markdown(
            '<div class="rm-notice"><b>Nothing to rank.</b><br/>'
            "The requirements passed, but no repository matched your description closely enough. "
            "Broaden the description or remove a topic.</div>",
            unsafe_allow_html=True,
        )
    else:
        h1, h2 = st.columns([2, 1])
        h1.markdown(
            f"**{eligible:,}** repositories meet your requirements. Showing the best "
            f"**{min(ss['fp_shown'], len(recs))}** of **{len(recs)}** matches."
        )
        sort = h2.selectbox("Sort", ["Best match", "Stars", "Forks"], key="fp_sort_by", label_visibility="collapsed")
        if sort == "Stars":
            recs = recs.sort_values("stars", ascending=False)
        elif sort == "Forks":
            recs = recs.sort_values("forks", ascending=False)
        render_result_list(recs.head(ss["fp_shown"]))
        if ss["fp_shown"] < len(recs):
            st.write("")
            st.button("Show more", on_click=_show_more)
