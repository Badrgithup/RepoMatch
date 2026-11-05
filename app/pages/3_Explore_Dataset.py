"""Phase 8.6 — Page 4: Explore Dataset (Interactive Plotly Visualizations & Quality Audit)."""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.charts import (
    plot_forks_distribution,
    plot_stars_distribution,
    plot_stars_vs_forks,
    plot_top_languages,
    plot_top_topics,
)
from app.components.styles import (
    apply_github_dark_style,
    get_clean_dataset,
    get_evaluation_tables,
    render_sidebar_header,
)
from src.utils.helpers import format_compact_number, parse_topic_list

st.set_page_config(page_title="Explore Dataset — RepoMatch", page_icon="R", layout="wide")
apply_github_dark_style()
render_sidebar_header("Explore Dataset")

df = get_clean_dataset()
tables = get_evaluation_tables()

st.title("Explore Dataset")
st.caption(
    "Interactive exploratory analysis and preprocessing audit of the Most Starred GitHub Repositories catalog."
)

all_topics = {t for val in df["topics"] for t in parse_topic_list(val)}
active_pct = (1.0 - df["is_archived"].mean()) * 100.0 if "is_archived" in df.columns else 100.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Unique Repositories", f"{len(df):,}")
c2.metric("Programming Languages", f"{df['language'].nunique()}")
c3.metric("Technical Topics", f"{len(all_topics)}")
c4.metric("Max Stars", format_compact_number(int(df["stars"].max())))
c5.metric("Active Repositories", f"{active_pct:.1f}%")

tab_overview, tab_popularity, tab_quality = st.tabs(
    ["Languages & Topics", "Popularity Distributions", "Data Quality & Top Repositories"]
)

with tab_overview:
    top_n = st.slider("Number of top categories to display", min_value=5, max_value=25, value=12, step=1)
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(plot_top_languages(df, top_n=top_n), use_container_width=True)
    with ch2:
        st.plotly_chart(plot_top_topics(df, top_n=top_n), use_container_width=True)

with tab_popularity:
    use_log = st.checkbox("Use log(1 + x) scale for histograms (reduces right-skewness)", value=True)
    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(plot_stars_distribution(df, use_log=use_log), use_container_width=True)
    with p2:
        st.plotly_chart(plot_forks_distribution(df, use_log=use_log), use_container_width=True)

    st.plotly_chart(plot_stars_vs_forks(df), use_container_width=True)

with tab_quality:
    st.subheader("Before vs. After Preprocessing Audit (Phase 2)")
    if "cleaning" in tables:
        st.dataframe(tables["cleaning"], use_container_width=True, hide_index=True)

    st.subheader("Top 15 Most Starred Repositories in Catalog")
    display_cols = [
        c
        for c in ("full_name", "language", "stars", "forks", "issues", "topics", "description", "url")
        if c in df.columns
    ]
    top_repos = df.sort_values("stars", ascending=False).head(15)[display_cols]
    st.dataframe(top_repos, use_container_width=True, hide_index=True)
