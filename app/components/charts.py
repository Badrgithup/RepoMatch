"""Plotly charts. One accent (#2F81F7); other series use neutral greys."""

from __future__ import annotations

from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.helpers import parse_topic_list

ACCENT = "#2F81F7"
GREYS = ["#79B8FF", "#8B949E", "#6E7681", "#484F58", "#30363D"]


def _style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(family="Inter, sans-serif", color="#C9D1D9", size=12),
        margin=dict(l=8, r=8, t=44, b=8),
        height=height,
        title_font=dict(size=14, color="#E6EDF3"),
        xaxis=dict(gridcolor="#21262D", zerolinecolor="#30363D"),
        yaxis=dict(gridcolor="#21262D", zerolinecolor="#30363D"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _hbar(frame: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    frame = frame.sort_values(x, ascending=True)
    fig = px.bar(frame, x=x, y=y, orientation="h", title=title, text=x, color_discrete_sequence=[ACCENT])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title=None)
    return _style_figure(fig, height=400)


def plot_top_languages(data: pd.DataFrame, top_n: int = 12) -> go.Figure:
    counts = data["language"].fillna("Unknown").value_counts().head(top_n).reset_index()
    counts.columns = ["Language", "Repositories"]
    return _hbar(counts, "Repositories", "Language", f"Top {top_n} languages")


def plot_top_topics(data: pd.DataFrame, top_n: int = 12) -> go.Figure:
    counter: Counter[str] = Counter()
    if "topics" in data.columns:
        for val in data["topics"]:
            counter.update(parse_topic_list(val))
    frame = pd.DataFrame(counter.most_common(top_n), columns=["Topic", "Repositories"])
    return _hbar(frame, "Repositories", "Topic", f"Top {top_n} topics")


def _hist(data: pd.DataFrame, col: str, label: str, title: str) -> go.Figure:
    fig = px.histogram(data, x=col, nbins=40, title=title, labels={col: label}, color_discrete_sequence=[ACCENT])
    fig.update_traces(marker_line_width=0)
    return _style_figure(fig, height=340)


def plot_stars_distribution(data: pd.DataFrame, use_log: bool = True) -> go.Figure:
    col = "log_stars" if (use_log and "log_stars" in data.columns) else "stars"
    return _hist(data, col, "log(1 + stars)" if col == "log_stars" else "Stars", "Stars")


def plot_forks_distribution(data: pd.DataFrame, use_log: bool = True) -> go.Figure:
    col = "log_forks" if (use_log and "log_forks" in data.columns) else "forks"
    return _hist(data, col, "log(1 + forks)" if col == "log_forks" else "Forks", "Forks")


def plot_stars_vs_forks(data: pd.DataFrame) -> go.Figure:
    top = list(data["language"].value_counts().head(5).index)
    frame = data.copy()
    frame["Language"] = frame["language"].where(frame["language"].isin(top), "Other")
    fig = px.scatter(
        frame, x="stars", y="forks", color="Language",
        hover_name="full_name" if "full_name" in frame.columns else "repo_name",
        log_x=True, log_y=True, title="Stars vs forks (log scale)",
        color_discrete_sequence=[ACCENT] + GREYS, opacity=0.8,
    )
    return _style_figure(fig, height=420)


def plot_ablation_comparison(ablation_df: pd.DataFrame) -> go.Figure:
    metrics = [c for c in ("Precision@10", "NDCG@10", "Diversity (ILD)", "Constraint Satisfaction") if c in ablation_df.columns]
    melted = ablation_df.melt(id_vars=["Experiment", "Model"], value_vars=metrics, var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="Experiment", y="Score", color="Metric", barmode="group", hover_data=["Model"],
        title="E1 to E6 compared", color_discrete_sequence=[ACCENT] + GREYS[:3],
    )
    fig.update_yaxes(range=[0, 1.05])
    return _style_figure(fig, height=380)


def render_language_counts(data: pd.DataFrame) -> None:
    if "language" in data.columns:
        st.plotly_chart(plot_top_languages(data), use_container_width=True)
