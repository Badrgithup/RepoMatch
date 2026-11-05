"""RepoMatch home: states the idea, then proves it with a live query."""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from src.utils.helpers import parse_topic_list

st.set_page_config(page_title="RepoMatch", page_icon="R", layout="wide")
apply_github_dark_style()
render_sidebar_header("Home")

df = get_clean_dataset()
bundle = get_feature_bundle()
n_topics = len({t for v in df["topics"] for t in parse_topic_list(v)})

st.title("RepoMatch")
st.markdown(
    '<p class="rm-lede">Stars tell you what is popular, not what fits. Set the requirements a repository '
    "must meet, then rank what is left by how closely it matches what you are building.</p>",
    unsafe_allow_html=True,
)
st.caption(f"{len(df):,} repositories · {df['language'].nunique()} languages · {n_topics} topic domains")

c1, c2, _ = st.columns([1, 1, 1.4])
with c1:
    st.page_link("pages/1_Find_Projects.py", label="Find a project by requirements")
with c2:
    st.page_link("pages/2_Similar_Repositories.py", label="Find repositories like one I know")

st.markdown('<hr class="rm-sep"/>', unsafe_allow_html=True)

PRESETS = {
    "Python machine learning, 5k+ stars": (
        {"description": "machine learning deep learning neural networks", "topics": ["machine-learning", "deep-learning"], "language": "Python"},
        {"languages": ["Python"], "min_stars": 5000, "include_archived": False},
    ),
    "JavaScript UI framework, 10k+ stars": (
        {"description": "frontend framework for building reactive user interfaces", "topics": ["frontend-ui", "web-framework"], "language": "JavaScript"},
        {"languages": ["JavaScript"], "min_stars": 10000, "include_archived": False},
    ),
    "Go cloud infrastructure, 3k+ stars": (
        {"description": "cloud native container orchestration kubernetes proxy", "topics": ["devops-cloud", "networking"], "language": "Go"},
        {"languages": ["Go"], "min_stars": 3000, "include_archived": False},
    ),
    "Rust command-line tools, 2k+ stars": (
        {"description": "fast command line terminal tool", "topics": ["cli-tools"], "language": "Rust"},
        {"languages": ["Rust"], "min_stars": 2000, "include_archived": False},
    ),
}

st.markdown('<div class="rm-h">LIVE EXAMPLE</div>', unsafe_allow_html=True)
choice = st.selectbox("Query", list(PRESETS), label_visibility="collapsed")
profile, constraints = PRESETS[choice]
out = hybrid_recommend(data=df, feature_bundle=bundle, user_profile=profile, constraints=constraints, top_k=3)

st.markdown(
    f'<div class="rm-funnel">'
    f'<div><b>{len(df):,}</b><span>repositories in the catalog</span></div>'
    f'<div><b>{out["eligible_count"]:,}</b><span>pass the hard constraints (Stage 1)</span></div>'
    f'<div class="hit"><b>3</b><span>ranked by content similarity (Stages 2-3)</span></div>'
    f"</div>",
    unsafe_allow_html=True,
)
render_result_list(out["recommendations"])

st.markdown('<hr class="rm-sep"/>', unsafe_allow_html=True)
st.markdown(
    '<div class="rm-steps">'
    '<div><span class="k">1</span><h4>Filter</h4><p>Language, minimum stars and forks, and activity are hard rules. '
    "A repository that breaks one is removed, never silently kept.</p></div>"
    '<div><span class="k">2</span><h4>Match</h4><p>Survivors are scored against your description, topics and language: '
    "65% text, 25% language and category, 10% metadata.</p></div>"
    '<div><span class="k">3</span><h4>Explain</h4><p>Each result states why it ranked, and every score can be '
    "opened up into its text, language and metadata parts.</p></div>"
    "</div>",
    unsafe_allow_html=True,
)
