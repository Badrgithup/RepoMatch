"""Similar Repositories: pick one repository, get its closest neighbours."""

from __future__ import annotations

import html
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.repository_card import repository_row_html, render_result_list
from app.components.styles import (
    apply_github_dark_style,
    get_clean_dataset,
    get_feature_bundle,
    render_sidebar_header,
)
from src.recommenders.content_based import resolve_repository_index
from src.recommenders.hybrid import hybrid_recommend

st.set_page_config(page_title="Similar Repositories - RepoMatch", page_icon="R", layout="wide")
apply_github_dark_style()
render_sidebar_header("Similar Repositories")

df = get_clean_dataset()
bundle = get_feature_bundle()
names = df["full_name"].tolist()
default_repo = "tensorflow/tensorflow" if "tensorflow/tensorflow" in names else names[0]

st.title("Similar repositories")
st.markdown(
    '<p class="rm-lede">Pick a repository you already know. Results are ranked by how close their '
    "description, topics, language and activity are to it.</p>",
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1, 2.2], gap="large")

with left:
    ref_name = st.selectbox(
        "Reference repository", names, index=names.index(default_repo),
        help="Type to search the catalog by owner/name.",
    )
    langs = st.multiselect(
        "Only these languages", sorted(df["language"].dropna().unique()), placeholder="Any language",
    )
    top_k = st.slider("Results", 3, 25, 10)
    min_sim = st.slider("Minimum match", 0.0, 0.8, 0.10, 0.05, help="Hide results scoring below this.")
    with st.expander("Score weights"):
        w_text = st.slider("Text", 0.0, 1.0, 0.65, 0.05)
        w_cat = st.slider("Language and category", 0.0, 1.0, 0.25, 0.05)
        w_meta = st.slider("Metadata", 0.0, 1.0, 0.10, 0.05)
        pop = st.slider("Popularity blend", 0.0, 0.4, 0.0, 0.05)
        st.caption("Weights are normalised, so only their ratio matters.")

ref = df.iloc[resolve_repository_index(df, ref_name)]

with right:
    st.markdown('<div class="rm-h">REFERENCE</div>', unsafe_allow_html=True)
    st.markdown(repository_row_html({**ref.to_dict(), "explanation": ""}), unsafe_allow_html=True)
    st.write("")

    out = hybrid_recommend(
        data=df, feature_bundle=bundle, repo_id=ref_name,
        constraints={"languages": langs or None},
        top_k=int(top_k),
        feature_weights={"text": w_text, "category": w_cat, "metadata": w_meta},
        popularity_weight=float(pop), min_similarity=float(min_sim),
    )
    recs = out["recommendations"]

    if recs.empty:
        st.markdown(
            '<div class="rm-notice"><b>No repository scores above your minimum match.</b><br/>'
            + ("Try removing the language filter or " if langs else "Try ")
            + "lowering Minimum match.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="rm-h">{len(recs)} MOST SIMILAR TO {html.escape(ref_name.upper())}</div>',
            unsafe_allow_html=True,
        )
        render_result_list(recs)
