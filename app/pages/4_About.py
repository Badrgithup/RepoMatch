"""About: method, formulas and the measured results."""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.charts import plot_ablation_comparison
from app.components.styles import apply_github_dark_style, get_evaluation_tables, render_sidebar_header

st.set_page_config(page_title="About - RepoMatch", page_icon="R", layout="wide")
apply_github_dark_style()
render_sidebar_header("About")
tables = get_evaluation_tables()

st.title("About")
st.markdown(
    '<p class="rm-lede">RepoMatch is a cascade hybrid recommender: a knowledge-based filter followed by '
    "a content-based ranker. No collaborative filtering, because the dataset has no user-repository interactions.</p>",
    unsafe_allow_html=True,
)

tab_method, tab_eval, tab_stack = st.tabs(["Method", "Evaluation", "Stack"])

with tab_method:
    st.markdown(
        """
**Stage 1, knowledge-based.** Language, minimum stars and forks, and activity are hard rules.
A repository that breaks one is removed. If nothing survives, the system names the limiting
requirement and proposes a relaxation instead of quietly loosening it.

**Stage 2, content-based.** Each repository is a vector in three spaces: TF-IDF over name,
description and topics (unigrams and bigrams), one-hot language and category, and scaled
log-popularity and recency. Similarity is cosine, computed per query against the surviving candidates only.

**Stage 3, ranking and explanation.** Soft preferences and an optional popularity term adjust the
order. Each result carries a sentence explaining why it ranked.
"""
    )
    st.latex(r"S(q,r)=0.65\,S_{text}+0.25\,S_{category}+0.10\,S_{metadata}")
    st.latex(r"Pop(r)=\lambda\,\mathrm{Norm}(\log(1+stars))+(1-\lambda)\,\mathrm{Norm}(\log(1+forks))")
    st.latex(r"Final(r)=\alpha\,S(q,r)+\beta\,Pop(r),\quad \alpha+\beta=1")
    st.caption(
        "Topics are derived from keyword rules over name and description, and 'dormant' means no commit in the "
        "365 days before the 2021-07-14 snapshot. The source CSV has neither field."
    )

with tab_eval:
    st.markdown(
        "Ten benchmark queries. Relevance labels are rule-based (language, popularity floor, topic or keyword "
        "overlap), not human judgments, so treat the numbers as a comparison between configurations rather than "
        "an absolute quality score."
    )
    if "ablation" in tables:
        st.plotly_chart(plot_ablation_comparison(tables["ablation"]), use_container_width=True)
        st.dataframe(tables["ablation"], use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation results found. Run notebooks/06_evaluation.ipynb to generate them.")
    if "weights" in tables:
        st.markdown("**Feature weight sensitivity**")
        st.dataframe(tables["weights"], use_container_width=True, hide_index=True)

with tab_stack:
    st.markdown(
        """
- **Python** with pandas and NumPy for cleaning and features
- **scikit-learn and SciPy** for TF-IDF, one-hot encoding, scaling and cosine similarity
- **joblib** to persist fitted vectorizers, so the app never refits on a query
- **Streamlit and Plotly** for the interface
- **pytest** for unit, integration and latency tests
"""
    )
