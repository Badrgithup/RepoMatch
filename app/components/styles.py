"""Design tokens, restrained CSS, and cached resource loaders for the RepoMatch UI.

Design contract
- One accent: #2F81F7. Green/amber are semantic only (constraint met / relaxation).
- One cool grey family (GitHub Primer dark).
- One corner radius: 6px (controls, tags, rows, panels).
- No emoji, no gradients, no shadows. Structure comes from spacing and 1px hairlines.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_clean_repositories
from src.features.metadata_features import load_or_build_feature_bundle
from src.utils.config import CLEANING_REPORT_PATH, EVALUATION_DIR

PALETTE = {
    "bg": "#0D1117",
    "surface": "#161B22",
    "border": "#30363D",
    "text": "#E6EDF3",
    "body": "#C9D1D9",
    "muted": "#8B949E",
    "primary": "#2F81F7",
    "success": "#3FB950",
    "warning": "#E3B341",
}


@st.cache_data(show_spinner=False)
def get_clean_dataset() -> pd.DataFrame:
    """Load the cleaned repository dataset once."""
    return load_clean_repositories()


@st.cache_resource(show_spinner="Loading feature matrices...")
def get_feature_bundle() -> dict[str, Any]:
    """Load persisted TF-IDF vectorizer, encoders and feature matrices once."""
    return load_or_build_feature_bundle(get_clean_dataset())


@st.cache_data(show_spinner=False)
def get_evaluation_tables() -> dict[str, pd.DataFrame]:
    """Load saved evaluation tables from reports/evaluation/."""
    tables: dict[str, pd.DataFrame] = {}
    for key, path in (
        ("ablation", EVALUATION_DIR / "ablation_summary_e1_e6.csv"),
        ("per_query", EVALUATION_DIR / "ablation_per_query.csv"),
        ("weights", EVALUATION_DIR / "weight_sensitivity.csv"),
        ("cleaning", CLEANING_REPORT_PATH),
    ):
        if path.exists():
            tables[key] = pd.read_csv(path)
    return tables


_CSS = """
<style>
html, body, [class*="css"], .stApp {
  font-family: Inter, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
}
.stApp { background: #0D1117; }
.block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #0D1117; border-right: 1px solid #30363D; }

/* Theme safeguards if .streamlit/config.toml is not picked up (cwd-dependent) */
button[kind="primary"], button[kind="primaryFormSubmit"] {
  background: #2F81F7 !important; border-color: #2F81F7 !important; color: #fff !important; border-radius: 6px;
}
button[kind="secondary"], button[kind="secondaryFormSubmit"] { border-radius: 6px; border-color: #30363D; }
span[data-baseweb="tag"] { background: rgba(47,129,247,.16) !important; border-radius: 6px !important; }
span[data-baseweb="tag"] span { color: #C9D1D9 !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {
  border-radius: 6px !important; background: #0D1117 !important; border-color: #30363D !important;
}
div[data-testid="stForm"] { border: 1px solid #30363D; border-radius: 6px; padding: 18px 18px 8px; background: transparent; }
div[data-testid="stPageLink"] a {
  border: 1px solid #30363D; border-radius: 6px; padding: 10px 16px; background: #161B22;
  font-weight: 600; text-decoration: none;
}
div[data-testid="stPageLink"] a:hover { border-color: #2F81F7; }
div[data-testid="stMetric"] { background: transparent; border: 0; padding: 0; }
div[data-testid="stMetricLabel"] p { color: #8B949E; font-size: .8rem; }
:focus-visible { outline: 2px solid #2F81F7 !important; outline-offset: 2px; }

/* Typography */
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -.02em; }
.rm-lede { color: #C9D1D9; font-size: 1.05rem; line-height: 1.55; max-width: 62ch; margin: 4px 0 0; }
.rm-h { font-size: .8rem; font-weight: 600; color: #8B949E; margin: 0 0 10px; letter-spacing: .02em; }
.rm-muted { color: #8B949E; font-size: .86rem; }
.rm-sep { border: 0; border-top: 1px solid #30363D; margin: 28px 0; }

/* Cascade funnel: the product's own idea, with real counts */
.rm-funnel { display: flex; flex-wrap: wrap; align-items: stretch; border: 1px solid #30363D; border-radius: 6px; margin: 14px 0 6px; }
.rm-funnel > div { flex: 1 1 180px; padding: 14px 18px; border-right: 1px solid #30363D; }
.rm-funnel > div:last-child { border-right: 0; }
.rm-funnel b { display: block; font-size: 1.7rem; line-height: 1.1; color: #E6EDF3; font-variant-numeric: tabular-nums; }
.rm-funnel span { color: #8B949E; font-size: .82rem; }
.rm-funnel .hit b { color: #2F81F7; }

/* Result rows (flat list, GitHub-search style) */
.rm-row { display: flex; gap: 24px; justify-content: space-between; padding: 18px 0; border-bottom: 1px solid #30363D; }
.rm-row:first-of-type { border-top: 1px solid #30363D; }
.rm-main { min-width: 0; flex: 1; }
.rm-title { font-size: 1.02rem; font-weight: 600; word-break: break-word; }
.rm-title a { color: #2F81F7; text-decoration: none; }
.rm-title a:hover { text-decoration: underline; }
.rm-title .own { color: #8B949E; font-weight: 500; }
.rm-title .n { color: #8B949E; font-weight: 500; margin-right: 8px; font-variant-numeric: tabular-nums; }
.rm-tag-dormant { margin-left: 8px; font-size: .72rem; font-weight: 600; color: #E3B341; border: 1px solid #E3B341; border-radius: 6px; padding: 1px 7px; }
.rm-desc { color: #C9D1D9; font-size: .92rem; line-height: 1.5; margin: 6px 0 10px;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.rm-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.rm-tag { font-size: .76rem; color: #8B949E; border: 1px solid #30363D; border-radius: 6px; padding: 1px 8px; }
.rm-tag.hit { color: #2F81F7; border-color: rgba(47,129,247,.6); }
.rm-meta { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .82rem; color: #8B949E; font-variant-numeric: tabular-nums; }
.rm-meta b { color: #E6EDF3; font-weight: 600; }
.rm-why { margin: 10px 0 0; font-size: .84rem; color: #8B949E; line-height: 1.5; }
.rm-why b { color: #3FB950; font-weight: 600; }
.rm-score { width: 128px; flex: none; text-align: right; }
.rm-score .v { font-size: 1.6rem; font-weight: 700; color: #E6EDF3; line-height: 1; font-variant-numeric: tabular-nums; }
.rm-score .v small { font-size: .8rem; color: #8B949E; font-weight: 500; margin-left: 2px; }
.rm-bar { height: 4px; background: #30363D; border-radius: 6px; margin: 8px 0 6px; overflow: hidden; }
.rm-bar > i { display: block; height: 100%; background: #2F81F7; }
.rm-score .l { font-size: .75rem; color: #8B949E; }
details.rm-det { margin-top: 10px; font-size: .8rem; color: #8B949E; }
details.rm-det summary { cursor: pointer; width: fit-content; }
details.rm-det summary:hover { color: #E6EDF3; }
.rm-parts { display: grid; grid-template-columns: 78px 1fr 36px; gap: 6px 10px; align-items: center; margin-top: 8px; max-width: 360px; }
.rm-parts .rm-bar { margin: 0; }

/* Notices */
.rm-notice { border: 1px solid #30363D; border-left: 3px solid #E3B341; border-radius: 6px; padding: 12px 16px; margin: 6px 0 14px; color: #C9D1D9; font-size: .92rem; }
.rm-notice b { color: #E6EDF3; }
.rm-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 24px; }
.rm-steps p { color: #C9D1D9; font-size: .9rem; line-height: 1.55; margin: 4px 0 0; }
.rm-steps .k { color: #2F81F7; font-weight: 700; font-size: .85rem; }
.rm-steps h4 { margin: 2px 0 0; font-size: 1rem; }

@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .rm-row { flex-direction: column-reverse; gap: 10px; }
  .rm-score { width: 100%; text-align: left; display: flex; align-items: center; gap: 12px; }
  .rm-score .rm-bar { flex: 1; margin: 0; }
  .rm-funnel > div { border-right: 0; border-bottom: 1px solid #30363D; }
  .rm-funnel > div:last-child { border-bottom: 0; }
}
</style>
"""


def apply_github_dark_style() -> None:
    """Inject the restrained stylesheet (kept under its original name for the pages)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar_header(active_section: str = "") -> None:
    """Quiet sidebar footer: real catalog facts only, no fake status."""
    df = get_clean_dataset()
    with st.sidebar:
        st.caption(
            f"{len(df):,} repositories · {df['language'].nunique()} languages · "
            "snapshot 2021-07-14"
        )
