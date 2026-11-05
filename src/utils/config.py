"""Project paths, constants, and default recommender parameters."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Most starred Github Repositories.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "repositories_clean.csv"
DATA_DICTIONARY_PATH = DATA_DIR / "processed" / "data_dictionary.csv"
EDA_SUMMARY_PATH = DATA_DIR / "processed" / "eda_summary.csv"
MISSING_VALUES_PATH = DATA_DIR / "processed" / "missing_values_analysis.csv"
CLEANING_REPORT_PATH = DATA_DIR / "processed" / "cleaning_before_after.csv"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
TFIDF_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
FEATURE_BUNDLE_PATH = ARTIFACTS_DIR / "feature_matrices" / "feature_bundle.joblib"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_DIR = REPORTS_DIR / "evaluation"

# Default multi-modal similarity weights (Phase 3.5: S = 0.65 S_text + 0.25 S_category + 0.10 S_metadata)
DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "text": 0.65,
    "category": 0.25,
    "metadata": 0.10,
}

# Default TF-IDF hyperparameters (Phase 3.2)
DEFAULT_TFIDF_PARAMS: dict[str, object] = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.90,
    "stop_words": "english",
    "sublinear_tf": True,
}

# Dormancy threshold (in days prior to dataset snapshot date) to treat a repo as archived/dormant
ARCHIVE_INACTIVITY_DAYS: int = 365
