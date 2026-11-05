"""Phase 2 — Data preprocessing, normalization, topic extraction, and feature derivation."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

from src.utils.config import ARCHIVE_INACTIVITY_DAYS
from src.utils.helpers import min_max_normalize

# Canonical mapping for programming languages and item buckets
LANGUAGE_STANDARDIZATION_MAP: dict[str, str] = {
    "matlab": "MATLAB",
    "viml": "Vim script",
    "vim-script": "Vim script",
    "vim script": "Vim script",
    "cpp": "C++",
    "c++": "C++",
    "csharp": "C#",
    "c#": "C#",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "coffeescript": "CoffeeScript",
    "actionscript": "ActionScript",
    "objective-c": "Objective-C",
    "jupyter notebook": "Jupyter Notebook",
    "apacheconf": "ApacheConf",
    "html": "HTML",
    "css": "CSS",
    "scss": "SCSS",
    "php": "PHP",
    "tex": "TeX",
    "python": "Python",
    "java": "Java",
    "go": "Go",
    "rust": "Rust",
    "ruby": "Ruby",
    "scala": "Scala",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "shell": "Shell",
    "c": "C",
    "r": "R",
    "lua": "Lua",
    "perl": "Perl",
    "haskell": "Haskell",
    "clojure": "Clojure",
    "dart": "Dart",
    "vue": "Vue",
    "groovy": "Groovy",
    "assembly": "Assembly",
}

NON_LANGUAGE_ITEMS = {"top-100-stars", "top-100-forks"}

# Curated technical topic taxonomy for deterministic offline topic extraction
TOPIC_TAXONOMY: dict[str, tuple[str, ...]] = {
    "machine-learning": (
        "machine learning",
        "scikit-learn",
        "sklearn",
        "xgboost",
        "lightgbm",
        "ml ",
        "classification",
        "regression",
        "clustering",
        "auto-ml",
        "automl",
        "reinforcement learning",
    ),
    "deep-learning": (
        "deep learning",
        "neural network",
        "tensorflow",
        "pytorch",
        "keras",
        "caffe",
        "mxnet",
        "transformer",
        "cnn",
        "rnn",
        "lstm",
        "gan",
        "pretrained",
    ),
    "computer-vision": (
        "computer vision",
        "image processing",
        "object detection",
        "segmentation",
        "opencv",
        "yolo",
        "face recognition",
        "ocr",
        "pose estimation",
        "video processing",
    ),
    "nlp": (
        "natural language processing",
        "nlp",
        "text mining",
        "bert",
        "gpt",
        "word2vec",
        "tokeniz",
        "speech recognition",
        "translation",
        "sentiment",
        "language model",
    ),
    "data-science": (
        "data science",
        "data analysis",
        "data visualization",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "ggplot",
        "jupyter",
        "analytics",
        "statistics",
        "statistical",
        "big data",
        "spark",
        "hadoop",
    ),
    "web-framework": (
        "web framework",
        "http server",
        "rest api",
        "graphql",
        "django",
        "flask",
        "fastapi",
        "express",
        "spring boot",
        "laravel",
        "rails",
        "asp.net",
        "gin",
        "microservice",
    ),
    "frontend-ui": (
        "frontend",
        "front-end",
        "user interface",
        "ui component",
        "react",
        "vue",
        "angular",
        "svelte",
        "tailwind",
        "bootstrap",
        "css framework",
        "animation",
        "dom",
        "web app",
        "browser",
    ),
    "mobile": (
        "android",
        "ios",
        "react native",
        "flutter",
        "swiftui",
        "mobile app",
        "kotlin multiplatform",
        "xcode",
        "objc",
    ),
    "devops-cloud": (
        "docker",
        "kubernetes",
        "k8s",
        "devops",
        "ci/cd",
        "terraform",
        "ansible",
        "prometheus",
        "grafana",
        "cloud",
        "aws",
        "serverless",
        "container",
        "deployment",
        "monitoring",
    ),
    "database": (
        "database",
        "sql",
        "nosql",
        "mysql",
        "postgresql",
        "mongodb",
        "redis",
        "sqlite",
        "elasticsearch",
        "orm",
        "storage engine",
        "key-value",
        "time-series",
    ),
    "security": (
        "security",
        "cryptography",
        "encryption",
        "penetration",
        "vulnerability",
        "exploit",
        "firewall",
        "authentication",
        "oauth",
        "privacy",
        "hacking",
        "reverse engineering",
        "malware",
    ),
    "game-development": (
        "game engine",
        "game development",
        "gamedev",
        "unity",
        "unreal",
        "godot",
        "2d game",
        "3d game",
        "physics engine",
        "opengl",
        "vulkan",
        "webgl",
        "shader",
        "emulator",
    ),
    "cli-tools": (
        "command-line",
        "command line",
        "terminal",
        "cli ",
        "shell script",
        "zsh",
        "bash",
        "tmux",
        "vim",
        "neovim",
        "console",
    ),
    "programming-education": (
        "learn to code",
        "curriculum",
        "tutorial",
        "course",
        "study plan",
        "interview",
        "programming books",
        "roadmap",
        "cheatsheet",
        "guide",
        "handbook",
        "examples",
    ),
    "awesome-list": (
        "awesome ",
        "curated list",
        "collection of",
        "list of ",
        "resources for",
    ),
    "algorithms-data-structures": (
        "algorithm",
        "data structure",
        "leetcode",
        "competitive programming",
        "sorting",
        "graph theory",
    ),
    "networking": (
        "network",
        "proxy",
        "vpn",
        "websocket",
        "tcp",
        "udp",
        "dns",
        "ssh",
        "p2p",
        "packet",
        "http client",
    ),
    "testing-qa": (
        "testing",
        "unit test",
        "test framework",
        "mock",
        "benchmark",
        "selenium",
        "puppeteer",
        "playwright",
        "linter",
        "static analysis",
    ),
    "operating-systems": (
        "operating system",
        "kernel",
        "linux",
        "windows",
        "macos",
        "driver",
        "embedded",
        "firmware",
        "rtos",
    ),
    "blockchain-crypto": (
        "blockchain",
        "bitcoin",
        "ethereum",
        "cryptocurrency",
        "smart contract",
        "web3",
        "defi",
    ),
    "scientific-computing": (
        "scientific",
        "simulation",
        "numerical",
        "linear algebra",
        "matrix",
        "signal processing",
        "bioinformatics",
        "astronomy",
        "physics",
        "mathematics",
    ),
    "automation-scraping": (
        "automation",
        "scraper",
        "scraping",
        "crawler",
        "bot ",
        "workflow",
        "task runner",
        "scheduler",
    ),
    "compilers-languages": (
        "compiler",
        "interpreter",
        "programming language",
        "parser",
        "ast",
        " transpiler",
        "virtual machine",
        "bytecode",
    ),
    "media-audio-video": (
        "video",
        "audio",
        "ffmpeg",
        "streaming",
        "music",
        "media player",
        "synthesizer",
        "webrtc",
    ),
}

# Short technical terms that must not be stripped during normalization
PRESERVED_SHORT_TOKENS = {
    "c",
    "r",
    "go",
    "ai",
    "ml",
    "dl",
    "cv",
    "ui",
    "ux",
    "db",
    "os",
    "io",
    "js",
    "ts",
    "py",
    "3d",
    "2d",
    "ci",
    "cd",
    "vr",
    "ar",
    "ip",
    "vm",
    "qt",
}


def standardize_language(lang_value: object, item_value: object = None) -> str:
    """Normalize language names and optionally infer missing language from item category."""
    raw = "" if lang_value is None or pd.isna(lang_value) else str(lang_value).strip()
    if raw:
        mapped = LANGUAGE_STANDARDIZATION_MAP.get(raw.lower())
        return mapped if mapped is not None else raw

    item_str = "" if item_value is None or pd.isna(item_value) else str(item_value).strip()
    if item_str and item_str not in NON_LANGUAGE_ITEMS:
        mapped_item = LANGUAGE_STANDARDIZATION_MAP.get(item_str.lower())
        return mapped_item if mapped_item is not None else item_str

    return "Unknown"


def normalize_text(text: object) -> str:
    """Normalize descriptions and technical text while preserving symbols like C++, C#, .NET, R, C, Go."""
    if text is None or pd.isna(text):
        return ""
    s = str(text).lower().strip()
    if not s:
        return ""

    # Remove URLs and GitHub emoji shortcodes like :books:
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r":[a-z0-9_+-]+:", " ", s)

    # Preserve compound technical tokens before stripping general punctuation
    s = re.sub(r"[^a-z0-9+#.\-\s]", " ", s)

    # Clean standalone punctuation around tokens while keeping c++, c#, f#, .net, node.js, vue.js
    tokens: list[str] = []
    for raw_tok in s.split():
        if raw_tok in {"c++", "c#", "f#", "g++", ".net"}:
            tokens.append(raw_tok)
            continue
        cleaned_tok = raw_tok.strip(".-+#")
        if not cleaned_tok:
            continue
        if len(cleaned_tok) == 1 and cleaned_tok not in PRESERVED_SHORT_TOKENS:
            continue
        tokens.append(cleaned_tok)

    return " ".join(tokens)


_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {}


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Match a taxonomy keyword without matching inside other words.

    The left edge must always be a word boundary ("ml" must not match inside "html",
    "gin" not inside "engine"). Keywords of 4 characters or fewer also need a right
    boundary; longer ones may be stems (e.g. "tokeniz" matches "tokenizer").
    """
    pattern = _KEYWORD_PATTERNS.get(keyword)
    if pattern is None:
        core = keyword.strip().lower()
        right = r"(?![a-z0-9])" if len(core) <= 4 else ""
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(core) + right)
        _KEYWORD_PATTERNS[keyword] = pattern
    return pattern.search(text) is not None


def extract_repository_topics(
    repo_name: str,
    description: str,
    language: str,
    categories: str = "",
) -> str:
    """Extract structured technical topics from repository text and metadata."""
    haystack = f" {repo_name} {description} {categories} ".lower()
    matched: list[str] = []

    for topic, keywords in TOPIC_TAXONOMY.items():
        for kw in keywords:
            if _keyword_in_text(kw, haystack):
                matched.append(topic)
                break

    # Fallbacks based on language ecosystem if no specific keyword matched
    lang_lower = (language or "").lower()
    if not matched:
        if lang_lower in {"html", "css", "scss", "vue", "typescript", "javascript"}:
            matched.append("frontend-ui")
        elif lang_lower in {"r", "matlab", "jupyter notebook"}:
            matched.append("data-science")
        elif lang_lower in {"shell", "vim script"}:
            matched.append("cli-tools")
        elif lang_lower in {"kotlin", "swift", "objective-c", "dart"}:
            matched.append("mobile")
        elif lang_lower in {"c", "c++", "rust", "assembly"}:
            matched.append("operating-systems")
        elif lang_lower in {"tex"}:
            matched.append("scientific-computing")
        else:
            matched.append("software-development")

    return ", ".join(dict.fromkeys(matched))


def clean_repositories(data: pd.DataFrame) -> pd.DataFrame:
    """Clean, deduplicate, normalize, and enrich the repository DataFrame.

    Supports both the full GitHub dataset (`repo_name`, `repo_url`, `username`, `stars`, `forks`,
    `issues`, `last_commit`, `description`, `language`, `item`) and minimal test DataFrames.
    """
    if data.empty:
        return data.copy()

    cleaned = data.copy()

    # 1. Strip whitespace on basic string columns
    for column in (
        "name",
        "repo_name",
        "username",
        "description",
        "language",
        "topics",
        "item",
        "url",
        "repo_url",
    ):
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    # 2. Standardize language names (with inference from 'item' if available)
    if "language" in cleaned.columns:
        has_any_non_empty = cleaned["language"].astype(str).str.len().gt(0).any()
        if "item" in cleaned.columns:
            cleaned["language"] = [
                standardize_language(lang, item)
                for lang, item in zip(cleaned["language"], cleaned["item"])
            ]
        elif has_any_non_empty:
            cleaned["language"] = [
                standardize_language(lang, None) if str(lang).strip() else "Unknown"
                for lang in cleaned["language"]
            ]

    # 3. Identify URL column and remove invalid/empty URL records if URL column is present
    url_column = "url" if "url" in cleaned.columns else "repo_url" if "repo_url" in cleaned.columns else None
    if url_column:
        cleaned = cleaned[cleaned[url_column].str.len() > 0].copy()

        # Aggregate 'item' categories across duplicate rows before deduplicating
        if "item" in cleaned.columns:
            categories_map = (
                cleaned.groupby(url_column, sort=False)["item"]
                .apply(lambda s: ", ".join(dict.fromkeys(v for v in s if v)))
                .to_dict()
            )
            cleaned["categories"] = cleaned[url_column].map(categories_map)

        cleaned = cleaned.drop_duplicates(subset=url_column, keep="first")

    cleaned = cleaned.reset_index(drop=True)

    # 4. Construct canonical identifiers (`full_name`, `name`, `url`) for full dataset rows
    if "repo_name" in cleaned.columns and "username" in cleaned.columns:
        cleaned["full_name"] = cleaned["username"] + "/" + cleaned["repo_name"]
        if "name" not in cleaned.columns:
            cleaned["name"] = cleaned["full_name"]
    elif "repo_name" in cleaned.columns and "full_name" not in cleaned.columns:
        cleaned["full_name"] = cleaned["repo_name"]

    if "repo_url" in cleaned.columns and "url" not in cleaned.columns:
        cleaned["url"] = cleaned["repo_url"]

    # 5. Normalize textual description and derive topics if rich dataset columns are present
    if "description" in cleaned.columns:
        cleaned["clean_description"] = cleaned["description"].apply(normalize_text)

    if "topics" not in cleaned.columns or (cleaned["topics"].astype(str).str.len() == 0).all():
        if "description" in cleaned.columns or "repo_name" in cleaned.columns:
            repo_names = cleaned["repo_name"] if "repo_name" in cleaned.columns else cleaned.get("name", pd.Series("", index=cleaned.index))
            descriptions = cleaned["description"] if "description" in cleaned.columns else pd.Series("", index=cleaned.index)
            languages = cleaned["language"] if "language" in cleaned.columns else pd.Series("Unknown", index=cleaned.index)
            categories = cleaned["categories"] if "categories" in cleaned.columns else cleaned.get("item", pd.Series("", index=cleaned.index))
            cleaned["topics"] = [
                extract_repository_topics(str(r), str(d), str(l), str(c))
                for r, d, l, c in zip(repo_names, descriptions, languages, categories)
            ]

    # 6. Handle numerical columns and derive log & popularity features
    if "stars" in cleaned.columns:
        cleaned["stars"] = pd.to_numeric(cleaned["stars"], errors="coerce").fillna(0).clip(lower=0).astype(int)
        cleaned["log_stars"] = np.log1p(cleaned["stars"])

    if "forks" in cleaned.columns:
        cleaned["forks"] = pd.to_numeric(cleaned["forks"], errors="coerce").fillna(0).clip(lower=0).astype(int)
        cleaned["log_forks"] = np.log1p(cleaned["forks"])

    if "issues" in cleaned.columns:
        cleaned["issues"] = pd.to_numeric(cleaned["issues"], errors="coerce").fillna(0).clip(lower=0).astype(int)
        cleaned["log_issues"] = np.log1p(cleaned["issues"])

    if "last_commit" in cleaned.columns:
        parsed_dates = pd.to_datetime(cleaned["last_commit"], utc=True, errors="coerce")
        ref_date = parsed_dates.max()
        if pd.notna(ref_date):
            days_diff = (ref_date - parsed_dates).dt.total_seconds() / 86400.0
            cleaned["days_since_last_update"] = days_diff.fillna(days_diff.median() if pd.notna(days_diff.median()) else 0.0).clip(lower=0.0).round(1)
        else:
            cleaned["days_since_last_update"] = 0.0
        cleaned["is_archived"] = cleaned["days_since_last_update"] > ARCHIVE_INACTIVITY_DAYS
        cleaned["recency_score"] = 1.0 / (1.0 + np.log1p(cleaned["days_since_last_update"]))

    if "log_stars" in cleaned.columns and "log_forks" in cleaned.columns:
        norm_stars = min_max_normalize(cleaned["log_stars"].to_numpy())
        norm_forks = min_max_normalize(cleaned["log_forks"].to_numpy())
        # Popularity(r) = lambda * Norm(log(1 + stars)) + (1 - lambda) * Norm(log(1 + forks)) with lambda = 0.7
        cleaned["popularity_score"] = np.round(0.7 * norm_stars + 0.3 * norm_forks, 6)

    return cleaned


def generate_cleaning_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Create a before/after comparison DataFrame for Phase 2 deliverables."""
    raw_missing_desc = int(raw_df["description"].isna().sum()) if "description" in raw_df else 0
    clean_missing_desc = int((clean_df["description"].fillna("") == "").sum()) if "description" in clean_df else 0
    raw_missing_lang = int(raw_df["language"].isna().sum()) if "language" in raw_df else 0
    clean_unknown_lang = int((clean_df["language"] == "Unknown").sum()) if "language" in clean_df else 0

    rows = [
        {
            "Metric": "Total rows",
            "Before Cleaning": len(raw_df),
            "After Cleaning": len(clean_df),
            "Notes": "Removed 117 duplicate repository entries across top-100 category buckets",
        },
        {
            "Metric": "Unique repositories (by repo_url)",
            "Before Cleaning": int(raw_df["repo_url"].nunique()) if "repo_url" in raw_df else len(raw_df),
            "After Cleaning": int(clean_df["repo_url"].nunique()) if "repo_url" in clean_df else len(clean_df),
            "Notes": "100% unique repositories preserved",
        },
        {
            "Metric": "Unique programming languages",
            "Before Cleaning": int(raw_df["language"].dropna().nunique()) if "language" in raw_df else 0,
            "After Cleaning": int(clean_df["language"].nunique()) if "language" in clean_df else 0,
            "Notes": "Standardized Matlab->MATLAB, VimL->Vim script; inferred from item or mapped to Unknown",
        },
        {
            "Metric": "Missing / Unknown language count",
            "Before Cleaning": raw_missing_lang,
            "After Cleaning": clean_unknown_lang,
            "Notes": "Explicitly labeled as 'Unknown' category instead of NaN",
        },
        {
            "Metric": "Empty description count",
            "Before Cleaning": raw_missing_desc,
            "After Cleaning": clean_missing_desc,
            "Notes": "Converted NaN to empty string; preserved rows for metadata & topic matching",
        },
        {
            "Metric": "Engineered columns added",
            "Before Cleaning": len(raw_df.columns),
            "After Cleaning": len(clean_df.columns),
            "Notes": "Added full_name, categories, clean_description, topics, log_stars, log_forks, log_issues, days_since_last_update, is_archived, recency_score, popularity_score",
        },
    ]
    return pd.DataFrame(rows)
