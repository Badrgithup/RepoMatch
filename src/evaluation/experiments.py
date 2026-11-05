"""Phase 7.1, 7.3 & 7.4 — Benchmark scenarios and experimental comparisons (E1-E6)."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    catalog_coverage,
    constraint_satisfaction_rate,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.features.metadata_features import build_feature_bundle
from src.recommenders.content_based import recommend_from_profile, recommend_similar
from src.recommenders.hybrid import hybrid_recommend
from src.recommenders.knowledge_based import recommend_by_constraints
from src.utils.helpers import parse_topic_list


def run_experiment(recommender: Callable, cases: Iterable) -> list:
    """Run a recommender callable against an iterable of test cases."""
    return [recommender(case) for case in cases]


def build_benchmark_scenarios(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Construct validated evaluation scenarios with ground-truth relevant sets.

    A candidate repository is judged relevant (grade >= 1) if it satisfies the scenario's
    domain topic overlap, primary language requirement, and minimum popularity threshold,
    with graded relevance (grade = 2) when it matches both primary language and >= 2 domain topics
    or strong lexical domain keywords.
    """
    scenarios_spec = [
        {
            "id": "Q1",
            "name": "Python Machine Learning & Deep Learning",
            "description": "machine learning deep learning neural network framework tensor training model",
            "topics": ["machine-learning", "deep-learning"],
            "language": "Python",
            "constraints": {"languages": ["Python"], "min_stars": 2000, "min_forks": 200, "include_archived": False},
            "keywords": ("machine learning", "deep learning", "neural", "tensor", "pytorch", "keras", "scikit", "model"),
        },
        {
            "id": "Q2",
            "name": "JavaScript Frontend UI Framework",
            "description": "progressive javascript framework for building reactive user interfaces web components dom",
            "topics": ["frontend-ui", "web-framework"],
            "language": "JavaScript",
            "constraints": {"languages": ["JavaScript"], "min_stars": 5000, "min_forks": 300, "include_archived": False},
            "keywords": ("react", "vue", "angular", "ui", "user interface", "component", "frontend", "dom", "browser"),
        },
        {
            "id": "Q3",
            "name": "Go Cloud & DevOps Infrastructure",
            "description": "cloud native container orchestration kubernetes docker microservice distributed server",
            "topics": ["devops-cloud", "networking"],
            "language": "Go",
            "constraints": {"languages": ["Go"], "min_stars": 3000, "min_forks": 200, "include_archived": False},
            "keywords": ("kubernetes", "docker", "container", "cloud", "proxy", "server", "distributed", "devops", "service"),
        },
        {
            "id": "Q4",
            "name": "C++ Game Engine & Graphics",
            "description": "3d game engine graphics rendering opengl vulkan physics simulation shader",
            "topics": ["game-development", "operating-systems"],
            "language": "C++",
            "constraints": {"languages": ["C++"], "min_stars": 1500, "min_forks": 150, "include_archived": True},
            "keywords": ("game", "engine", "graphics", "opengl", "renderer", "3d", "2d", "physics", "emulator", "shader"),
        },
        {
            "id": "Q5",
            "name": "Rust Systems & CLI Tools",
            "description": "fast command line terminal tool systems programming operating system shell",
            "topics": ["cli-tools", "operating-systems"],
            "language": "Rust",
            "constraints": {"languages": ["Rust"], "min_stars": 2000, "min_forks": 100, "include_archived": False},
            "keywords": ("command-line", "terminal", "cli", "fast", "shell", "rust", "system", "editor", "console"),
        },
        {
            "id": "Q6",
            "name": "Data Science & Statistical Computing in R",
            "description": "data visualization plotting statistical analysis grammar of graphics dataframe",
            "topics": ["data-science", "scientific-computing"],
            "language": "R",
            "constraints": {"languages": ["R"], "min_stars": 500, "min_forks": 100, "include_archived": True},
            "keywords": ("data", "plot", "ggplot", "visualization", "statistical", "analysis", "package", "tidy", "r "),
        },
        {
            "id": "Q7",
            "name": "TypeScript Web & Fullstack Applications",
            "description": "typescript web application framework compiler static types frontend state management",
            "topics": ["frontend-ui", "web-framework"],
            "language": "TypeScript",
            "constraints": {"languages": ["TypeScript"], "min_stars": 3000, "min_forks": 200, "include_archived": False},
            "keywords": ("typescript", "react", "angular", "vue", "web", "framework", "ui", "state", "node"),
        },
        {
            "id": "Q8",
            "name": "Java Enterprise Backend & Distributed Data",
            "description": "distributed database big data stream processing backend microservice spring framework",
            "topics": ["database", "web-framework"],
            "language": "Java",
            "constraints": {"languages": ["Java"], "min_stars": 3000, "min_forks": 500, "include_archived": True},
            "keywords": ("spring", "distributed", "database", "server", "java", "microservice", "search", "data", "netty"),
        },
        {
            "id": "Q9",
            "name": "Shell Scripting & Developer Productivity",
            "description": "bash zsh shell script command line automation dotfiles linux environment",
            "topics": ["cli-tools", "devops-cloud"],
            "language": "Shell",
            "constraints": {"languages": ["Shell"], "min_stars": 1500, "min_forks": 150, "include_archived": True},
            "keywords": ("shell", "bash", "zsh", "script", "linux", "command", "terminal", "install", "docker"),
        },
        {
            "id": "Q10",
            "name": "Mobile Development in Kotlin & Swift",
            "description": "android ios mobile application ui networking reactive library",
            "topics": ["mobile"],
            "language": "Kotlin",
            "constraints": {"languages": ["Kotlin"], "min_stars": 1500, "min_forks": 150, "include_archived": True},
            "keywords": ("android", "kotlin", "mobile", "ui", "app", "coroutines", "http", "image", "view"),
        },
    ]

    url_col = "url" if "url" in data.columns else "repo_url"
    built_scenarios: list[dict[str, Any]] = []

    for spec in scenarios_spec:
        req_lang = spec["language"].lower()
        req_topics = set(spec["topics"])
        min_stars = spec["constraints"]["min_stars"]
        min_forks = spec["constraints"]["min_forks"]
        include_archived = spec["constraints"]["include_archived"]

        graded_relevance: dict[str, float] = {}
        binary_relevant: set[str] = set()
        anchor_repo_id = None
        best_anchor_stars = -1

        for idx, row in data.iterrows():
            repo_key = str(row.get(url_col, idx))
            lang_match = str(row.get("language", "")).lower() == req_lang
            stars_ok = float(row.get("stars", 0)) >= min_stars
            forks_ok = float(row.get("forks", 0)) >= min_forks
            arch_ok = True if include_archived else not bool(row.get("is_archived", False))

            row_topics = set(parse_topic_list(row.get("topics", "")))
            topic_overlap = len(req_topics & row_topics)
            desc_text = f"{row.get('repo_name', '')} {row.get('description', '')}".lower()
            kw_hits = sum(1 for kw in spec["keywords"] if kw in desc_text)

            if lang_match and stars_ok and forks_ok and arch_ok and (topic_overlap >= 1 or kw_hits >= 1):
                binary_relevant.add(repo_key)
                grade = 2.0 if (topic_overlap >= 1 and kw_hits >= 2) else 1.0
                graded_relevance[repo_key] = grade
                if grade >= 2.0 and float(row.get("stars", 0)) > best_anchor_stars:
                    best_anchor_stars = float(row.get("stars", 0))
                    anchor_repo_id = row.get("full_name", repo_key)

        if binary_relevant:
            built_scenarios.append(
                {
                    **spec,
                    "anchor_repo_id": anchor_repo_id,
                    "relevant_set": binary_relevant,
                    "graded_relevance": graded_relevance,
                }
            )

    return built_scenarios


def run_ablation_experiments(
    data: pd.DataFrame,
    feature_bundle: dict[str, Any] | None = None,
    k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the 6 academic experiments (E1-E6) from Phase 7.3 across all benchmark scenarios.

    Returns:
      - `summary_df`: Aggregated metrics table for E1 through E6
      - `per_query_df`: Detailed per-scenario metrics table
    """
    bundle = feature_bundle if feature_bundle is not None else build_feature_bundle(data, save=False)
    scenarios = build_benchmark_scenarios(data)
    url_col = "url" if "url" in data.columns else "repo_url"
    url_to_pos = {str(u): idx for idx, u in enumerate(data[url_col])}

    experiments_config = [
        {
            "exp_id": "E1",
            "model": "E1: TF-IDF Only",
            "type": "content",
            "weights": {"text": 1.0, "category": 0.0, "metadata": 0.0},
            "popularity_weight": 0.0,
        },
        {
            "exp_id": "E2",
            "model": "E2: Text + Categorical",
            "type": "content",
            "weights": {"text": 0.72, "category": 0.28, "metadata": 0.0},
            "popularity_weight": 0.0,
        },
        {
            "exp_id": "E3",
            "model": "E3: Content-Based Full",
            "type": "content",
            "weights": {"text": 0.65, "category": 0.25, "metadata": 0.10},
            "popularity_weight": 0.0,
        },
        {
            "exp_id": "E4",
            "model": "E4: Knowledge-Based Only",
            "type": "knowledge",
            "weights": None,
            "popularity_weight": 0.0,
        },
        {
            "exp_id": "E5",
            "model": "E5: Cascade Hybrid (KB + CB)",
            "type": "hybrid",
            "weights": {"text": 0.65, "category": 0.25, "metadata": 0.10},
            "popularity_weight": 0.0,
        },
        {
            "exp_id": "E6",
            "model": "E6: Cascade Hybrid + Popularity",
            "type": "hybrid",
            "weights": {"text": 0.65, "category": 0.25, "metadata": 0.10},
            "popularity_weight": 0.20,
        },
    ]

    summary_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []

    for exp in experiments_config:
        all_rec_lists: list[list[str]] = []
        precisions: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        diversities: list[float] = []
        satisfactions: list[float] = []
        latencies_ms: list[float] = []

        for scen in scenarios:
            t0 = time.perf_counter()
            if exp["type"] == "content":
                rec_df = recommend_from_profile(
                    description=scen["description"],
                    topics=scen["topics"],
                    language=scen["language"] if exp["weights"]["category"] > 0 else "",
                    data=data,
                    feature_bundle=bundle,
                    top_k=k,
                    feature_weights=exp["weights"],
                )
            elif exp["type"] == "knowledge":
                rec_df = recommend_by_constraints(
                    df=data,
                    languages=scen["constraints"]["languages"],
                    min_stars=scen["constraints"]["min_stars"],
                    min_forks=scen["constraints"]["min_forks"],
                    include_archived=scen["constraints"]["include_archived"],
                    preferred_topics=scen["topics"],
                    top_k=k,
                )
            else:
                out = hybrid_recommend(
                    data=data,
                    feature_bundle=bundle,
                    user_profile={
                        "description": scen["description"],
                        "topics": scen["topics"],
                        "language": scen["language"],
                    },
                    constraints=scen["constraints"],
                    top_k=k,
                    feature_weights=exp["weights"],
                    popularity_weight=exp["popularity_weight"],
                )
                rec_df = out["recommendations"]

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            rec_urls = rec_df[url_col].astype(str).tolist() if not rec_df.empty and url_col in rec_df.columns else []
            rec_indices = [url_to_pos[u] for u in rec_urls if u in url_to_pos]
            all_rec_lists.append(rec_urls)

            p_k = precision_at_k(rec_urls, scen["relevant_set"], k=k)
            r_k = recall_at_k(rec_urls, scen["relevant_set"], k=k)
            n_k = ndcg_at_k(rec_urls, scen["graded_relevance"], k=k)
            ild = intra_list_diversity(rec_indices, bundle["text_matrix"])
            csat = constraint_satisfaction_rate(rec_df, scen["constraints"])

            precisions.append(p_k)
            recalls.append(r_k)
            ndcgs.append(n_k)
            diversities.append(ild)
            satisfactions.append(csat)

            per_query_rows.append(
                {
                    "Experiment": exp["exp_id"],
                    "Model": exp["model"],
                    "Query ID": scen["id"],
                    "Query Name": scen["name"],
                    f"Precision@{k}": round(p_k, 4),
                    f"Recall@{k}": round(r_k, 4),
                    f"NDCG@{k}": round(n_k, 4),
                    "Diversity (ILD)": round(ild, 4),
                    "Constraint Satisfaction": round(csat, 4),
                    "Latency (ms)": round(elapsed_ms, 2),
                }
            )

        cov = catalog_coverage(all_rec_lists, total_catalog_size=len(data))
        summary_rows.append(
            {
                "Experiment": exp["exp_id"],
                "Model": exp["model"],
                f"Precision@{k}": round(float(np.mean(precisions)), 4),
                f"Recall@{k}": round(float(np.mean(recalls)), 4),
                f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
                "Diversity (ILD)": round(float(np.mean(diversities)), 4),
                "Coverage": round(cov, 4),
                "Constraint Satisfaction": round(float(np.mean(satisfactions)), 4),
                "Avg Latency (ms)": round(float(np.mean(latencies_ms)), 2),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(per_query_rows)


def evaluate_weight_sensitivity(
    data: pd.DataFrame,
    feature_bundle: dict[str, Any] | None = None,
    k: int = 10,
) -> pd.DataFrame:
    """Evaluate how varying (w_text, w_category, w_metadata) impacts Precision@K, NDCG@K, and Diversity."""
    bundle = feature_bundle if feature_bundle is not None else build_feature_bundle(data, save=False)
    scenarios = build_benchmark_scenarios(data)
    url_col = "url" if "url" in data.columns else "repo_url"
    url_to_pos = {str(u): idx for idx, u in enumerate(data[url_col])}

    weight_configs = [
        {"label": "Text Heavy (0.85 / 0.10 / 0.05)", "text": 0.85, "category": 0.10, "metadata": 0.05},
        {"label": "Balanced Default (0.65 / 0.25 / 0.10)", "text": 0.65, "category": 0.25, "metadata": 0.10},
        {"label": "Equal Text & Cat (0.45 / 0.45 / 0.10)", "text": 0.45, "category": 0.45, "metadata": 0.10},
        {"label": "Category Heavy (0.30 / 0.55 / 0.15)", "text": 0.30, "category": 0.55, "metadata": 0.15},
        {"label": "Metadata Heavy (0.40 / 0.20 / 0.40)", "text": 0.40, "category": 0.20, "metadata": 0.40},
    ]

    rows: list[dict[str, Any]] = []
    for cfg in weight_configs:
        weights = {"text": cfg["text"], "category": cfg["category"], "metadata": cfg["metadata"]}
        precisions, ndcgs, diversities = [], [], []
        for scen in scenarios:
            rec_df = recommend_from_profile(
                description=scen["description"],
                topics=scen["topics"],
                language=scen["language"],
                data=data,
                feature_bundle=bundle,
                top_k=k,
                feature_weights=weights,
            )
            rec_urls = rec_df[url_col].astype(str).tolist() if not rec_df.empty else []
            rec_indices = [url_to_pos[u] for u in rec_urls if u in url_to_pos]
            precisions.append(precision_at_k(rec_urls, scen["relevant_set"], k=k))
            ndcgs.append(ndcg_at_k(rec_urls, scen["graded_relevance"], k=k))
            diversities.append(intra_list_diversity(rec_indices, bundle["text_matrix"]))

        rows.append(
            {
                "Configuration": cfg["label"],
                "w_text": cfg["text"],
                "w_category": cfg["category"],
                "w_metadata": cfg["metadata"],
                f"Precision@{k}": round(float(np.mean(precisions)), 4),
                f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
                "Diversity (ILD)": round(float(np.mean(diversities)), 4),
            }
        )
    return pd.DataFrame(rows)
