"""Phase 8.1, 8.2 & 8.3 — Unit, integration, and performance tests for Cascade Hybrid & Evaluation."""

import time
import numpy as np
import pandas as pd

from src.data.loader import load_clean_repositories
from src.evaluation.metrics import (
    catalog_coverage,
    constraint_satisfaction_rate,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.features.metadata_features import load_or_build_feature_bundle
from src.recommenders.hybrid import combine, hybrid_recommend


def test_hybrid_combines_scores():
    result = combine(np.array([1.0]), np.array([0.0]), content_weight=0.75)
    assert result.tolist() == [0.75]


def test_evaluation_metrics_correctness():
    recommended = ["r1", "r2", "r3", "r4"]
    relevant = {"r1", "r3", "r5"}
    assert np.isclose(precision_at_k(recommended, relevant, k=2), 0.5)
    assert np.isclose(recall_at_k(recommended, relevant, k=4), 2.0 / 3.0)
    assert 0.0 < ndcg_at_k(recommended, {"r1": 2.0, "r3": 1.0}, k=4) <= 1.0

    feat_mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    ild = intra_list_diversity([0, 1], feat_mat)
    assert np.isclose(ild, 1.0)
    assert np.isclose(catalog_coverage([["r1", "r2"], ["r2", "r3"]], 6), 0.5)


def test_end_to_end_hybrid_workflows_and_performance():
    df = load_clean_repositories()
    bundle = load_or_build_feature_bundle(df)

    # 1. Mode 1: Reference repository + hard constraints
    t0 = time.perf_counter()
    res_mode1 = hybrid_recommend(
        data=df,
        feature_bundle=bundle,
        repo_id="tensorflow/tensorflow",
        constraints={"languages": ["Python", "C++"], "min_stars": 5000, "include_archived": False},
        top_k=5,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    recs1 = res_mode1["recommendations"]
    assert len(recs1) == 5
    assert "tensorflow/tensorflow" not in recs1["full_name"].tolist()
    assert constraint_satisfaction_rate(recs1, {"languages": ["Python", "C++"], "min_stars": 5000, "include_archived": False}) == 1.0
    assert all(recs1["explanation"].str.startswith("Recommended because"))
    assert latency_ms < 500.0

    # 2. Mode 2: User profile + constraints
    res_mode2 = hybrid_recommend(
        data=df,
        feature_bundle=bundle,
        user_profile={
            "description": "machine learning deep learning neural networks",
            "topics": ["machine-learning", "deep-learning"],
            "language": "Python",
        },
        constraints={"languages": ["Python"], "min_stars": 2000, "min_forks": 200, "include_archived": False},
        top_k=5,
    )
    recs2 = res_mode2["recommendations"]
    assert len(recs2) == 5
    assert set(recs2["language"].tolist()) == {"Python"}
    assert (recs2["stars"] >= 2000).all()

    # 3. Conflict resolution when constraints are impossible
    res_conflict = hybrid_recommend(
        data=df,
        feature_bundle=bundle,
        user_profile={"description": "web framework", "language": "Python"},
        constraints={"languages": ["Python"], "min_stars": 999_999},
        top_k=5,
    )
    assert res_conflict["recommendations"].empty
    assert res_conflict["conflict_diagnosis"] is not None
    assert res_conflict["conflict_diagnosis"]["has_conflict"] is True
