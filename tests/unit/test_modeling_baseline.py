"""Tests for src.modeling.baseline (Story 7-2)."""

import numpy as np
import pandas as pd
import pytest

from src.modeling import baseline as bl
from src.modeling.baseline import compute_metrics, make_pipeline, run_models


def test_compute_metrics_perfect_prediction():
    y = np.array([0.1, 0.2, 0.3, 0.4])
    m = compute_metrics(y, y)
    assert m["rmse"] == 0.0 and m["mae"] == 0.0
    assert m["r2"] == 1.0
    assert m["qlike"] == 0.0  # log(1)=0


def test_compute_metrics_r2_negative_for_bad_pred():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([4.0, 4.0, 4.0, 4.0])  # constant mean → r2 ~ 0 or negative-ish
    m = compute_metrics(y, pred)
    assert m["r2"] <= 0.0 + 1e-6
    assert m["rmse"] > 0


def test_compute_metrics_directional_accuracy():
    y = np.array([1.0, 2.0, 3.0, 2.0])
    yprev = np.array([0.5, 1.0, 2.0, 3.0])  # actual up,up,up,down
    pred = np.array([0.6, 1.5, 4.0, 1.0])   # pred up,up,up,down → 100% dir acc
    m = compute_metrics(y, pred, yprev)
    assert m["dir_acc"] == 1.0


def test_compute_metrics_empty():
    m = compute_metrics(np.array([]), np.array([]))
    assert m["rmse"] is None


def test_make_pipeline_fit_predict():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50)})
    y = pd.Series(2 * X["a"] + rng.normal(0, 0.1, 50))
    pipe = make_pipeline().fit(X, y)
    pred = pipe.predict(X)
    assert len(pred) == 50


def test_real_run_models_smoke():
    df = run_models()
    if df.empty:
        pytest.skip("no modeling panel (run EDA phases first)")
    assert {"target", "model", "feature_set", "rmse", "r2", "qlike"} <= set(df.columns)
    assert set(df["model"]) == {"ridge", "gbm"}
    assert set(df["feature_set"]) == {"price", "price+news_basic", "price+news_adv"}


def test_real_run_writes_report_smoke():
    written = bl.run()
    if not written:
        pytest.skip("no panel")
    names = {p.name for p in written}
    assert {"metrics.csv", "comparison_report.md"} <= names
