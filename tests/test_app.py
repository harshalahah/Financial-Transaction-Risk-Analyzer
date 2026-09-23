"""Tests: feature correctness, leakage guard, rule firing, API contract."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from features import build_features, build_features_online, ALL_FEATURES
from generate_data import generate
import api as api_module


@pytest.fixture(scope="module")
def client():
    with TestClient(api_module.app) as c:
        yield c


# ------------------------------------------------------------- features ---
def test_batch_features_have_no_nans_and_expected_columns():
    df = generate(n_customers=40, avg_txn=25, seed=7)
    feats = build_features(df)
    assert set(ALL_FEATURES).issubset(feats.columns)
    assert feats[ALL_FEATURES].select_dtypes("number").isna().sum().sum() == 0


def test_velocity_window_excludes_current_transaction():
    """First transaction of a customer must have zero prior-hour activity."""
    df = generate(n_customers=10, avg_txn=20, seed=3)
    feats = build_features(df).sort_values(["customer_id", "timestamp"])
    first = feats.groupby("customer_id").head(1)
    assert (first["txn_count_1h"] == 0).all()
    assert (first["amount_sum_24h"] == 0).all()


def test_online_and_batch_features_agree():
    """The API path and the training path must compute the same vector."""
    df = generate(n_customers=5, avg_txn=30, seed=11)
    feats = build_features(df)
    cust = feats.customer_id.value_counts().idxmax()
    sub = feats[feats.customer_id == cust].sort_values("timestamp")
    split = len(sub) // 2
    target = sub.iloc[split]
    history = [{"timestamp": r.timestamp, "amount": r.amount, "country": r.country,
                "device_id": r.device_id, "channel": r.channel}
               for r in sub.iloc[:split].itertuples()]
    online = build_features_online(target.to_dict(), history)
    for col in ["amount", "txn_count_24h", "amount_sum_24h", "is_night",
                "near_reporting_limit", "is_foreign_country"]:
        assert online[col].iloc[0] == pytest.approx(target[col], rel=1e-6), col


# ------------------------------------------------------------------ API ---
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_score_normal_transaction(client):
    r = client.post("/score", json={
        "transaction_id": "T-normal", "customer_id": "C00001",
        "timestamp": "2024-06-01T13:00:00", "amount": 425.0,
        "channel": "UPI", "merchant_category": "kirana", "country": "IN",
        "device_id": "D0001", "home_country": "IN", "tenure_months": 60, "age": 40})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["decision"] in {"ALLOW", "REVIEW", "BLOCK"}


def test_structuring_rule_fires(client):
    r = client.post("/score", json={
        "transaction_id": "T-struct", "customer_id": "C99999",
        "timestamp": "2024-06-01T10:00:00", "amount": 48500.0,
        "channel": "BRANCH_CASH", "merchant_category": "cash", "country": "IN",
        "device_id": "D7777", "home_country": "IN"})
    codes = [h["code"] for h in r.json()["rule_hits"]]
    assert "R02_STRUCTURING" in codes
    assert r.json()["decision"] != "ALLOW"


def test_validation_rejects_bad_payload(client):
    r = client.post("/score", json={
        "transaction_id": "T-bad", "customer_id": "C1",
        "timestamp": "2024-06-01T10:00:00", "amount": -5,
        "channel": "TELEPATHY", "merchant_category": "cash",
        "country": "IND", "device_id": "D1"})
    assert r.status_code == 422


def test_batch_scoring(client):
    txns = [{
        "transaction_id": f"T-b{i}", "customer_id": "C00002",
        "timestamp": f"2024-06-02T0{i}:00:00", "amount": 300.0 + i,
        "channel": "UPI", "merchant_category": "kirana", "country": "IN",
        "device_id": "D0002", "home_country": "IN"} for i in range(5)]
    r = client.post("/score/batch", json={"transactions": txns})
    assert r.status_code == 200 and r.json()["count"] == 5


def test_model_metrics_endpoint(client):
    r = client.get("/metrics/model")
    assert r.status_code == 200
    assert r.json()["model_comparison"]
