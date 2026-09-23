"""
Scoring service layer — everything the API needs that is not HTTP.

Three responsibilities:

1. **History store.** Behavioural features need the customer's recent past.
   In production this would be a feature store (Redis / Feast). Here it is
   an in-memory ring buffer per customer, seeded from the CSV at startup,
   behind a small interface so swapping the backend touches one class.

2. **Hybrid decisioning.** Banks do not ship a bare ML score. A
   deterministic rule layer runs alongside the model: rules give auditable,
   regulator-explainable coverage of known typologies, and the model
   generalises to patterns nobody wrote a rule for. The final score is
   max(model score, rule score), which can only ever raise risk.

3. **Explanation.** For every alert we return the rules that fired plus the
   features that pushed the score up, so an investigator sees *why*.
"""
from __future__ import annotations

import pathlib
import time
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd

from features import build_features_online, ALL_FEATURES, REPORTING_LIMIT

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
HISTORY_LEN = 200          # transactions retained per customer


# ---------------------------------------------------------------- history --
class CustomerHistory:
    """Per-customer rolling window of recent transactions."""

    def __init__(self, maxlen: int = HISTORY_LEN):
        self._store: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def get(self, customer_id: str) -> list[dict]:
        return list(self._store[customer_id])

    def add(self, txn: dict) -> None:
        self._store[txn["customer_id"]].append({
            "timestamp": pd.Timestamp(txn["timestamp"]),
            "amount": float(txn["amount"]),
            "country": txn["country"],
            "device_id": txn["device_id"],
            "channel": txn["channel"],
        })

    def seed_from_csv(self, csv_path: pathlib.Path, per_customer: int = 60) -> int:
        if not csv_path.exists():
            return 0
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").groupby("customer_id").tail(per_customer)
        for r in df.itertuples():
            self.add({"customer_id": r.customer_id, "timestamp": r.timestamp,
                      "amount": r.amount, "country": r.country,
                      "device_id": r.device_id, "channel": r.channel})
        return len(df)

    @property
    def customers(self) -> int:
        return len(self._store)


# ------------------------------------------------------------ rule engine --
HIGH_RISK_COUNTRIES = {"NG", "RU", "AE"}


def run_rules(txn: dict, history: list[dict], f: pd.Series) -> list[dict]:
    """Deterministic typology rules. Each returns a code, reason and weight."""
    hits: list[dict] = []

    if f.txn_count_1h >= 4 and txn["channel"] == "CARD_ONLINE":
        hits.append({"code": "R01_VELOCITY_BURST", "weight": 0.55,
                     "description": f"{int(f.txn_count_1h)} card-not-present transactions in the last hour"})

    if f.near_reporting_limit and txn["channel"] in {"BRANCH_CASH", "ATM"}:
        hits.append({"code": "R02_STRUCTURING", "weight": 0.45,
                     "description": "cash transaction just below the Rs 50,000 PAN threshold"})

    recent_cash = sum(1 for h in history
                      if h["channel"] in {"BRANCH_CASH", "ATM"}
                      and 0.6 * REPORTING_LIMIT <= h["amount"] <= REPORTING_LIMIT
                      and (pd.Timestamp(txn["timestamp"]) - h["timestamp"]).days <= 7)
    if recent_cash >= 2 and txn["channel"] in {"BRANCH_CASH", "ATM"}:
        hits.append({"code": "R03_REPEAT_CASH_PATTERN", "weight": 0.6,
                     "description": f"{recent_cash} similar sub-PAN-threshold cash deposits in 7 days"})

    if f.amount_zscore_cust >= 6 and f.is_new_device_for_customer:
        hits.append({"code": "R04_AMOUNT_ANOMALY_NEW_DEVICE", "weight": 0.5,
                     "description": "amount far above the customer's baseline on an unrecognised device"})

    if txn["country"] in HIGH_RISK_COUNTRIES and f.is_new_country_for_customer \
            and f.amount_ratio_to_cust_median >= 3:
        hits.append({"code": "R05_HIGH_RISK_GEOGRAPHY", "weight": 0.45,
                     "description": f"large first-time transaction in {txn['country']}"})

    if txn["merchant_category"] in {"crypto", "p2p_transfer"} and f.is_new_beneficiary \
            and f.amount_ratio_to_cust_median >= 3:
        hits.append({"code": "R06_NEW_BENEFICIARY_VALUE", "weight": 0.35,
                     "description": "large transfer to a beneficiary seen for the first time"})

    if f.unique_countries_24h >= 3:
        hits.append({"code": "R07_IMPOSSIBLE_TRAVEL", "weight": 0.4,
                     "description": "transactions from 3+ countries within 24 hours"})

    return hits


# ---------------------------------------------------------------- scoring --
class RiskScorer:
    def __init__(self, model_path: pathlib.Path = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.model_name = bundle["model_name"]
        self.history = CustomerHistory()
        self._baseline = None

    # -- explanation -------------------------------------------------------
    def _top_signals(self, X: pd.DataFrame, k: int = 4) -> list[dict]:
        """
        Feature attribution by single-feature ablation: replace one feature
        with the training-median value and see how far the score falls.
        Model-agnostic, cheap for a handful of features, and easy to explain
        to a non-technical reviewer.
        """
        base = float(self.pipeline.predict_proba(X)[:, 1][0])
        watch = ["amount_zscore_cust", "amount_ratio_to_cust_median",
                 "txn_count_1h", "txn_count_24h", "amount_sum_24h",
                 "is_new_device_for_customer", "is_new_country_for_customer",
                 "is_foreign_country", "near_reporting_limit",
                 "seconds_since_prev_txn", "is_new_beneficiary",
                 "unique_countries_24h"]
        deltas = []
        for col in watch:
            probe = X.copy()
            neutral = {"amount_zscore_cust": 0.0, "amount_ratio_to_cust_median": 1.0,
                       "txn_count_1h": 0.0, "txn_count_24h": 1.0,
                       "amount_sum_24h": 0.0, "is_new_device_for_customer": 0,
                       "is_new_country_for_customer": 0, "is_foreign_country": 0,
                       "near_reporting_limit": 0, "seconds_since_prev_txn": 86400.0,
                       "is_new_beneficiary": 0, "unique_countries_24h": 1.0}[col]
            probe[col] = neutral
            drop = base - float(self.pipeline.predict_proba(probe)[:, 1][0])
            if drop > 1e-4:
                deltas.append({"feature": col,
                               "value": float(X[col].iloc[0]),
                               "contribution": round(drop, 4)})
        return sorted(deltas, key=lambda d: -d["contribution"])[:k]

    # -- main entry point --------------------------------------------------
    def score(self, txn: dict, persist: bool = True) -> dict:
        t0 = time.perf_counter()
        history = self.history.get(txn["customer_id"])
        X = build_features_online(txn, history)
        f = X.iloc[0]

        model_score = float(self.pipeline.predict_proba(X)[:, 1][0])
        hits = run_rules(txn, history, f)
        rule_score = max([h["weight"] for h in hits], default=0.0)
        # rules can only raise risk, never lower the model's view
        score = max(model_score, rule_score)

        if score >= max(self.threshold * 3, 0.5):
            band, decision = "HIGH", "BLOCK"
        elif score >= self.threshold:
            band, decision = "MEDIUM", "REVIEW"
        else:
            band, decision = "LOW", "ALLOW"

        signals = self._top_signals(X) if band != "LOW" else []
        if persist:
            self.history.add(txn)

        return {
            "transaction_id": txn["transaction_id"],
            "customer_id": txn["customer_id"],
            "risk_score": round(score, 4),
            "risk_band": band,
            "decision": decision,
            "model_version": self.model_name,
            "threshold": round(self.threshold, 4),
            "rule_hits": hits,
            "top_signals": signals,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
