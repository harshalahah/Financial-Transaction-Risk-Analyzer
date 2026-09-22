"""
Feature engineering.

Design rule that governs this whole module: **every feature must be
computable at the moment the transaction arrives, using only that
transaction and the customer's history strictly before it.**

That rule is what keeps the offline training features identical to what
the FastAPI service can compute in real time, and it is what stops
target leakage (the classic mistake of building a "customer fraud rate"
column that already contains the answer).

Feature families
----------------
raw            : amount, hour, weekday, night flag
velocity       : count / sum of the customer's txns in the last 1h, 24h, 7d
deviation      : amount vs the customer's own rolling mean and std (z-score)
recency        : seconds since the customer's previous transaction
novelty        : first time we've seen this device / country for the customer
threshold      : closeness to the Rs 50,000 PAN cash limit (structuring cue)
categorical    : channel, merchant category, account type (one-hot later)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REPORTING_LIMIT = 50_000.0   # Rs 50,000: PAN required for cash deposits at/above this

NUMERIC_FEATURES = [
    "amount", "log_amount", "hour", "day_of_week", "is_night", "is_weekend",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_24h", "amount_sum_7d",
    "amount_zscore_cust", "amount_ratio_to_cust_median",
    "seconds_since_prev_txn", "unique_countries_24h",
    "is_new_device_for_customer", "is_foreign_country", "is_new_country_for_customer",
    "pct_of_reporting_limit", "near_reporting_limit",
    "is_new_beneficiary", "tenure_months", "age",
]
CATEGORICAL_FEATURES = ["channel", "merchant_category", "account_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------
# batch path (training)
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised, leak-free feature build over a full transaction ledger."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    g = df.groupby("customer_id", sort=False)

    # --- recency -----------------------------------------------------------
    df["seconds_since_prev_txn"] = (
        g["timestamp"].diff().dt.total_seconds().fillna(7 * 24 * 3600)
    )

    # --- rolling velocity windows (shifted -> excludes the current row) ----
    parts = []
    for cid, grp in g:
        grp = grp.set_index("timestamp")
        amt = grp["amount"]
        out = pd.DataFrame(index=grp.index)
        for win, tag in [("1h", "1h"), ("24h", "24h"), ("7D", "7d")]:
            out[f"txn_count_{tag}"] = amt.rolling(win).count() - 1
            out[f"amount_sum_{tag}"] = amt.rolling(win).sum() - amt
        # expanding customer profile, shifted so the current txn is excluded
        out["_cust_mean"] = amt.expanding().mean().shift(1)
        out["_cust_std"] = amt.expanding().std().shift(1)
        out["_cust_median"] = amt.expanding().median().shift(1)
        ccode = pd.Series(pd.factorize(grp["country"])[0].astype(float),
                          index=grp.index)
        out["unique_countries_24h"] = (
            ccode.rolling("24h").apply(lambda s: len(np.unique(s)), raw=True)
        )
        # first-seen novelty flags
        out["is_new_device_for_customer"] = (
            ~grp["device_id"].duplicated()).astype(int).values
        out["is_new_country_for_customer"] = (
            ~grp["country"].duplicated()).astype(int).values
        out["transaction_id"] = grp["transaction_id"].values
        parts.append(out.reset_index(drop=True))

    roll = pd.concat(parts, ignore_index=True)
    df = df.merge(roll, on="transaction_id", how="left")

    # --- deviation from the customer's own baseline ------------------------
    std = df["_cust_std"].replace(0, np.nan)
    df["amount_zscore_cust"] = ((df["amount"] - df["_cust_mean"]) / std).fillna(0.0)
    df["amount_ratio_to_cust_median"] = (
        df["amount"] / df["_cust_median"].replace(0, np.nan)).fillna(1.0)

    # --- threshold / geography cues ---------------------------------------
    df["pct_of_reporting_limit"] = df["amount"] / REPORTING_LIMIT
    df["near_reporting_limit"] = (
        df["amount"].between(0.85 * REPORTING_LIMIT, REPORTING_LIMIT)).astype(int)
    df["is_foreign_country"] = (df["country"] != df["home_country"]).astype(int)

    df = df.drop(columns=["_cust_mean", "_cust_std", "_cust_median"])
    df[NUMERIC_FEATURES] = (df[NUMERIC_FEATURES]
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0)
                            .clip(-1e6, 1e6))
    return df


# --------------------------------------------------------------------------
# online path (inference) — same definitions, one transaction at a time
# --------------------------------------------------------------------------
def build_features_online(txn: dict, history: list[dict]) -> pd.DataFrame:
    """
    Compute the identical feature vector for a single incoming transaction
    given the customer's prior transactions (already time-ordered, oldest
    first). `history` may be empty for a brand-new customer.
    """
    ts = pd.Timestamp(txn["timestamp"])
    amt = float(txn["amount"])
    hist = pd.DataFrame(history)
    if not hist.empty:
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        hist = hist[hist["timestamp"] < ts].sort_values("timestamp")

    def window(hours: float) -> pd.DataFrame:
        if hist.empty:
            return hist
        return hist[hist["timestamp"] >= ts - pd.Timedelta(hours=hours)]

    h1, h24, d7 = window(1), window(24), window(24 * 7)
    prev_amounts = hist["amount"].astype(float) if not hist.empty else pd.Series(dtype=float)

    mean = prev_amounts.mean() if len(prev_amounts) else np.nan
    std = prev_amounts.std() if len(prev_amounts) > 1 else np.nan
    median = prev_amounts.median() if len(prev_amounts) else np.nan

    row = {
        "amount": amt,
        "log_amount": float(np.log1p(amt)),
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "is_night": int(0 <= ts.hour <= 5),
        "is_weekend": int(ts.dayofweek >= 5),
        "txn_count_1h": float(len(h1)),
        "txn_count_24h": float(len(h24)),
        "txn_count_7d": float(len(d7)),
        "amount_sum_24h": float(h24["amount"].astype(float).sum()) if len(h24) else 0.0,
        "amount_sum_7d": float(d7["amount"].astype(float).sum()) if len(d7) else 0.0,
        "amount_zscore_cust": float((amt - mean) / std) if std and not np.isnan(std) and std > 0 else 0.0,
        "amount_ratio_to_cust_median": float(amt / median) if median and median > 0 else 1.0,
        "seconds_since_prev_txn": float((ts - hist["timestamp"].iloc[-1]).total_seconds())
        if not hist.empty else 7 * 24 * 3600.0,
        "unique_countries_24h": float(len(set(h24["country"])) + 1) if len(h24) else 1.0,
        "is_new_device_for_customer": int(hist.empty or txn["device_id"] not in set(hist["device_id"])),
        "is_foreign_country": int(txn["country"] != txn.get("home_country", txn["country"])),
        "is_new_country_for_customer": int(hist.empty or txn["country"] not in set(hist["country"])),
        "pct_of_reporting_limit": amt / REPORTING_LIMIT,
        "near_reporting_limit": int(0.85 * REPORTING_LIMIT <= amt <= REPORTING_LIMIT),
        "is_new_beneficiary": int(txn.get("is_new_beneficiary") or 0),
        "tenure_months": float(txn.get("tenure_months") or 0),
        "age": float(txn.get("age") or 0),
        "channel": txn["channel"],
        "merchant_category": txn["merchant_category"],
        "account_type": txn.get("account_type", "savings"),
    }
    return pd.DataFrame([row])[ALL_FEATURES]
