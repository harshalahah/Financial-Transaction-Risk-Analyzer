"""
Synthetic transaction generator — Indian retail banking context.

Amounts are in INR. Channels match how money actually moves in India:
UPI dominates by count, cards and net-banking transfers (IMPS/NEFT) by
value, with ATM and branch cash still material.

Two regulatory thresholds matter for the crime patterns below:

  * PAN_THRESHOLD = Rs 50,000      -- cash deposits at or above this require
                                      the depositor to furnish a PAN, so
                                      smurfing clusters just under it
  * CTR_THRESHOLD = Rs 10,00,000   -- cash transactions above this trigger a
                                      Cash Transaction Report to FIU-IND

Three financial-crime typologies are injected:

  1. card-not-present fraud burst  (small test txn, then fast escalating spend)
  2. structuring / smurfing        (repeated cash-ins just under the PAN limit)
  3. mule pass-through             (large credit in, fast full debit out)

Genuine transactions also include deliberate look-alikes -- festival
shopping, gold purchases, legitimate large cash deposits, travel windows --
so the two classes are not trivially separable.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

RNG_SEED = 42

CHANNELS = ["UPI", "CARD_POS", "CARD_ONLINE", "IMPS", "NEFT", "ATM", "BRANCH_CASH"]
CHANNEL_P = [0.46, 0.17, 0.11, 0.09, 0.05, 0.08, 0.04]

CATEGORIES = ["kirana", "fuel", "food_delivery", "mobile_recharge", "utilities",
              "electronics", "travel", "gold_jewellery", "gaming", "crypto",
              "restaurant", "cash", "p2p_transfer"]
CATEGORY_P = [0.19, 0.09, 0.11, 0.07, 0.08, 0.04, 0.03,
              0.02, 0.02, 0.01, 0.12, 0.10, 0.12]

FOREIGN = ["AE", "SG", "HK", "US", "GB", "NG", "RU"]
PAN_THRESHOLD = 50_000.0
CTR_THRESHOLD = 10_00_000.0


def _customers(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n_customers)],
        "home_country": ["IN"] * n_customers,
        "age": rng.integers(19, 72, n_customers),
        "tenure_months": rng.integers(1, 240, n_customers),
        "account_type": rng.choice(["savings", "current", "salary"],
                                   n_customers, p=[.55, .2, .25]),
        # per-customer spending scale in INR: Rs 40,000 is routine for one
        # customer and extraordinary for another, so every amount feature
        # must be measured relative to the customer, never in absolute rupees
        "spend_scale": rng.lognormal(mean=6.2, sigma=0.6, size=n_customers),
    })


def _genuine_rows(cust: pd.Series, n: int, start: pd.Timestamp,
                  rng: np.random.Generator) -> list[dict]:
    """Ordinary day-to-day behaviour, plus unlabelled look-alike anomalies."""
    gaps = rng.exponential(scale=26.0, size=n).cumsum()
    ts = start + pd.to_timedelta(gaps, unit="h")

    travel_start = start + pd.to_timedelta(rng.uniform(300, 1500), unit="h")
    travel_end = travel_start + pd.to_timedelta(rng.uniform(72, 336), unit="h")
    travel_country = rng.choice(FOREIGN)
    travels = rng.random() < 0.30

    home_device = f"D{abs(hash(cust.customer_id)) % 9999:04d}"
    rows = []
    for t in ts:
        amount = float(np.round(cust.spend_scale * rng.lognormal(0, 0.8), 2))
        channel = rng.choice(CHANNELS, p=CHANNEL_P)
        category = rng.choice(CATEGORIES, p=CATEGORY_P)

        r = rng.random()
        if r < 0.02:                       # festival / big-ticket purchase
            amount = float(np.round(cust.spend_scale * rng.uniform(8, 25), 2))
            channel = rng.choice(["CARD_ONLINE", "CARD_POS", "IMPS"])
            category = rng.choice(["electronics", "gold_jewellery", "travel"])
        elif r < 0.032:                    # genuine large cash deposit
            amount = float(np.round(PAN_THRESHOLD * rng.uniform(0.7, 3.0), 2))
            channel = rng.choice(["BRANCH_CASH", "ATM"])
            category = "cash"

        on_holiday = travels and travel_start <= t <= travel_end
        rows.append({
            "customer_id": cust.customer_id,
            "timestamp": t,
            "amount": amount,
            "channel": channel,
            "merchant_category": category,
            "country": travel_country if on_holiday else (
                "IN" if rng.random() < 0.95 else rng.choice(FOREIGN)),
            "device_id": home_device if rng.random() < 0.87
            else f"D{rng.integers(0, 9999):04d}",
            "is_new_beneficiary": int(rng.random() < 0.12),
            "label": 0,
            "typology": "none",
        })
    return rows


def _fraud_burst(cust, start, rng) -> list[dict]:
    """Card details tested with a tiny charge, then drained online."""
    n = int(rng.integers(4, 9))
    t = start + pd.to_timedelta(rng.uniform(0, 5), unit="m")
    dev = f"D{rng.integers(0, 9999):04d}"
    ctry = rng.choice(["NG", "RU", "US", "AE"]) if rng.random() < 0.6 else "IN"
    rows = []
    for i in range(n):
        amt = float(np.round(rng.uniform(1, 25), 2)) if i == 0 else float(
            np.round(cust.spend_scale * rng.uniform(1.5, 7.0), 2))
        rows.append({
            "customer_id": cust.customer_id, "timestamp": t,
            "amount": amt, "channel": "CARD_ONLINE",
            "merchant_category": rng.choice(
                ["electronics", "crypto", "travel", "gaming"]),
            "country": ctry, "device_id": dev,
            "is_new_beneficiary": int(rng.random() < 0.7),
            "label": 1, "typology": "cnp_fraud",
        })
        t += pd.to_timedelta(rng.uniform(2, 95), unit="m")
    return rows


def _structuring(cust, start, rng) -> list[dict]:
    """Repeated cash deposits kept under the Rs 50,000 PAN requirement."""
    n = int(rng.integers(3, 7))
    t = start
    rows = []
    for _ in range(n):
        amt = float(np.round(PAN_THRESHOLD * rng.uniform(0.62, 0.99), 2))
        rows.append({
            "customer_id": cust.customer_id, "timestamp": t,
            "amount": amt,
            "channel": rng.choice(["BRANCH_CASH", "ATM"], p=[.6, .4]),
            "merchant_category": "cash", "country": "IN",
            "device_id": f"D{abs(hash(cust.customer_id)) % 9999:04d}",
            "is_new_beneficiary": 0, "label": 1, "typology": "structuring",
        })
        t += pd.to_timedelta(rng.uniform(6, 60), unit="h")
    return rows


def _mule(cust, start, rng) -> list[dict]:
    """Large inbound credit, then near-total outflow within hours."""
    inflow = float(np.round(cust.spend_scale * rng.uniform(8, 30), 2))
    t = start
    rows = [{
        "customer_id": cust.customer_id, "timestamp": t, "amount": inflow,
        "channel": "IMPS", "merchant_category": "p2p_transfer", "country": "IN",
        "device_id": f"D{abs(hash(cust.customer_id)) % 9999:04d}",
        "is_new_beneficiary": 1, "label": 1, "typology": "mule",
    }]
    remaining, k = inflow * rng.uniform(0.9, 0.99), int(rng.integers(2, 5))
    for _ in range(k):
        t += pd.to_timedelta(rng.uniform(0.5, 20), unit="h")
        rows.append({
            "customer_id": cust.customer_id, "timestamp": t,
            "amount": float(np.round(remaining / k, 2)),
            "channel": rng.choice(["IMPS", "UPI"]),
            "merchant_category": rng.choice(["p2p_transfer", "crypto", "cash"]),
            "country": rng.choice(["IN", "AE", "HK", "NG"]),
            "device_id": f"D{rng.integers(0, 9999):04d}",
            "is_new_beneficiary": 1, "label": 1, "typology": "mule",
        })
    return rows


def generate(n_customers: int = 1200, avg_txn: int = 60,
             suspicious_customer_rate: float = 0.06,
             seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = _customers(n_customers, rng)
    start = pd.Timestamp("2024-01-01")
    rows: list[dict] = []

    for _, cust in customers.iterrows():
        n = max(12, int(rng.normal(avg_txn, 14)))
        rows += _genuine_rows(cust, n, start, rng)

        if rng.random() < suspicious_customer_rate:
            attack_at = start + pd.to_timedelta(rng.uniform(200, 1600), unit="h")
            kind = rng.choice(["cnp_fraud", "structuring", "mule"], p=[.5, .3, .2])
            rows += {"cnp_fraud": _fraud_burst,
                     "structuring": _structuring,
                     "mule": _mule}[kind](cust, attack_at, rng)

    df = pd.DataFrame(rows)
    df = df.merge(customers.drop(columns=["spend_scale"]), on="customer_id")
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(df))])
    df.insert(4, "currency", "INR")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/transactions.csv")
    p.add_argument("--customers", type=int, default=1200)
    args = p.parse_args()

    df = generate(n_customers=args.customers)
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} transactions -> {args.out}")
    print(f"suspicious rate: {df.label.mean():.3%}")
    print(f"median amount: Rs {df.amount.median():,.0f}")
    print(df.typology.value_counts().to_string())
