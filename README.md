# AI-Powered Financial Transaction Risk Analyzer

A hybrid machine-learning and rules system that scores banking transactions for
financial-crime risk in real time, and serves that score over a FastAPI backend
with an auditable explanation attached to every alert.

**Stack:** Python 3.12 · FastAPI · scikit-learn · pandas · pydantic v2 · pytest
**Data:** Indian retail banking context — amounts in INR, UPI / cards / IMPS / NEFT / ATM / branch cash, Rs 50,000 PAN cash-deposit threshold

---

## What it does

| Layer | Responsibility |
|---|---|
| `src/generate_data.py` | Builds a synthetic Indian retail banking ledger (INR, UPI/IMPS/cards/cash) with three injected crime typologies and realistic look-alike confounders |
| `src/features.py` | 25 point-in-time behavioural features, computed identically for training and live inference |
| `src/train.py` | Time-based split, three candidate models, selection and threshold set against an alert budget |
| `src/service.py` | Model loading, customer history store, deterministic rule layer, feature attribution |
| `src/api.py` | FastAPI endpoints, validation, correlation IDs, structured logging |
| `tests/test_app.py` | 9 tests covering feature leakage, online/offline parity, rules and API contract |

## Results (held-out later time period, 21,244 transactions, 0.44% suspicious)

| Model | PR-AUC | ROC-AUC | Recall @ 1% alert budget |
|---|---|---|---|
| Logistic regression | 0.321 | 0.987 | 0.667 |
| **Random forest (selected)** | 0.828 | 0.999 | **0.978** |
| Gradient boosting | 0.838 | 0.998 | 0.892 |

The model is selected on recall inside the alert budget, not on PR-AUC: only
the top ~1% of scores will ever be reviewed, so that is the part of the curve
that matters.

At the chosen threshold: **recall 0.946, precision 0.500, alert rate 0.83%** —
88 of 93 suspicious transactions caught while sending fewer than 1 in 100
transactions to a human reviewer. Per-typology recall: fraud burst 0.968,
mule pass-through 0.909, structuring 0.895. Median scoring latency **19.4 ms**.

## Quickstart

```bash
pip install -r requirements.txt
python src/generate_data.py --out data/transactions.csv   # build dataset
cd src && python train.py                                  # train + evaluate
python demo.py                                             # scored scenarios
uvicorn api:app --reload --port 8000                       # serve; docs at /docs
python -m pytest tests -q                                  # run tests
```

## Example

```bash
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "transaction_id":"T-1","customer_id":"C00012","timestamp":"2024-06-03T11:15:00",
  "amount":48200.0,"channel":"BRANCH_CASH","merchant_category":"cash",
  "country":"IN","device_id":"D0012","home_country":"IN"}'
```

```json
{
  "risk_score": 0.6, "risk_band": "HIGH", "decision": "BLOCK",
  "rule_hits": [
    {"code": "R02_STRUCTURING",
     "description": "cash transaction just below the Rs 50,000 PAN threshold"},
    {"code": "R03_REPEAT_CASH_PATTERN",
     "description": "2 similar sub-PAN-threshold cash deposits in 7 days"}
  ],
  "top_signals": [
    {"feature": "amount_ratio_to_cust_median", "value": 120.13, "contribution": 0.2858}
  ],
  "latency_ms": 19.4
}
```

New to this? Read `START_HERE.md` first — plain-English explanation, a 10-day
study plan and resume wording. See `DOCUMENTATION.md` for the full design rationale, feature dictionary,
evaluation methodology, API reference and interview talking points.
