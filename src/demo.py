"""End-to-end demo: start the app in-process and score three scenarios."""
import json, statistics, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
import api

SCENARIOS = {
    "1. normal kirana spend": [{
        "transaction_id": "T-D1", "customer_id": "C00010",
        "timestamp": "2024-06-01T12:30:00", "amount": 420.00, "channel": "UPI",
        "merchant_category": "kirana", "country": "IN", "device_id": "D0010",
        "home_country": "IN", "account_type": "savings", "age": 34, "tenure_months": 75}],

    "2. card-not-present fraud burst": [{
        "transaction_id": f"T-D2{i}", "customer_id": "C00011",
        "timestamp": f"2024-06-01T22:{10 + i * 4:02d}:00",
        "amount": 12.0 if i == 0 else 18000.0 + i * 9500,
        "channel": "CARD_ONLINE", "merchant_category": "electronics", "country": "NG",
        "device_id": "D9911", "is_new_beneficiary": 1, "home_country": "IN",
        "account_type": "savings", "age": 29, "tenure_months": 18} for i in range(5)],

    "3. structuring / smurfing": [{
        "transaction_id": f"T-D3{i}", "customer_id": "C00012",
        "timestamp": f"2024-06-0{1 + i}T11:15:00", "amount": 47000.0 + i * 600,
        "channel": "BRANCH_CASH", "merchant_category": "cash", "country": "IN",
        "device_id": "D0012", "home_country": "IN", "account_type": "savings",
        "age": 51, "tenure_months": 120} for i in range(4)],
}

with TestClient(api.app) as c:
    print(json.dumps(c.get("/health").json(), indent=2), "\n")
    for name, txns in SCENARIOS.items():
        print("=" * 70, f"\n{name}")
        for t in txns:
            r = c.post("/score", json=t).json()
            print(f"  {r['transaction_id']:8s} Rs {t['amount']:>10,.2f}  "
                  f"score={r['risk_score']:.3f}  {r['risk_band']:<6} {r['decision']}")
            for h in r["rule_hits"]:
                print(f"      rule  {h['code']}: {h['description']}")
            for s in r["top_signals"][:3]:
                print(f"      signal {s['feature']}={s['value']:.2f} (+{s['contribution']})")

    # latency benchmark
    t = SCENARIOS["1. normal kirana spend"][0]
    lat = []
    for i in range(50):
        t2 = dict(t, transaction_id=f"T-L{i}")
        lat.append(c.post("/score", json=t2).json()["latency_ms"])
    print("=" * 70)
    print(f"scoring latency over 50 calls: p50={statistics.median(lat):.1f} ms "
          f"p95={sorted(lat)[int(.95 * len(lat))]:.1f} ms")
