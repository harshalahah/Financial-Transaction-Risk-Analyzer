"""Request / response contracts for the risk API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Transaction(BaseModel):
    transaction_id: str = Field(..., examples=["T0000001"])
    customer_id: str = Field(..., examples=["C00001"])
    timestamp: datetime
    amount: float = Field(..., gt=0, le=10_000_000)
    channel: Literal["UPI", "CARD_POS", "CARD_ONLINE", "IMPS", "NEFT", "ATM", "BRANCH_CASH"]
    merchant_category: str
    currency: str = "INR"
    country: str = Field(..., min_length=2, max_length=2)
    device_id: str
    is_new_beneficiary: int = 0
    home_country: Optional[str] = None
    account_type: str = "savings"
    age: Optional[int] = None
    tenure_months: Optional[int] = None

    @field_validator("country", "home_country")
    @classmethod
    def upper(cls, v):
        return v.upper() if v else v


class RuleHit(BaseModel):
    code: str
    description: str
    weight: float


class ScoreResponse(BaseModel):
    transaction_id: str
    customer_id: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["ALLOW", "REVIEW", "BLOCK"]
    model_version: str
    threshold: float
    rule_hits: list[RuleHit]
    top_signals: list[dict]
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    flagged: int
    count: int
