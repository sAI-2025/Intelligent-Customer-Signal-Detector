# customer_signal/ml_pipeline.py
"""
Phase 3 — ML pipeline wrapper around your trained decisiontree_churn.pkl.
Runs synchronously, no LLM calls. Triggered only by the "Process Customers"
button / /api/process/ endpoint.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from django.conf import settings
from django.utils import timezone

from .models import Customer
from .signal_logic import risk_band


MODEL_PATH = getattr(
    settings,
    "CHURN_MODEL_PATH",
    Path(settings.BASE_DIR) / "model" / "decisiontree_churn.pkl"
)

_model_cache = {}


def _load_model():
    if "bundle" not in _model_cache:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        _model_cache["bundle"] = bundle
    return _model_cache["bundle"]


def preprocess(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Placeholder that mirrors the shape of your real preprocess() function.
    Replace this body with your actual training-time preprocessing logic —
    it MUST be identical to what was used when decisiontree_churn.pkl was
    trained, or predictions will be meaningless. Left here as a stub so the
    pipeline is wired end-to-end; swap in your real implementation.
    """
    out = df.copy()

    # Example categorical -> numeric mappings consistent with the spec
    satisfaction_map = {"Low": 1, "Medium": 2, "High": 3}
    if "satisfaction_level" in out.columns:
        out["satisfaction_level"] = out["satisfaction_level"].map(satisfaction_map).fillna(0)

    bool_like_cols = [
        "senior_citizen", "partner", "dependents", "phone_service",
        "paperless_billing", "unlimited_data", "streaming_tv",
        "streaming_movies", "streaming_music", "married",
    ]
    for col in bool_like_cols:
        if col in out.columns:
            out[col] = out[col].map({"Yes": 1, "No": 0, "Y": 1, "N": 0, 1: 1, 0: 0}).fillna(0)

    one_hot_cols = [c for c in ["contract_type", "payment_method", "internet_service"] if c in out.columns]
    if one_hot_cols:
        out = pd.get_dummies(out, columns=one_hot_cols, prefix=["is_" + c for c in one_hot_cols])

    return out


def run_pipeline(queryset=None) -> dict:
    """
    Scores every Customer with a null or stale predicted_churn_score.
    Returns a summary dict: {processed, high_risk, attention, low}
    """
    bundle = _load_model()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    qs = queryset if queryset is not None else Customer.objects.filter(predicted_churn_score__isnull=True)
    customers = list(qs)
    if not customers:
        return {"processed": 0, "high_risk": 0, "attention": 0, "low": 0}

    rows = []
    for c in customers:
        row = {**c.raw_json}
        row["customer_id"] = c.customer_id
        rows.append(row)

    df = pd.DataFrame(rows)
    processed_df = preprocess(df, is_training=False)
    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)

    preds = model.predict(processed_df)
    preds = np.clip(preds, 0, 100)

    counts = {"High": 0, "Attention": 0, "Low": 0}
    now = timezone.now()
    for customer, score in zip(customers, preds):
        band = risk_band(float(score))
        customer.predicted_churn_score = float(score)
        customer.risk_band = band
        customer.processed_at = now
        counts[band if band in counts else "Low"] += 1

    Customer.objects.bulk_update(
        customers, ["predicted_churn_score", "risk_band", "processed_at"]
    )

    return {
        "processed": len(customers),
        "high_risk": counts["High"],
        "attention": counts["Attention"],
        "low": counts["Low"],
    }
