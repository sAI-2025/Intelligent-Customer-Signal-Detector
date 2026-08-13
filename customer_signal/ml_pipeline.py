# customer_signal/ml_pipeline.py
"""
Phase 3 — ML pipeline wrapper around your trained decisiontree_churn.pkl.
This preprocess() is now identical to your Colab Cell 1, so training and
inference use the exact same feature engineering.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from django.conf import settings
from django.utils import timezone

from .models import Customer
from .signal_logic import risk_band

MODEL_PATH = Path(settings.BASE_DIR) / "model" / "decisiontree_churn.pkl"

_model_cache = {}


def _load_model():
    if "bundle" not in _model_cache:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        _model_cache["bundle"] = bundle
    return _model_cache["bundle"]


SATISFACTION_MAP = {'Low': 1, 'Medium': 2, 'High': 3}

DROP_COLS = [
    'avg_monthly_long_distance_charges',
    'churn',
    'churn_category',
    'churn_rate',
    'churn_reason',
    'city',
    'close_issue_count',
    'cltv',
    'count',
    'country',
    'customer_id',
    'customer_status',
    'device_protection',
    'lat_long',
    'latitude',
    'longitude',
    'online_backup',
    'online_security',
    'open_issue_count',
    'premium_tech_support',
    'referred_a_friend',
    'resolution_status_open_closed',
    'service_count',
    'state',
    'support_interaction_count',
    'tech_support',
    'total_long_distance_charges',
    'under_30',
    'zip_code',
    'total_charges',
    'total_revenue',
]


def preprocess(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Exact copy of your Colab Cell 1 preprocess(), adapted for your reduced
    37-column CSV schema. This MUST match training exactly.
    """
    df = df.copy()

    # total_charges median-fill (kept for parity even if column absent)
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())

    # 1. gender -> ismale
    if 'gender' in df.columns:
        df['ismale'] = (df['gender'].astype(str).str.strip() == 'Male').astype(int)
        df.drop(columns=['gender'], inplace=True)

    # 2. partner -> ispartner
    if 'partner' in df.columns:
        df['ispartner'] = (df['partner'] == 'Yes').astype(int)
        df.drop(columns=['partner'], inplace=True)

    # 3. dependents -> isdependents
    if 'dependents' in df.columns:
        df['isdependents'] = (df['dependents'] == 'Yes').astype(int)
        df.drop(columns=['dependents'], inplace=True)

    # 4. phone_service -> isphone_service
    if 'phone_service' in df.columns:
        df['isphone_service'] = (df['phone_service'] == 'Yes').astype(int)
        df.drop(columns=['phone_service'], inplace=True)

    # 5. multiple_lines -> ismultiple_lines
    if 'multiple_lines' in df.columns:
        df['multiple_lines'] = df['multiple_lines'].replace('No phone service', 'No')
        df['ismultiple_lines'] = (df['multiple_lines'] == 'Yes').astype(int)
        df.drop(columns=['multiple_lines'], inplace=True)

    # 6. internet_service -> isFiberOpticInternetService, DSLInternetService
    if 'internet_service' in df.columns:
        df['isFiberOpticInternetService'] = (df['internet_service'] == 'Fiber optic').astype(int)
        df['DSLInternetService'] = (df['internet_service'] == 'DSL').astype(int)
        df.drop(columns=['internet_service'], inplace=True)

    # 7. streaming_tv -> isstreaming_tv
    if 'streaming_tv' in df.columns:
        df['streaming_tv'] = df['streaming_tv'].replace('No internet service', 'No')
        df['isstreaming_tv'] = (df['streaming_tv'] == 'Yes').astype(int)
        df.drop(columns=['streaming_tv'], inplace=True)

    # 8. streaming_movies -> isstreaming_movies (engineered then dropped later)
    if 'streaming_movies' in df.columns:
        df['streaming_movies'] = df['streaming_movies'].replace('No internet service', 'No')
        df['isstreaming_movies'] = (df['streaming_movies'] == 'Yes').astype(int)
        df.drop(columns=['streaming_movies'], inplace=True)

    # 9. contract_type -> 3 one-hot columns
    if 'contract_type' in df.columns:
        df['iscontract_typeMonth-to-month'] = (df['contract_type'] == 'Month-to-month').astype(int)
        df['iscontract_typeTwo year'] = (df['contract_type'] == 'Two year').astype(int)
        df['iscontract_typeOne year'] = (df['contract_type'] == 'One year').astype(int)
        df.drop(columns=['contract_type'], inplace=True)

    # 10. paperless_billing -> ispaperless_billing
    if 'paperless_billing' in df.columns:
        df['ispaperless_billing'] = (df['paperless_billing'] == 'Yes').astype(int)
        df.drop(columns=['paperless_billing'], inplace=True)

    # 11. payment_method -> 4 one-hot columns
    if 'payment_method' in df.columns:
        df['ispayment_methodElectronic check'] = (df['payment_method'] == 'Electronic check').astype(int)
        df['ispayment_methodMailed check'] = (df['payment_method'] == 'Mailed check').astype(int)
        df['ispayment_methodBank transfer'] = (df['payment_method'] == 'Bank transfer (automatic)').astype(int)
        df['ispayment_methodCredit card'] = (df['payment_method'] == 'Credit card (automatic)').astype(int)
        df.drop(columns=['payment_method'], inplace=True)

    # 12. married -> ismarried
    if 'married' in df.columns:
        df['ismarried'] = (df['married'] == 'Yes').astype(int)
        df.drop(columns=['married'], inplace=True)

    # 13. offer -> isPromoOffer
    if 'offer' in df.columns:
        df['isPromoOffer'] = (df['offer'] != 'None').astype(int)
        df.drop(columns=['offer'], inplace=True)

    # 14. streaming_music -> isstreaming_music
    if 'streaming_music' in df.columns:
        df['isstreaming_music'] = (df['streaming_music'] == 'Yes').astype(int)
        df.drop(columns=['streaming_music'], inplace=True)

    # 15. unlimited_data -> isunlimited_data
    if 'unlimited_data' in df.columns:
        df['isunlimited_data'] = (df['unlimited_data'] == 'Yes').astype(int)
        df.drop(columns=['unlimited_data'], inplace=True)

    # 16. satisfaction_level -> label encode
    if 'satisfaction_level' in df.columns:
        df['satisfaction_level'] = df['satisfaction_level'].map(SATISFACTION_MAP)

    # Drop the fixed drop-list columns AND the two engineered-then-dropped
    # columns from the original script (isstreaming_movies, ispartner).
    cols_to_drop = DROP_COLS + ['isstreaming_movies', 'ispartner']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # Guard: at inference time the target usually won't be present.
    TARGET_COL = 'churn_score_1'
    if TARGET_COL in df.columns and not is_training:
        df.drop(columns=[TARGET_COL], inplace=True)

    return df


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
