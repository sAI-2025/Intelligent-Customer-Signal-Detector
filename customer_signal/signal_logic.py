# customer_signal/signal_logic.py
"""
Phase 2 — Ingestion (CSV + transcript parsing) and Phase 2b — deterministic
rule engine (risk banding + signal flags). No ML, no LLM calls here.
"""
import re
import pandas as pd
from django.utils import timezone

from .models import Customer, Transcript, UploadBatch

# ------------------------------------------------------------------
# CSV -> Customer field mapping
# ------------------------------------------------------------------

_FIELD_MAP = {
    "customer_id": "customer_id",
    "gender": "gender",
    "senior_citizen": "senior_citizen",
    "age": "age",
    "married": "married",
    "dependents": "dependents",
    "partner": "partner",
    "tenure_months": "tenure_months",
    "contract_type": "contract_type",
    "payment_method": "payment_method",
    "paperless_billing": "paperless_billing",
    "monthly_charges": "monthly_charges",
    "total_charges": "total_charges",
    "phone_service": "phone_service",
    "internet_service": "internet_service",
    "service_count": "service_count",
    "unlimited_data": "unlimited_data",
    "streaming_tv": "streaming_tv",
    "streaming_movies": "streaming_movies",
    "streaming_music": "streaming_music",
    "satisfaction_level": "satisfaction_level",
    "open_issue_count": "open_issue_count",
    "close_issue_count": "close_issue_count",
    "support_interaction_count": "support_interaction_count",
    "resolution_status_open_closed": "resolution_status_open_closed",
    "customer_status": "customer_status",
    "churn": "churn",
    "churn_score_target": "churn_score_target",
    "cltv": "cltv",
}

_INT_FIELDS = {"senior_citizen", "age", "tenure_months", "service_count",
               "open_issue_count", "close_issue_count", "support_interaction_count"}
_FLOAT_FIELDS = {"monthly_charges", "total_charges", "churn_score_target", "cltv"}


def _coerce(field, value):
    if pd.isna(value):
        return None if field in (_INT_FIELDS | _FLOAT_FIELDS) else ""
    if field in _INT_FIELDS:
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    if field in _FLOAT_FIELDS:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    return str(value).strip()


def extract_mapped_fields(row: pd.Series) -> dict:
    defaults = {}
    for csv_col, model_field in _FIELD_MAP.items():
        if csv_col in row.index:
            defaults[model_field] = _coerce(model_field, row[csv_col])
    return defaults


def ingest_csv(file, batch: UploadBatch) -> dict:
    """
    Parses an uploaded CSV, upserts Customer rows keyed on customer_id.
    Bad/missing-id rows are skipped, never fatal to the batch.
    """
    df = pd.read_csv(file)
    added, updated, skipped = 0, 0, 0

    for _, row in df.iterrows():
        try:
            cid = str(row.get("customer_id", "")).strip()
            if not cid or cid.lower() == "nan":
                skipped += 1
                continue

            defaults = extract_mapped_fields(row)
            defaults.pop("customer_id", None)
            defaults["raw_json"] = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}

            obj, created = Customer.objects.update_or_create(
                customer_id=cid, defaults=defaults
            )
            if created:
                added += 1
                # link any transcript that arrived before this customer row
                Transcript.objects.filter(
                    pending_customer_id=cid, customer__isnull=True
                ).update(customer=obj, pending_customer_id="")
            else:
                updated += 1
        except Exception:
            skipped += 1
            continue

    batch.rows_added = added
    batch.rows_updated = updated
    batch.rows_skipped_duplicate = skipped
    batch.save()
    return {"added": added, "updated": updated, "skipped": skipped}


# ------------------------------------------------------------------
# Transcript parsing (matches confirmed .txt format exactly)
# ------------------------------------------------------------------

SEP = "-" * 40  # matches "-" * 40 used when the .txt files were generated


def parse_transcript(raw_text: str) -> dict:
    parts = raw_text.split(SEP)
    # Fallback split in case the separator length differs slightly
    if len(parts) < 3:
        parts = re.split(r"-{10,}", raw_text)

    header = parts[0] if len(parts) > 0 else ""
    body = parts[1] if len(parts) > 1 else ""
    footer = parts[2] if len(parts) > 2 else ""

    customer_id_m = re.search(r"Customer ID:\s*(.+)", header)
    topic_m = re.search(r"Topic:\s*(.+)", header)

    turns = []
    for line in body.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(Customer|Agent)\s*:?\s*(.*)", line)
        if m:
            turns.append({"speaker": m.group(1), "text": m.group(2).strip()})

    feedback_m = re.search(r"Feedback:\s*(.+)", footer)
    sentiment_m = re.search(r"Sentiment:\s*(\w+)", footer)

    return {
        "customer_id": customer_id_m.group(1).strip() if customer_id_m else None,
        "topic": topic_m.group(1).strip() if topic_m else "",
        "turns_json": turns,
        "feedback_text": feedback_m.group(1).strip() if feedback_m else "",
        "stated_sentiment": sentiment_m.group(1).strip() if sentiment_m else "",
        "raw_text": raw_text,
    }


def ingest_transcript_file(filename: str, raw_text: str) -> str:
    """
    Ingests one .txt file. Filename (minus extension) is the fallback
    customer_id if the header doesn't parse one out. Returns 'linked' or
    'pending' depending on whether a matching Customer row exists yet.
    """
    parsed = parse_transcript(raw_text)
    cid = parsed["customer_id"] or filename.rsplit(".", 1)[0].strip()

    customer = Customer.objects.filter(customer_id=cid).first()

    defaults = {
        "topic": parsed["topic"],
        "raw_text": parsed["raw_text"],
        "turns_json": parsed["turns_json"],
        "feedback_text": parsed["feedback_text"],
        "stated_sentiment": parsed["stated_sentiment"],
    }

    if customer:
        Transcript.objects.update_or_create(customer=customer, defaults=defaults)
        return "linked"
    else:
        defaults["pending_customer_id"] = cid
        Transcript.objects.update_or_create(
            pending_customer_id=cid, customer__isnull=True, defaults=defaults
        )
        return "pending"


# ------------------------------------------------------------------
# Rule engine — deterministic, documented, never invented by the LLM
# ------------------------------------------------------------------

def risk_band(score) -> str:
    if score is None:
        return "Unscored"
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Attention"
    else:
        return "Low"


def compute_signal_flags(customer: Customer, transcript: "Transcript | None") -> list:
    flags = []
    if customer.satisfaction_level == "Low":
        flags.append({"label": "Satisfaction", "value": "Low", "concern": "High"})
    if customer.open_issue_count and customer.open_issue_count > 0:
        flags.append({
            "label": "Open Support Issues",
            "value": str(customer.open_issue_count),
            "concern": "High",
        })
    if transcript and transcript.stated_sentiment == "Negative":
        flags.append({"label": "Sentiment", "value": "Negative", "concern": "High"})
    if customer.resolution_status_open_closed == "NotResolved":
        flags.append({"label": "Resolution Status", "value": "Not Resolved", "concern": "High"})
    if customer.contract_type == "Month-to-month":
        flags.append({"label": "Contract Type", "value": "Month-to-month", "concern": "Moderate"})
    if customer.tenure_months is not None and customer.tenure_months < 12:
        flags.append({
            "label": "Tenure",
            "value": f"{customer.tenure_months} months",
            "concern": "Moderate",
        })
    return flags
