# customer_signal/views.py
"""
Phase 5 — Views + APIs for all 5 screens per route map in the spec.
"""
import json
from django.core.paginator import Paginator
from django.db.models import Q, Sum, Count
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from django.views.decorators.http import require_POST

from .models import Customer, Transcript, SignalAnalysis, UploadBatch
from .signal_logic import ingest_csv, ingest_transcript_file, compute_signal_flags, risk_band
from .ml_pipeline import run_pipeline
from .llm_layer import get_or_generate_analysis


# ------------------------------------------------------------------
# SCREEN 1 — Overview
# ------------------------------------------------------------------

def overview(request):
    analyzed_qs = Customer.objects.filter(predicted_churn_score__isnull=False)
    analyzed_count = analyzed_qs.count()

    if analyzed_count == 0:
        return render(request, "customer_signal/overview.html", {"empty": True})

    high_risk = analyzed_qs.filter(risk_band="High").count()
    attention = analyzed_qs.filter(risk_band="Attention").count()
    low = analyzed_qs.filter(risk_band="Low").count()
    open_issues = analyzed_qs.aggregate(total=Sum("open_issue_count"))["total"] or 0
    negative_signals = Transcript.objects.filter(stated_sentiment="Negative").count()

    last_processed = analyzed_qs.order_by("-processed_at").first()
    top10 = analyzed_qs.order_by("-predicted_churn_score")[:10]

    top10_rows = []
    for rank, c in enumerate(top10, start=1):
        transcript = getattr(c, "transcript", None)
        top10_rows.append({
            "rank": rank,
            "customer": c,
            "sentiment": transcript.stated_sentiment if transcript else "—",
        })

    context = {
        "empty": False,
        "analyzed_count": analyzed_count,
        "high_risk": high_risk,
        "attention": attention,
        "low": low,
        "open_issues": open_issues,
        "negative_signals": negative_signals,
        "last_processed": last_processed.processed_at if last_processed else None,
        "top10_rows": top10_rows,
        "risk_max": max(high_risk, attention, low, 1),
    }
    return render(request, "customer_signal/overview.html", context)


# ------------------------------------------------------------------
# SCREEN 2 — Customer Signals (list + filters)
# ------------------------------------------------------------------

def customer_signals(request):
    qs = Customer.objects.filter(predicted_churn_score__isnull=False)

    search = request.GET.get("search", "").strip()
    risk = request.GET.get("risk", "All")
    sentiment = request.GET.get("sentiment", "All")
    resolution = request.GET.get("resolution", "All")
    contract = request.GET.get("contract", "All")
    sort = request.GET.get("sort", "risk_desc")

    active_filters = 0

    if search:
        qs = qs.filter(customer_id__icontains=search)
        active_filters += 1
    if risk != "All":
        qs = qs.filter(risk_band=risk)
        active_filters += 1
    if sentiment != "All":
        qs = qs.filter(transcript__stated_sentiment=sentiment)
        active_filters += 1
    if resolution != "All":
        qs = qs.filter(resolution_status_open_closed=resolution)
        active_filters += 1
    if contract != "All":
        qs = qs.filter(contract_type=contract)
        active_filters += 1

    sort_map = {
        "risk_desc": "-predicted_churn_score",
        "risk_asc": "predicted_churn_score",
        "tenure": "-tenure_months",
        "monthly_charges": "-monthly_charges",
    }
    qs = qs.order_by(sort_map.get(sort, "-predicted_churn_score"))

    total_count = Customer.objects.filter(predicted_churn_score__isnull=False).count()
    filtered_count = qs.count()

    paginator = Paginator(qs, 25)
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    context = {
        "page_obj": page_obj,
        "search": search,
        "risk": risk,
        "sentiment": sentiment,
        "resolution": resolution,
        "contract": contract,
        "sort": sort,
        "active_filters": active_filters,
        "total_count": total_count,
        "filtered_count": filtered_count,
    }
    return render(request, "customer_signal/customer_signals.html", context)


# ------------------------------------------------------------------
# Customer Detail (reached via row click, LLM triggered here)
# ------------------------------------------------------------------

def customer_detail(request, customer_id):
    customer = get_object_or_404(Customer, customer_id=customer_id)
    transcript = getattr(customer, "transcript", None)

    flags = compute_signal_flags(customer, transcript)
    analysis = get_or_generate_analysis(customer)

    raw_json_items = sorted(customer.raw_json.items()) if customer.raw_json else []

    context = {
        "customer": customer,
        "transcript": transcript,
        "flags": flags,
        "analysis": analysis,
        "raw_json_items": raw_json_items,
    }
    return render(request, "customer_signal/customer_detail.html", context)


@require_POST
def reanalyze_customer(request, customer_id):
    customer = get_object_or_404(Customer, customer_id=customer_id)
    get_or_generate_analysis(customer, force=True)
    return redirect("customer_detail", customer_id=customer_id)


# ------------------------------------------------------------------
# SCREEN 3 — Model Insights
# ------------------------------------------------------------------

def model_insights(request):
    # Feature importances — replace with your actual logged values/artifact
    feature_importances = [
        {"label": "Satisfaction Level", "value": 0.314},
        {"label": "Contract: Month-to-month", "value": 0.067},
        {"label": "Fiber Optic Internet", "value": 0.064},
        {"label": "Number of Referrals", "value": 0.041},
        {"label": "Contract: Two year", "value": 0.029},
        {"label": "Unlimited Data", "value": 0.027},
        {"label": "Payment: Electronic Check", "value": 0.027},
        {"label": "Paperless Billing", "value": 0.026},
    ]
    max_importance = max(f["value"] for f in feature_importances)

    context = {
        "model_name": "Decision Tree Regressor",
        "target_col": "churn_score_1 (0-100 continuous)",
        "feature_importances": feature_importances,
        "max_importance": max_importance,
    }
    return render(request, "customer_signal/model_insights.html", context)


# ------------------------------------------------------------------
# SCREEN 4 — Data / Upload
# ------------------------------------------------------------------

def data_upload(request):
    customer_count = Customer.objects.count()
    transcript_count = Transcript.objects.filter(customer__isnull=False).count()
    last_batch = UploadBatch.objects.first()
    upload_history = UploadBatch.objects.all()[:20]

    context = {
        "customer_count": customer_count,
        "transcript_count": transcript_count,
        "last_batch": last_batch,
        "upload_history": upload_history,
    }
    return render(request, "customer_signal/data_upload.html", context)


@require_POST
def api_upload_csv(request):
    file = request.FILES.get("file")
    if not file:
        return JsonResponse({"error": "No file provided"}, status=400)

    batch = UploadBatch.objects.create(filename=file.name, file_type="csv")
    try:
        result = ingest_csv(file, batch)
        return JsonResponse({"status": "ok", **result})
    except Exception as e:
        return JsonResponse({"status": "error", "error": str(e)}, status=400)


@require_POST
def api_upload_transcripts(request):
    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse({"error": "No files provided"}, status=400)

    batch = UploadBatch.objects.create(
        filename=f"{len(files)} transcript file(s)", file_type="txt"
    )
    linked, pending, failed = 0, 0, 0
    for f in files:
        try:
            raw_text = f.read().decode("utf-8", errors="replace")
            status = ingest_transcript_file(f.name, raw_text)
            if status == "linked":
                linked += 1
            else:
                pending += 1
        except Exception:
            failed += 1

    batch.rows_added = linked
    batch.rows_updated = pending
    batch.rows_skipped_duplicate = failed
    batch.save()

    return JsonResponse({"status": "ok", "linked": linked, "pending": pending, "failed": failed})


@require_POST
def api_manual_customer(request):
    data = request.POST
    cid = data.get("customer_id", "").strip()
    if not cid:
        return JsonResponse({"error": "customer_id is required"}, status=400)

    defaults = {
        "age": data.get("age") or None,
        "tenure_months": data.get("tenure_months") or None,
        "contract_type": data.get("contract_type", ""),
        "monthly_charges": data.get("monthly_charges") or None,
        "satisfaction_level": data.get("satisfaction_level", ""),
        "open_issue_count": data.get("open_issue_count") or 0,
        "resolution_status_open_closed": data.get("resolution_status_open_closed", ""),
        "customer_status": data.get("customer_status", ""),
    }
    customer, created = Customer.objects.update_or_create(customer_id=cid, defaults=defaults)

    transcript_text = data.get("transcript_text", "").strip()
    if transcript_text:
        from .signal_logic import parse_transcript
        parsed = parse_transcript(transcript_text)
        Transcript.objects.update_or_create(
            customer=customer,
            defaults={
                "topic": parsed["topic"] or data.get("topic", ""),
                "raw_text": transcript_text,
                "turns_json": parsed["turns_json"],
                "feedback_text": parsed["feedback_text"] or data.get("feedback_text", ""),
                "stated_sentiment": parsed["stated_sentiment"] or data.get("sentiment", ""),
            },
        )

    return JsonResponse({"status": "ok", "created": created, "customer_id": cid})


@require_POST
def api_process(request):
    result = run_pipeline()
    return JsonResponse({"status": "ok", **result})


@require_POST
def api_data_reset(request):
    target = request.POST.get("target")
    if target == "customers":
        Customer.objects.all().delete()
    elif target == "transcripts":
        Transcript.objects.all().delete()
    elif target == "all":
        Customer.objects.all().delete()
        Transcript.objects.all().delete()
        SignalAnalysis.objects.all().delete()
        UploadBatch.objects.all().delete()
    else:
        return JsonResponse({"error": "invalid target"}, status=400)
    return JsonResponse({"status": "ok", "target": target})
