# customer_signal/views.py
"""
Phase 5 — Views + APIs for all 5 screens per route map in the spec.
"""
import io
from pathlib import Path

import json
from django.core.paginator import Paginator
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.db.models import Q, Sum, Count
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from django.utils.text import get_valid_filename
from django.views.decorators.http import require_POST

from .models import Customer, Transcript, SignalAnalysis, UploadBatch
from .signal_logic import ingest_csv, ingest_transcript_file, compute_signal_flags, risk_band
from .ml_pipeline import run_pipeline
from .llm_layer import get_or_generate_analysis


def _save_upload_copy(uploaded_file, folder_name, batch_subfolder=None):
    """Persist a copy of an uploaded file under MEDIA_ROOT for traceability.

    If `batch_subfolder` is provided, files are stored under
    `uploads/<folder_name>/<batch_subfolder>/<original_name>` so a single
    upload batch creates its own subfolder.
    Returns the saved path (relative to MEDIA_ROOT) and raw bytes.
    """
    raw_content = uploaded_file.read()
    original_name = get_valid_filename(Path(uploaded_file.name).name)
    stamp = timezone.now().strftime("%Y%m%d_%H%M%S_%f")
    if batch_subfolder:
        storage_dir = f"uploads/{folder_name}/{batch_subfolder}"
        storage_path = f"{storage_dir}/{original_name}"
    else:
        storage_path = f"uploads/{folder_name}/{stamp}_{original_name}"
    saved_path = default_storage.save(storage_path, ContentFile(raw_content))
    return saved_path, raw_content


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
    upload_history_qs = UploadBatch.objects.all().order_by('-uploaded_at')
    from django.core.paginator import Paginator
    page = request.GET.get('batch_page', 1)
    paginator = Paginator(upload_history_qs, 10)
    upload_history = paginator.get_page(page)

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
    # create a per-batch subfolder
    batch_sub = f"batch_{batch.id}_{timezone.now().strftime('%Y%m%d_%H%M%S') }"
    try:
        saved_path, raw_content = _save_upload_copy(file, "csv", batch_subfolder=batch_sub)
        # record storage folder on the batch (relative to MEDIA_ROOT)
        storage_folder = str(Path(saved_path).parent).replace('\\', '/')
        batch.storage_folder = storage_folder
        batch.save()

        result = ingest_csv(io.BytesIO(raw_content), batch)

        # store last upload folder in session for UI convenience
        request.session['last_upload_folder'] = storage_folder

        # optionally auto-run processing if client requested
        auto_process = request.POST.get('auto_process') or request.GET.get('auto_process')
        pipeline_result = None
        if auto_process:
            pipeline_result = run_pipeline()

        resp = {"status": "ok", "saved_path": saved_path, **result}
        if pipeline_result is not None:
            resp['pipeline'] = pipeline_result
        return JsonResponse(resp)
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
    batch_sub = f"batch_{batch.id}_{timezone.now().strftime('%Y%m%d_%H%M%S') }"
    linked, pending, failed = 0, 0, 0
    for f in files:
        try:
            saved_path, raw_content = _save_upload_copy(f, "transcripts", batch_subfolder=batch_sub)
            raw_text = raw_content.decode("utf-8", errors="replace")
            status = ingest_transcript_file(f.name, raw_text)
            if status == "linked":
                linked += 1
            else:
                pending += 1
        except Exception:
            failed += 1

    storage_folder = f"uploads/transcripts/{batch_sub}"
    batch.storage_folder = storage_folder
    batch.rows_added = linked
    batch.rows_updated = pending
    batch.rows_skipped_duplicate = failed
    batch.save()

    # session storage for UI convenience
    request.session['last_upload_folder'] = storage_folder

    # optional auto-process
    auto_process = request.POST.get('auto_process') or request.GET.get('auto_process')
    pipeline_result = None
    if auto_process:
        pipeline_result = run_pipeline()

    resp = {"status": "ok", "linked": linked, "pending": pending, "failed": failed, "storage_folder": storage_folder}
    if pipeline_result is not None:
        resp['pipeline'] = pipeline_result
    return JsonResponse(resp)


@require_POST
def api_upload_all(request):
    """Accept an optional CSV (`file`) and zero-or-more transcript files (`files`) in one request.
    Saves files under per-batch subfolders and ingests accordingly. Supports `auto_process`.
    """
    csv_file = request.FILES.get("file")
    txt_files = request.FILES.getlist("files")
    if not csv_file and not txt_files:
        return JsonResponse({"error": "No files provided"}, status=400)

    batch = UploadBatch.objects.create(filename=(csv_file.name if csv_file else f"{len(txt_files)} transcript file(s)"), file_type="mixed")
    batch_sub = f"batch_{batch.id}_{timezone.now().strftime('%Y%m%d_%H%M%S') }"

    csv_result = None
    linked, pending, failed = 0, 0, 0

    try:
        print(f"\n--- [UPLOAD START] Received request to upload {batch.filename} ---")
        # handle CSV if present
        if csv_file:
            # save under uploads/csv/<batch_sub>/
            saved_path, raw_content = _save_upload_copy(csv_file, "csv", batch_subfolder=batch_sub)
            print(f"[Upload] Saved CSV to {saved_path}. Ingesting...")
            csv_result = ingest_csv(io.BytesIO(raw_content), batch)
            print(f"[Upload] CSV result: {csv_result}")

        # handle transcripts if present
        if txt_files:
            print(f"[Upload] Processing {len(txt_files)} transcript files...")
        for f in txt_files:
            try:
                saved_path, raw_content = _save_upload_copy(f, "transcripts", batch_subfolder=batch_sub)
                raw_text = raw_content.decode("utf-8", errors="replace")
                status = ingest_transcript_file(f.name, raw_text)
                if status == "linked":
                    linked += 1
                else:
                    pending += 1
                print(f"  -> Transcript {f.name}: {status}")
            except Exception as e:
                failed += 1
                print(f"  -> Transcript {f.name}: Failed ({e})")

        # record a top-level storage folder for UI (points to a mixed batch root)
        storage_folder = f"uploads/mixed/{batch_sub}"
        batch.storage_folder = storage_folder
        # if csv_result present, copy counts into batch
        if csv_result:
            batch.rows_added = csv_result.get('added', 0)
            batch.rows_updated = csv_result.get('updated', 0)
            batch.rows_skipped_duplicate = csv_result.get('skipped', 0)
        # accumulate transcript counts into rows_added/updated/skipped as well
        batch.rows_added = (batch.rows_added or 0) + linked
        batch.rows_updated = (batch.rows_updated or 0) + pending
        batch.rows_skipped_duplicate = (batch.rows_skipped_duplicate or 0) + failed
        batch.storage_folder = storage_folder
        batch.save()

        request.session['last_upload_folder'] = storage_folder

        # optional auto-process
        auto_process = request.POST.get('auto_process') or request.GET.get('auto_process')
        pipeline_result = None
        if auto_process:
            print("[Upload] Auto-process flag detected. Running ML pipeline...")
            pipeline_result = run_pipeline()
            print("[Upload] Pipeline result:", pipeline_result)

        resp = {"status": "ok", "csv": csv_result, "linked": linked, "pending": pending, "failed": failed, "storage_folder": storage_folder}
        print("--- [UPLOAD END] Successfully processed request. ---\n")
        if pipeline_result is not None:
            resp['pipeline'] = pipeline_result
        return JsonResponse(resp)
    except Exception as e:
        print(f"--- [UPLOAD ERROR] Failed: {e} ---")
        return JsonResponse({"status": "error", "error": str(e)}, status=400)


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
    print("\n--- [PROCESS START] Running ML pipeline manually ---")
    result = run_pipeline()
    print("[PROCESS END] Result:", result, "\n")
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


def coming_soon(request):
    feature = request.GET.get("feature", "This feature")
    # Pretty names and descriptions
    feature_meta = {
        "AskAssistant": {
            "title": "AskAssistant Chatbot",
            "desc": "An AI-powered assistant trained on your workspace to answer questions, analyze risk signals, and draft customer response playbooks on the fly.",
            "icon": "🤖"
        },
        "ConnectTeam": {
            "title": "Connect Team",
            "desc": "Deep Slack & Discord integration to alert account executives, triage customer issues instantly, and trigger workflows on custom events.",
            "icon": "💬"
        }
    }
    meta = feature_meta.get(feature, {
        "title": feature,
        "desc": "We are currently building this feature to help streamline your operations workflow. Stay tuned!",
        "icon": "⚡"
    })
    return render(request, "customer_signal/coming_soon.html", {"meta": meta})
