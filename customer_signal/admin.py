# customer_signal/admin.py
from django.contrib import admin
from .models import Customer, Transcript, SignalAnalysis, UploadBatch


@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ("customer_id", "risk_band", "predicted_churn_score",
                     "satisfaction_level", "customer_status", "uploaded_at")
    search_fields = ("customer_id",)
    list_filter = ("risk_band", "customer_status", "contract_type")


@admin.register(Transcript)
class TranscriptAdmin(admin.ModelAdmin):
    list_display = ("customer", "pending_customer_id", "topic", "stated_sentiment", "uploaded_at")
    search_fields = ("customer__customer_id", "pending_customer_id")


@admin.register(SignalAnalysis)
class SignalAnalysisAdmin(admin.ModelAdmin):
    list_display = ("customer", "llm_sentiment", "model_used", "generated_at")


@admin.register(UploadBatch)
class UploadBatchAdmin(admin.ModelAdmin):
    list_display = ("filename", "file_type", "rows_added", "rows_updated",
                     "rows_skipped_duplicate", "uploaded_at")
