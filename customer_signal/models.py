# customer_signal/models.py
from django.db import models


class Customer(models.Model):
    customer_id = models.CharField(max_length=20, unique=True, db_index=True)

    # Demographics
    gender = models.CharField(max_length=10, blank=True)
    senior_citizen = models.IntegerField(default=0)
    age = models.IntegerField(null=True, blank=True)
    married = models.CharField(max_length=5, blank=True)
    dependents = models.CharField(max_length=5, blank=True)
    partner = models.CharField(max_length=5, blank=True)

    # Account
    tenure_months = models.IntegerField(null=True, blank=True)
    contract_type = models.CharField(max_length=30, blank=True)
    payment_method = models.CharField(max_length=40, blank=True)
    paperless_billing = models.CharField(max_length=5, blank=True)
    monthly_charges = models.FloatField(null=True, blank=True)
    total_charges = models.FloatField(null=True, blank=True)

    # Services
    phone_service = models.CharField(max_length=5, blank=True)
    internet_service = models.CharField(max_length=20, blank=True)
    service_count = models.IntegerField(null=True, blank=True)
    unlimited_data = models.CharField(max_length=5, blank=True)
    streaming_tv = models.CharField(max_length=5, blank=True)
    streaming_movies = models.CharField(max_length=5, blank=True)
    streaming_music = models.CharField(max_length=5, blank=True)

    # Support / satisfaction (high-signal fields)
    satisfaction_level = models.CharField(max_length=10, blank=True)  # Low/Medium/High
    open_issue_count = models.IntegerField(default=0)
    close_issue_count = models.IntegerField(default=0)
    support_interaction_count = models.IntegerField(default=0)
    resolution_status_open_closed = models.CharField(max_length=15, blank=True)

    # Business / status (ground truth, validation only)
    customer_status = models.CharField(max_length=15, blank=True)  # Stayed/Churned
    churn = models.CharField(max_length=5, blank=True)
    churn_score_target = models.FloatField(null=True, blank=True)
    cltv = models.FloatField(null=True, blank=True)

    # Model output
    predicted_churn_score = models.FloatField(null=True, blank=True)
    risk_band = models.CharField(max_length=12, blank=True)  # High / Attention / Low

    raw_json = models.JSONField(default=dict, blank=True)

    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-predicted_churn_score"]

    def __str__(self):
        return self.customer_id


class Transcript(models.Model):
    customer = models.OneToOneField(
        Customer, on_delete=models.CASCADE, related_name="transcript", null=True, blank=True
    )
    # Kept in case a transcript arrives before its matching Customer row
    pending_customer_id = models.CharField(max_length=20, blank=True, db_index=True)

    topic = models.CharField(max_length=255, blank=True)
    raw_text = models.TextField()
    turns_json = models.JSONField(default=list, blank=True)
    feedback_text = models.TextField(blank=True)
    stated_sentiment = models.CharField(max_length=10, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Transcript({self.customer_id or self.pending_customer_id})"


class SignalAnalysis(models.Model):
    """Cached LLM output — generated once per customer, reused after that."""
    customer = models.OneToOneField(Customer, on_delete=models.CASCADE, related_name="analysis")
    signals = models.JSONField(default=list, blank=True)
    rationale = models.TextField(blank=True)
    evidence = models.JSONField(default=list, blank=True)
    llm_sentiment = models.CharField(max_length=10, blank=True)
    suggested_action = models.TextField(blank=True)
    generated_at = models.DateTimeField(auto_now_add=True)
    model_used = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return f"Analysis({self.customer_id})"


class UploadBatch(models.Model):
    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=10)  # csv / txt
    rows_added = models.IntegerField(default=0)
    rows_updated = models.IntegerField(default=0)
    rows_skipped_duplicate = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.filename} ({self.file_type})"
