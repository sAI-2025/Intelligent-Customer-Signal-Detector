# customer_signal/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.overview, name="overview"),
    path("customers/", views.customer_signals, name="customer_signals"),
    path("customer/<str:customer_id>/", views.customer_detail, name="customer_detail"),
    path("model-insights/", views.model_insights, name="model_insights"),
    path("data/", views.data_upload, name="data_upload"),
    path("coming-soon/", views.coming_soon, name="coming_soon"),

    path("api/upload/csv/", views.api_upload_csv, name="api_upload_csv"),
    path("api/upload/transcripts/", views.api_upload_transcripts, name="api_upload_transcripts"),
    path("api/upload/all/", views.api_upload_all, name="api_upload_all"),
    path("api/customer/manual/", views.api_manual_customer, name="api_manual_customer"),
    path("api/process/", views.api_process, name="api_process"),
    path("api/customer/<str:customer_id>/reanalyze/", views.reanalyze_customer, name="reanalyze_customer"),
    path("api/data/reset/", views.api_data_reset, name="api_data_reset"),
]
