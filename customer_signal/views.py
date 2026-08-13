from django.shortcuts import render


def dashboard(request):
    context = {
        "total_customers": 200,
        "high_risk": 27,
        "open_issues": 18,
        "negative_signals": 43,
    }

    return render(
        request,
        "customer_signal/dashboard.html",
        context,
    )
