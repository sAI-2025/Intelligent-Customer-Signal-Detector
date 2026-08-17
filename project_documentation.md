# Intelligent Customer Signal Detector — Documentation

## Project Purpose
The **Intelligent Customer Signal Detector** is a full-stack Django application designed to ingest customer operational data and call transcripts, run a pre-trained regression model to calculate risk (churn) scores, flag critical operational warning signs (e.g., unresolved tickets, month-to-month contract risks, negative transcript sentiment), and utilize LLMs (via Groq API) to generate easily readable operational rationales and recommended next actions for customer support teams.

---

## Base File Structure (Depth 2)

```text
Intelligent-Customer-Signal-Detector/
├── Dockerfile                  # Multi-stage production runtime container configuration
├── Setting/                    # Django project core settings and routing configuration
│   ├── asgi.py
│   ├── settings.py             # Loads environments from .env and configures Django
│   ├── urls.py                 # Maps top-level routes to apps
│   └── wsgi.py
├── customer_signal/            # Django app for core business logic, ML, and LLMs
│   ├── admin.py                # Registers database models for Django Admin panel
│   ├── apps.py                 # App configuration registry
│   ├── llm_layer.py            # LangChain + Groq API prompt generation & caching
│   ├── migrations/             # Database migration schemas
│   ├── ml_pipeline.py          # Machine learning preprocessing and model inference
│   ├── models.py               # Database schemas (Customer, Transcript, Analysis, etc.)
│   ├── signal_logic.py         # CSV parser and rule-based concern flag evaluator
│   ├── urls.py                 # App-level routing mappings
│   └── views.py                # Page-level endpoints, bulk file uploads, & reset actions
├── datasets/                   # Sample datasets for customer uploads and test runs
│   ├── Telco_customer_dashboard.csv
│   ├── Telco_customer_dashboard1.csv  # 50% random split dataset
│   └── Telco_customer_dashboard2.csv  # 50% random split dataset
├── media/                      # Uploaded files and transcripts registry (for traceability)
│   └── uploads/
├── static/                     # Global static assets
│   ├── css/                    # Stylesheets (dashboard.css design system)
│   └── js/                     # Interactivity scripts (dashboard.js, data_upload.js)
├── templates/                  # Page-level HTML template layouts
│   └── customer_signal/        # Page templates (overview, details, uploads, coming-soon)
├── manage.py                   # Django CLI administrative entrypoint
├── requirements.txt            # Virtual environment python package requirements
└── start.py                    # Container startup wrapper for database prep and Gunicorn
```

---

## Detailed Feature Index

### 1. Overview Dashboard (`/`)
* **Purpose**: Serves as the primary operational command center showing a snapshot of risk across the customer base.
* **Functionality**:
  * Displays high-level analytics cards for **Customers Analyzed**, **High Risk** volume, **Open Issues**, and **Negative Sentiments**.
  * Each card is clickable, applying pre-filtered values (e.g. `risk=High`, `sentiment=Negative`) when navigating to the Customer Signals panel.
  * Shows a visual bar graph of the **Risk Distribution** (High vs Attention vs Low).
  * Lists the **Top 10 Prioritized Customer Signals** with the highest predicted risk scores.

### 2. Customer Signals Panel (`/customers/`)
* **Purpose**: Allows analysts to search, filter, and drill down on custom segments of the customer base.
* **Functionality**:
  * Real-time text search matches target Customer IDs.
  * Dropdown filtering options isolate records by **Risk Band** (High/Attention/Low), **Sentiment** (Positive/Neutral/Negative), **Resolution Status** (Resolved/Not Resolved), and **Contract Type**.
  * Generates clean table rows displaying predicted risk scores and color-coded status badges.
  * Complete pagination (25 records per page).

### 3. Customer Detail Profile (`/customer/<customer_id>/`)
* **Purpose**: Delivers a 360-degree diagnostic view of a single flagged customer.
* **Functionality**:
  * **Core Metrics**: Displays age, tenure, monthly charges, contract type, and satisfaction level.
  * **Raw Properties**: Expands to show all imported columns, automatically filtering out empty, invalid (`"—"`, `None`, `nan`) fields.
  * **System Flags**: Calculates rule-based flags (e.g. `Open Issues Concern`, `Month-to-Month Contract Warning`) and rates their threat levels.
  * **AI Rationale**: Integrates with Groq (`llama-3.1-8b-instant` or customized) using a structured LangChain prompt to output a plain-English, jargon-free summary explaining the score and suggesting exactly one action. If Groq is offline, it falls back to a template ruleset.
  * **Support Transcript**: Formats and prints color-coded chat bubbles showing the dialogue history between the customer and agent.

### 4. Model Insights Dashboard (`/model-insights/`)
* **Purpose**: Provides auditability and explainability of the machine learning engine.
* **Functionality**:
  * Names the active algorithm (Decision Tree Regressor) and target features.
  * Renders a custom horizontal bar chart showing calculated **Feature Importances** (e.g. how heavily Customer Satisfaction, month-to-month contracts, or fiber optic usage weight risk calculations).

### 5. Data Ingestion & Uploads (`/data/`)
* **Purpose**: Handles database management, bulk data loading, and manual simulations.
* **Functionality**:
  * **Bulk CSV/TXT**: Allows dragging-and-dropping customer CSV databases and folder transcripts. Files are structured under media folders per batch for traceability.
  * **Auto-Process**: Checkbox triggers the machine learning pipeline to run inference immediately after ingest.
  * **Manual Sandbox**: Form inputs let administrators add individual customer profiles and copy/paste support conversations to immediately test rule configurations and Groq APIs.
  * **Database Control**: "Danger Zone" features allow purging customer data, wiping transcripts, or performing a factory reset.

### 6. Channels transition view (`/coming-soon/`)
* **Purpose**: Holds spaces for upcoming application expansions without presenting broken links.
* **Functionality**:
  * Keeps `AskAssistant` and `Connect Team` links in the sidebar.
  * Renders a premium, animated transition page within the layout that details what the feature will do once released.
