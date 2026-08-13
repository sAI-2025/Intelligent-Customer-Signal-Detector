# Intelligent Customer Signal Detector — Complete Build Specification

**Stack:** Django (backend + templates) · SQLite · Regression model (your trained `decisiontree_churn.pkl` / XGBoost) · LangChain + Groq API (LLM reasoning) · HTML/CSS/JS (vanilla, no framework needed for a 3-day POC)

This document is written so you can build directly from it — every screen, every field, every function, every prompt template, every API route, every DB table.

---

## 0. What the brief actually asks for (grounding, so nothing drifts)

From the POC brief:

- **Core functionality:** accept customer data inputs (structured records, text transcripts, or both), analyze sentiment/behavioural patterns, and generate a **prioritised list of at-risk customers or flagged issues with reasoning**.
- **Output format:** a signal summary — flagged customers, a **risk/urgency score**, and a **brief AI-generated rationale** — presented clearly for a customer-ops/retention team.
- **Optional enhancements (do NOT prioritize over the core):** multi-signal correlation, visual risk heatmap, suggested retention action per flag.
- **Documentation:** one-page README — approach, tools, assumptions, one input/output example.
- **Evaluation weights** — this should directly drive where your hours go:

| Focus | Weight |
|---|---|
| Signal detection logic & AI reasoning quality; accuracy/relevance of flags | **40%** |
| Clarity & usefulness of output for operational decisions | **30%** |
| Communication clarity & presentation structure | 15% |
| Creativity, initiative, reflection in documentation | 15% |

**Reading of the weights:** 70% of your grade is "does the system correctly flag the right customers and explain why, in a way an ops person can act on." Visual polish matters but is a minority of the score — don't let animation/theming eat time that should go into signal quality and the rationale text.

---

## 1. Your actual data (grounded in the uploaded files, not assumptions)

### 1.1 CSV — `Telco_customer_final_sample.csv`
2000 rows × 57 columns. Full column list:

```
customer_id, gender, senior_citizen, partner, dependents, tenure_months,
phone_service, multiple_lines, internet_service, online_security, online_backup,
device_protection, tech_support, streaming_tv, streaming_movies, contract_type,
paperless_billing, payment_method, monthly_charges, total_charges, churn,
churn_rate, churn_score_target, cltv, churn_reason, count, country, state, city,
zip_code, lat_long, latitude, longitude, age, under_30, married,
referred_a_friend, number_of_referrals, offer, avg_monthly_long_distance_charges,
avg_monthly_gb_download, streaming_music, premium_tech_support, unlimited_data,
total_refunds, total_extra_data_charges, total_long_distance_charges,
total_revenue, customer_status, churn_score_1, churn_category,
satisfaction_level, service_count, open_issue_count, close_issue_count,
support_interaction_count, resolution_status_open_closed
```

Your training target is `churn_score_1` (per your preprocessing script). `customer_status` is `Stayed` / `Churned`. `satisfaction_level` is `Low` / `Medium` / `High` (mapped 1/2/3 in preprocessing). `resolution_status_open_closed` is `Resolved` / `NotResolved`.

### 1.2 Support transcripts — `.txt` files, one per customer
Confirmed format from your actual files (`0014-BMAQU.txt`, `0020-INWCK.txt`):

```
Customer ID: 0014-BMAQU
Topic: incoming calls not connecting
Customer Status: Stayed
----------------------------------------

Customer: <line>
Agent: <line>
Customer: <line>
Agent: <line>

----------------------------------------
Feedback: <one paragraph, customer's own words>
Sentiment: Positive | Negative | (Neutral presumably also occurs)
```

This is a clean, parseable format:
- Header block → `Customer ID`, `Topic`, `Customer Status`
- Body between the two `---` separators → alternating `Customer:` / `Agent:` turns
- Footer → `Feedback:` (free text) and `Sentiment:` (pre-labelled, so you don't even need to re-run sentiment classification unless you want the LLM to independently verify it)

**Filename convention:** `{customer_id}.txt` — this is your join key back to the CSV's `customer_id`.

### 1.3 Your trained model (from the inference script you pasted)
- `decisiontree_churn.pkl` → pickle containing `{model, feature_columns, target_col}`
- `preprocess()` is your single source of truth, used identically at train and inference time
- Inference: `preprocess(df, is_training=False)` → `reindex(columns=feature_columns, fill_value=0)` → `model.predict()` → `np.clip(preds, 0, 100)` → append as `predicted_churn_score`
- Top feature by far: `satisfaction_level` (importance 0.31, correlation **-0.62** with churn — lower satisfaction → higher churn). Next: `iscontract_typeMonth-to-month` (0.067, +0.37 — month-to-month contracts churn more).

This feature importance table is gold for your "Why Flagged" rationale — use it to decide which features the LLM is *allowed* to cite as evidence (see §6.3).

---

## 2. System architecture

```
                         ┌─────────────────────────┐
                         │   UPLOAD (CSV + TXTs)   │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │   Django views: parse,   │
                         │   validate, dedupe        │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                     ▼
        ┌───────────────────────┐            ┌───────────────────────┐
        │  SQLite: Customer      │            │  SQLite: Transcript    │
        │  (57 raw + engineered) │            │  (raw text + parsed    │
        │                        │            │   turns + feedback)    │
        └───────────┬────────────┘            └───────────┬────────────┘
                    │                                     │
                    │            "Process" button          │
                    └───────────────┬─────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  ML PIPELINE (on click, batch)  │
                    │  preprocess() → reindex →        │
                    │  model.predict() → clip 0-100    │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  predicted_churn_score written   │
                    │  back to Customer row            │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  RULE-BASED risk banding          │
                    │  (High/Attention/Low, see §5.3)   │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  DASHBOARD (prioritized table)   │
                    └───────────────┬───────────────┘
                                    │  click a customer
                                    ▼
                    ┌───────────────────────────────┐
                    │  ON-DEMAND LLM CALL (LangChain   │
                    │  + Groq) → structured JSON:       │
                    │  signals[], rationale, evidence[] │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  CUSTOMER DETAIL SCREEN           │
                    │  (profile + score + signals +     │
                    │   rationale + transcript + fdbk)  │
                    └───────────────────────────────┘
```

**Key design decision — LLM call timing:** Run the LLM rationale **on-demand when a customer card is clicked**, not for all 200 customers up front. This is why the brief's "in a few seconds" framing matters — cache the result after first generation (see §6.4) so re-opening the same customer is instant, but the first open runs a live call. This keeps your Groq API usage sane and your "Process" button fast (it should only run the ML regression, which is local and instant — no API calls needed for that step).

---

## 3. Database schema (SQLite via Django ORM)

```python
# models.py

class Customer(models.Model):
    # Identity
    customer_id = models.CharField(max_length=20, unique=True, db_index=True)  # dedup key

    # Demographics
    gender = models.CharField(max_length=10, blank=True)
    senior_citizen = models.IntegerField(default=0)
    age = models.IntegerField(null=True)
    married = models.CharField(max_length=5, blank=True)
    dependents = models.CharField(max_length=5, blank=True)

    # Account
    tenure_months = models.IntegerField(null=True)
    contract_type = models.CharField(max_length=30, blank=True)
    payment_method = models.CharField(max_length=40, blank=True)
    paperless_billing = models.CharField(max_length=5, blank=True)
    monthly_charges = models.FloatField(null=True)
    total_charges = models.FloatField(null=True)

    # Services
    phone_service = models.CharField(max_length=5, blank=True)
    internet_service = models.CharField(max_length=20, blank=True)
    service_count = models.IntegerField(null=True)
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

    # Business / status (ground truth, kept for validation only — never shown as if predicted)
    customer_status = models.CharField(max_length=15, blank=True)   # Stayed/Churned
    churn = models.CharField(max_length=5, blank=True)
    churn_score_target = models.FloatField(null=True)  # if present in file — ground truth label
    cltv = models.FloatField(null=True)

    # Model output
    predicted_churn_score = models.FloatField(null=True)   # 0-100, from your regression model
    risk_band = models.CharField(max_length=12, blank=True)  # High / Attention / Low

    # Raw row backup (so "View more details" can show anything not modeled above)
    raw_json = models.JSONField(default=dict)

    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True)


class Transcript(models.Model):
    customer = models.OneToOneField(Customer, on_delete=models.CASCADE, related_name='transcript')
    topic = models.CharField(max_length=255, blank=True)
    raw_text = models.TextField()               # full original .txt content
    turns_json = models.JSONField(default=list)  # [{"speaker": "Customer", "text": "..."}, ...]
    feedback_text = models.TextField(blank=True)
    stated_sentiment = models.CharField(max_length=10, blank=True)  # from the file's own "Sentiment:" line
    uploaded_at = models.DateTimeField(auto_now_add=True)


class SignalAnalysis(models.Model):
    """Cached LLM output — generated once per customer, reused after that."""
    customer = models.OneToOneField(Customer, on_delete=models.CASCADE, related_name='analysis')
    signals = models.JSONField(default=list)      # ["Negative sentiment", "2 unresolved issues", ...]
    rationale = models.TextField(blank=True)       # 1-2 sentence "why flagged"
    evidence = models.JSONField(default=list)      # [{"label": "Satisfaction", "value": "Low", "concern": "High"}, ...]
    llm_sentiment = models.CharField(max_length=10, blank=True)  # LLM's own read, for comparison to stated_sentiment
    suggested_action = models.TextField(blank=True)  # optional enhancement, P2
    generated_at = models.DateTimeField(auto_now_add=True)
    model_used = models.CharField(max_length=50, blank=True)  # e.g. "llama-3.3-70b-versatile"


class UploadBatch(models.Model):
    """Tracks each upload/process run for the Data/Demo screen."""
    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=10)  # csv / txt
    rows_added = models.IntegerField(default=0)
    rows_skipped_duplicate = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
```

**Deduplication rule:** `customer_id` is the unique key. On CSV upload, use `Customer.objects.update_or_create(customer_id=row['customer_id'], defaults={...})` — this means re-uploading the same file is safe (no duplicate rows), and re-uploading a file with updated values for an existing customer refreshes that row. Same pattern for `Transcript` via `customer_id` filename match. Track skip counts in `UploadBatch` so the Data screen can honestly report "180 new, 20 updated" instead of just "200 processed."

---

## 4. Screen-by-screen specification

### Navigation (persistent left sidebar, all screens)

```
┌───────────────────────────────┐
│  ● Intelligent Customer        │
│    Signal Detector             │
├───────────────────────────────┤
│  🏠  Overview                  │
│  👥  Customer Signals          │
│  📊  Model Insights            │
│  ⚙   Data / Upload             │
└───────────────────────────────┘
```

Four screens total. Customer Detail is not in the sidebar — it's reached only by clicking a row in Customer Signals (keeps nav minimal per the brief's "don't overbuild" spirit).

---

### SCREEN 1 — Overview (hero / landing screen)

**Purpose:** answer "how many customers, how bad is it, who's worst" in under 5 seconds of looking.

**Layout, top to bottom:**

1. **Header band**
   - Title: `Intelligent Customer Signal Detector`
   - Subtitle: `Proactively identify customer signals before they escalate.`
   - Right-aligned: `Last processed: <timestamp>` + a `Refresh` button (re-runs risk banding from current DB state, does not re-call the ML model unless data changed)

2. **KPI card row** — 4 cards, equal width, horizontally laid out (on mobile: horizontal scroll per your stated UI preference)
   | Card | Value | Definition |
   |---|---|---|
   | Customers Analyzed | `Customer.objects.filter(predicted_churn_score__isnull=False).count()` | Rows that have been through the ML pipeline |
   | High Risk | count where `risk_band == 'High'` | See banding rule §5.3 |
   | Open Issues | `sum(open_issue_count)` across all customers | Raw support signal |
   | Negative Signals | count of `Transcript.stated_sentiment == 'Negative'` | Feedback-derived signal |

   Each card: large number top, label below, no chart clutter. Use a subtle colored left-border accent (red/amber/green) only on High Risk and Negative Signals cards — not on every card.

3. **Risk distribution** — one horizontal bar chart, 3 bars (High / Medium / Low), counts + proportional bar length. This is the *only* chart on this screen. Do not add a second chart here.

4. **Prioritized Customer Signals table** (top 10 by risk score, "View all →" link to Customer Signals screen)
   Columns: `Rank | Customer ID | Risk Score | Risk Band | Sentiment | Open Issues | Resolution | →`
   Sorted descending by `predicted_churn_score`. Each row clickable → Customer Detail.

**Empty state:** if no customers processed yet, replace KPI row + table with a single centered card: *"No customers processed yet. Go to Data / Upload to get started."* + button linking to Screen 4.

---

### SCREEN 2 — Customer Signals (full table + filters)

**Purpose:** the operational workhorse screen — browse, filter, search, triage the full customer base.

**Layout:**

1. **Filter bar** (sticky under header)
   ```
   [ 🔍 Search customer ID... ]   Risk: [All ▾]   Sentiment: [All ▾]
   Resolution: [All ▾]   Contract: [All ▾]           Sort: [Risk (high→low) ▾]
   ```
   - Search: matches `customer_id` (contains, case-insensitive)
   - Risk: `All / High / Attention / Low` (filters on `risk_band`)
   - Sentiment: `All / Positive / Neutral / Negative` (filters on `Transcript.stated_sentiment`)
   - Resolution: `All / Resolved / NotResolved`
   - Contract: `All / Month-to-month / One year / Two year`
   - Sort: Risk desc (default) / Risk asc / Tenure / Monthly charges
   - Filters combine with AND logic; show active filter count as a badge, with a one-click "Clear filters"

2. **Result count line:** `Showing 43 of 200 customers`

3. **Table** (full width, paginated 25/page)
   | Column | Notes |
   |---|---|
   | Customer ID | monospace font, clickable |
   | Risk Score | `91 / 100` + small colored risk-band pill (`HIGH RISK` / `ATTENTION` / `LOW`) |
   | Sentiment | colored dot + label (green/gray/red) |
   | Open Issues | plain integer |
   | Resolution | `Resolved` (gray) / `Not Resolved` (amber, bold) |
   | Contract | plain text |
   | → | chevron affordance, whole row is clickable |

4. Row click → navigate to `/customer/<customer_id>/` (Customer Detail, Screen 3-equivalent described in §"Customer Detail" below).

**Loading state:** if a large CSV was just processed, show a subtle top progress bar rather than blocking the whole screen.

---

### CUSTOMER DETAIL (reached via row click — the most important screen for the 40% "signal quality" score)

**Purpose:** answer "why was THIS person flagged" with evidence, not a black-box number.

**Layout, top to bottom:**

1. **Header**
   ```
   ← Back to Customer Signals
   CUST-104                                    [ HIGH RISK ]
   Customer Signal Profile                       91 / 100
   ```

2. **Risk card** (prominent, own section)
   ```
   ┌─────────────────────────────┐
   │ PREDICTED CUSTOMER RISK      │
   │        91 / 100              │
   │        HIGH RISK             │
   │ Multiple attention signals   │
   │ detected                     │
   └─────────────────────────────┘
   ```
   Label thresholds — see §5.3. Never call this "Probability of Churn" — your model is a regression on `churn_score_1`, not a calibrated probability. Label it **Predicted Churn Score** or **Predicted Risk Score**.

3. **Customer Profile** — 7 fields only, in a 2-column grid (per your own instinct to not dump all 57 columns):
   ```
   Age            Tenure           Contract
   42             8 months         Month-to-month

   Plan/Services   Satisfaction     Monthly Charges
   5 services      Low (3-tier)     $70.50
   ```
   Below: a collapsed `▸ View all customer details` disclosure that expands to a plain key-value dump of every remaining CSV column (from `raw_json`) — satisfies completeness without cluttering the primary view.

4. **Signal Breakdown** — a small table contrasting model output vs. observed evidence:
   ```
   SIGNAL BREAKDOWN
   Satisfaction        Low               ● High concern
   Open Support Issues  2                 ● High concern
   Sentiment            Negative          ● High concern
   Resolution Status    Not Resolved      ● High concern
   Contract Type        Month-to-month    ● Moderate concern
   Tenure                8 months          ● Moderate concern
   ```
   Each row's "concern" level is computed by the rule-engine in §5.4 (NOT invented freely by the LLM — the LLM explains signals the rule engine already flagged, so your rationale is grounded and reproducible, which matters a lot under the 40% evaluation criterion for reasoning quality).

5. **Why This Customer Was Flagged** (the AI-generated rationale — LLM call output, cached)
   ```
   WHY THIS CUSTOMER WAS FLAGGED

   Low satisfaction, an unresolved support issue, and negative
   feedback are combining to drive a high predicted risk score.
   The customer is also on a month-to-month contract, which
   historically correlates with higher churn.
   ```
   1–3 sentences max. See §6 for the exact prompt template and JSON contract.

   Below it, a compact **Evidence** list (bullet form, pulled straight from the same LLM JSON — do not let the LLM free-write this, generate it from the structured `evidence` field so it's consistent with the Signal Breakdown table above):
   ```
   Evidence
   • Satisfaction level: Low
   • Open issues: 2 (unresolved)
   • Sentiment: Negative
   • Contract: Month-to-month
   ```

6. **Support Conversation** (renders `Transcript.turns_json`, collapsible if long)
   ```
   SUPPORT CONVERSATION                         Topic: <topic>

   Customer   I've been facing this problem several times.
   Agent      Can you explain what happened?
   Customer   I already contacted support twice...
   Agent      I'm sorry about that. Let me check this for you.
   ```
   Speaker labels visually distinguished (e.g. Customer left-aligned/blue accent, Agent right-aligned/gray — like a chat bubble UI, matching your "best UI" ask). Metadata row above: `Issue Type` (from Topic) · `Resolution` · `Interactions count`.

7. **Customer Feedback**
   ```
   CUSTOMER FEEDBACK
   "I've contacted support several times and the issue is
   still not fixed."

   Sentiment: Negative        (as stated in transcript)
   LLM read:  Negative        (independent check, optional)
   ```
   If `stated_sentiment` and `llm_sentiment` disagree, show both plainly rather than picking one — this is a legitimate, honest thing to surface in a POC and shows reasoning transparency (good for the "reasoning quality" evaluation line).

8. *(P2, optional, only if time remains)* **Suggested Next Step** — one line, e.g. `Review unresolved support issue and offer retention credit.` Keep this OFF by default / behind a toggle so it doesn't distract from the core deliverable, per the brief explicitly calling this optional.

---

### SCREEN 3 — Model Insights

**Purpose:** give the evaluator technical credibility without letting it dominate the product experience.

**Layout:**

1. **Model summary card**
   ```
   Model            Decision Tree / XGBoost Regressor
   Target           churn_score_1 (0-100 continuous)
   Training rows    <n from your split>
   Test rows        <n>
   ```

2. **Evaluation metrics card** — MAE / RMSE / R² (pull whatever you logged during training; if not logged, compute quickly with a holdout split before demo day — don't fabricate numbers).

3. **Top Feature Importance** — horizontal bar chart, top 8-10 features only (not all 27), using your actual XGBoost importances:
   ```
   Satisfaction Level              ████████████████ 0.314
   Contract: Month-to-month        ███ 0.067
   Fiber Optic Internet            ███ 0.064
   Number of Referrals             ██ 0.041
   Contract: Two year              ██ 0.029
   Unlimited Data                  █ 0.027
   Payment: Electronic Check       █ 0.027
   Paperless Billing               █ 0.026
   ```
   Caption underneath, verbatim wording matters: *"Feature importance reflects influence on the model's prediction, not proven causation."* Never phrase this section as "these features caused churn."

4. *(optional)* Correlation direction note: satisfaction level is inversely correlated with churn risk (r ≈ −0.62); month-to-month contracts are positively correlated (r ≈ +0.37). This one sentence demonstrates you understand your own model, which is worth more than a chart.

---

### SCREEN 4 — Data / Upload

**Purpose:** the ingestion + processing control panel. This is where your explicitly requested upload/process feature lives.

**Layout:**

1. **Current data status card**
   ```
   200 customers loaded · 200 support transcripts linked
   Last processed: Aug 13, 2026, 11:04 AM
   ```

2. **Upload panel — two independent upload zones:**

   **A. Bulk CSV upload**
   ```
   ┌───────────────────────────────────────────┐
   │  Drop CSV file here or click to browse      │
   │  Expected: customer records (any columns     │
   │  except target are fine — extra/missing       │
   │  columns are handled automatically)           │
   └───────────────────────────────────────────┘
   [ Upload CSV ]
   ```
   - Accepts the full raw schema (57 cols) or a subset — reuses your `preprocess()` function, which already fills missing training columns with 0 via `reindex`.
   - On upload: parse → `update_or_create` per row on `customer_id` → report `{added, updated, skipped_invalid}`.
   - **Explicitly per your request:** during processing, invalid/malformed rows are logged and skipped, not fatal — one bad row must never kill the whole batch.

   **B. Transcript upload (bulk or single)**
   ```
   ┌───────────────────────────────────────────┐
   │  Drop one or more .txt files here            │
   │  Filename must match customer_id, e.g.        │
   │  0014-BMAQU.txt                                │
   └───────────────────────────────────────────┘
   [ Upload Transcripts ]
   ```
   - Multi-file input (`<input type="file" multiple>`)
   - Each file parsed with the transcript parser (§5.2), linked to `Customer` by filename-derived `customer_id`
   - If a transcript's `customer_id` has no matching `Customer` row yet, still store it (nullable FK or a "pending" queue) so upload order doesn't matter — link it retroactively when/if the matching CSV row arrives.

   **C. Manual single-customer entry** (per your "manual upload each data point" request)
   A simple form below the two drop zones: `+ Add customer manually` expands a form with the ~15 decision-relevant fields (not all 57 — match the Customer Profile card fields from the detail screen) plus a free-text box for pasting a transcript directly. Good for demo-day live entry of a new example.

3. **`[ Process Customers ]` button** — the core requested action:
   - Runs `preprocess()` → `reindex(feature_columns, fill_value=0)` → `model.predict()` → `clip(0,100)` on every `Customer` row with a null or stale `predicted_churn_score`
   - Writes `predicted_churn_score` + computed `risk_band` back to each row
   - Does **not** call the LLM (LLM rationale is generated lazily per-customer on detail-page open, per §2) — keeps this button fast even for hundreds of rows
   - Shows a simple progress indicator, then a summary toast: `187 customers processed · 27 flagged High Risk`
   - Auto-redirects (or offers a button) to the Overview screen so the "process → see dashboard update" loop the brief wants is visible and fast

4. **Danger zone (per your settings/delete request)**
   ```
   ⚠ Danger Zone
   [ Delete all customer data ]   [ Delete all transcripts ]   [ Reset database ]
   ```
   Each requires a confirm-dialog ("Type DELETE to confirm") — this is a demo tool, but don't make data loss a single misclick.

5. **Upload history table** (from `UploadBatch`): filename, type, rows added/updated/skipped, timestamp — small, at the bottom, collapsed by default.

---

## 5. Backend logic — pseudocode you can implement directly

### 5.1 CSV ingestion

```python
def ingest_csv(file, batch: UploadBatch):
    df = pd.read_csv(file)
    added, updated, skipped = 0, 0, 0
    for _, row in df.iterrows():
        cid = str(row.get('customer_id', '')).strip()
        if not cid:
            skipped += 1
            continue
        defaults = extract_mapped_fields(row)   # maps CSV columns -> model fields
        defaults['raw_json'] = row.to_dict()
        obj, created = Customer.objects.update_or_create(
            customer_id=cid, defaults=defaults
        )
        added += 1 if created else 0
        updated += 0 if created else 1
    batch.rows_added, batch.rows_skipped_duplicate = added, skipped
    batch.save()
```

### 5.2 Transcript parsing (matches your confirmed `.txt` format exactly)

```python
import re

def parse_transcript(raw_text: str) -> dict:
    header, _, rest = raw_text.partition('----------------------------------------')
    body, _, footer = rest.partition('----------------------------------------')

    customer_id = re.search(r'Customer ID:\s*(.+)', header)
    topic       = re.search(r'Topic:\s*(.+)', header)
    status      = re.search(r'Customer Status:\s*(.+)', header)

    turns = []
    for line in body.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'(Customer|Agent):\s*(.*)', line)
        if m:
            turns.append({'speaker': m.group(1), 'text': m.group(2)})

    feedback  = re.search(r'Feedback:\s*(.+)', footer)
    sentiment = re.search(r'Sentiment:\s*(\w+)', footer)

    return {
        'customer_id': customer_id.group(1).strip() if customer_id else None,
        'topic': topic.group(1).strip() if topic else '',
        'turns_json': turns,
        'feedback_text': feedback.group(1).strip() if feedback else '',
        'stated_sentiment': sentiment.group(1).strip() if sentiment else '',
        'raw_text': raw_text,
    }
```

### 5.3 Risk banding (rule, not LLM — must be deterministic and documented for your README)

```python
def risk_band(score: float) -> str:
    if score is None:
        return 'Unscored'
    if score >= 70:
        return 'High'
    elif score >= 40:
        return 'Attention'
    else:
        return 'Low'
```
Document this threshold choice explicitly in your README ("High risk = predicted score ≥ 70") — the brief specifically warns not to change your threshold ad hoc during a demo.

### 5.4 Signal Breakdown concern levels (deterministic rule engine — this is what the LLM will explain, not invent)

```python
def compute_signal_flags(customer: Customer, transcript: Transcript | None) -> list[dict]:
    flags = []
    if customer.satisfaction_level == 'Low':
        flags.append({'label': 'Satisfaction', 'value': 'Low', 'concern': 'High'})
    if customer.open_issue_count and customer.open_issue_count > 0:
        flags.append({'label': 'Open Support Issues', 'value': str(customer.open_issue_count), 'concern': 'High'})
    if transcript and transcript.stated_sentiment == 'Negative':
        flags.append({'label': 'Sentiment', 'value': 'Negative', 'concern': 'High'})
    if customer.resolution_status_open_closed == 'NotResolved':
        flags.append({'label': 'Resolution Status', 'value': 'Not Resolved', 'concern': 'High'})
    if customer.contract_type == 'Month-to-month':
        flags.append({'label': 'Contract Type', 'value': 'Month-to-month', 'concern': 'Moderate'})
    if customer.tenure_months is not None and customer.tenure_months < 12:
        flags.append({'label': 'Tenure', 'value': f'{customer.tenure_months} months', 'concern': 'Moderate'})
    return flags
```

---

## 6. LLM layer (LangChain + Groq)

### 6.1 Why on-demand + cached, not batch
200 upfront LLM calls burns free-tier quota and adds latency to your "Process" button, which should stay ML-only and fast. Generating the rationale when a card is opened (and caching in `SignalAnalysis`) matches your own "generate instruction on them... showing those card in few seconds" requirement — the *first* open takes a couple seconds (one Groq call), every subsequent open of that same customer is instant (DB read).

### 6.2 LangChain setup

```python
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class SignalOutput(BaseModel):
    signals: list[str] = Field(description="3-5 short flag phrases, e.g. 'Negative sentiment'")
    rationale: str = Field(description="1-3 sentences, plain language, no jargon")
    evidence: list[dict] = Field(description="label/value pairs the rationale is grounded in")
    llm_sentiment: str = Field(description="Positive, Neutral, or Negative")
    suggested_action: str = Field(description="one short actionable sentence")

parser = PydanticOutputParser(pydantic_object=SignalOutput)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
```

### 6.3 Prompt template — grounded in the rule-engine output, not free association

```python
PROMPT = ChatPromptTemplate.from_template("""
You are assisting a customer operations analyst. You are given a customer's
profile data, a rule-based list of concerning signals already detected by
the system, and their support conversation. Do not invent signals that are
not present in the provided data.

CUSTOMER PROFILE
{profile_summary}

PREDICTED RISK SCORE (from a trained regression model, 0-100)
{risk_score}

SYSTEM-DETECTED SIGNALS (rule-based, already computed — explain these, do not contradict them)
{signal_flags}

SUPPORT CONVERSATION
{transcript_text}

CUSTOMER FEEDBACK
"{feedback_text}"
(stated sentiment in source data: {stated_sentiment})

Task:
1. Write a 1-3 sentence plain-language rationale for why this customer is
   flagged, using only the signals and evidence given above.
2. List 3-5 short evidence bullets (label + value) drawn only from the data given.
3. State your own independent read of the feedback's sentiment (Positive/Neutral/Negative).
4. Suggest one short, concrete next action for the ops team.

{format_instructions}
""")
```

Feed `signal_flags` from `compute_signal_flags()` (§5.4) directly into the prompt — this is the single most important design choice for the 40%-weighted "reasoning quality" criterion: the LLM is *explaining* a deterministic, auditable rule output in natural language, not hallucinating its own risk assessment from scratch. If asked in an interview "how do you know the AI isn't making things up," this is your answer.

### 6.4 Caching / invocation flow

```python
def get_or_generate_analysis(customer: Customer) -> SignalAnalysis:
    existing = getattr(customer, 'analysis', None)
    if existing:
        return existing
    transcript = getattr(customer, 'transcript', None)
    flags = compute_signal_flags(customer, transcript)
    chain = PROMPT | llm | parser
    result = chain.invoke({
        'profile_summary': build_profile_summary(customer),
        'risk_score': customer.predicted_churn_score,
        'signal_flags': flags,
        'transcript_text': format_turns(transcript.turns_json) if transcript else 'No transcript available.',
        'feedback_text': transcript.feedback_text if transcript else '',
        'stated_sentiment': transcript.stated_sentiment if transcript else 'Unknown',
        'format_instructions': parser.get_format_instructions(),
    })
    return SignalAnalysis.objects.create(
        customer=customer,
        signals=result.signals,
        rationale=result.rationale,
        evidence=result.evidence,
        llm_sentiment=result.llm_sentiment,
        suggested_action=result.suggested_action,
        model_used="llama-3.3-70b-versatile",
    )
```

**Failure handling:** if the Groq call errors or times out (rate limit, network), fall back to a template-generated rationale built purely from `signal_flags` (e.g. `"Flagged due to: " + ", ".join(f['label'] for f in flags)`), so the demo never shows a broken screen. Log the failure but don't block the UI.

---

## 7. UI design system

### 7.1 Color palette

```
Background        #FAFAFA (very light neutral)
Surface / cards    #FFFFFF
Border              #E5E7EB
Text primary        #111827
Text secondary      #6B7280

High risk accent    #DC2626 (red)
Attention accent    #D97706 (amber)
Low risk accent     #16A34A (green)
Positive sentiment  #16A34A
Neutral sentiment   #6B7280
Negative sentiment  #DC2626

Primary action      #2563EB (blue) — buttons, links, active nav item
```

Rule: color is a **small accent** (pill background, left-border, dot, small badge) — never a full-card or full-row background fill. Keeps the dashboard scannable rather than looking like an alarm panel.

### 7.2 Typography

```
Page title       28px / 700 weight
Section heading  16px / 600 weight, uppercase, letter-spacing 0.03em, text-secondary color
Metric (big #)   36-44px / 700 weight
Body text        14px / 400 weight
Supporting text  12-13px / 400 weight, text-secondary
Monospace        Customer IDs only — e.g. 'JetBrains Mono' or 'Roboto Mono' via CDN
```

### 7.3 Component states & micro-animation (matching your "best UI, minimalist, good animation" ask)

- **Cards:** `transition: box-shadow 150ms ease, transform 150ms ease;` — subtle lift (`translateY(-2px)` + soft shadow) on hover for clickable cards/rows. Nothing more elaborate.
- **Risk pill badges:** rounded-full, small caps, background = accent color at 12% opacity, text = accent color at full opacity — not solid fill.
- **Table rows:** hover background `#F9FAFB`, cursor pointer, whole row clickable.
- **Loading (Process button / LLM generation):** skeleton shimmer on the specific card being populated (e.g. the rationale box shows 3 animated gray bars for ~1-2s while the Groq call is in flight) — never a full-page spinner for a single-card wait.
- **Filter dropdowns:** simple native `<select>` styled with a custom chevron, or a lightweight custom dropdown with a 100ms fade-in — don't build a heavy component library for this.
- **Page transitions:** none needed — Django server-rendered pages are fine; a 200ms fade-in on `.main-content` on load is enough polish.
- **Toasts** (upload/process confirmations): slide in from top-right, auto-dismiss after 4s, manual close (×) available.

### 7.4 Layout grid
- Sidebar: fixed 240px, collapses to icon-only rail under 768px width
- Main content: max-width 1200px, centered, 24px gutter padding
- KPI cards: CSS grid, `grid-template-columns: repeat(4, 1fr)` desktop → horizontal scroll snap on mobile (per your stated mobile preference), each card `min-width: 200px`
- Table: full width, horizontal scroll on mobile rather than column stacking (preserves scanability)

---

## 8. Route map (Django URLs)

```
/                          → Overview
/customers/                → Customer Signals (list + filters)
/customer/<customer_id>/   → Customer Detail (triggers LLM if not cached)
/model-insights/           → Model Insights
/data/                     → Data / Upload

/api/upload/csv/           POST  → ingest_csv
/api/upload/transcripts/   POST  → ingest transcripts (multi-file)
/api/customer/manual/      POST  → manual single-customer entry
/api/process/              POST  → run ML pipeline on all unscored/stale customers
/api/customer/<id>/reanalyze/  POST → force-regenerate SignalAnalysis (bypass cache)
/api/data/reset/           POST  → danger-zone deletes
```

---

## 9. Priority order for your remaining build time

**P0 — must ship (this is what 70% of your score depends on):**
Upload CSV/TXT → Process button → ML scoring → Overview KPIs → Customer Signals table + filters → Customer Detail with Signal Breakdown + LLM rationale + transcript + feedback.

**P1 — strong enhancement, do if P0 is solid:**
Model Insights screen with real feature importances, risk distribution chart, manual entry form, upload history log.

**P2 — only if time remains, keep hidden/off by default:**
Suggested retention action, sentiment-disagreement badge, danger-zone reset polish, animated transitions beyond the basics above.

**Do not build (explicitly out of scope per the brief):** multi-signal correlation beyond what's described here, a chatbot, a heatmap, historical trend charts. These are the brief's own "optional enhancement" list — they exist to reward extra time, not to define your core deliverable.

---

## 10. What to say in your 3-minute demo (mapped to this build)

1. **0:00–0:20** — Overview screen: *"The system processes customer records and support conversations, scores each customer's churn risk with a trained regression model, and surfaces who needs attention first."*
2. **0:20–0:50** — Customer Signals: filter to High Risk, point at the table. *"Each customer is ranked by predicted risk, with sentiment and open-issue signals visible at a glance."*
3. **0:50–1:40** — Click the top customer → Customer Detail: walk Signal Breakdown → Why Flagged rationale → transcript → feedback. *"The rationale is generated by an LLM that explains signals already detected by a deterministic rule engine — so the explanation is grounded in the same evidence you can see in this table, not a black box."*
4. **1:40–2:10** — Data/Upload: show the Process button flow briefly (or mention it was used to build this dataset).
5. **2:10–2:40** — Model Insights: feature importance chart, one sentence on satisfaction level being the dominant driver.
6. **2:40–3:00** — Close: *"This is a prototype demonstrating how the signal-detection workflow could support continuous, automated monitoring instead of the manual, siloed review process described in the brief."*

---

## 11. README skeleton (one page, per submission requirement)

```markdown
# Intelligent Customer Signal Detector — README

## Approach
[2-3 sentences: structured customer data + support transcripts →
regression model for risk score + LLM for grounded natural-language rationale]

## Tools used
Django, SQLite, scikit-learn/XGBoost (trained model), LangChain + Groq
(llama-3.3-70b-versatile), vanilla HTML/CSS/JS

## Assumptions made
- customer_id is the join key between structured records and transcripts
- High risk threshold = predicted score >= 70, Attention = 40-69, Low = <40
- LLM rationale is generated from and constrained to rule-detected signals,
  not generated independently, to keep explanations auditable
- [any others specific to your build]

## Example input → output
Input: CUST-XXXX profile (satisfaction: Low, open issues: 2, contract:
Month-to-month) + support transcript with Negative stated sentiment
Output: Risk score 91/100 (High Risk). Rationale: "Low satisfaction, an
unresolved support issue, and negative feedback are combining to drive a
high predicted risk score..."

## Setup instructions
[pip install -r requirements.txt, migrate, set GROQ_API_KEY, runserver]
```

---

This spec is grounded entirely in your uploaded brief, your actual 57-column CSV, your confirmed `.txt` transcript format, and your own inference/preprocessing code — nothing here assumes a dataset or model shape you haven't already built.
