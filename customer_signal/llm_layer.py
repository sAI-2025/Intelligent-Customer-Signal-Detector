# customer_signal/llm_layer.py
"""
Phase 4 — LangChain + Groq rationale generation, on-demand + cached.
Never called from the Process button. Only called when a Customer Detail
page is opened for the first time (or via /reanalyze/ to force refresh).
"""
import logging
from pydantic import BaseModel, Field

from .models import Customer, SignalAnalysis
from .signal_logic import compute_signal_flags

logger = logging.getLogger(__name__)

MODEL_NAME = "llama-3.3-70b-versatile"


class SignalOutput(BaseModel):
    signals: list[str] = Field(description="3-5 short flag phrases, e.g. 'Negative sentiment'")
    rationale: str = Field(description="1-3 sentences, plain language, no jargon")
    evidence: list[dict] = Field(description="label/value pairs the rationale is grounded in")
    llm_sentiment: str = Field(description="Positive, Neutral, or Negative")
    suggested_action: str = Field(description="one short actionable sentence")


PROMPT_TEMPLATE = """
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
"""


def build_profile_summary(customer: Customer) -> str:
    return (
        f"Age: {customer.age}, Tenure: {customer.tenure_months} months, "
        f"Contract: {customer.contract_type}, Monthly charges: {customer.monthly_charges}, "
        f"Satisfaction: {customer.satisfaction_level}, "
        f"Open issues: {customer.open_issue_count}, "
        f"Resolution status: {customer.resolution_status_open_closed}"
    )


def format_turns(turns_json: list) -> str:
    if not turns_json:
        return "No transcript available."
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns_json)


def _template_fallback(customer: Customer, flags: list, transcript) -> SignalOutput:
    """Used if Groq errors/times out — never leave the UI broken."""
    labels = [f["label"] for f in flags] or ["No strong signals detected"]
    return SignalOutput(
        signals=labels[:5],
        rationale="Flagged due to: " + ", ".join(labels) + ".",
        evidence=flags,
        llm_sentiment=(transcript.stated_sentiment if transcript else "Neutral") or "Neutral",
        suggested_action="Review account manually — automated rationale unavailable.",
    )


def _get_llm_chain():
    from langchain_groq import ChatGroq
    from langchain.prompts import ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=SignalOutput)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatGroq(model=MODEL_NAME, temperature=0.2)
    return prompt | llm | parser, parser


def get_or_generate_analysis(customer: Customer, force: bool = False) -> SignalAnalysis:
    """
    Returns cached SignalAnalysis unless force=True (used by /reanalyze/).
    Falls back to a deterministic template rationale on any LLM failure.
    """
    existing = getattr(customer, "analysis", None)
    if existing and not force:
        return existing

    transcript = getattr(customer, "transcript", None)
    flags = compute_signal_flags(customer, transcript)

    try:
        chain, parser = _get_llm_chain()
        result = chain.invoke({
            "profile_summary": build_profile_summary(customer),
            "risk_score": customer.predicted_churn_score,
            "signal_flags": flags,
            "transcript_text": format_turns(transcript.turns_json if transcript else []),
            "feedback_text": transcript.feedback_text if transcript else "",
            "stated_sentiment": transcript.stated_sentiment if transcript else "Unknown",
            "format_instructions": parser.get_format_instructions(),
        })
        model_used = MODEL_NAME
    except Exception as e:
        logger.warning("Groq LLM call failed for %s, using fallback: %s", customer.customer_id, e)
        result = _template_fallback(customer, flags, transcript)
        model_used = "rule-based-fallback"

    if existing:
        existing.signals = result.signals
        existing.rationale = result.rationale
        existing.evidence = result.evidence
        existing.llm_sentiment = result.llm_sentiment
        existing.suggested_action = result.suggested_action
        existing.model_used = model_used
        existing.save()
        return existing

    return SignalAnalysis.objects.create(
        customer=customer,
        signals=result.signals,
        rationale=result.rationale,
        evidence=result.evidence,
        llm_sentiment=result.llm_sentiment,
        suggested_action=result.suggested_action,
        model_used=model_used,
    )
