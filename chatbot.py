"""
chatbot.py
----------
Bonus Task: "Ask the data" chatbot.

Same rule as ai_explain.py: the LLM never invents numbers. Before every
chat turn we (re)compute a broad set of real aggregates straight from the
cleaned/filtered Pandas dataframe and hand them to the model as grounding
context, along with the user's question and the conversation so far. The
model's job is only to interpret / explain / compare real numbers in plain
English -- never to guess a statistic it wasn't given.

Supports the same three providers as ai_explain.py (Groq, Anthropic,
OpenAI) and reads API keys the same way -- from environment variables /
Streamlit secrets, never hard-coded.
"""

from __future__ import annotations
import os
import json
import pandas as pd

import analysis as A


def _sanitize_keys(obj):
    """Recursively convert non-JSON-safe dict keys (tuples from a MultiIndex
    .to_dict(), numpy scalar keys, etc.) into plain strings so json.dumps
    never chokes on them."""
    if isinstance(obj, dict):
        return {
            (" + ".join(str(part) for part in k) if isinstance(k, tuple) else str(k)): _sanitize_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_sanitize_keys(v) for v in obj]
    return obj


def build_data_context(df: pd.DataFrame) -> dict:
    """A broad set of REAL, precomputed aggregates the chatbot can draw on,
    so it can answer common follow-up questions without hallucinating
    numbers. Everything here is computed live, with plain Pandas groupbys,
    on whatever slice of data is currently selected on the dashboard."""
    d = df.copy()
    ctx = {}

    ctx["overall_stats"] = A.basic_stats(d.copy())
    ctx["avg_time_by_traffic"] = A.q1_traffic_impact(d).to_dict()

    q2 = A.q2_distance_impact(d)
    ctx["distance_vs_time"] = {
        "correlation": q2["correlation"],
        "avg_time_by_distance_bucket": q2["avg_time_by_distance_bucket"].to_dict(),
    }
    ctx["worst_weather_traffic_combos"] = A.q3_combined_conditions(d, top_n=5).to_dict()

    if "City" in d.columns:
        ctx["avg_time_by_city"] = d.groupby("City", observed=True)["Time_taken (min)"].mean().round(2).to_dict()
    if "Type_of_vehicle" in d.columns:
        ctx["avg_time_by_vehicle_type"] = d.groupby("Type_of_vehicle", observed=True)["Time_taken (min)"].mean().round(2).to_dict()
    if "Type_of_order" in d.columns:
        ctx["avg_time_by_order_type"] = d.groupby("Type_of_order", observed=True)["Time_taken (min)"].mean().round(2).to_dict()
    if "Festival" in d.columns:
        ctx["avg_time_by_festival"] = d.groupby("Festival", observed=True)["Time_taken (min)"].mean().round(2).to_dict()
    if "Vehicle_condition" in d.columns:
        ctx["avg_time_by_vehicle_condition"] = d.groupby("Vehicle_condition", observed=True)["Time_taken (min)"].mean().round(2).to_dict()
    if "Order_Hour" in d.columns:
        ctx["avg_time_by_order_hour"] = {
            str(int(k)): v for k, v in d.groupby("Order_Hour", observed=True)["Time_taken (min)"].mean().round(2).items()
            if pd.notna(k)
        }
    if "multiple_deliveries" in d.columns:
        ctx["avg_time_by_multiple_deliveries"] = d.groupby("multiple_deliveries", observed=True)["Time_taken (min)"].mean().round(2).to_dict()

    ctx["rating_stats"] = {
        "avg_rating": round(float(d["Delivery_person_Ratings"].mean()), 2),
        "min_rating": float(d["Delivery_person_Ratings"].min()),
        "max_rating": float(d["Delivery_person_Ratings"].max()),
    }
    ctx["age_stats"] = {
        "avg_age": round(float(d["Delivery_person_Age"].mean()), 1),
        "min_age": float(d["Delivery_person_Age"].min()),
        "max_age": float(d["Delivery_person_Age"].max()),
    }
    ctx["rows_in_current_filter"] = int(len(d))

    return _sanitize_keys(ctx)


SYSTEM_PROMPT_TEMPLATE = """You are "RouteBot", a data assistant embedded in a food-delivery
analytics dashboard (RouteWise). You help a non-technical business user
understand the delivery data shown on screen.

Rules:
- ONLY use the numbers inside DATA_CONTEXT below. Never invent, estimate,
  or guess a statistic that isn't there.
- If the answer isn't in DATA_CONTEXT, say so plainly and suggest which
  filter or tab on the dashboard would show it, instead of making a
  number up.
- Keep answers short and conversational (2-6 sentences). Plain English,
  no markdown headers. Short bullet points are fine for lists.
- All numbers reflect only the currently selected filters on the
  dashboard ({n_rows} rows in the current view), not necessarily the
  full raw dataset.

DATA_CONTEXT:
{context_json}
"""


def build_system_prompt(df: pd.DataFrame) -> str:
    ctx = build_data_context(df)
    return SYSTEM_PROMPT_TEMPLATE.format(
        n_rows=ctx["rows_in_current_filter"],
        context_json=json.dumps(ctx, indent=2, default=str),
    )


def chat_with_anthropic(system: str, history: list[dict], model: str = "claude-sonnet-4-5-20250929") -> str:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=500,
        system=system,
        messages=history,
    )
    return resp.content[0].text


def chat_with_openai(system: str, history: list[dict], model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + history,
        max_tokens=500,
    )
    return resp.choices[0].message.content


def chat_with_groq(system: str, history: list[dict], model: str = "openai/gpt-oss-20b") -> str:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + history,
        max_tokens=500,
    )
    return resp.choices[0].message.content


CHAT_PROVIDERS = {
    "Groq": chat_with_groq,
    "Claude (Anthropic)": chat_with_anthropic,
    "OpenAI": chat_with_openai,
}


NO_KEY_FALLBACK = (
    "⚠️ No API key found in the environment for this provider, so I can't chat live. "
    "Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY as an environment variable "
    "(or Streamlit secret) and try again."
)


def _safe_error_text(e: Exception) -> str:
    """Turn ANY exception into a short, safe, printable string.
    Some provider SDKs raise error objects whose own __str__/repr can itself
    blow up (e.g. when the HTTP response wasn't fully populated). We never
    want the *error handler* to crash the app, so every step here is
    wrapped defensively."""
    for getter in (lambda: str(e), lambda: repr(e), lambda: e.__class__.__name__):
        try:
            text = getter()
            if text:
                return text
        except Exception:
            continue
    return "Unknown error"


def ask(question: str, chat_history: list[dict], df: pd.DataFrame, provider: str = "Groq") -> str:
    """chat_history: list of {'role': 'user'|'assistant', 'content': str},
    NOT including the new question yet. Returns the assistant's reply text.

    This never raises -- any failure (missing key, network issue, bad SDK
    response, etc.) is caught here and turned into a friendly message, so
    the calling UI code never has to build error strings itself."""
    fn = CHAT_PROVIDERS.get(provider, chat_with_groq)
    try:
        system = build_system_prompt(df)
        messages = chat_history + [{"role": "user", "content": question}]
        return fn(system, messages)
    except RuntimeError as e:
        # Raised by us above when the relevant API key env var / secret is missing.
        return f"{NO_KEY_FALLBACK}\n\n_Detail: {_safe_error_text(e)}_"
    except Exception as e:
        return (
            f"⚠️ Sorry, I couldn't reach **{provider}** right now.\n\n"
            f"_Detail: {_safe_error_text(e)}_\n\n"
            "This usually means the API key is missing/invalid, the model name "
            "changed, or you've hit a rate limit. Check your Streamlit "
            "secrets or try a different provider from the sidebar."
        )
