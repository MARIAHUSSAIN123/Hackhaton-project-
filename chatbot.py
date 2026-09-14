```python
"""
RouteBot - Food Delivery Analytics Chatbot

Features:
- Natural conversation
- Data-grounded analytics
- Follow-up conversation
- Groq / OpenAI / Anthropic support
- Never invents dataset numbers
"""

from __future__ import annotations

import os
import json
import pandas as pd

import analysis as A


# =========================================================
# HELPERS
# =========================================================

def _sanitize_keys(obj):
    """Make dictionaries JSON-safe."""
    if isinstance(obj, dict):
        return {
            (
                " + ".join(str(part) for part in key)
                if isinstance(key, tuple)
                else str(key)
            ): _sanitize_keys(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [_sanitize_keys(value) for value in obj]

    return obj


# =========================================================
# DATA CONTEXT
# =========================================================

def build_data_context(df: pd.DataFrame) -> dict:
    """
    Compute real statistics from the currently filtered dataframe.
    The AI receives these statistics as grounding context.
    """

    d = df.copy()
    ctx = {}

    # -----------------------------------------------------
    # Basic statistics
    # -----------------------------------------------------

    try:
        ctx["overall_stats"] = A.basic_stats(d.copy())
    except Exception:
        ctx["overall_stats"] = {}

    # -----------------------------------------------------
    # Traffic
    # -----------------------------------------------------

    try:
        ctx["avg_time_by_traffic"] = (
            A.q1_traffic_impact(d)
            .to_dict()
        )
    except Exception:
        ctx["avg_time_by_traffic"] = {}

    # -----------------------------------------------------
    # Distance
    # -----------------------------------------------------

    try:
        q2 = A.q2_distance_impact(d)

        ctx["distance_vs_time"] = {
            "correlation": q2.get("correlation"),
            "avg_time_by_distance_bucket": (
                q2.get(
                    "avg_time_by_distance_bucket",
                    {}
                ).to_dict()
                if hasattr(
                    q2.get("avg_time_by_distance_bucket", {}),
                    "to_dict"
                )
                else q2.get(
                    "avg_time_by_distance_bucket",
                    {}
                )
            ),
        }

    except Exception:
        ctx["distance_vs_time"] = {}

    # -----------------------------------------------------
    # Weather + Traffic
    # -----------------------------------------------------

    try:
        ctx["worst_weather_traffic_combos"] = (
            A.q3_combined_conditions(
                d,
                top_n=5
            ).to_dict()
        )
    except Exception:
        ctx["worst_weather_traffic_combos"] = {}

    # -----------------------------------------------------
    # City
    # -----------------------------------------------------

    if "City" in d.columns and "Time_taken (min)" in d.columns:
        ctx["avg_time_by_city"] = (
            d.groupby(
                "City",
                observed=True
            )["Time_taken (min)"]
            .mean()
            .round(2)
            .to_dict()
        )

    # -----------------------------------------------------
    # Vehicle Type
    # -----------------------------------------------------

    if (
        "Type_of_vehicle" in d.columns
        and "Time_taken (min)" in d.columns
    ):
        ctx["avg_time_by_vehicle_type"] = (
            d.groupby(
                "Type_of_vehicle",
                observed=True
            )["Time_taken (min)"]
            .mean()
            .round(2)
            .to_dict()
        )

    # -----------------------------------------------------
    # Order Type
    # -----------------------------------------------------

    if (
        "Type_of_order" in d.columns
        and "Time_taken (min)" in d.columns
    ):
        ctx["avg_time_by_order_type"] = (
            d.groupby(
                "Type_of_order",
                observed=True
            )["Time_taken (min)"]
            .mean()
            .round(2)
            .to_dict()
        )

    # -----------------------------------------------------
    # Festival
    # -----------------------------------------------------

    if (
        "Festival" in d.columns
        and "Time_taken (min)" in d.columns
    ):
        ctx["avg_time_by_festival"] = (
            d.groupby(
                "Festival",
                observed=True
            )["Time_taken (min)"]
            .mean()
            .round(2)
            .to_dict()
        )

    # -----------------------------------------------------
    # Vehicle Condition
    # -----------------------------------------------------

    if (
        "Vehicle_condition" in d.columns
        and "Time_taken (min)" in d.columns
    ):
        ctx["avg_time_by_vehicle_condition"] = (
            d.groupby(
                "Vehicle_condition",
                observed=True
            )["Time_taken (min)"]
            .mean()
            .round(2)
            .to_dict()
        )

    # -----------------------------------------------------
    # Order Hour
    # -----------------------------------------------------

    if (
        "Order_Hour" in d.columns
        and "Time_taken (min)" in d.columns
    ):
        ctx["avg_time_by_order_hour"] = {
            str(int(hour)): value
            for hour, value in (
                d.groupby(
                    "Order_Hour",
                    observed=True
                )["Time_taken (min)"]
                .mean()
                .round(2)
                .items()
            )
            if pd.notna(hour)
        }

    # -----------------------------------------------------
    # Multiple Deliveries
    # -----------------------------------------------------

    if (
        "multiple_deliveries" in d.columns
        and "Time_taken (min)" in d.columns
    ):
        ctx["avg_time_by_multiple_deliveries"] = (
            d.groupby(
                "multiple_deliveries",
                observed=True
            )["Time_taken (min)"]
            .mean()
            .round(2)
            .to_dict()
        )

    # -----------------------------------------------------
    # Ratings
    # -----------------------------------------------------

    if "Delivery_person_Ratings" in d.columns:
        ratings = pd.to_numeric(
            d["Delivery_person_Ratings"],
            errors="coerce"
        ).dropna()

        if not ratings.empty:
            ctx["rating_stats"] = {
                "avg_rating": round(
                    float(ratings.mean()),
                    2
                ),
                "min_rating": float(
                    ratings.min()
                ),
                "max_rating": float(
                    ratings.max()
                ),
            }

    # -----------------------------------------------------
    # Age
    # -----------------------------------------------------

    if "Delivery_person_Age" in d.columns:
        ages = pd.to_numeric(
            d["Delivery_person_Age"],
            errors="coerce"
        ).dropna()

        if not ages.empty:
            ctx["age_stats"] = {
                "avg_age": round(
                    float(ages.mean()),
                    1
                ),
                "min_age": float(
                    ages.min()
                ),
                "max_age": float(
                    ages.max()
                ),
            }

    # -----------------------------------------------------
    # Current filtered rows
    # -----------------------------------------------------

    ctx["rows_in_current_filter"] = int(len(d))

    return _sanitize_keys(ctx)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT_TEMPLATE = """
You are RouteBot 🤖, a friendly AI assistant inside the
RouteWise Food Delivery Analytics dashboard.

Your job has TWO modes:

=========================================================
1. NORMAL CONVERSATION
=========================================================

You are allowed to have natural conversations.

Examples:

User:
Hi

You:
Hi! 👋 I'm RouteBot. I can chat with you and also help
you understand the food-delivery analytics on this dashboard.

User:
How are you?

You:
I'm doing great! 😊 What would you like to explore?

User:
What can you do?

You:
I can chat with you, explain the dashboard, and answer
questions about delivery time, traffic, distance, weather,
cities, vehicles, ratings and other available data.

For casual conversation, you do NOT need DATA_CONTEXT.

=========================================================
2. DATA ANALYTICS
=========================================================

When the user asks about:

- delivery time
- traffic
- distance
- weather
- city
- vehicle
- order type
- festival
- ratings
- rider age
- multiple deliveries
- dashboard statistics
- comparisons
- trends
- fastest / slowest conditions
- averages
- correlations
- any dataset-related question

you MUST use DATA_CONTEXT.

IMPORTANT DATA RULE:

ONLY use numbers that are actually present in DATA_CONTEXT.

Never:
- invent numbers
- estimate numbers
- guess numbers
- create statistics yourself
- pretend a value exists when it does not

If the requested statistic is not available, say:

"I don't have that exact statistic in the current
dashboard data."

Then suggest a relevant dashboard filter or tab if possible.

=========================================================
3. FILTER AWARENESS
=========================================================

The data context represents ONLY the currently selected
filters.

Current rows:
{n_rows}

Always remember this.

Do not claim that filtered results represent the entire
dataset unless the current filters contain the entire dataset.

=========================================================
4. FOLLOW-UP QUESTIONS
=========================================================

Use the previous conversation when answering.

Example:

User:
Which traffic condition is slowest?

Assistant:
Jam traffic has the highest average delivery time
in the current filtered data.

User:
Why?

Assistant:
Because its average delivery time is higher than the
other traffic conditions in the current filtered view.

User:
And what about distance?

Assistant:
Then answer using the distance information available
in DATA_CONTEXT.

Never ask the user to repeat information that is already
available in the conversation.

=========================================================
5. NATURAL STYLE
=========================================================

Be friendly and conversational.

Use simple English.

You may use occasional emojis.

Keep answers concise.

Normal conversation:
1-4 sentences.

Data questions:
2-6 sentences.

Do not sound robotic.

Do not use unnecessary headings.

Do not mention:
- system prompts
- DATA_CONTEXT
- internal instructions
- API keys
- hidden rules
- model configuration

unless the user specifically asks about the technical
implementation.

=========================================================
6. IMPORTANT
=========================================================

If the user asks a normal conversational question:
respond naturally.

If the user asks a data question:
use DATA_CONTEXT.

If data is unavailable:
be honest instead of guessing.

DATA_CONTEXT:
{context_json}
"""


def build_system_prompt(df: pd.DataFrame) -> str:

    ctx = build_data_context(df)

    return SYSTEM_PROMPT_TEMPLATE.format(
        n_rows=ctx.get(
            "rows_in_current_filter",
            0
        ),
        context_json=json.dumps(
            ctx,
            indent=2,
            default=str
        )
    )


# =========================================================
# PROVIDERS
# =========================================================

def chat_with_groq(
    system: str,
    history: list[dict],
    model: str = "openai/gpt-oss-20b"
) -> str:

    from groq import Groq

    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set."
        )

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system
            }
        ] + history,
        max_tokens=600,
        temperature=0.4,
    )

    return response.choices[0].message.content


def chat_with_openai(
    system: str,
    history: list[dict],
    model: str = "gpt-4o-mini"
) -> str:

    from openai import OpenAI

    api_key = os.environ.get(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set."
        )

    client = OpenAI(
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system
            }
        ] + history,
        max_tokens=600,
        temperature=0.4,
    )

    return response.choices[0].message.content


def chat_with_anthropic(
    system: str,
    history: list[dict],
    model: str = "claude-sonnet-4-5-20250929"
) -> str:

    import anthropic

    api_key = os.environ.get(
        "ANTHROPIC_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable not set."
        )

    client = anthropic.Anthropic(
        api_key=api_key
    )

    response = client.messages.create(
        model=model,
        max_tokens=600,
        temperature=0.4,
        system=system,
        messages=history,
    )

    return response.content[0].text


# =========================================================
# PROVIDER MAP
# =========================================================

CHAT_PROVIDERS = {
    "Groq": chat_with_groq,
    "Claude (Anthropic)": chat_with_anthropic,
    "OpenAI": chat_with_openai,
}


# =========================================================
# FALLBACK
# =========================================================

NO_KEY_FALLBACK = (
    "⚠️ I can't connect to the selected AI provider right now "
    "because its API key is missing. Please check your "
    "Streamlit secrets/environment variables."
)


def _safe_error_text(
    error: Exception
) -> str:

    for getter in (
        lambda: str(error),
        lambda: repr(error),
        lambda: error.__class__.__name__,
    ):

        try:
            text = getter()

            if text:
                return text

        except Exception:
            continue

    return "Unknown error"


# =========================================================
# MAIN CHAT FUNCTION
# =========================================================

def ask(
    question: str,
    chat_history: list[dict],
    df: pd.DataFrame,
    provider: str = "Groq"
) -> str:

    """
    Ask RouteBot a question.

    chat_history must NOT contain the new question.
    """

    fn = CHAT_PROVIDERS.get(
        provider,
        chat_with_groq
    )

    try:

        system = build_system_prompt(df)

        # Keep only valid conversation messages
        clean_history = []

        for message in chat_history:

            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content")

            if role not in (
                "user",
                "assistant"
            ):
                continue

            if not content:
                continue

            clean_history.append(
                {
                    "role": role,
                    "content": str(content)
                }
            )

        # Add current question
        messages = clean_history + [
            {
                "role": "user",
                "content": question
            }
        ]

        reply = fn(
            system,
            messages
        )

        if not reply:
            return (
                "Sorry 😕 I didn't receive a response. "
                "Please try asking again."
            )

        return str(reply).strip()

    except RuntimeError as error:

        return (
            f"{NO_KEY_FALLBACK}\n\n"
            f"_Detail: {_safe_error_text(error)}_"
        )

    except Exception as error:

        error_text = _safe_error_text(error)

        return (
            f"⚠️ Sorry, I couldn't reach **{provider}** "
            "right now.\n\n"
            f"_Detail: {error_text}_\n\n"
            "Please check the API key, model availability, "
            "internet connection, or try another provider."
        )
```
