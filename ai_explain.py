"""
ai_explain.py
-------------
Task G: "AI-Powered Explanation".

Python/Pandas has ALREADY done the real analysis (see analysis.py).
This module's job is to hand the already-calculated numbers to an LLM
and ask it to explain the results.

It also provides a chatbot for the RouteWise food-delivery analytics app.

Supports Anthropic (Claude), OpenAI, and Groq.
API keys are read from environment variables / Streamlit secrets.
"""

from __future__ import annotations

import os
import json


# ============================================================
# API KEY HELPER
# ============================================================

def _get_secret(name: str):
    """
    Read API key from:
    1. Environment variable
    2. Streamlit secrets
    """

    value = os.environ.get(name)

    if value:
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]

    except Exception:
        pass

    return None


# ============================================================
# SANITIZE DATA
# ============================================================

def _sanitize_keys(obj):
    """
    Recursively convert non-JSON-safe dictionary keys into strings.
    """

    if isinstance(obj, dict):
        return {
            (
                " + ".join(str(part) for part in k)
                if isinstance(k, tuple)
                else str(k)
            ): _sanitize_keys(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [_sanitize_keys(v) for v in obj]

    return obj


# ============================================================
# BUILD ANALYSIS PROMPT
# ============================================================

def _build_prompt(
    stats: dict,
    q1: dict,
    q2: dict,
    q3: dict,
    insights: list[dict],
) -> str:

    payload = _sanitize_keys({
        "overall_stats": stats,
        "avg_delivery_time_by_traffic": q1,
        "distance_vs_time": q2,
        "worst_weather_traffic_combos": q3,
        "insights_already_identified": [
            i["title"] for i in insights
        ],
    })

    return f"""
You are a data analyst explaining results to a
non-technical food-delivery business audience.

Below is JSON containing numbers ALREADY calculated
with Python/Pandas.

Do not invent new numbers.
Only explain the numbers provided.

DATA:

{json.dumps(payload, indent=2, default=str)}

Write:

1. A 2-3 sentence executive summary of overall
   delivery performance.

2. A short explanation of what the traffic,
   distance, and weather findings mean for operations.

3. Two concrete and actionable recommendations
   for the business.

Keep it under 200 words.
Use plain English.
No complicated jargon.
"""


# ============================================================
# ANTHROPIC
# ============================================================

def explain_with_anthropic(
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929"
) -> str:

    import anthropic

    api_key = _get_secret("ANTHROPIC_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured."
        )

    client = anthropic.Anthropic(
        api_key=api_key
    )

    resp = client.messages.create(
        model=model,
        max_tokens=750,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return resp.content[0].text


# ============================================================
# OPENAI
# ============================================================

def explain_with_openai(
    prompt: str,
    model: str = "gpt-4o-mini"
) -> str:

    from openai import OpenAI

    api_key = _get_secret("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=750,
    )

    return resp.choices[0].message.content


# ============================================================
# GROQ
# ============================================================

def explain_with_groq(
    prompt: str,
    model: str = "openai/gpt-oss-20b"
) -> str:

    from groq import Groq

    api_key = _get_secret("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    client = Groq(
        api_key=api_key
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_completion_tokens=1000,
        temperature=0.4,
    )

    return resp.choices[0].message.content


# ============================================================
# PROVIDERS
# ============================================================

PROVIDERS = {
    "Groq": explain_with_groq,
    "Claude (Anthropic)": explain_with_anthropic,
    "OpenAI": explain_with_openai,
}


# ============================================================
# AI EXPLANATION
# ============================================================

def generate_explanation(
    stats,
    q1,
    q2,
    q3,
    insights,
    provider: str = "Groq"
) -> str:

    prompt = _build_prompt(
        stats,
        q1,
        q2,
        q3,
        insights
    )

    fn = PROVIDERS.get(
        provider,
        explain_with_groq
    )

    return fn(prompt)


# ============================================================
# CHATBOT
# ============================================================

def chatbot_response(
    user_message: str,
    stats=None,
    q1=None,
    q2=None,
    q3=None,
    insights=None,
    chat_history=None,
    provider: str = "Groq",
) -> str:
    """
    Generate a chatbot response for the RouteWise dashboard.

    The chatbot can answer:
    - General questions
    - Questions about the dashboard
    - Questions about delivery performance
    - Questions about traffic
    - Questions about distance
    - Questions about weather
    - Questions about business insights
    """

    if not user_message or not user_message.strip():
        return "Please type a question."

    # --------------------------------------------------------
    # Prepare analytics context
    # --------------------------------------------------------

    analytics_context = _sanitize_keys({
        "overall_stats": stats or {},
        "traffic_analysis": q1 or {},
        "distance_analysis": q2 or {},
        "weather_traffic_analysis": q3 or {},
        "business_insights": insights or [],
    })

    # --------------------------------------------------------
    # System instructions
    # --------------------------------------------------------

    system_prompt = """
You are RouteWise AI Assistant.

RouteWise is a food-delivery analytics dashboard.

Your job is to help the user understand the dashboard
and its food-delivery data.

You can answer questions about:

- Delivery time
- Traffic
- Distance
- Weather
- Ratings
- Delivery performance
- Business insights
- Operational recommendations

IMPORTANT RULES:

1. Be friendly and conversational.

2. Answer the user's exact question first.

3. Use the provided RouteWise data whenever
   the question is about the dashboard.

4. NEVER invent statistics.

5. If a number is not available in the provided data,
   clearly say that the available dataset does not
   provide that information.

6. Do not pretend that you calculated something
   that is not present in the supplied analysis.

7. Keep answers concise and easy to understand.

8. You can explain technical concepts simply if
   the user asks about the dashboard.

9. If the user says hello, greet them naturally.

10. If the user asks for recommendations,
    provide practical business suggestions based
    on the available data.
"""

    # --------------------------------------------------------
    # Build messages
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # --------------------------------------------------------
    # Add previous conversation
    # --------------------------------------------------------

    if chat_history:

        for message in chat_history[-10:]:

            role = message.get("role")
            content = message.get("content")

            if role in ("user", "assistant") and content:

                messages.append({
                    "role": role,
                    "content": content
                })

    # --------------------------------------------------------
    # Current user question
    # --------------------------------------------------------

    user_prompt = f"""
Here is the current RouteWise analytics data:

{json.dumps(
    analytics_context,
    indent=2,
    default=str
)}

User question:

{user_message}

Answer the user naturally.
"""

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    if provider == "Groq":

        from groq import Groq

        api_key = _get_secret(
            "GROQ_API_KEY"
        )

        if not api_key:
            return (
                "⚠️ Groq API key is not configured. "
                "Please add GROQ_API_KEY in Streamlit "
                "Secrets."
            )

        try:

            client = Groq(
                api_key=api_key
            )

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                max_completion_tokens=1000,
                temperature=0.4,
            )

            answer = response.choices[0].message.content

            if answer:
                return answer.strip()

            return "Sorry, I couldn't generate a response."

        except Exception as e:

            return (
                "⚠️ I couldn't connect to the AI service.\n\n"
                f"Error: {str(e)}"
            )

    # --------------------------------------------------------
    # OPENAI
    # --------------------------------------------------------

    if provider == "OpenAI":

        from openai import OpenAI

        api_key = _get_secret(
            "OPENAI_API_KEY"
        )

        if not api_key:
            return (
                "⚠️ OpenAI API key is not configured."
            )

        try:

            client = OpenAI(
                api_key=api_key
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
                temperature=0.4,
            )

            answer = response.choices[0].message.content

            if answer:
                return answer.strip()

            return "Sorry, I couldn't generate a response."

        except Exception as e:

            return (
                "⚠️ I couldn't connect to OpenAI.\n\n"
                f"Error: {str(e)}"
            )

    # --------------------------------------------------------
    # ANTHROPIC
    # --------------------------------------------------------

    if provider == "Claude (Anthropic)":

        import anthropic

        api_key = _get_secret(
            "ANTHROPIC_API_KEY"
        )

        if not api_key:
            return (
                "⚠️ Anthropic API key is not configured."
            )

        try:

            client = anthropic.Anthropic(
                api_key=api_key
            )

            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
            )

            answer = response.content[0].text

            if answer:
                return answer.strip()

            return "Sorry, I couldn't generate a response."

        except Exception as e:

            return (
                "⚠️ I couldn't connect to Claude.\n\n"
                f"Error: {str(e)}"
            )

    return "Please select a valid AI provider."


# ============================================================
# FALLBACK MESSAGE
# ============================================================

FALLBACK_EXPLANATION_NOTE = (
    "⚠️ No API key found in the environment, so this "
    "is a template placeholder, not a real AI response. "
    "Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY "
    "as an environment variable or Streamlit secret "
    "and rerun to get a live model-generated explanation."
)
