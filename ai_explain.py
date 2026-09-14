"""
ai_explain.py
-------------
RouteWise AI Explanation + AI Analyst Chatbot

This module provides:
1. AI-powered dashboard explanation
2. RouteWise AI Analyst chatbot
3. Support for Groq, Claude, and OpenAI
4. Environment variables + Streamlit Secrets support
"""

from __future__ import annotations

import os
import json


# ============================================================
# API KEY HELPER
# ============================================================

def _get_api_key(name: str):
    """
    Get API key from environment variables.
    If not available, try Streamlit secrets.
    """

    # First try environment variable
    key = os.environ.get(name)

    if key:
        return key

    # Then try Streamlit secrets
    try:
        import streamlit as st

        key = st.secrets.get(name)

        if key:
            return key

    except Exception:
        pass

    return None


# ============================================================
# DATA SANITIZER
# ============================================================

def _sanitize_keys(obj):
    """
    Convert dictionary keys into JSON-safe strings.

    Example:
    ('Fog', 'Jam') -> 'Fog + Jam'
    """

    if isinstance(obj, dict):

        cleaned = {}

        for key, value in obj.items():

            if isinstance(key, tuple):
                new_key = " + ".join(
                    str(part) for part in key
                )
            else:
                new_key = str(key)

            cleaned[new_key] = _sanitize_keys(value)

        return cleaned

    if isinstance(obj, list):
        return [
            _sanitize_keys(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return [
            _sanitize_keys(value)
            for value in obj
        ]

    return obj


# ============================================================
# AI EXPLANATION PROMPT
# ============================================================

def _build_prompt(
    stats: dict,
    q1: dict,
    q2: dict,
    q3: dict,
    insights: list[dict]
) -> str:

    payload = _sanitize_keys({

        "overall_stats": stats,

        "avg_delivery_time_by_traffic": q1,

        "distance_vs_time": q2,

        "worst_weather_traffic_combos": q3,

        "insights_already_identified": [
            item["title"]
            for item in insights
        ],

    })

    return f"""
You are a data analyst explaining results to a
non-technical food-delivery business audience.

Python and Pandas have already calculated all the numbers.

You must NOT invent any new numbers.

Use ONLY the data provided below.

DATA:

{json.dumps(payload, indent=2, default=str)}

Write:

1. A 2-3 sentence executive summary of
overall delivery performance.

2. A short explanation of what the traffic,
distance, and weather findings mean for operations.

3. Two concrete and actionable recommendations
for the food delivery business.

Keep the answer under 200 words.

Use simple professional English.

Do not use unnecessary technical jargon.
"""


# ============================================================
# GROQ
# ============================================================

def explain_with_groq(
    prompt: str,
    model: str = "openai/gpt-oss-20b"
) -> str:

    from groq import Groq

    api_key = _get_api_key(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(

        model=model,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=750,

    )

    return response.choices[0].message.content


# ============================================================
# CLAUDE / ANTHROPIC
# ============================================================

def explain_with_anthropic(
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929"
) -> str:

    import anthropic

    api_key = _get_api_key(
        "ANTHROPIC_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured."
        )

    client = anthropic.Anthropic(
        api_key=api_key
    )

    response = client.messages.create(

        model=model,

        max_tokens=750,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

    )

    return response.content[0].text


# ============================================================
# OPENAI
# ============================================================

def explain_with_openai(
    prompt: str,
    model: str = "gpt-4o-mini"
) -> str:

    from openai import OpenAI

    api_key = _get_api_key(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key
    )

    response = client.chat.completions.create(

        model=model,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=750,

    )

    return response.choices[0].message.content


# ============================================================
# PROVIDERS
# ============================================================

# IMPORTANT:
# app.py uses this dictionary.
# Do NOT remove or rename it.

PROVIDERS = {

    "Groq": explain_with_groq,

    "Claude (Anthropic)": explain_with_anthropic,

    "OpenAI": explain_with_openai,

}


# ============================================================
# GENERATE AI EXPLANATION
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

        insights,

    )

    fn = PROVIDERS.get(

        provider,

        explain_with_groq

    )

    return fn(prompt)


# ============================================================
# ROUTEWISE AI CHATBOT
# ============================================================

def chatbot_response(
    question: str,
    stats: dict,
    q1: dict,
    q2: dict,
    q3: dict,
    insights: list[dict],
    provider: str = "Groq"
) -> str:

    """
    RouteWise AI Analyst chatbot.

    Answers questions using the CURRENT
    filtered dashboard data.
    """

    # --------------------------------------------------------
    # Prepare dashboard data
    # --------------------------------------------------------

    payload = _sanitize_keys({

        "overall_stats": stats,

        "traffic_analysis": q1,

        "distance_analysis": q2,

        "weather_traffic_analysis": q3,

        "business_insights": insights,

    })


    # --------------------------------------------------------
    # Build chatbot prompt
    # --------------------------------------------------------

    prompt = f"""

You are RouteWise AI Analyst.

You are an intelligent AI assistant inside
a food delivery analytics dashboard.

The user is asking:

"{question}"


CURRENT FILTERED DASHBOARD DATA:

{json.dumps(payload, indent=2, default=str)}


IMPORTANT RULES:

1. Answer ONLY using the data provided above.

2. Do NOT invent numbers.

3. Do NOT make up information.

4. If the exact answer cannot be determined
   from the provided data, clearly say:

   "I don't have enough data in the dashboard
   to answer that exactly."

5. You may perform simple calculations such as:

   - differences
   - comparisons
   - percentages
   - averages

   ONLY when the required numbers are
   available in the provided data.

6. Always mention actual numbers when
   they are relevant.

7. Explain the answer in simple,
   professional English.

8. If the user asks for recommendations,
   give practical business recommendations
   based ONLY on the available dashboard data.

9. Keep the answer concise.

10. Normally answer in 2-6 sentences.

11. Do not mention JSON.

12. Do not say that you are an AI model
    unless the user specifically asks.

13. Do not invent customer information,
    restaurant information, driver information,
    or business information that is not present
    in the dashboard data.

14. Focus on food delivery analytics,
    delivery time, distance, traffic,
    weather, ratings, and business insights.


Now answer the user's question.
"""


    # --------------------------------------------------------
    # Select provider
    # --------------------------------------------------------

    fn = PROVIDERS.get(

        provider,

        explain_with_groq

    )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    return fn(prompt)


# ============================================================
# FALLBACK MESSAGE
# ============================================================

FALLBACK_EXPLANATION_NOTE = (

    "⚠️ No API key found in the environment, "
    "so this is a template placeholder, not a real "
    "AI response. Set GROQ_API_KEY, "
    "ANTHROPIC_API_KEY, or OPENAI_API_KEY "
    "as an environment variable or Streamlit secret "
    "and rerun the app."

)
