"""
ai_explain.py
-------------
Task G: "AI-Powered Explanation".

Python/Pandas has ALREADY done the real analysis (see analysis.py). This
module's only job is to hand the already-calculated numbers to an LLM and
ask it to write a short, plain-English business explanation. The AI never
computes anything itself — it only explains numbers we give it.

Supports Anthropic (Claude), OpenAI, and Groq — pick whichever key you have.
The API key is ALWAYS read from an environment variable / Streamlit secret,
never hard-coded, per the hackathon rules.
"""

from __future__ import annotations
import os
import json


def _build_prompt(stats: dict, q1: dict, q2: dict, q3: dict, insights: list[dict]) -> str:
    payload = {
        "overall_stats": stats,
        "avg_delivery_time_by_traffic": q1,
        "distance_vs_time": q2,
        "worst_weather_traffic_combos": q3,
        "insights_already_identified": [i["title"] for i in insights],
    }
    return f"""You are a data analyst explaining results to a non-technical food-delivery
business audience. Below is JSON containing numbers ALREADY calculated with
Python/Pandas. Do not invent any new numbers — only explain the ones given.

DATA:
{json.dumps(payload, indent=2, default=str)}

Write:
1. A 2-3 sentence executive summary of overall delivery performance.
2. A short explanation (3-5 sentences) of what the traffic, distance, and
   weather findings mean for operations.
3. Two concrete, actionable recommendations for the business.

Keep it under 200 words, plain English, no jargon, no markdown headers."""


def explain_with_anthropic(prompt: str, model: str = "claude-sonnet-4-5-20250929") -> str:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def explain_with_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return resp.choices[0].message.content


def explain_with_groq(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return resp.choices[0].message.content


PROVIDERS = {
    "Groq": explain_with_groq,
    "Claude (Anthropic)": explain_with_anthropic,
    "OpenAI": explain_with_openai,
}


def generate_explanation(stats, q1, q2, q3, insights, provider: str = "Groq") -> str:
    """Main entry point used by both the notebook and the Streamlit app."""
    prompt = _build_prompt(stats, q1, q2, q3, insights)
    fn = PROVIDERS.get(provider, explain_with_anthropic)
    return fn(prompt)


FALLBACK_EXPLANATION_NOTE = (
    "⚠️ No API key found in the environment, so this is a template placeholder, "
    "not a real AI response. Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY "
    "as an environment variable (or Streamlit secret) and rerun to get a live, "
    "model-generated explanation."
)
