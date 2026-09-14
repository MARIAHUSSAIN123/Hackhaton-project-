def chatbot_response(question: str, stats: dict, q1: dict, q2: dict, q3: dict, insights: list[dict], provider: str = "Groq") -> str:
    """
    RouteWise AI Analyst chatbot.
    Answers questions using the currently filtered dashboard data.
    """

    payload = _sanitize_keys({
        "overall_stats": stats,
        "traffic_analysis": q1,
        "distance_analysis": q2,
        "weather_traffic_analysis": q3,
        "business_insights": insights,
    })

    prompt = f"""
You are RouteWise AI Analyst, an intelligent assistant for a food delivery
analytics dashboard.

The user is asking:
"{question}"

Below is the CURRENT filtered dashboard data calculated by Python/Pandas:

{json.dumps(payload, indent=2, default=str)}

Rules:
- Answer using ONLY the data provided above.
- Do not invent numbers.
- If the exact answer cannot be determined from the provided data, say so.
- You can calculate simple comparisons or differences from the provided numbers.
- Explain answers in simple, professional English.
- Give the actual numbers when relevant.
- If the user asks for recommendations, give practical business recommendations
  based on the available data.
- Keep the answer concise, normally 2-6 sentences.
"""

    fn = PROVIDERS.get(provider, explain_with_groq)
    return fn(prompt)
