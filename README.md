# 🛵 Food Delivery Analytics Challenge

Hackathon Task A — AI & DS (Beginner–Intermediate) submission.
Analyzes 38,964 food-delivery records with **Python & Pandas only** (no ML)
to help a delivery company understand and improve delivery performance.

## What's in this repo

| File | Purpose |
|---|---|
| `Food_Delivery_Analysis.ipynb` | The main notebook deliverable — run on **Google Colab** or Jupyter. Load → Clean → Analyze → Visualize → Interpret → Explain, all in one self-contained notebook. |
| `app.py` | Polished, interactive **Streamlit dashboard** for the live demo (filters, KPIs, Plotly charts, AI explanation button). |
| `analysis.py` | Shared analysis engine (loading, cleaning, the 3 questions, charts, insights) reused by the dashboard. |
| `ai_explain.py` | Task G — sends the *already-computed* numbers to an LLM (Claude / OpenAI / Groq) for a plain-English explanation. |
| `data/food_delivery_dataset.csv` | The provided dataset. |
| `charts/` | The two required generated charts (PNG). |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Template for API keys — **copy to `.env`**, never commit real keys. |

## How to run

### Option 1 — Notebook (Colab)
1. Open [Google Colab](https://colab.research.google.com) → Upload notebook → select `Food_Delivery_Analysis.ipynb`.
2. Upload `food_delivery_dataset.csv` when prompted (see the "Get the dataset" cell), or place it next to the notebook.
3. Run all cells. The AI-explanation cell will ask for an API key with a hidden `getpass` prompt if one isn't already set as an environment variable — no key is ever hard-coded in the notebook.

### Option 2 — Streamlit dashboard (for the demo)
```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in ONE api key
streamlit run app.py
```
Open the local URL Streamlit prints. Use the sidebar to filter by city, weather,
traffic, festival day, and distance — every KPI, chart, and insight recalculates live.

### Deploying the dashboard
- **Streamlit Community Cloud** (recommended, free, made for this): push this
  folder to a public GitHub repo → [share.streamlit.io](https://share.streamlit.io) →
  "New app" → point at `app.py` → add your API key under *Settings → Secrets*
  as `ANTHROPIC_API_KEY = "sk-..."` (or `OPENAI_API_KEY` / `GROQ_API_KEY`).
- **Vercel**: Vercel is built for Next.js/serverless JS apps, not long-running
  Python/Streamlit servers, so it isn't a natural fit for this dashboard as-is.
  If you specifically want a Vercel deployment, the clean path is a small
  **FastAPI backend** (wrapping `analysis.py`) deployed as a Vercel serverless
  function, with a **Next.js/React frontend** calling it — happy to build that
  version too if you want a JS/Vercel-native stack instead of Streamlit.

## Cleaning decisions (Task B)

| Issue found | Decision | Reasoning |
|---|---|---|
| Whitespace in text columns (e.g. `"Jam "`) | Stripped | So `"Jam"` and `"Jam "` aren't treated as different categories |
| 0 duplicate rows | N/A (checked, none found) | Pipeline still checks in case the CSV is refreshed |
| `Delivery_person_Age` missing (1,019 rows) | Filled with **median** (30.0) | Roughly symmetric distribution — median won't distort the average-age KPI |
| `Delivery_person_Ratings` missing (1,055 rows) | Filled with **median** (4.7) | Ratings are bounded 1–5 and left-skewed — median is safer than mean |
| `Time_Orderd` stored in **two different encodings** — normal `HH:MM` strings *and* Excel fraction-of-day floats (e.g. `0.4583333` = 11:00) | Both formats parsed into real timestamps | Without this fix, ~3,500 valid timestamps would have looked "missing" just because of encoding |
| `Time_Orderd` still missing after parsing (835 rows) | Left blank (`NaT`), excluded from time-of-day calculations | A genuinely missing timestamp shouldn't be guessed |
| Distance ≤ 0 or delivery time ≤ 0 | Dropped (safety check) | Physically impossible values (none currently present, but the check guards against a refreshed dataset) |
| Categorical columns (`Weather_conditions`, `Road_traffic_density`, `City`, etc.) | Cast to `category` dtype | Faster `groupby`, smaller memory footprint |

## Answers to the 3 competition questions

1. **Traffic impact:** `Jam` traffic has the highest average delivery time (~31.4 min),
   vs. ~21.5 min under `Low` traffic — computed via `groupby("Road_traffic_density")`.
2. **Distance impact:** Positive correlation (~0.32) between distance and delivery
   time; average time rises from ~22 min (0–5 km) to ~30 min (15+ km), then levels off.
3. **Combined conditions:** `Fog` weather + `Jam` traffic is the slowest combination
   (~37 min average), ahead of `Cloudy` + `Jam` (~37 min).

*(Exact figures are recalculated live in both the notebook and the dashboard — nothing above is hard-coded in the code.)*

## Business insights (Task F)

1. **Jam traffic adds ~10 minutes per delivery** → pad ETAs dynamically during
   high-traffic windows; pre-position riders on jam-prone routes before the rush.
2. **Delivery time rises steadily with distance** → route longer orders to
   riders already nearby; scale the promised delivery window with distance.
3. **Fog + Jam is the worst combination** → when forecast, proactively message
   customers with realistic ETAs and temporarily boost rider incentives.
4. **Poorer vehicle condition correlates with slower deliveries** → tie vehicle
   maintenance checks to rider performance incentives.

## AI integration (Task G)

`ai_explain.py` builds a prompt containing **only the numbers Python/Pandas already
calculated** and asks an LLM (Claude, OpenAI, or Groq — swappable) to turn them into
a short business narrative. The model performs no calculation of its own. The API
key is always read from an environment variable / Streamlit secret — never hard-coded,
per the hackathon rules.

## Notes on rules compliance
- No ML model is trained anywhere in this project.
- All three competition questions are answered programmatically (`groupby` on the
  live dataframe), not hard-coded strings.
- The project runs end-to-end without manual source changes — the notebook was
  executed top-to-bottom with no errors before submission.
