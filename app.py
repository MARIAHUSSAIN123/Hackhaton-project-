"""
Food Delivery Analytics Challenge — Streamlit Dashboard
=========================================================
Run locally:   streamlit run app.py
Deploy free:   push this repo to GitHub -> streamlit.io/cloud -> New app
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import analysis as A
import ai_explain as AI

# --------------------------------------------------------------------------- #
# PAGE CONFIG + THEME
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="RouteWise · Delivery Analytics",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

:root{
  --bg:#0E1015; --panel:#1B1F27; --panel2:#20242E; --line:#2B303B;
  --amber:#F4A261; --teal:#2FBF9F; --red:#EF5B5B; --blue:#5B8DEF; --ink:#EAECEF; --sub:#8B93A3;
}
html, body, [class*="css"]  { font-family:'Inter', sans-serif; }

/* ---------- animated aurora background (safe: no fixed overlay blocking content) ---------- */
.stApp{
  background:
    radial-gradient(circle at 10% -8%, rgba(244,162,97,0.16) 0%, transparent 38%),
    radial-gradient(circle at 92% 8%, rgba(47,191,159,0.13) 0%, transparent 42%),
    radial-gradient(circle at 30% 105%, rgba(91,141,239,0.10) 0%, transparent 45%),
    var(--bg);
  background-size: 200% 200%, 200% 200%, 200% 200%, auto;
  animation: auroraShift 22s ease-in-out infinite;
  color:var(--ink);
}
@keyframes auroraShift{
  0%,100%{ background-position: 0% 0%, 100% 0%, 30% 100%, 0 0; }
  50%{ background-position: 15% 20%, 80% 25%, 45% 85%, 0 0; }
}

h1,h2,h3,h4{ font-family:'Space Grotesk', sans-serif !important; letter-spacing:-0.01em; }
[data-testid="stMainBlockContainer"]{ padding-top:4.5rem; position:relative; z-index:1; }

/* entrance animation */
@keyframes fadeInUp{ from{ opacity:0; transform:translateY(14px);} to{ opacity:1; transform:translateY(0);} }

/* dashed road divider — animated like a moving lane line */
.roadline{ height:0; border-top:3px dashed var(--amber); opacity:.55; margin:0.4rem 0 1.4rem 0;
  background-size: 40px 3px; animation: drive 3s linear infinite; }
@keyframes drive{ from{ background-position:0 0; } to{ background-position:-40px 0; } }

/* hero */
.hero-badge{ display:inline-block; padding:5px 14px; border-radius:999px; background:var(--panel2);
  border:1px solid var(--line); color:var(--amber); font-family:'JetBrains Mono',monospace; font-size:0.72rem;
  letter-spacing:.08em; text-transform:uppercase; box-shadow:0 0 18px rgba(244,162,97,0.25);
  animation: fadeInUp .6s ease both, glowPulse 2.4s ease-in-out infinite; }
@keyframes glowPulse{ 0%,100%{ box-shadow:0 0 12px rgba(244,162,97,0.18);} 50%{ box-shadow:0 0 26px rgba(244,162,97,0.4);} }
.hero-title{ font-size:3.1rem; font-weight:800; margin:0.5rem 0 0.2rem 0; line-height:1.05;
  background:linear-gradient(100deg, var(--ink) 30%, var(--amber) 55%, var(--teal) 75%, var(--ink) 95%);
  background-size:300% auto;
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  animation: fadeInUp .7s ease .05s both, shimmer 7s linear infinite; }
@keyframes shimmer{ to{ background-position:300% center; } }
.hero-sub{ color:var(--sub); font-size:1.05rem; max-width:660px; line-height:1.55;
  animation: fadeInUp .7s ease .12s both; }

/* KPI ticket cards — glass + 3D tilt + icon chip + staggered entrance */
.kpi{ background:linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%);
  border:1px solid var(--line); border-radius:14px; padding:16px 18px 14px 18px; position:relative;
  overflow:hidden; transform-style:preserve-3d; perspective:600px;
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
  animation: fadeInUp .55s ease both; }
.kpi::before{ content:""; position:absolute; inset:0 0 auto 0; height:3px;
  background:linear-gradient(90deg, var(--accent, var(--teal)), transparent 85%); }
.kpi::after{ content:""; position:absolute; inset:0; border-radius:14px; opacity:0; pointer-events:none;
  background:radial-gradient(160px circle at 50% 0%, color-mix(in srgb, var(--accent, var(--teal)) 22%, transparent), transparent 70%);
  transition:opacity .25s ease; }
.kpi:hover{ transform:translateY(-5px) rotateX(3deg); border-color:var(--accent, var(--teal));
  box-shadow:0 14px 30px -10px rgba(0,0,0,0.55), 0 0 0 1px var(--accent, var(--teal)) inset; }
.kpi:hover::after{ opacity:1; }
.kpi-icon{ font-size:1.15rem; opacity:.9; margin-bottom:6px; display:block;
  filter:drop-shadow(0 0 6px color-mix(in srgb, var(--accent, var(--teal)) 60%, transparent)); }
.kpi-label{ color:var(--sub); font-size:0.68rem; text-transform:uppercase; letter-spacing:.07em; font-family:'JetBrains Mono',monospace; white-space:nowrap; }
.kpi-value{ font-family:'Space Grotesk',sans-serif; font-weight:700; margin-top:4px; color:var(--ink);
  display:flex; align-items:baseline; gap:4px; white-space:nowrap; }
.kpi-num{ font-size:1.55rem; }
.kpi-unit{ font-size:0.85rem; font-weight:600; color:var(--sub); }

/* insight card */
.insight{ background:linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%); border:1px solid var(--line);
  border-radius:14px; padding:20px 22px; margin-bottom:14px; transition:border-color .18s ease, transform .18s ease;
  animation: fadeInUp .5s ease both; }
.insight:hover{ border-color:var(--amber); transform:translateY(-3px) scale(1.005); }
.insight-title{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.08rem; color:var(--amber); margin-bottom:6px;}
.insight-so{ color:var(--sub); font-size:0.92rem; border-top:1px dashed var(--line); margin-top:10px; padding-top:8px;}

/* answer callout */
.answer{ background:linear-gradient(135deg, rgba(47,191,159,0.14), rgba(91,141,239,0.08));
  border:1px solid var(--teal); border-radius:12px; padding:16px 20px; font-size:1.02rem; margin:10px 0 18px 0;
  box-shadow:0 8px 24px -12px rgba(47,191,159,0.35); animation: fadeInUp .5s ease both; }

section[data-testid="stSidebar"]{ background:linear-gradient(180deg, var(--panel) 0%, #171A21 100%);
  border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"]{ background:var(--panel2) !important;
  border:1px solid var(--amber) !important; border-radius:999px !important; transition:transform .15s ease; }
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"]:hover{ transform:translateY(-1px) scale(1.03); }

.stTabs [data-baseweb="tab-list"]{ gap: 6px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ background:var(--panel); border-radius:10px 10px 0 0; padding:10px 18px;
  color:var(--sub); transition:color .2s ease, background .2s ease, transform .2s ease; }
.stTabs [data-baseweb="tab"]:hover{ color:var(--ink); background:var(--panel2); transform:translateY(-2px); }
.stTabs [aria-selected="true"]{ color:var(--amber) !important; background:var(--panel2) !important;
  box-shadow:inset 0 -3px 0 var(--amber); animation: tabPulse .4s ease; }
@keyframes tabPulse{ from{ box-shadow:inset 0 -3px 0 transparent; } to{ box-shadow:inset 0 -3px 0 var(--amber); } }

/* buttons */
.stButton>button, .stDownloadButton>button{ border-radius:999px !important; border:1px solid var(--amber) !important;
  color:var(--amber) !important; background:transparent !important; transition:.2s ease !important; }
.stButton>button:hover, .stDownloadButton>button:hover{ background:var(--amber) !important; color:#12151B !important;
  transform:translateY(-2px) scale(1.02); box-shadow:0 8px 20px -6px rgba(244,162,97,0.5) !important; }

footer, #MainMenu {visibility:hidden;}

/* AI explanation card */
.ai-card{ background:linear-gradient(160deg, rgba(47,191,159,0.10), rgba(91,141,239,0.05) 60%), var(--panel);
  border:1px solid var(--teal); border-radius:16px; padding:22px 26px; margin:14px 0 18px 0; position:relative;
  box-shadow:0 12px 32px -14px rgba(47,191,159,0.4); animation: fadeInUp .5s ease both; }
.ai-card-head{ display:flex; align-items:center; gap:10px; margin-bottom:14px; }
.ai-avatar{ width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:1.2rem; background:linear-gradient(135deg, var(--teal), var(--blue)); box-shadow:0 0 16px rgba(47,191,159,0.5); }
.ai-live{ display:inline-flex; align-items:center; gap:6px; font-family:'JetBrains Mono',monospace; font-size:0.7rem;
  text-transform:uppercase; letter-spacing:.08em; color:var(--teal); }
.ai-dot{ width:7px; height:7px; border-radius:50%; background:var(--teal); box-shadow:0 0 8px var(--teal);
  animation: dotPulse 1.4s ease-in-out infinite; }
@keyframes dotPulse{ 0%,100%{ opacity:1; transform:scale(1);} 50%{ opacity:.4; transform:scale(0.7);} }
.ai-provider{ color:var(--sub); font-size:0.8rem; margin-left:auto; font-family:'JetBrains Mono',monospace; }
.ai-body p{ line-height:1.7; font-size:1.0rem; color:var(--ink); margin:0 0 12px 0;
  animation: fadeInUp .45s ease both; opacity:0; }
.ai-body p:last-child{ margin-bottom:0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor="#1B1F27", plot_bgcolor="#1B1F27",
        font=dict(color="#EAECEF", family="Inter"),
        xaxis=dict(gridcolor="#2B303B", zerolinecolor="#2B303B"),
        yaxis=dict(gridcolor="#2B303B", zerolinecolor="#2B303B"),
        colorway=["#2FBF9F", "#F4A261", "#5B8DEF", "#EF5B5B", "#B497D6", "#7FD1B9"],
    )
)

TRAFFIC_COLOR = {"Jam": "#EF5B5B", "High": "#F4A261", "Medium": "#2FBF9F", "Low": "#5B8DEF"}

# --------------------------------------------------------------------------- #
# DATA LOADING (cached)
# --------------------------------------------------------------------------- #
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "food_delivery_dataset.csv")


@st.cache_data(show_spinner="Loading and cleaning delivery records...")
def get_data():
    raw = A.load_data(DATA_PATH)
    overview = A.data_overview(raw)
    clean, log = A.clean_data(raw)
    return raw, overview, clean, log


raw_df, overview, df, clean_log = get_data()

# --------------------------------------------------------------------------- #
# SIDEBAR — FILTERS
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🛵 RouteWise")
    st.caption("Food Delivery Analytics Challenge")
    st.markdown('<div class="roadline"></div>', unsafe_allow_html=True)
    st.markdown("**Filters**")

    cities = st.multiselect("City type", sorted(df["City"].unique()), default=list(df["City"].unique()))
    weathers = st.multiselect("Weather", sorted(df["Weather_conditions"].unique()), default=list(df["Weather_conditions"].unique()))
    traffics = st.multiselect("Traffic density", sorted(df["Road_traffic_density"].unique()), default=list(df["Road_traffic_density"].unique()))
    festival = st.selectbox("Festival day?", ["All", "Yes", "No"], index=0)
    dist_range = st.slider("Distance (km)", float(df["distance_km"].min()), float(df["distance_km"].max()),
                            (float(df["distance_km"].min()), float(df["distance_km"].max())))

    st.markdown('<div class="roadline"></div>', unsafe_allow_html=True)
    st.markdown("**AI explanation provider**")
    provider = st.selectbox("Provider", list(AI.PROVIDERS.keys()), index=0, label_visibility="collapsed")
    st.caption("Reads the matching API key from environment variables — never hard-coded.")

# apply filters
f = df[
    df["City"].isin(cities) & df["Weather_conditions"].isin(weathers) &
    df["Road_traffic_density"].isin(traffics) &
    df["distance_km"].between(dist_range[0], dist_range[1])
]
if festival != "All":
    f = f[f["Festival"] == festival]
if len(f) == 0:
    st.warning("No rows match these filters — widen your selection.")
    st.stop()

# --------------------------------------------------------------------------- #
# HERO
# --------------------------------------------------------------------------- #
st.markdown('<span class="hero-badge">Hackathon Task A · AI &amp; DS</span>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Food Delivery Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">38,964 delivery records, cleaned and analyzed with Pandas — '
    'turning raw trip data into decisions a delivery business can act on today.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="roadline"></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# KPI ROW (Task C — recalculated live on the filtered slice)
# --------------------------------------------------------------------------- #
stats = A.basic_stats(f.copy())
kpi_defs = [
    ("📦", "Deliveries", f"{stats['total_deliveries']:,}", "", "var(--teal)"),
    ("⏱️", "Avg time", f"{stats['avg_delivery_time_min']}", "min", "var(--amber)"),
    ("📏", "Avg distance", f"{stats['avg_distance_km']}", "km", "var(--blue)"),
    ("⚡", "Avg speed", f"{stats['avg_speed_kmph']}", "km/h", "var(--red)"),
    ("⭐", "Avg rating", f"{stats['avg_rating']}", "★", "var(--teal)"),
    ("🧑", "Avg rider age", f"{stats['avg_age']}", "yrs", "var(--amber)"),
]
cols = st.columns(6)
for i, (c, (icon, label, num, unit, accent)) in enumerate(zip(cols, kpi_defs)):
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    c.markdown(
        f'<div class="kpi" style="--accent:{accent}; animation-delay:{i*0.07:.2f}s">'
        f'<span class="kpi-icon">{icon}</span>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value"><span class="kpi-num">{num}</span>{unit_html}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------------------------------- #
# TABS
# --------------------------------------------------------------------------- #
tab_overview, tab_q1, tab_q2, tab_q3, tab_insights, tab_ai = st.tabs(
    ["📋 Data & Cleaning", "🚦 Traffic Impact", "📏 Distance Impact",
     "🌦️ Combined Conditions", "💡 Business Insights", "🤖 AI Explanation"]
)

# ---- TAB: Overview & cleaning -----------------------------------------------
with tab_overview:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Dataset overview")
        st.write(f"**{overview['n_rows']:,} rows × {overview['n_cols']} columns**, "
                 f"**{overview['duplicate_rows']}** duplicate rows found.")
        dtype_df = pd.DataFrame({
            "column": overview["columns"],
            "dtype": [overview["dtypes"][c] for c in overview["columns"]],
            "missing": [overview["missing_values"][c] for c in overview["columns"]],
            "missing %": [overview["missing_pct"][c] for c in overview["columns"]],
        })
        st.dataframe(dtype_df, height=380, width='stretch')

    with c2:
        st.subheader("Cleaning decisions")
        st.markdown(f"""
- **Text columns** stripped of stray whitespace (e.g. `"Jam "` → `"Jam"`).
- **Duplicates dropped:** {clean_log['duplicates_dropped']}
- **Age** missing ({clean_log['age_missing_filled']} rows) → filled with median = **{clean_log['age_fill_value']}**
- **Rating** missing ({clean_log['rating_missing_filled']} rows) → filled with median = **{clean_log['rating_fill_value']}**
- **Order time** stored in two formats in the raw file (clock strings *and*
  Excel fraction-of-day floats) — both are now parsed correctly.
- **Time_Orderd** still missing for {clean_log['time_orderd_missing_left_as_na']} rows
  → left blank rather than guessed, and excluded from time-of-day calculations.
- **Impossible rows** (distance ≤ 0 or time ≤ 0) dropped: {clean_log['impossible_rows_dropped']}
- Categorical columns (`Weather_conditions`, `Road_traffic_density`, `City`, …) cast to `category` dtype.

Full rationale is in **README.md**.
        """)
    st.subheader("Preview of cleaned data")
    st.dataframe(f.head(20), width='stretch')

# ---- TAB: Q1 traffic --------------------------------------------------------
with tab_q1:
    st.subheader("Q1 — Which road traffic condition has the highest average delivery time?")
    q1 = A.q1_traffic_impact(f)
    worst, worst_v = q1.index[0], q1.iloc[0]
    st.markdown(f'<div class="answer">🚦 <b>{worst}</b> traffic has the highest average delivery time at '
                f'<b>{worst_v} minutes</b>, computed live from the current filter selection.</div>',
                unsafe_allow_html=True)
    fig = px.bar(q1, x=q1.index, y=q1.values, template=PLOTLY_TEMPLATE,
                 labels={"x": "Road traffic density", "y": "Avg delivery time (min)"},
                 color=q1.index, color_discrete_map=TRAFFIC_COLOR, text=q1.values)
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", showlegend=False)
    fig.update_layout(title="Average delivery time by traffic density", height=460)
    st.plotly_chart(fig, width='stretch')
    st.download_button("⬇ Download this chart as HTML", fig.to_html(), "chart1_traffic.html")

# ---- TAB: Q2 distance --------------------------------------------------------
with tab_q2:
    st.subheader("Q2 — How does delivery distance affect delivery time?")
    q2 = A.q2_distance_impact(f)
    st.markdown(f'<div class="answer">📏 Correlation between distance and delivery time = '
                f'<b>{q2["correlation"]}</b> — delivery time <b>increases</b> as distance increases, '
                f'most sharply up to ~10-15 km before leveling off.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sample = f.sample(min(4000, len(f)), random_state=42)
        fig2 = px.scatter(sample, x="distance_km", y="Time_taken (min)", template=PLOTLY_TEMPLATE,
                           opacity=0.35, trendline="ols", trendline_color_override="#EF5B5B",
                           labels={"distance_km": "Distance (km)", "Time_taken (min)": "Delivery time (min)"})
        fig2.update_traces(marker=dict(color="#2FBF9F", size=5))
        fig2.update_layout(title="Distance vs. delivery time", height=430)
        st.plotly_chart(fig2, width='stretch')
    with c2:
        bucket = q2["avg_time_by_distance_bucket"]
        fig3 = px.bar(bucket, x=bucket.index, y=bucket.values, template=PLOTLY_TEMPLATE,
                      labels={"x": "Distance bucket", "y": "Avg delivery time (min)"}, text=bucket.values)
        fig3.update_traces(marker_color="#5B8DEF", texttemplate="%{text:.1f}", textposition="outside")
        fig3.update_layout(title="Average time by distance bucket", height=430)
        st.plotly_chart(fig3, width='stretch')

# ---- TAB: Q3 combined --------------------------------------------------------
with tab_q3:
    st.subheader("Q3 — Which weather × traffic combination is slowest?")
    pivot = f.groupby(["Weather_conditions", "Road_traffic_density"], observed=True)["Time_taken (min)"] \
             .mean().round(2).unstack()
    top = A.q3_combined_conditions(f, top_n=1)
    (w, t), v = top.index[0], top.iloc[0]
    st.markdown(f'<div class="answer">🌦️ <b>{w} weather + {t} traffic</b> is the slowest combination '
                f'at <b>{v} minutes</b> average delivery time.</div>', unsafe_allow_html=True)
    fig4 = px.imshow(pivot, text_auto=".1f", color_continuous_scale="Sunsetdark",
                      labels=dict(color="Avg min"), aspect="auto")
    fig4.update_layout(template=PLOTLY_TEMPLATE, title="Average delivery time (min): weather × traffic", height=480)
    st.plotly_chart(fig4, width='stretch')
    st.caption("Top 5 slowest combinations")
    st.dataframe(A.q3_combined_conditions(f, top_n=5).rename("avg_minutes").reset_index(), width='stretch')

# ---- TAB: Business insights --------------------------------------------------
with tab_insights:
    st.subheader("💡 Business insights")
    insights = A.business_insights(f.copy())
    for i, ins in enumerate(insights):
        st.markdown(f"""
<div class="insight" style="animation-delay:{i*0.08:.2f}s">
  <div class="insight-title">{ins['title']}</div>
  <div>{ins['finding']}</div>
  <div class="insight-so"><b>So what:</b> {ins['so_what']}</div>
</div>
        """, unsafe_allow_html=True)

# ---- TAB: AI explanation ------------------------------------------------------
with tab_ai:
    st.subheader("🤖 AI-generated business explanation")
    st.caption("Python/Pandas already computed every number below — the model only turns them into plain-English narrative.")
    if st.button("✨ Generate explanation", type="primary"):
        insights = A.business_insights(f.copy())
        q1d, q2d, q3d = A.q1_traffic_impact(f).to_dict(), A.q2_distance_impact(f), A.q3_combined_conditions(f).to_dict()
        q2_payload = {"correlation": q2d["correlation"],
                      "avg_time_by_distance_bucket": q2d["avg_time_by_distance_bucket"].to_dict()}
        try:
            with st.spinner(f"Asking {provider}..."):
                text = AI.generate_explanation(stats, q1d, q2_payload, q3d, insights, provider=provider)
            paragraphs = [p.strip() for p in text.strip().split("\n") if p.strip()]
            paras_html = "".join(
                f'<p style="animation-delay:{i*0.12:.2f}s">{p}</p>' for i, p in enumerate(paragraphs)
            )
            st.markdown(f"""
<div class="ai-card">
  <div class="ai-card-head">
    <div class="ai-avatar">🤖</div>
    <span class="ai-live"><span class="ai-dot"></span>Live model response</span>
    <span class="ai-provider">{provider}</span>
  </div>
  <div class="ai-body">{paras_html}</div>
</div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(AI.FALLBACK_EXPLANATION_NOTE)
            st.code(str(e))
    else:
        st.info("Click the button to call the LLM API with the computed statistics.")

st.markdown('<div class="roadline"></div>', unsafe_allow_html=True)
st.caption("Built with Pandas, Plotly & Streamlit · Food Delivery Analytics Challenge")
