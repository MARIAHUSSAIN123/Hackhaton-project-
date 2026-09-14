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
import chatbot as CB

# short "new message" notification sound (self-contained, base64 wav)
_DING_B64 = "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"

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

/* ---------- chat tab: nicer bubbles ---------- */
[data-testid="stChatMessage"]{
  background:linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%);
  border:1px solid var(--line); border-radius:16px; padding:6px 6px;
  margin-bottom:10px; animation: fadeInUp .35s ease both;
  box-shadow:0 6px 18px -10px rgba(0,0,0,0.5);
}
[data-testid="stChatMessage"]:has(img[alt="🧑"]){ border-color: var(--blue); }
[data-testid="stChatMessage"]:has(img[alt="🤖"]){ border-color: var(--teal); }
[data-testid="stChatInput"] textarea{
  background:var(--panel2) !important; border:1px solid var(--line) !important;
  border-radius:14px !important; color:var(--ink) !important;
}
[data-testid="stChatInput"]{ border-radius:14px !important; }
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
tab_overview, tab_q1, tab_q2, tab_q3, tab_insights, tab_ai, tab_chat = st.tabs(
    ["📋 Data & Cleaning", "🚦 Traffic Impact", "📏 Distance Impact",
     "🌦️ Combined Conditions", "💡 Business Insights", "🤖 AI Explanation", "💬 Chatbot"]
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

# ---- TAB: Chatbot --------------------------------------------------------
with tab_chat:
    st.subheader("💬 RouteBot — ask questions about this data")
    st.caption(
        "Every reply is grounded in real numbers computed live with Pandas from the "
        "currently filtered data (see the sidebar) — the model explains those numbers, "
        "it never invents its own."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role": ..., "content": ...}

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    # replay previous turns
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant",
                              avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    example_qs = [
        "Which traffic condition is slowest?",
        "How does distance affect delivery time?",
        "Does festival season slow deliveries down?",
    ]
    st.caption("Try: " + " · ".join(f"*{q}*" for q in example_qs))

    user_q = st.chat_input("Ask about traffic, distance, weather, city, riders…")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_q)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(f"Asking {provider}..."):
                # CB.ask() never raises -- it returns a friendly message on any
                # failure (missing key, network issue, bad SDK response, etc.)
                reply = CB.ask(
                    user_q,
                    st.session_state.chat_history[:-1],  # history before this question
                    f,  # currently filtered dataframe
                    provider=provider,
                )
            st.markdown(reply)
            # play a short notification sound when the bot's reply lands
            st.markdown(
                f'<audio autoplay style="display:none"><source src="data:audio/wav;base64,{_DING_B64}" type="audio/wav"></audio>',
                unsafe_allow_html=True,
            )
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

st.markdown('<div class="roadline"></div>', unsafe_allow_html=True)
st.caption("Built with Pandas, Plotly & Streamlit · Food Delivery Analytics Challenge")
