"""
Food Delivery Analytics Challenge — Streamlit Dashboard
Run locally: streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import analysis as A
import ai_explain as AI


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RouteWise · Delivery Analytics",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# THEME / CSS
# ---------------------------------------------------------------------------

CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --bg: #0E1015;
    --panel: #1B1F27;
    --panel2: #20242E;
    --line: #2B303B;
    --amber: #F4A261;
    --teal: #2FBF9F;
    --red: #EF5B5B;
    --blue: #5B8DEF;
    --ink: #EAECEF;
    --sub: #8B93A3;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% -8%, rgba(244,162,97,0.16) 0%, transparent 38%),
        radial-gradient(circle at 92% 8%, rgba(47,191,159,0.13) 0%, transparent 42%),
        radial-gradient(circle at 30% 105%, rgba(91,141,239,0.10) 0%, transparent 45%),
        var(--bg);
    color: var(--ink);
}

h1, h2, h3, h4 {
    font-family: "Space Grotesk", sans-serif !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 4.5rem;
    position: relative;
    z-index: 1;
}

.roadline {
    height: 0;
    border-top: 3px dashed var(--amber);
    opacity: 0.55;
    margin: 0.4rem 0 1.4rem 0;
}

.hero-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    background: var(--panel2);
    border: 1px solid var(--line);
    color: var(--amber);
    font-family: "JetBrains Mono", monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero-title {
    font-size: 3.1rem;
    font-weight: 800;
    margin: 0.5rem 0 0.2rem 0;
    line-height: 1.05;
    background: linear-gradient(
        100deg,
        var(--ink) 30%,
        var(--amber) 55%,
        var(--teal) 75%,
        var(--ink) 95%
    );
    background-size: 300% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: var(--sub);
    font-size: 1.05rem;
    max-width: 700px;
    line-height: 1.55;
}

.kpi {
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 18px 14px 18px;
    position: relative;
    overflow: hidden;
    transition: 0.25s ease;
    min-height: 125px;
}

.kpi::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: var(--accent, var(--teal));
}

.kpi:hover {
    transform: translateY(-5px);
    border-color: var(--accent, var(--teal));
    box-shadow: 0 14px 30px -10px rgba(0,0,0,0.55);
}

.kpi-icon {
    font-size: 1.15rem;
    display: block;
    margin-bottom: 6px;
}

.kpi-label {
    color: var(--sub);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-family: "JetBrains Mono", monospace;
}

.kpi-value {
    font-family: "Space Grotesk", sans-serif;
    font-weight: 700;
    margin-top: 4px;
    color: var(--ink);
    display: flex;
    align-items: baseline;
    gap: 4px;
    white-space: nowrap;
}

.kpi-num {
    font-size: 1.55rem;
}

.kpi-unit {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--sub);
}

.answer {
    background: linear-gradient(
        135deg,
        rgba(47,191,159,0.14),
        rgba(91,141,239,0.08)
    );
    border: 1px solid var(--teal);
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 1.02rem;
    margin: 10px 0 18px 0;
}

.insight {
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel2) 100%);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
}

.insight:hover {
    border-color: var(--amber);
    transform: translateY(-2px);
}

.insight-title {
    font-family: "Space Grotesk", sans-serif;
    font-weight: 700;
    font-size: 1.08rem;
    color: var(--amber);
    margin-bottom: 6px;
}

.insight-so {
    color: var(--sub);
    font-size: 0.92rem;
    border-top: 1px dashed var(--line);
    margin-top: 10px;
    padding-top: 8px;
}

.ai-card {
    background:
        linear-gradient(
            160deg,
            rgba(47,191,159,0.10),
            rgba(91,141,239,0.05) 60%
        ),
        var(--panel);
    border: 1px solid var(--teal);
    border-radius: 16px;
    padding: 22px 26px;
    margin: 14px 0 18px 0;
    box-shadow: 0 12px 32px -14px rgba(47,191,159,0.4);
}

.ai-card-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}

.ai-avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    background: linear-gradient(135deg, var(--teal), var(--blue));
}

.ai-live {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--teal);
}

.ai-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--teal);
}

.ai-provider {
    color: var(--sub);
    font-size: 0.8rem;
    margin-left: auto;
    font-family: "JetBrains Mono", monospace;
}

.chat-header {
    background:
        linear-gradient(
            135deg,
            rgba(47,191,159,.14),
            rgba(91,141,239,.12)
        );
    border: 1px solid var(--teal);
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 18px;
}

.chat-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
}

.chat-subtitle {
    color: var(--sub);
    margin-top: 4px;
    font-size: .9rem;
}

.chat-info {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 15px;
    color: var(--sub);
    font-size: .85rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--panel) 0%, #171A21 100%);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background: var(--panel2) !important;
    border: 1px solid var(--amber) !important;
    border-radius: 999px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    background: var(--panel);
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    color: var(--sub);
}

.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    background: var(--panel2) !important;
    box-shadow: inset 0 -3px 0 var(--amber);
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 999px !important;
    border: 1px solid var(--amber) !important;
    color: var(--amber) !important;
    background: transparent !important;
    transition: 0.2s ease !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--amber) !important;
    color: #12151B !important;
}

footer, #MainMenu {
    visibility: hidden;
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 2.2rem;
    }

    .kpi {
        min-height: 110px;
    }
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PLOTLY THEME
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor="#1B1F27",
        plot_bgcolor="#1B1F27",
        font=dict(
            color="#EAECEF",
            family="Inter"
        ),
        xaxis=dict(
            gridcolor="#2B303B",
            zerolinecolor="#2B303B"
        ),
        yaxis=dict(
            gridcolor="#2B303B",
            zerolinecolor="#2B303B"
        ),
        colorway=[
            "#2FBF9F",
            "#F4A261",
            "#5B8DEF",
            "#EF5B5B",
            "#B497D6",
            "#7FD1B9"
        ],
    )
)

TRAFFIC_COLOR = {
    "Jam": "#EF5B5B",
    "High": "#F4A261",
    "Medium": "#2FBF9F",
    "Low": "#5B8DEF"
}


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "food_delivery_dataset.csv"
)


@st.cache_data(show_spinner="Loading and cleaning delivery records...")
def get_data():
    raw = A.load_data(DATA_PATH)
    overview = A.data_overview(raw)
    clean, log = A.clean_data(raw)
    return raw, overview, clean, log


raw_df, overview, df, clean_log = get_data()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:

    st.markdown("### 🛵 RouteWise")
    st.caption("Food Delivery Analytics Challenge")

    st.markdown(
        '<div class="roadline"></div>',
        unsafe_allow_html=True
    )

    st.markdown("**Filters**")

    cities = st.multiselect(
        "City type",
        sorted(df["City"].unique()),
        default=list(df["City"].unique())
    )

    weathers = st.multiselect(
        "Weather",
        sorted(df["Weather_conditions"].unique()),
        default=list(df["Weather_conditions"].unique())
    )

    traffics = st.multiselect(
        "Traffic density",
        sorted(df["Road_traffic_density"].unique()),
        default=list(df["Road_traffic_density"].unique())
    )

    festival = st.selectbox(
        "Festival day?",
        ["All", "Yes", "No"],
        index=0
    )

    dist_range = st.slider(
        "Distance (km)",
        float(df["distance_km"].min()),
        float(df["distance_km"].max()),
        (
            float(df["distance_km"].min()),
            float(df["distance_km"].max())
        )
    )

    st.markdown(
        '<div class="roadline"></div>',
        unsafe_allow_html=True
    )

    st.markdown("**AI explanation provider**")

    provider = st.selectbox(
        "Provider",
        list(AI.PROVIDERS.keys()),
        index=0,
        label_visibility="collapsed"
    )

    st.caption(
        "Reads the matching API key from environment variables "
        "or Streamlit secrets."
    )


# ---------------------------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------------------------

f = df[
    df["City"].isin(cities)
    & df["Weather_conditions"].isin(weathers)
    & df["Road_traffic_density"].isin(traffics)
    & df["distance_km"].between(
        dist_range[0],
        dist_range[1]
    )
]

if festival != "All":
    f = f[f["Festival"] == festival]


if len(f) == 0:
    st.warning("No rows match these filters — widen your selection.")
    st.stop()


# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------

st.markdown(
    '<span class="hero-badge">Hackathon Task A · AI &amp; DS</span>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Food Delivery Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-sub">
        38,964 delivery records, cleaned and analyzed with Pandas —
        turning raw trip data into decisions a delivery business can act on today.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="roadline"></div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------------

stats = A.basic_stats(f.copy())

kpi_defs = [
    (
        "📦",
        "Deliveries",
        f"{stats['total_deliveries']:,}",
        "",
        "var(--teal)"
    ),
    (
        "⏱️",
        "Avg time",
        f"{stats['avg_delivery_time_min']}",
        "min",
        "var(--amber)"
    ),
    (
        "📏",
        "Avg distance",
        f"{stats['avg_distance_km']}",
        "km",
        "var(--blue)"
    ),
    (
        "⚡",
        "Avg speed",
        f"{stats['avg_speed_kmph']}",
        "km/h",
        "var(--red)"
    ),
    (
        "⭐",
        "Avg rating",
        f"{stats['avg_rating']}",
        "★",
        "var(--teal)"
    ),
    (
        "🧑",
        "Avg rider age",
        f"{stats['avg_age']}",
        "yrs",
        "var(--amber)"
    ),
]

cols = st.columns(6)

for i, (c, item) in enumerate(zip(cols, kpi_defs)):

    icon, label, num, unit, accent = item

    unit_html = (
        f'<span class="kpi-unit">{unit}</span>'
        if unit
        else ""
    )

    c.markdown(
        f"""
        <div
            class="kpi"
            style="
                --accent:{accent};
                animation-delay:{i * 0.07:.2f}s;
            "
        >
            <span class="kpi-icon">{icon}</span>

            <div class="kpi-label">
                {label}
            </div>

            <div class="kpi-value">
                <span class="kpi-num">{num}</span>
                {unit_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

(
    tab_overview,
    tab_q1,
    tab_q2,
    tab_q3,
    tab_insights,
    tab_ai,
    tab_chat
) = st.tabs(
    [
        "📋 Data & Cleaning",
        "🚦 Traffic Impact",
        "📏 Distance Impact",
        "🌦️ Combined Conditions",
        "💡 Business Insights",
        "🤖 AI Explanation",
        "💬 RouteWise Chatbot"
    ]
)


# ---------------------------------------------------------------------------
# TAB: OVERVIEW
# ---------------------------------------------------------------------------

with tab_overview:

    c1, c2 = st.columns([1.1, 1])

    with c1:

        st.subheader("Dataset overview")

        st.write(
            f"**{overview['n_rows']:,} rows × "
            f"{overview['n_cols']} columns**, "
            f"**{overview['duplicate_rows']}** "
            f"duplicate rows found."
        )

        dtype_df = pd.DataFrame({
            "column": overview["columns"],
            "dtype": [
                overview["dtypes"][c]
                for c in overview["columns"]
            ],
            "missing": [
                overview["missing_values"][c]
                for c in overview["columns"]
            ],
            "missing %": [
                overview["missing_pct"][c]
                for c in overview["columns"]
            ],
        })

        st.dataframe(
            dtype_df,
            height=380,
            width="stretch"
        )

    with c2:

        st.subheader("Cleaning decisions")

        st.markdown(
            f"""
            - **Text columns** stripped of stray whitespace.
            - **Duplicates dropped:** {clean_log['duplicates_dropped']}
            - **Age** missing ({clean_log['age_missing_filled']} rows)
              → filled with median = **{clean_log['age_fill_value']}**
            - **Rating** missing ({clean_log['rating_missing_filled']} rows)
              → filled with median = **{clean_log['rating_fill_value']}**
            - **Order time** stored in two formats in the raw file —
              both are now parsed correctly.
            - **Time_Orderd** still missing for
              {clean_log['time_orderd_missing_left_as_na']} rows
              → left blank rather than guessed.
            - **Impossible rows** (distance ≤ 0 or time ≤ 0) dropped:
              {clean_log['impossible_rows_dropped']}
            - Categorical columns cast to `category` dtype.
            - Full rationale is in **README.md**.
            """
        )

    st.subheader("Preview of cleaned data")

    st.dataframe(
        f.head(20),
        width="stretch"
    )


# ---------------------------------------------------------------------------
# TAB: Q1 TRAFFIC
# ---------------------------------------------------------------------------

with tab_q1:

    st.subheader(
        "Q1 — Which road traffic condition has the highest "
        "average delivery time?"
    )

    q1 = A.q1_traffic_impact(f)

    worst = q1.index[0]
    worst_v = q1.iloc[0]

    st.markdown(
        f"""
        <div class="answer">
            🚦 <b>{worst}</b> traffic has the highest
            average delivery time at
            <b>{worst_v} minutes</b>,
            computed live from the current filter selection.
        </div>
        """,
        unsafe_allow_html=True
    )

    fig = px.bar(
        q1,
        x=q1.index,
        y=q1.values,
        template=PLOTLY_TEMPLATE,
        labels={
            "x": "Road traffic density",
            "y": "Avg delivery time (min)"
        },
        color=q1.index,
        color_discrete_map=TRAFFIC_COLOR,
        text=q1.values
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        showlegend=False
    )

    fig.update_layout(
        title="Average delivery time by traffic density",
        height=460
    )

    st.plotly_chart(fig, width="stretch")

    st.download_button(
        "⬇ Download this chart as HTML",
        fig.to_html(),
        "chart1_traffic.html"
    )


# ---------------------------------------------------------------------------
# TAB: Q2 DISTANCE
# ---------------------------------------------------------------------------

with tab_q2:

    st.subheader(
        "Q2 — How does delivery distance affect delivery time?"
    )

    q2 = A.q2_distance_impact(f)

    st.markdown(
        f"""
        <div class="answer">
            📏 Correlation between distance and delivery time =
            <b>{q2["correlation"]}</b>.
            Delivery time <b>increases</b> as distance increases,
            most sharply up to ~10-15 km before leveling off.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        sample = f.sample(
            min(4000, len(f)),
            random_state=42
        )

        fig2 = px.scatter(
            sample,
            x="distance_km",
            y="Time_taken (min)",
            template=PLOTLY_TEMPLATE,
            opacity=0.35,
            trendline="ols",
            trendline_color_override="#EF5B5B",
            labels={
                "distance_km": "Distance (km)",
                "Time_taken (min)": "Delivery time (min)"
            }
        )

        fig2.update_traces(
            marker=dict(
                color="#2FBF9F",
                size=5
            )
        )

        fig2.update_layout(
            title="Distance vs. delivery time",
            height=430
        )

        st.plotly_chart(fig2, width="stretch")

    with c2:

        bucket = q2["avg_time_by_distance_bucket"]

        fig3 = px.bar(
            bucket,
            x=bucket.index,
            y=bucket.values,
            template=PLOTLY_TEMPLATE,
            labels={
                "x": "Distance bucket",
                "y": "Avg delivery time (min)"
            },
            text=bucket.values
        )

        fig3.update_traces(
            marker_color="#5B8DEF",
            texttemplate="%{text:.1f}",
            textposition="outside"
        )

        fig3.update_layout(
            title="Average time by distance bucket",
            height=430
        )

        st.plotly_chart(fig3, width="stretch")


# ---------------------------------------------------------------------------
# TAB: Q3 COMBINED CONDITIONS
# ---------------------------------------------------------------------------

with tab_q3:

    st.subheader(
        "Q3 — Which weather × traffic combination is slowest?"
    )

    pivot = (
        f.groupby(
            [
                "Weather_conditions",
                "Road_traffic_density"
            ],
            observed=True
        )["Time_taken (min)"]
        .mean()
        .round(2)
        .unstack()
    )

    top = A.q3_combined_conditions(
        f,
        top_n=1
    )

    (w, t), v = top.index[0], top.iloc[0]

    st.markdown(
        f"""
        <div class="answer">
            🌦️ <b>{w} weather + {t} traffic</b>
            is the slowest combination at
            <b>{v} minutes</b> average delivery time.
        </div>
        """,
        unsafe_allow_html=True
    )

    fig4 = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale="Sunsetdark",
        labels={"color": "Avg min"},
        aspect="auto"
    )

    fig4.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Average delivery time (min): weather × traffic",
        height=480
    )

    st.plotly_chart(fig4, width="stretch")

    st.caption("Top 5 slowest combinations")

    st.dataframe(
        A.q3_combined_conditions(
            f,
            top_n=5
        )
        .rename("avg_minutes")
        .reset_index(),
        width="stretch"
    )


# ---------------------------------------------------------------------------
# TAB: BUSINESS INSIGHTS
# ---------------------------------------------------------------------------

with tab_insights:

    st.subheader("💡 Business insights")

    insights = A.business_insights(f.copy())

    for i, ins in enumerate(insights):

        st.markdown(
            f"""
            <div class="insight">
                <div class="insight-title">
                    {ins['title']}
                </div>

                <div>
                    {ins['finding']}
                </div>

                <div class="insight-so">
                    <b>So what:</b>
                    {ins['so_what']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# TAB: AI EXPLANATION
# ---------------------------------------------------------------------------

with tab_ai:

    st.subheader("🤖 AI-generated business explanation")

    st.caption(
        "Python/Pandas already computed every number below — "
        "the model only turns them into plain-English narrative."
    )

    if st.button(
        "✨ Generate explanation",
        type="primary",
        key="generate_explanation"
    ):

        insights = A.business_insights(f.copy())

        q1d = (
            A.q1_traffic_impact(f)
            .to_dict()
        )

        q2d = A.q2_distance_impact(f)

        q3d = (
            A.q3_combined_conditions(f)
            .to_dict()
        )

        q2_payload = {
            "correlation": q2d["correlation"],
            "avg_time_by_distance_bucket":
                q2d["avg_time_by_distance_bucket"].to_dict()
        }

        try:

            with st.spinner(f"Asking {provider}..."):

                text = AI.generate_explanation(
                    stats,
                    q1d,
                    q2_payload,
                    q3d,
                    insights,
                    provider=provider
                )

            paragraphs = [
                p.strip()
                for p in text.strip().split("\n")
                if p.strip()
            ]

            paras_html = "".join(
                f"<p>{p}</p>"
                for p in paragraphs
            )

            st.markdown(
                f"""
                <div class="ai-card">

                    <div class="ai-card-head">

                        <div class="ai-avatar">
                            🤖
                        </div>

                        <span class="ai-live">
                            <span class="ai-dot"></span>
                            Live model response
                        </span>

                        <span class="ai-provider">
                            {provider}
                        </span>

                    </div>

                    <div class="ai-body">
                        {paras_html}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.warning(AI.FALLBACK_EXPLANATION_NOTE)

            st.code(str(e))

    else:

        st.info(
            "Click the button to call the LLM API "
            "with the computed statistics."
        )


# ---------------------------------------------------------------------------
# TAB: ROUTEWISE AI CHATBOT
# ---------------------------------------------------------------------------

with tab_chat:

    st.markdown(
        """
        <div class="chat-header">
            <div class="chat-title">
                🤖 RouteWise AI Analyst
            </div>

            <div class="chat-subtitle">
                Your intelligent assistant for food delivery analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="chat-info">
            📊 The chatbot uses the <b>currently filtered dashboard data</b>.
            Change the filters from the sidebar and ask again for updated analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "routewise_chat" not in st.session_state:
        st.session_state.routewise_chat = []

    clear_col, status_col = st.columns([1, 4])

    with clear_col:

        if st.button(
            "🗑️ Clear Chat",
            key="clear_routewise_chat"
        ):

            st.session_state.routewise_chat = []
            st.rerun()

    with status_col:

        st.caption(
            f"📦 Analyzing {len(f):,} currently filtered deliveries"
        )

    st.markdown("### 💡 Try asking")

    examples = [
        "What is the overall delivery performance?",
        "Which traffic condition has the highest delivery time?",
        "Which weather condition is worst for deliveries?",
        "How does distance affect delivery time?",
        "Give me 3 business recommendations."
    ]

    example_cols = st.columns(5)

    for col, question in zip(example_cols, examples):

        with col:

            if st.button(
                question,
                key=f"example_{question}"
            ):

                st.session_state.routewise_question = question
                st.rerun()

    st.write("")

    # Display previous messages
    for message in st.session_state.routewise_chat:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # Chat input
    user_question = st.chat_input(
        "Ask RouteWise about your delivery data..."
    )

    # Example button question
    if "routewise_question" in st.session_state:

        user_question = st.session_state.pop(
            "routewise_question"
        )

    # Process question
    if user_question:

        st.session_state.routewise_chat.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        # Calculate current analytics
        insights = A.business_insights(f.copy())

        q1_data = (
            A.q1_traffic_impact(f)
            .to_dict()
        )

        q2_data = A.q2_distance_impact(f)

        q2_payload = {
            "correlation": q2_data["correlation"],
            "avg_time_by_distance_bucket":
                q2_data["avg_time_by_distance_bucket"].to_dict()
        }

        q3_data = (
            A.q3_combined_conditions(
                f,
                top_n=10
            )
            .to_dict()
        )

        # Ask AI
        with st.chat_message("assistant"):

            with st.spinner("🤖 RouteWise AI is analyzing..."):

                try:

                    answer = AI.chatbot_response(
                        question=user_question,
                        stats=stats,
                        q1=q1_data,
                        q2=q2_payload,
                        q3=q3_data,
                        insights=insights,
                        provider=provider
                    )

                    st.markdown(answer)

                    st.session_state.routewise_chat.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(
                        "Sorry, I couldn't connect to the AI service. "
                        "Please check your API key and try again."
                    )

                    st.code(str(e))


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="roadline"></div>',
    unsafe_allow_html=True
)

st.caption(
    "Built with Pandas, Plotly & Streamlit · "
    "Food Delivery Analytics Challenge"
)
