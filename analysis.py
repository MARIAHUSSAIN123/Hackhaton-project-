"""
analysis.py
------------
Core data-analysis engine for the Food Delivery Analytics Challenge.

This module is imported by BOTH:
  1) Food_Delivery_Analysis.ipynb  (the Colab / Jupyter notebook deliverable)
  2) app.py                       (the Streamlit dashboard)

so the two never drift apart and every number shown on the dashboard is
produced by the exact same Pandas code that the notebook demonstrates.

No hard-coded answers: every question is answered by grouping / aggregating
the actual dataframe at run time.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --------------------------------------------------------------------------- #
# 1. LOAD
# --------------------------------------------------------------------------- #

RAW_COLUMNS = [
    "ID", "Delivery_person_ID", "Delivery_person_Age", "Delivery_person_Ratings",
    "Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude",
    "Delivery_location_longitude", "Order_Date", "Time_Orderd", "Time_Order_picked",
    "Weather_conditions", "Road_traffic_density", "Vehicle_condition", "Type_of_order",
    "Type_of_vehicle", "multiple_deliveries", "Festival", "City", "Time_taken (min)",
    "distance_km", "delivery_speed",
]


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as provided."""
    df = pd.read_csv(path)
    return df


def data_overview(df: pd.DataFrame) -> dict:
    """Everything required by Task A: shape, dtypes, missing values, duplicates."""
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# --------------------------------------------------------------------------- #
# 2. CLEAN
# --------------------------------------------------------------------------- #

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Cleaning decisions (also written out in README.md):

      * Strip whitespace from every text/categorical column (source files of
        this kind commonly ship with values like "Jam " or " Fog").
      * Drop exact duplicate rows.
      * Delivery_person_Age: missing -> filled with the median age (numeric,
        roughly symmetric distribution, so median is a safe fill that will
        not distort the average-age KPI or the age based aggregates).
      * Delivery_person_Ratings: missing -> filled with the median rating
        (ratings are bounded 1-5 and left-skewed, so median beats mean).
      * Time_Orderd: missing -> left as NaT (True missing timestamps aren't
        safely imputable — we simply exclude them from any calculation that
        specifically needs the order-placed clock time).
      * Order_Date / Time_Orderd / Time_Order_picked: parsed into proper
        datetime/time dtypes so they can be used for time-based analysis.
      * Road_traffic_density, Weather_conditions: category dtype for memory
        + groupby performance; any stray blank string is treated as missing.
      * Rows with an impossible / non-physical value (negative distance,
        distance == 0, delivery time <= 0) are dropped — there are none in
        this dataset today, but the check is kept so the pipeline is safe if
        the CSV is refreshed.

    Returns the cleaned dataframe plus a dict log of what changed, so the
    notebook / README can report it transparently.
    """
    log = {}
    df = df.copy()
    log["rows_before"] = len(df)

    # --- strip whitespace on text columns -----------------------------------
    text_cols = df.select_dtypes(include="object").columns.tolist() + \
        [c for c in df.columns if str(df[c].dtype) == "string"]
    text_cols = list(dict.fromkeys(text_cols))
    for c in text_cols:
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].replace({"NaN": pd.NA, "": pd.NA, "nan": pd.NA})

    # --- drop exact duplicates ----------------------------------------------
    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log["duplicates_dropped"] = dup_count

    # --- numeric coercion -----------------------------------------------------
    numeric_cols = ["Delivery_person_Age", "Delivery_person_Ratings",
                     "Time_taken (min)", "distance_km", "multiple_deliveries",
                     "Vehicle_condition"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- fill missing age / rating with median --------------------------------
    age_missing = int(df["Delivery_person_Age"].isna().sum())
    rating_missing = int(df["Delivery_person_Ratings"].isna().sum())
    age_median = df["Delivery_person_Age"].median()
    rating_median = df["Delivery_person_Ratings"].median()
    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(age_median)
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(rating_median)
    log["age_missing_filled"] = age_missing
    log["age_fill_value"] = float(age_median)
    log["rating_missing_filled"] = rating_missing
    log["rating_fill_value"] = float(rating_median)

    # --- parse dates / times ---------------------------------------------------
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")

    def _parse_time_column(series: pd.Series) -> pd.Series:
        """
        Time_Orderd / Time_Order_picked ship in two different encodings in
        this dataset: normal 'HH:MM' strings AND, for a subset of rows, an
        Excel-style fraction-of-a-day float (e.g. 0.4583333 == 11:00). Both
        are parsed into real datetimes so no legitimate timestamp is thrown
        away as "missing" just because of its encoding.
        """
        s = series.astype(str).str.strip()
        as_clock = pd.to_datetime(s, format="%H:%M", errors="coerce")
        frac = pd.to_numeric(s, errors="coerce")
        as_fraction = pd.to_datetime(
            (frac * 24 * 60 * 60).round(), unit="s", origin="1970-01-01", errors="coerce"
        )
        return as_clock.fillna(as_fraction)

    df["Time_Orderd_parsed"] = _parse_time_column(df["Time_Orderd"])
    df["Time_Order_picked_parsed"] = _parse_time_column(df["Time_Order_picked"])
    time_orderd_missing = int(df["Time_Orderd_parsed"].isna().sum())
    log["time_orderd_missing_left_as_na"] = time_orderd_missing

    # order hour, useful for time-of-day analysis (bonus)
    df["Order_Hour"] = df["Time_Orderd_parsed"].dt.hour

    # --- categorical dtype ------------------------------------------------------
    for c in ["Weather_conditions", "Road_traffic_density", "Type_of_order",
              "Type_of_vehicle", "Festival", "City", "delivery_speed"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # --- drop physically impossible rows (safety net) ---------------------------
    before_physical = len(df)
    df = df[(df["distance_km"] > 0) & (df["Time_taken (min)"] > 0)]
    log["impossible_rows_dropped"] = before_physical - len(df)

    df = df.reset_index(drop=True)
    log["rows_after"] = len(df)
    return df, log


# --------------------------------------------------------------------------- #
# 3. BASIC ANALYSIS (Task C)
# --------------------------------------------------------------------------- #

def basic_stats(df: pd.DataFrame) -> dict:
    df["delivery_speed_kmph"] = df["distance_km"] / (df["Time_taken (min)"] / 60)
    return {
        "total_deliveries": int(len(df)),
        "avg_delivery_time_min": round(df["Time_taken (min)"].mean(), 2),
        "min_delivery_time_min": int(df["Time_taken (min)"].min()),
        "max_delivery_time_min": int(df["Time_taken (min)"].max()),
        "avg_distance_km": round(df["distance_km"].mean(), 2),
        "avg_speed_kmph": round(df["delivery_speed_kmph"].mean(), 2),
        "avg_rating": round(df["Delivery_person_Ratings"].mean(), 2),
        "avg_age": round(df["Delivery_person_Age"].mean(), 1),
    }


# --------------------------------------------------------------------------- #
# 4. THE 3 COMPETITION QUESTIONS (Task D) — computed programmatically
# --------------------------------------------------------------------------- #

def q1_traffic_impact(df: pd.DataFrame) -> pd.Series:
    """Average delivery time by traffic density, sorted descending."""
    return (
        df.groupby("Road_traffic_density", observed=True)["Time_taken (min)"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )


def q2_distance_impact(df: pd.DataFrame) -> dict:
    """
    Correlation between distance and delivery time, plus average delivery
    time broken down by distance bucket, to support the conclusion with data.
    """
    corr = df["distance_km"].corr(df["Time_taken (min)"])
    bins = [0, 5, 10, 15, 20, np.inf]
    labels = ["0-5 km", "5-10 km", "10-15 km", "15-20 km", "20+ km"]
    df["_distance_bucket"] = pd.cut(df["distance_km"], bins=bins, labels=labels)
    by_bucket = (
        df.groupby("_distance_bucket", observed=True)["Time_taken (min)"]
        .mean()
        .round(2)
    )
    df.drop(columns="_distance_bucket", inplace=True)
    return {"correlation": round(float(corr), 3), "avg_time_by_distance_bucket": by_bucket}


def q3_combined_conditions(df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """Average delivery time by (weather, traffic) combination, sorted descending."""
    combo = (
        df.groupby(["Weather_conditions", "Road_traffic_density"], observed=True)["Time_taken (min)"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )
    return combo.head(top_n)


# --------------------------------------------------------------------------- #
# 5. VISUALIZATIONS (Task E) — static PNGs for the notebook deliverable
# --------------------------------------------------------------------------- #

PALETTE = {
    "Jam": "#E63946", "High": "#F4A261", "Medium": "#2A9D8F", "Low": "#457B9D",
}


def chart_traffic_bar(df: pd.DataFrame, save_path: str | None = None):
    data = q1_traffic_impact(df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [PALETTE.get(k, "#6D6875") for k in data.index]
    bars = ax.bar(data.index.astype(str), data.values, color=colors, edgecolor="white")
    ax.set_title("Average Delivery Time by Road Traffic Density", fontsize=13, fontweight="bold")
    ax.set_xlabel("Road Traffic Density")
    ax.set_ylabel("Average Delivery Time (minutes)")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, data.values):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                     va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def chart_distance_scatter(df: pd.DataFrame, save_path: str | None = None, sample: int = 4000):
    plot_df = df.sample(min(sample, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(plot_df["distance_km"], plot_df["Time_taken (min)"],
               alpha=0.25, s=14, color="#2A9D8F", edgecolors="none")
    z = np.polyfit(df["distance_km"], df["Time_taken (min)"], 1)
    xs = np.linspace(df["distance_km"].min(), df["distance_km"].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), color="#E63946", linewidth=2, label="Trend line")
    ax.set_title("Delivery Distance vs. Delivery Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Delivery Time (minutes)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def generate_all_charts(df: pd.DataFrame, out_dir: str):
    import os
    os.makedirs(out_dir, exist_ok=True)
    p1 = f"{out_dir}/chart1_traffic_vs_time.png"
    p2 = f"{out_dir}/chart2_distance_vs_time.png"
    chart_traffic_bar(df, p1)
    chart_distance_scatter(df, p2)
    return p1, p2


# --------------------------------------------------------------------------- #
# 6. BUSINESS INSIGHTS (Task F) — data-driven, generated from the frame
# --------------------------------------------------------------------------- #

def business_insights(df: pd.DataFrame) -> list[dict]:
    insights = []

    # Insight 1 — traffic
    traffic = q1_traffic_impact(df)
    worst_traffic, worst_val = traffic.index[0], traffic.iloc[0]
    best_traffic, best_val = traffic.index[-1], traffic.iloc[-1]
    gap = round(worst_val - best_val, 1)
    insights.append({
        "title": f"'{worst_traffic}' traffic adds ~{gap} minutes per delivery",
        "finding": f"Average delivery time under '{worst_traffic}' traffic is "
                   f"{worst_val} min vs. {best_val} min under '{best_traffic}' traffic.",
        "so_what": "Dispatch and ETAs should dynamically pad estimated times during "
                   "high-traffic windows, and riders on jam-prone routes could be "
                   "pre-assigned before the rush to avoid a backlog.",
    })

    # Insight 2 — distance
    q2 = q2_distance_impact(df)
    insights.append({
        "title": f"Delivery time rises steadily with distance (corr = {q2['correlation']})",
        "finding": "Average time by distance band: " + ", ".join(
            f"{k} → {v} min" for k, v in q2["avg_time_by_distance_bucket"].items()
        ),
        "so_what": "Longer orders should be routed to riders already positioned nearby, "
                   "and the app's promised delivery window should scale with distance "
                   "rather than using one flat estimate for all orders.",
    })

    # Insight 3 — weather x traffic combo
    combo = q3_combined_conditions(df, top_n=1)
    (weather, traffic_lvl), worst_combo_val = combo.index[0], combo.iloc[0]
    insights.append({
        "title": f"'{weather}' weather + '{traffic_lvl}' traffic is the worst combination",
        "finding": f"Deliveries in {weather} weather with {traffic_lvl} traffic take "
                   f"{worst_combo_val} minutes on average — the slowest of any combination.",
        "so_what": "On days this combination is forecast, the platform could proactively "
                   "message customers with realistic ETAs and temporarily boost rider "
                   "incentives to keep supply matched with demand.",
    })

    # Insight 4 — rider condition / rating (bonus, still data-driven)
    if "Vehicle_condition" in df.columns:
        veh = df.groupby("Vehicle_condition", observed=True)["Time_taken (min)"].mean().round(2)
        worst_v = veh.idxmax()
        insights.append({
            "title": "Poorer vehicle condition is associated with slower deliveries",
            "finding": f"Average delivery time by vehicle-condition score: "
                       + ", ".join(f"{k}: {v} min" for k, v in veh.items()),
            "so_what": "Vehicle maintenance/condition checks for riders could be tied "
                       "to performance incentives, since condition score "
                       f"{worst_v} sees the slowest average deliveries.",
        })

    return insights
