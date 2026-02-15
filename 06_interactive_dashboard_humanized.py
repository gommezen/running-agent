#!/usr/bin/env python

# ===================================================
# Interactive Dashboard (Humanized UX — Dark Athletic Theme)
# ===================================================
#
# Dark athletic-themed Streamlit dashboard with electric green accents.
# Translates technical ML outputs into human-readable running insights.
#
# Features:
# - Model toggle: RF classifier (run type) vs RF regressor (pace)
# - Metric cards (weekly mileage, avg pace, streak, total runs)
# - Sidebar filters (date range, run type, distance)
# - SHAP-powered single-run analysis, run comparison, global insights
# - Custom Altair dark theme with Exo 2 typography
# ===================================================

from __future__ import annotations

# standard libraries
import contextlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

# third-party libraries
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---- Typed shims so mypy treats cache decorators as identity ----
F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:

    def cache_data(func: F) -> F: ...
    def cache_resource(func: F) -> F: ...
else:
    cache_data = st.cache_data
    cache_resource = st.cache_resource

# ---------------------------------------------------
# Project Root Finder
# ---------------------------------------------------

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.db_utils import get_engine  # noqa: E402
from src.xai_utils import shap_vector_for_sample  # noqa: E402

with contextlib.suppress(Exception):
    st.set_page_config(
        page_title="Running Insights Dashboard",
        page_icon="\U0001f3c3",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _find_project_root() -> Path:
    anchors = [Path(__file__).resolve(), Path.cwd().resolve()]
    visited: set[Path] = set()
    for base in anchors:
        for p in [base, *list(base.parents)]:
            if p in visited:
                continue
            visited.add(p)
            if (p / "src").exists() and (p / "models").exists():
                return p
    for p in [Path(__file__).resolve(), *list(Path(__file__).resolve().parents)]:
        if p.name.lower() == "running-agent":
            return p
    return Path(__file__).resolve().parents[1]


project_root = _find_project_root()
sys.path.append(str(project_root))


# ---------------------------------------------------
# Custom CSS Injection
# ---------------------------------------------------
def inject_custom_css() -> None:
    """Inject dark athletic theme CSS with Exo 2 typography."""
    st.markdown(
        """
        <style>
        @import url(
            'https://fonts.googleapis.com/css2?family=Exo+2:'
            'wght@400;600;700;800&display=swap'
        );

        :root {
            --bg-primary: #0E1117;
            --bg-secondary: #161B22;
            --bg-card: #1C2128;
            --accent: #00FF87;
            --accent-dim: #00CC6A;
            --text-primary: #E6EDF3;
            --text-secondary: #8B949E;
            --text-muted: #484F58;
            --border: #30363D;
            --font-stack: "Exo 2", -apple-system,
                          BlinkMacSystemFont, sans-serif;
        }

        html, body, [class*="css"] {
            font-family: var(--font-stack) !important;
        }

        h1 {
            font-family: var(--font-stack) !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px !important;
            color: var(--text-primary) !important;
        }

        h2, h3 {
            font-family: var(--font-stack) !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }

        [data-testid="stMetricLabel"] {
            font-family: var(--font-stack) !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            color: var(--text-secondary) !important;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        [data-testid="stMetricValue"] {
            font-family: var(--font-stack) !important;
            font-weight: 800 !important;
            font-size: 2rem !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stMetricDelta"] > div {
            font-weight: 600 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 2px solid var(--border);
        }

        .stTabs [data-baseweb="tab"] {
            font-family: var(--font-stack) !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 10px 24px !important;
            border-radius: 8px 8px 0 0;
            color: var(--text-secondary) !important;
        }

        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom: 3px solid var(--accent) !important;
        }

        [data-testid="stSidebar"] {
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--accent) !important;
        }

        hr {
            border-color: var(--border) !important;
        }

        [data-baseweb="select"] {
            border-radius: 8px !important;
        }

        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--text-muted);
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


# ---------------------------------------------------
# Altair Dark Athletic Theme
# ---------------------------------------------------
def _running_dark_theme() -> dict[str, Any]:
    """Dark athletic theme for all Altair charts."""
    return {
        "config": {
            "background": "#0E1117",
            "view": {"stroke": "transparent"},
            "axis": {
                "domainColor": "#30363D",
                "gridColor": "#21262D",
                "tickColor": "#30363D",
                "labelColor": "#8B949E",
                "titleColor": "#E6EDF3",
                "labelFont": "Exo 2, sans-serif",
                "titleFont": "Exo 2, sans-serif",
                "labelFontSize": 11,
                "titleFontSize": 13,
                "titleFontWeight": 600,
            },
            "legend": {
                "labelColor": "#8B949E",
                "titleColor": "#E6EDF3",
                "labelFont": "Exo 2, sans-serif",
                "titleFont": "Exo 2, sans-serif",
            },
            "title": {
                "color": "#E6EDF3",
                "font": "Exo 2, sans-serif",
                "fontSize": 16,
                "fontWeight": 700,
            },
            "mark": {"color": "#00FF87"},
            "bar": {"color": "#00FF87"},
            "line": {"color": "#00FF87", "strokeWidth": 2.5},
            "point": {"color": "#00FF87", "filled": True, "size": 60},
            "range": {
                "category": [
                    "#00FF87",
                    "#00BFFF",
                    "#FF6B6B",
                    "#FFD93D",
                    "#C084FC",
                    "#FF8C42",
                    "#6EE7B7",
                    "#F472B6",
                ],
            },
        },
    }


# Support both Altair 5.5+ (alt.theme.register) and 5.4.x (alt.themes.register)
if hasattr(alt, "theme") and hasattr(alt.theme, "register"):
    alt.theme.register("running_dark", enable=True)(_running_dark_theme)
else:
    alt.themes.register("running_dark", _running_dark_theme)  # type: ignore[attr-defined]
    alt.themes.enable("running_dark")  # type: ignore[attr-defined]


# ---------------------------------------------------
# Model Paths
# ---------------------------------------------------
model_dir = project_root / "models"
_model_candidates = [
    "random_forest_classifier.pkl",
    "random_forest_clf.pkl",
]
rf_model_path = None
for name in _model_candidates:
    cand = model_dir / name
    if cand.exists():
        rf_model_path = cand
        break

explainer_path = model_dir / "shap_explainer_clf.pkl"
rf_reg_model_path = model_dir / "random_forest_regressor.pkl"
explainer_reg_path = model_dir / "shap_explainer_reg.pkl"
feature_cols_path = model_dir / "feature_columns.pkl"


# ---------------------------------------------------
# Load Models + Data
# ---------------------------------------------------
@cache_resource
def load_models() -> tuple[Any, Any | None, Any | None, Any | None, list[str]]:
    """Load classifier, regressor, both SHAP explainers, and feature list."""
    if rf_model_path is None:
        raise FileNotFoundError(f"No classifier found in {model_dir}")
    rf_clf: Any = joblib.load(rf_model_path)
    explainer_clf: Any | None = joblib.load(explainer_path) if explainer_path.exists() else None
    rf_reg: Any | None = joblib.load(rf_reg_model_path) if rf_reg_model_path.exists() else None
    explainer_reg: Any | None = (
        joblib.load(explainer_reg_path) if explainer_reg_path.exists() else None
    )
    feature_cols: list[str] = joblib.load(feature_cols_path) if feature_cols_path.exists() else []
    return rf_clf, explainer_clf, rf_reg, explainer_reg, feature_cols


@cache_data
def load_data() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM runs_summary ORDER BY date DESC", engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


rf_clf, explainer_clf, rf_reg, explainer_reg, _all_feature_cols = load_models()
summary_df = load_data()

# Derive feature name lists for each model
CLF_FEATURES: list[str] = list(rf_clf.feature_names_in_)

if rf_reg is not None and hasattr(rf_reg, "feature_names_in_"):
    REG_FEATURES: list[str] = list(rf_reg.feature_names_in_)
elif rf_reg is not None and _all_feature_cols:
    REG_FEATURES = _all_feature_cols[: rf_reg.n_features_in_]
else:
    REG_FEATURES = []

_REG_AVAILABLE = rf_reg is not None and len(REG_FEATURES) > 0

# Add predicted run type column for sidebar filtering (always uses classifier)
_X_fill = summary_df[CLF_FEATURES].fillna(summary_df[CLF_FEATURES].mean())
summary_df["predicted_type"] = rf_clf.predict(_X_fill)

# Add predicted pace column if regressor is available
if _REG_AVAILABLE and rf_reg is not None:
    _reg_X = summary_df[REG_FEATURES].fillna(summary_df[REG_FEATURES].mean())
    summary_df["predicted_pace"] = rf_reg.predict(_reg_X)


# ---------------------------------------------------
# Feature & Cluster Labels
# ---------------------------------------------------
FEATURE_LABELS: dict[str, str] = {
    "total_distance_km": "Distance (km)",
    "duration_min": "Duration (min)",
    "avg_pace_min_km": "Average Pace (min/km)",
    "avg_cadence": "Cadence (steps/min)",
    "total_elev_gain": "Elevation Gain (m)",
    "avg_stride_len_m": "Stride Length (m)",
    "avg_gct_est_ms": "Ground Contact Time (ms)",
    "pace_variability": "Pace Consistency",
    "cadence_drift": "Cadence Stability",
    "load_7d": "Training Load (7-day)",
    "load_28d": "Training Load (28-day)",
    "fastest_1km_pace": "Fastest 1 km Pace (min/km)",
    "fastest_5min_pace": "Fastest 5-min Pace (min/km)",
}

RUN_TYPE_LABELS: dict[int, str] = {
    0: "Recovery / Easy Run",
    1: "Endurance Run",
    2: "Interval / Tempo Run",
    3: "Long Run",
}


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def _active_model_context(
    mode: str,
) -> tuple[Any, Any | None, list[str]]:
    """Return (model, explainer, feature_names) for the active mode."""
    if mode == "regressor" and _REG_AVAILABLE:
        return rf_reg, explainer_reg, REG_FEATURES
    return rf_clf, explainer_clf, CLF_FEATURES


def _format_pace(pace_min_km: float) -> str:
    """Format a pace float (min/km) as M:SS /km."""
    if pd.isna(pace_min_km):
        return "--:-- /km"
    minutes = int(pace_min_km)
    seconds = int((pace_min_km - minutes) * 60)
    return f"{minutes}:{seconds:02d} /km"


def explain_prediction(pred_label: int, top_features: pd.Series) -> str:
    """Return a short, human explanation for the prediction."""
    if pred_label == 0:
        return "Low intensity and short duration suggest this was an easy recovery run."
    if pred_label == 1:
        return (
            "Steady pace, moderate elevation, and consistent cadence indicate endurance training."
        )
    if pred_label == 2:
        return "High cadence and variable pace point toward interval or tempo work."
    if pred_label == 3:
        return "Long distance and steady rhythm match a classic long run profile."
    return "The model detected a mixed running pattern based on your data."


# ---------------------------------------------------
# Metric Computation Helpers
# ---------------------------------------------------
def _compute_weekly_mileage(
    df: pd.DataFrame,
) -> tuple[str, str | None]:
    """Sum distance last 7 days, delta vs prior 7 days."""
    if df.empty:
        return "0.0 km", None
    now = df["date"].max()
    week_ago = now - pd.Timedelta(days=7)
    two_weeks_ago = now - pd.Timedelta(days=14)
    current = df.loc[df["date"] > week_ago, "total_distance_km"].sum()
    previous = df.loc[
        (df["date"] > two_weeks_ago) & (df["date"] <= week_ago),
        "total_distance_km",
    ].sum()
    delta = None
    if previous > 0:
        pct = ((current - previous) / previous) * 100
        delta = f"{pct:+.1f}%"
    return f"{current:.1f} km", delta


def _compute_avg_pace(
    df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> tuple[str, str | None]:
    """Mean pace formatted as M:SS, delta vs overall mean."""
    if df.empty or "avg_pace_min_km" not in df.columns:
        return "--:-- /km", None
    avg = df["avg_pace_min_km"].mean()
    overall = full_df["avg_pace_min_km"].mean()
    delta = None
    if not pd.isna(overall) and not pd.isna(avg):
        diff = avg - overall
        delta = f"{diff:+.2f} min/km"
    if pd.isna(avg):
        return "--:-- /km", None
    minutes = int(avg)
    seconds = int((avg - minutes) * 60)
    return f"{minutes}:{seconds:02d} /km", delta


def _compute_streak(df: pd.DataFrame) -> tuple[str, None]:
    """Count consecutive days with runs from most recent date."""
    if df.empty:
        return "0 days", None
    dates = sorted(df["date"].dt.date.unique(), reverse=True)
    streak = 1
    for i in range(1, len(dates)):
        if (dates[i - 1] - dates[i]).days == 1:
            streak += 1
        else:
            break
    unit = "day" if streak == 1 else "days"
    return f"{streak} {unit}", None


def _compute_total_runs(
    df: pd.DataFrame,
) -> tuple[str, str | None]:
    """Total runs with delta vs prior 30-day period."""
    if df.empty:
        return "0", None
    total = len(df)
    now = df["date"].max()
    month_ago = now - pd.Timedelta(days=30)
    two_months_ago = now - pd.Timedelta(days=60)
    recent = len(df[df["date"] > month_ago])
    prior = len(df[(df["date"] > two_months_ago) & (df["date"] <= month_ago)])
    delta = None
    if prior > 0:
        diff = recent - prior
        delta = f"{diff:+d} vs prev 30d"
    return str(total), delta


# ---------------------------------------------------
# Sidebar — Model Selection + Filters
# ---------------------------------------------------
with st.sidebar:
    st.markdown("## Model")

    _model_options = ["Run Type (Classifier)"]
    if _REG_AVAILABLE:
        _model_options.append("Pace (Regressor)")

    selected_model_label = st.radio(
        "Prediction model",
        options=_model_options,
        index=0,
        key="sidebar_model_select",
        help=(
            "Classifier predicts run type (Recovery, Endurance, etc.). "
            "Regressor predicts pace (min/km)."
        ),
    )
    model_mode = "regressor" if selected_model_label == "Pace (Regressor)" else "classifier"

    st.markdown("---")
    st.markdown("## Filters")
    st.caption("Narrow the runs displayed in charts and metrics.")

    # Date range
    min_date = summary_df["date"].min().date()
    max_date = summary_df["date"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="sidebar_date_range",
    )

    # Run type filter (classifier mode only)
    if model_mode == "classifier":
        all_run_types = list(RUN_TYPE_LABELS.values())
        selected_types = st.multiselect(
            "Run type",
            options=all_run_types,
            default=all_run_types,
            key="sidebar_run_type",
        )
    else:
        selected_types = list(RUN_TYPE_LABELS.values())

    # Distance range slider
    dist_min = float(summary_df["total_distance_km"].min())
    dist_max = float(summary_df["total_distance_km"].max())
    if dist_min < dist_max:
        dist_range = st.slider(
            "Distance (km)",
            min_value=dist_min,
            max_value=dist_max,
            value=(dist_min, dist_max),
            step=0.5,
            key="sidebar_distance",
        )
    else:
        dist_range = (dist_min, dist_max)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8B949E; "
        "font-size:0.75rem; padding:8px 0;'>"
        "Running Insights v2.1<br>"
        "RF Classifier + Regressor &middot; SHAP Analytics"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------
# Apply Filters -> filtered_df
# ---------------------------------------------------
_reverse_type_map = {v: k for k, v in RUN_TYPE_LABELS.items()}
_mask = pd.Series(True, index=summary_df.index)

# Date filter
if isinstance(date_range, tuple) and len(date_range) == 2:
    _mask &= (summary_df["date"].dt.date >= date_range[0]) & (
        summary_df["date"].dt.date <= date_range[1]
    )

# Run type filter
_selected_ids = [_reverse_type_map[t] for t in selected_types if t in _reverse_type_map]
_mask &= summary_df["predicted_type"].isin(_selected_ids)

# Distance filter
_mask &= (summary_df["total_distance_km"] >= dist_range[0]) & (
    summary_df["total_distance_km"] <= dist_range[1]
)

filtered_df = summary_df[_mask].copy()


# ---------------------------------------------------
# Layout — Title + Metric Cards
# ---------------------------------------------------
st.title("Running Insights Dashboard")
st.write(
    "Explore how your model interprets training data "
    "\u2014 from single run analysis to overall performance patterns."
)

if filtered_df.empty:
    st.warning("No runs match the current filters.")
else:
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    weekly_val, weekly_delta = _compute_weekly_mileage(filtered_df)
    pace_val, pace_delta = _compute_avg_pace(filtered_df, summary_df)
    streak_val, _ = _compute_streak(filtered_df)
    total_val, total_delta = _compute_total_runs(filtered_df)

    with m1:
        st.metric("Weekly Mileage", weekly_val, weekly_delta)
    with m2:
        st.metric(
            "Avg Pace",
            pace_val,
            pace_delta,
            delta_color="inverse",
        )
    with m3:
        st.metric("Run Streak", streak_val)
    with m4:
        st.metric("Total Runs", total_val, total_delta)

st.caption(f"Loaded {summary_df.shape[0]} runs ({len(filtered_df)} shown after filters)")


# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Single Run", "Compare Runs", "Global Insights"])

# ---------------------------------------------------
# Tab 1 — Single Run
# ---------------------------------------------------
with tab1:
    st.subheader("Analyze This Run")
    options = filtered_df["date"].dt.strftime("%Y-%m-%d").tolist()
    if not options:
        st.warning("No runs available.")
    else:
        selected_date = st.selectbox("Select run date", options)
        case = summary_df[summary_df["date"].dt.strftime("%Y-%m-%d") == selected_date]
        if not case.empty:
            model, explainer, feat_names = _active_model_context(model_mode)
            X_means = summary_df[feat_names].mean()
            case_X = case[feat_names].fillna(X_means).iloc[[0]]

            if model_mode == "classifier":
                pred_label = int(model.predict(case_X)[0])
                label_text = RUN_TYPE_LABELS.get(pred_label, f"Cluster {pred_label}")
                st.markdown(f"### Model predicts: **{label_text}**")
                st.caption(explain_prediction(pred_label, pd.Series()))
            else:
                pred_pace = float(model.predict(case_X)[0])
                st.markdown(f"### Predicted pace: **{_format_pace(pred_pace)}**")
                actual = case["avg_pace_min_km"].iloc[0]
                if not pd.isna(actual):
                    diff = pred_pace - actual
                    st.caption(f"Actual: {_format_pace(actual)}  |  Difference: {diff:+.2f} min/km")

            # Per-sample SHAP chart
            if explainer is not None:
                shap_vec, _ = shap_vector_for_sample(explainer, model, case_X)
                shap_df = pd.DataFrame(
                    {
                        "Feature": [FEATURE_LABELS.get(f, f) for f in feat_names],
                        "SHAP": shap_vec,
                    }
                ).sort_values("SHAP", key=lambda x: abs(x), ascending=False)

                chart_title = (
                    "What drives this run type prediction"
                    if model_mode == "classifier"
                    else "What drives this pace prediction"
                )
                chart = (
                    alt.Chart(shap_df.head(10))
                    .mark_bar()
                    .encode(
                        x=alt.X("SHAP:Q", title="SHAP value"),
                        y=alt.Y("Feature:N", sort="-x"),
                        color=alt.condition(
                            alt.datum["SHAP"] > 0,
                            alt.value("#00FF87"),
                            alt.value("#FF6B6B"),
                        ),
                    )
                    .properties(title=chart_title)
                )
                st.altair_chart(chart, use_container_width=True)
                st.caption("Green = pushes prediction higher, Red = pushes prediction lower.")


# ---------------------------------------------------
# Tab 2 — Compare Two Runs
# ---------------------------------------------------
with tab2:
    st.subheader("Compare Runs")
    dates = filtered_df["date"].dt.strftime("%Y-%m-%d").tolist()
    if len(dates) < 2:
        st.info("At least two runs required.")
    else:
        col1, col2 = st.columns(2)
        d1 = col1.selectbox("First run", dates, index=0, key="cmp1")
        d2 = col2.selectbox("Second run", dates, index=1, key="cmp2")
        df1 = summary_df[summary_df["date"].dt.strftime("%Y-%m-%d") == d1]
        df2 = summary_df[summary_df["date"].dt.strftime("%Y-%m-%d") == d2]
        model, explainer, feat_names = _active_model_context(model_mode)
        if not df1.empty and not df2.empty and explainer is not None:
            X_means = summary_df[feat_names].mean()
            x1 = df1[feat_names].fillna(X_means).iloc[[0]]
            x2 = df2[feat_names].fillna(X_means).iloc[[0]]
            shap1, _ = shap_vector_for_sample(explainer, model, x1)
            shap2, _ = shap_vector_for_sample(explainer, model, x2)
            diff = pd.Series(shap2 - shap1, index=feat_names).sort_values(
                key=lambda x: abs(x), ascending=False
            )
            diff_title = (
                "Differences in run type interpretation"
                if model_mode == "classifier"
                else "Differences in pace prediction"
            )
            st.markdown(f"**{diff_title} (Run 2 \u2212 Run 1):**")
            diff_df = (
                diff.head(10)
                .rename(index=lambda f: FEATURE_LABELS.get(f, f))
                .reset_index()
                .rename(columns={"index": "Feature", 0: "\u0394 SHAP"})
            )

            diff_chart = (
                alt.Chart(diff_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "\u0394 SHAP:Q",
                        title="SHAP Difference",
                    ),
                    y=alt.Y(
                        "Feature:N",
                        sort=None,
                        title="Feature",
                    ),
                    color=alt.condition(
                        alt.datum["\u0394 SHAP"] > 0,
                        alt.value("#00FF87"),
                        alt.value("#FF6B6B"),
                    ),
                )
            )
            st.altair_chart(diff_chart, use_container_width=True)
        else:
            st.info("Could not compute comparison (missing SHAP or runs).")

# ---------------------------------------------------
# Tab 3 — Global Insights
# ---------------------------------------------------
with tab3:
    model, explainer, feat_names = _active_model_context(model_mode)
    section_title = (
        "Global Feature Importance (Run Type)"
        if model_mode == "classifier"
        else "Global Feature Importance (Pace)"
    )
    st.subheader(section_title)

    if explainer is not None:
        X = summary_df[feat_names].fillna(summary_df[feat_names].mean())
        sv = explainer.shap_values(X)
        arr = np.asarray(sv)
        if isinstance(sv, list):
            stacked = np.stack([np.abs(s) for s in sv], axis=0)
            mean_abs = stacked.mean(axis=(0, 1))
        elif arr.ndim == 3 and arr.shape[-1] == X.shape[1]:
            mean_abs = np.abs(arr).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(arr).mean(axis=0)
        mean_abs = np.array(mean_abs)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=-1).ravel()

        global_df = pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS.get(f, f) for f in feat_names],
                "Importance": mean_abs.tolist(),
            }
        ).sort_values("Importance", ascending=False)

        chart = (
            alt.Chart(global_df.head(10))
            .mark_bar()
            .encode(
                x=alt.X("Importance:Q", title="Mean |SHAP|"),
                y=alt.Y("Feature:N", sort="-x"),
            )
        )
        st.altair_chart(chart, use_container_width=True)
        shap_context = (
            "run type classification" if model_mode == "classifier" else "pace prediction"
        )
        st.caption(f"Average influence of each feature across all runs ({shap_context}).")
    else:
        st.info("Global SHAP importance unavailable \u2014 explainer not loaded.")

    st.markdown("#### Performance Trend (Average Pace Over Time)")
    pace_df = filtered_df.sort_values("date")
    if not pace_df.empty and "avg_pace_min_km" in pace_df.columns:
        trend = (
            alt.Chart(pace_df)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y(
                    "avg_pace_min_km:Q",
                    title="Avg Pace (min/km)",
                    scale=alt.Scale(reverse=True),
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip(
                        "avg_pace_min_km:Q",
                        title="Pace (min/km)",
                        format=".2f",
                    ),
                ],
            )
        )
        st.altair_chart(trend, use_container_width=True)
    else:
        st.info("No pace data found.")
