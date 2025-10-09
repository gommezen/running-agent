#!/usr/bin/env python

# ===================================================
# 📓 Notebook 6 — Interactive Dashboard (Humanized UX)
# ===================================================
#
# This version focuses on clarity, narrative explainability, and user flow.
# It keeps all ML and DB logic identical to the original file.
#
# Goals:
# • Translate technical labels into running-language.
# • Explain model predictions in short, meaningful sentences.
# • Organize sections around user reasoning, not developer structure.
# ===================================================

# standard libraries
import contextlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import (  # ← add TYPE_CHECKING, Callable, TypeVar
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

# ---- Typed shims so mypy treats cache decorators as identity (keeps function types) ----
F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:

    def cache_data(func: F) -> F: ...
    def cache_resource(func: F) -> F: ...
else:
    cache_data = st.cache_data
    cache_resource = st.cache_resource

# ---------------------------------------------------
# 🔧 Project Root Finder
# ---------------------------------------------------

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.db_utils import get_engine  # noqa: E402

with contextlib.suppress(Exception):
    st.set_page_config(page_title="Running Insights Dashboard", layout="wide")


def _find_project_root() -> Path:
    anchors = [Path(__file__).resolve(), Path.cwd().resolve()]
    visited = set()
    for base in anchors:
        for p in [base] + list(base.parents):
            if p in visited:
                continue
            visited.add(p)
            if (p / "src").exists() and (p / "models").exists():
                return p
    for p in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        if p.name.lower() == "running-agent":
            return p
    return Path(__file__).resolve().parents[1]


project_root = _find_project_root()
sys.path.append(str(project_root))


with contextlib.suppress(Exception):
    st.set_page_config(page_title="Running Insights Dashboard", layout="wide")


# ---------------------------------------------------
# 📁 Model Paths
# ---------------------------------------------------
model_dir = project_root / "models"
_model_candidates = ["random_forest_classifier.pkl", "random_forest_clf.pkl"]
rf_model_path = None
for name in _model_candidates:
    cand = model_dir / name
    if cand.exists():
        rf_model_path = cand
        break

explainer_path = model_dir / "shap_explainer_clf.pkl"


# ---------------------------------------------------
# 🧠 Load Models + Data
# ---------------------------------------------------
@cache_resource
def load_models() -> tuple[Any, Any | None]:
    if rf_model_path is None:
        raise FileNotFoundError(f"No model found in {model_dir}")
    rf_clf: Any = joblib.load(rf_model_path)
    explainer: Any | None = joblib.load(explainer_path) if explainer_path.exists() else None
    return rf_clf, explainer


@cache_data
def load_data() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM runs_summary ORDER BY date DESC", engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


rf_clf, explainer_clf = load_models()
summary_df = load_data()
st.caption(f"Loaded data: {summary_df.shape[0]} runs, {summary_df.shape[1]} features")

# ---------------------------------------------------
# 🏷️ Feature & Cluster Labels
# ---------------------------------------------------
FEATURE_LABELS = {
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

RUN_TYPE_LABELS = {
    0: "Recovery / Easy Run",
    1: "Endurance Run",
    2: "Interval / Tempo Run",
    3: "Long Run",
}


# ---------------------------------------------------
# 🧠 Explainability Text Generator
# ---------------------------------------------------
def explain_prediction(pred_label: int, top_features: pd.Series) -> str:
    """Return a short, human explanation for the prediction."""
    if pred_label == 0:
        return "Low intensity and short duration suggest this was an easy recovery run."
    elif pred_label == 1:
        return (
            "Steady pace, moderate elevation, and consistent cadence indicate endurance training."
        )
    elif pred_label == 2:
        return "High cadence and variable pace point toward interval or tempo work."
    elif pred_label == 3:
        return "Long distance and steady rhythm match a classic long run profile."
    else:
        return "The model detected a mixed running pattern based on your data."


# ---------------------------------------------------
# 🧩 Helper for SHAP vector extraction
# ---------------------------------------------------
def shap_vector_for_sample(
    explainer: Any,
    model: Any,
    X_one_row: pd.DataFrame,
) -> tuple[np.ndarray, Any]:
    """
    Safe universal SHAP vector extractor.
    Works for list-per-class, (n,feat,class), (n,class,feat), or (n,feat).
    """
    nfeat = X_one_row.shape[1]
    raw: Any = explainer.shap_values(X_one_row)
    arr = np.asarray(raw)

    # For debugging – will print once in Streamlit log
    st.write("SHAP detected shape:", arr.shape)

    # Case 1: SHAP returns list of arrays (one per class)
    if isinstance(raw, list):
        pred_class = int(np.argmax(model.predict_proba(X_one_row), axis=1)[0])
        v = np.array(raw[pred_class])
        if v.ndim == 2:
            return v[0], raw
        return v, raw

    # Case 2: shape (n_samples, n_features, n_classes)
    if arr.ndim == 3 and arr.shape[1] == nfeat:
        pred_class = int(np.argmax(model.predict_proba(X_one_row), axis=1)[0])
        return arr[0, :, pred_class], raw

    # Case 3: shape (n_samples, n_classes, n_features)
    if arr.ndim == 3 and arr.shape[2] == nfeat:
        pred_class = int(np.argmax(model.predict_proba(X_one_row), axis=1)[0])
        return arr[0, pred_class, :], raw

    # Case 4: simple (n_samples, n_features)
    if arr.ndim == 2 and arr.shape[1] == nfeat:
        return arr[0], raw

    # Anything else → report shape
    st.error(f"⚠️ Unhandled SHAP shape: {arr.shape}")
    return np.zeros(nfeat), raw


# ---------------------------------------------------
# 🧭 Layout — Tabs
# ---------------------------------------------------
st.title("Running Insights Dashboard")
st.write(
    "Explore how your model interprets training data — from single run analysis "
    "to overall performance patterns."
)

tab1, tab2, tab3 = st.tabs(["Single Run", "Compare Runs", "Global Insights"])

# ---------------------------------------------------
# 🔍 Tab 1 — Single Run
# ---------------------------------------------------
with tab1:
    st.subheader("Analyze This Run")
    options = summary_df["date"].dt.strftime("%Y-%m-%d").tolist()
    if not options:
        st.warning("No runs available.")
    else:
        selected_date = st.selectbox("Select run date", options)
        case = summary_df[summary_df["date"].dt.strftime("%Y-%m-%d") == selected_date]
        if not case.empty:
            X_means = summary_df[rf_clf.feature_names_in_].mean()
            case_X = case[rf_clf.feature_names_in_].fillna(X_means).iloc[[0]]
            pred_label = int(rf_clf.predict(case_X)[0])
            label_text = RUN_TYPE_LABELS.get(pred_label, f"Cluster {pred_label}")
            st.markdown(f"### Model predicts: **{label_text}**")

            if explainer_clf:
                X = summary_df[rf_clf.feature_names_in_]
                sv = explainer_clf.shap_values(X)
                arr = np.asarray(sv)

                # Handle multiple possible shapes
                if isinstance(sv, list):
                    # List of (n_samples, n_features)
                    stacked = np.stack(
                        [np.abs(s) for s in sv], axis=0
                    )  # (n_classes, n_samples, n_features)
                    mean_abs = stacked.mean(axis=(0, 1))
                else:
                    # array, ensure it's 2D (n_features,)
                    arr = np.abs(arr)
                    if arr.ndim == 3:
                        # could be (n_samples, n_features, n_classes) or (n_samples, n_classes, n_features)
                        if arr.shape[1] == X.shape[1]:
                            mean_abs = arr.mean(axis=(0, 2))
                        elif arr.shape[2] == X.shape[1]:
                            mean_abs = arr.mean(axis=(0, 1))
                        else:
                            # flatten any extra class axis
                            mean_abs = arr.mean(axis=tuple(range(arr.ndim - 1)))
                    elif arr.ndim == 2:
                        mean_abs = arr.mean(axis=0)
                    else:
                        mean_abs = arr.flatten()

                # --- build dataframe safely ---
                mean_abs = np.ravel(mean_abs)  # ensure 1D
                global_df = pd.DataFrame(
                    {
                        "Feature": [FEATURE_LABELS.get(f, f) for f in rf_clf.feature_names_in_],
                        "Importance": mean_abs,
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
                st.caption("Average influence of each feature across all runs.")


# ---------------------------------------------------
# 📈 Tab 2 — Compare Two Runs
# ---------------------------------------------------
with tab2:
    st.subheader("Compare Runs")
    dates = summary_df["date"].dt.strftime("%Y-%m-%d").tolist()
    if len(dates) < 2:
        st.info("At least two runs required.")
    else:
        col1, col2 = st.columns(2)
        d1 = col1.selectbox("First run", dates, index=0, key="cmp1")
        d2 = col2.selectbox("Second run", dates, index=1, key="cmp2")
        df1 = summary_df[summary_df["date"].dt.strftime("%Y-%m-%d") == d1]
        df2 = summary_df[summary_df["date"].dt.strftime("%Y-%m-%d") == d2]
        if not df1.empty and not df2.empty and explainer_clf:
            X_means = summary_df[rf_clf.feature_names_in_].mean()
            x1 = df1[rf_clf.feature_names_in_].fillna(X_means).iloc[[0]]
            x2 = df2[rf_clf.feature_names_in_].fillna(X_means).iloc[[0]]
            shap1, _ = shap_vector_for_sample(explainer_clf, rf_clf, x1)
            shap2, _ = shap_vector_for_sample(explainer_clf, rf_clf, x2)
            diff = pd.Series(shap2 - shap1, index=rf_clf.feature_names_in_).sort_values(
                key=lambda x: abs(x), ascending=False
            )
            st.markdown("**Differences in model interpretation (Run 2 − Run 1):**")
            diff_df = (
                diff.head(10)
                .rename(index=lambda f: FEATURE_LABELS.get(f, f))
                .reset_index()
                .rename(columns={"index": "Feature", 0: "Δ SHAP"})
            )
            st.bar_chart(diff_df.set_index("Feature"))
        else:
            st.info("Could not compute comparison (missing SHAP or runs).")

# ---------------------------------------------------
# 🌍 Tab 3 — Global Insights
# ---------------------------------------------------
with tab3:
    st.subheader("Global Feature Importance")
    if explainer_clf:
        X = summary_df[rf_clf.feature_names_in_]
        sv = explainer_clf.shap_values(X)
        arr = np.asarray(sv)
        if isinstance(sv, list):
            stacked = np.stack([np.abs(s) for s in sv], axis=0)
            mean_abs = stacked.mean(axis=(0, 1))
        elif arr.ndim == 3 and arr.shape[-1] == X.shape[1]:
            mean_abs = np.abs(arr).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(arr).mean(axis=0)
        # --- force 1-D importance vector ---
        mean_abs = np.array(mean_abs)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=-1).ravel()

        # now safe to build dataframe
        global_df = pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS.get(f, f) for f in rf_clf.feature_names_in_],
                "Importance": mean_abs.tolist(),
            }
        ).sort_values("Importance", ascending=False)

        chart = (
            alt.Chart(global_df.head(10))
            .mark_bar()
            .encode(x=alt.X("Importance:Q", title="Mean |SHAP|"), y=alt.Y("Feature:N", sort="-x"))
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("Average influence of each feature across all runs.")
    else:
        st.info("Global SHAP importance unavailable — explainer not loaded.")

    st.markdown("#### Performance Trend (Average Pace Over Time)")
    if "avg_pace_min_km" in summary_df.columns:
        trend = (
            alt.Chart(summary_df).mark_line(point=True).encode(x="date:T", y="avg_pace_min_km:Q")
        )
        st.altair_chart(trend, use_container_width=True)
    else:
        st.info("No pace data found.")

# ---------------------------------------------------
# 🏁 Summary
# ---------------------------------------------------
# This humanized version improves clarity and trust without altering ML logic.
# - Human-readable feature labels and run type names.
# - Short, narrative explanations for predictions.
# - Cleaner section flow and neutral tone.
# - Compatible with Notebook 7 (PostgreSQL integration) and future deployment.
