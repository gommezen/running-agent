#!/usr/bin/env python
# coding: utf-8

# ===================================================
# 📓 Notebook 6 — Interactive Dashboard
# ===================================================

# This notebook moves from explainability and validation (Notebook 5)
# → to interactive exploration and deployment readiness.
#
# Purpose:
# • Turn SHAP-based explanations and model predictions into an interactive web app.  
# • Enable real-time exploration of individual runs, comparisons, and global insights.  
# • Bridge the gap between data science outputs and user-facing analytics tools.
#
# Inputs:
# • Trained Random Forest Classifier + SHAP explainer (Notebook 5).  
# • PostgreSQL database table `runs_summary` (Notebook 7) containing processed runs.  
# • Feature descriptions and metadata defined in the project’s `src/` folder.
#
# Components:
# 1. Model + data loading with caching (for fast dashboard refresh).  
# 2. Streamlit layout with tabs for single-run, comparison, and global insights.  
# 3. Interactive SHAP visualizations for local and global explanations.  
# 4. Integration hooks for PostgreSQL and model files in `/models`.
#
# Outcomes:
# • A functional Streamlit dashboard that interprets model predictions live.  
# • Tools for comparing runs and understanding what drives performance changes.  
# • A deployable foundation for end-user analytics and future Strava/Garmin integration.
#
# Next step (Notebook 7):
# • Connect the dashboard to the PostgreSQL backend for dynamic queries.  
# • Add data-lineage logging and automated model updates.  
# • Prepare deployment via Docker or Streamlit Cloud.
# ===================================================


# ---------------------------------------------------
# 📦 1) Setup & Imports
# ---------------------------------------------------
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import shap
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 🔧 Robust project root detection
# ---------------------------------------------------
def _find_project_root() -> Path:
    """
    Walk up from __file__ and CWD to find a directory that
    contains both 'src' and 'models'. Fallback to any parent
    named 'running-agent', else parent of this file.
    """
    anchors = [Path(__file__).resolve(), Path.cwd().resolve()]
    visited = set()
    for base in anchors:
        for p in [base] + list(base.parents):
            if p in visited:
                continue
            visited.add(p)
            if (p / "src").exists() and (p / "models").exists():
                return p
    # Fallback by folder name
    for p in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        if p.name.lower() == "running-agent":
            return p
    return Path(__file__).resolve().parents[1]

project_root = _find_project_root()
sys.path.append(str(project_root))

# ---------------------------------------------------
# 🧩 Custom Project Imports (after root on path)
# ---------------------------------------------------
# db_utils is your thin PostgreSQL adapter (engine factory).
# If you later swap DBs (e.g., DuckDB, SQLite), you only need to change db_utils.py
from src.db_utils import get_engine  # noqa: E402

# ---------------------------------------------------
# ⚙️ Streamlit Setup (safe to ignore warnings in Jupyter)
# ---------------------------------------------------
# This config only applies when launched via `streamlit run`.
# In notebooks, Streamlit “widgets” no-op; we guard with try/except to avoid noise.
try:
    st.set_page_config(page_title="🏃‍♂️ Running Insights Dashboard", layout="wide")
except Exception:
    pass  # fine in Jupyter

# ---------------------------------------------------
# 📁 Model paths (with filename fallbacks)
# ---------------------------------------------------
# Where we look for model artifacts. If you rename the model file, add it below.
# Tip: standardize artifact names in Notebook 4/5 when saving.
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

print("✅ Project root added to path:", project_root)
print("📂 Model directory:", model_dir)
print(" - Classifier exists:", bool(rf_model_path), "→", rf_model_path)
print(" - SHAP explainer exists:", explainer_path.exists(), "→", explainer_path)

with st.sidebar:
    st.caption("🔎 Paths (debug)")
    st.write("Project root:", str(project_root))
    st.write("Model dir:", str(model_dir))
    st.write("Classifier:", str(rf_model_path))
    st.write("Explainer:", str(explainer_path))
    try:
        listing = sorted([p.name for p in model_dir.iterdir()])
        st.write("models/ contents:", listing)
    except Exception as e:
        st.write("models/ not accessible:", e)

# ---------------------------------------------------
# 🧠 2) Load models + data
# ---------------------------------------------------
@st.cache_resource
def load_models():
    """Load trained RF classifier + SHAP explainer (if present)."""
    if rf_model_path is None:
        raise FileNotFoundError(
            f"Could not find a classifier model in {model_dir}. Tried: {_model_candidates}"
        )
    rf_clf = joblib.load(rf_model_path)
    explainer = joblib.load(explainer_path) if explainer_path.exists() else None
    if explainer is None:
        st.warning("No SHAP explainer found — SHAP plots will be skipped.")
    return rf_clf, explainer


@st.cache_data
def load_data():
    """Read processed runs from PostgreSQL."""
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM runs_summary ORDER BY date DESC", engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


rf_clf, explainer_clf = load_models()
summary_df = load_data()

st.success("Models and data loaded — ready for dashboard.")
st.caption(f"Data shape: {summary_df.shape}")

# ---------------------------------------------------
# 🧾 Feature dictionary (for headers) + feature alignment
# ---------------------------------------------------
# Human-readable names for your features (shown in expander panes).
# Extend this dict as you add engineered features in earlier notebooks.
FEATURE_DESCRIPTIONS = {
    "total_distance_km": "Distance of run (km)",
    "duration_min": "Duration (minutes)",
    "avg_pace_min_km": "Average pace (min/km)",
    "avg_cadence": "Steps per minute",
    "total_elev_gain": "Elevation gain (m)",
    "avg_stride_len_m": "Average stride length (m)",
    "avg_gct_est_ms": "Ground contact time (ms)",
    "pace_variability": "Pace consistency index",
    "cadence_drift": "Cadence stability index",
    "load_7d": "7-day rolling training load",
    "load_28d": "28-day rolling training load",
    "fastest_1km_pace": "Fastest 1 km segment (min/km)",
    "fastest_5min_pace": "Fastest 5 min pace (min/km)",
}

# ✅ Use the model's training feature order to avoid mismatches
if hasattr(rf_clf, "feature_names_in_"):
    available_features = list(rf_clf.feature_names_in_)
else:
    # fallback: intersect with known feature descriptions
    available_features = [f for f in FEATURE_DESCRIPTIONS if f in summary_df.columns]

# ---------------------------------------------------
# 🧮 SHAP helper (robust to various output shapes)
# ---------------------------------------------------
def shap_vector_for_sample(explainer, model, X_one_row: pd.DataFrame):
    """
    Return a 1-D SHAP vector (len = n_features) + the raw SHAP output.

    Why this exists:
      • SHAP returns different shapes depending on backend, classes, and sample size.
        This function normalizes those cases so the rest of the UI can stay simple.

    Input contract:
      • X_one_row must be a DataFrame with EXACTLY one row and columns aligned to the model.

    Output:
      • v: np.ndarray of length n_features — contribution per feature for the predicted class (if classifier)
      • raw: the unmodified object returned by explainer.shap_values (list/array), useful for debugging/plots
    """
    assert X_one_row.shape[0] == 1, "Pass exactly one sample"
    nfeat = X_one_row.shape[1]

    try:
        c = int(np.argmax(model.predict_proba(X_one_row), axis=1)[0])
    except Exception:
        c = 0

    raw = explainer.shap_values(X_one_row)
    arr = np.asarray(raw)

    if isinstance(raw, list):
        v = np.array(raw[c]).reshape(-1)
        if v.shape[0] == nfeat:
            return v, raw
        elif v.ndim > 1 and v.shape[1] == nfeat:
            return v[0], raw
        else:
            raise ValueError(f"List SHAP shape {v.shape} mismatch (expected {nfeat})")
    elif arr.ndim == 3 and arr.shape[1] == nfeat:
        v = arr[0, :, c]
    elif arr.ndim == 3 and arr.shape[2] == nfeat:
        v = arr[0, c, :]
    elif arr.ndim == 2 and arr.shape == (1, nfeat):
        v = arr[0, :]
    elif arr.ndim == 1 and arr.shape[0] == nfeat:
        v = arr
    else:
        raise ValueError(f"Unexpected SHAP shape {arr.shape}, expected 1D of len {nfeat}")
    return v, raw

# ---------------------------------------------------
# 🧭 3) Layout — Tabs
# ---------------------------------------------------
st.title("🏃‍♂️ Running Insights Dashboard")
st.markdown("""
Use the tabs to explore:
1. **Single run** prediction + SHAP explanation  
2. **Compare two runs** to see what changed  
3. **Global insights** (feature importance + trends)
""")

tab1, tab2, tab3 = st.tabs(["🔍 Single Run", "📈 Compare Runs", "🌍 Global Insights"])

# ---------------------------------------------------
# 🔍 Tab 1 — Single Run Explainability
# ---------------------------------------------------
with tab1:
    st.subheader("Single Run SHAP Explanation")

    # UX: pick two dates; we compute SHAP for each and show (Run2 − Run1) impact difference.
    options = summary_df["date"].dt.strftime("%Y-%m-%d").tolist()
    if not options:
        st.warning("No runs available.")
    else:
        selected_date = st.selectbox("Select a run date", options)
        case = summary_df[summary_df["date"].dt.strftime("%Y-%m-%d") == selected_date]

        if case.empty:
            st.warning("No data for selected date.")
        else:
            # Build X (align to training features, impute simple means)
            X_means = summary_df[available_features].mean()
            case_X = case[available_features].fillna(X_means).iloc[[0]]

            # Predict
            pred_label = rf_clf.predict(case_X)[0]
            st.markdown(f"### 🏷️ Predicted Cluster: **{pred_label}**")

            # Explain
            if explainer_clf is None:
                st.info("SHAP explainer missing — skipping local explanation.")
            else:
                shap_vec, _ = shap_vector_for_sample(explainer_clf, rf_clf, case_X)
                contrib = (
                    pd.Series(shap_vec, index=case_X.columns)
                    .sort_values(key=lambda x: x.abs(), ascending=False)
                )
                st.bar_chart(contrib.head(10))
                with st.expander("Feature descriptions"):
                    desc_df = pd.DataFrame({
                        "feature": case_X.columns,
                        "description": [FEATURE_DESCRIPTIONS.get(c, "") for c in case_X.columns]
                    })
                    st.dataframe(desc_df)

# ---------------------------------------------------
# 📈 Tab 2 — Compare Two Runs
# ---------------------------------------------------
with tab2:
    st.subheader("Compare Two Runs (SHAP Difference)")

    dates = summary_df["date"].dt.strftime("%Y-%m-%d").tolist()
    if len(dates) < 2:
        st.info("Need at least two runs to compare.")
    else:
        col1, col2 = st.columns(2)
        d1 = col1.selectbox("Run 1 date", dates, index=0, key="cmp1")
        d2 = col2.selectbox("Run 2 date", dates, index=1, key="cmp2")

        df1 = summary_df[summary_df["date"].dt.strftime("%Y-%m-%d") == d1]
        df2 = summary_df[summary_df["date"].dt.strftime("%Y-%m-%d") == d2]

        if explainer_clf and (not df1.empty and not df2.empty):
            X_means = summary_df[available_features].mean()
            x1 = df1[available_features].fillna(X_means).iloc[[0]]
            x2 = df2[available_features].fillna(X_means).iloc[[0]]
            shap1, _ = shap_vector_for_sample(explainer_clf, rf_clf, x1)
            shap2, _ = shap_vector_for_sample(explainer_clf, rf_clf, x2)
            diff = pd.Series(shap2 - shap1, index=available_features).sort_values(
                key=lambda x: x.abs(), ascending=False
            )
            st.bar_chart(diff.head(10))
            st.caption("Difference in SHAP contributions between the two runs (Run 2 − Run 1).")
        else:
            st.warning("Ensure both runs and the SHAP explainer are available.")

# ---------------------------------------------------
# 🌍 Tab 3 — Global Insights
# ---------------------------------------------------
with tab3:
    st.subheader("Global Feature Importance and Trends")
    
    # GLOBAL SHAP:
    # We compute mean absolute SHAP per feature across the dataset shown in the dashboard.
    # This gives a high-level “which features matter most on average?” view.
    if explainer_clf:
        X = summary_df[available_features]
        # For multi-class TreeExplainer, shap_values can be list or 3D array
        sv = explainer_clf.shap_values(X)
        arr = np.asarray(sv)

        # Compute mean |SHAP| per feature
        if isinstance(sv, list):
            # list of (n_samples, n_features)
            stacked = np.stack([np.abs(s) for s in sv], axis=0)  # (n_classes, n_samples, n_features)
            mean_abs = stacked.mean(axis=(0, 1))  # (n_features,)
        else:
            if arr.ndim == 3 and arr.shape[1] == X.shape[1]:
                # (n_samples, n_features, n_classes)
                mean_abs = np.abs(arr).mean(axis=(0, 2))
            elif arr.ndim == 3 and arr.shape[2] == X.shape[1]:
                # (n_samples, n_classes, n_features)
                mean_abs = np.abs(arr).mean(axis=(0, 1))
            elif arr.ndim == 2 and arr.shape[1] == X.shape[1]:
                # (n_samples, n_features)
                mean_abs = np.abs(arr).mean(axis=0)
            else:
                raise ValueError(f"Unexpected global SHAP shape {arr.shape}")

        global_df = pd.DataFrame({"feature": available_features, "importance": mean_abs})
        global_df = global_df.sort_values("importance", ascending=False)

        chart = alt.Chart(global_df.head(10)).mark_bar().encode(
            x=alt.X("importance:Q", title="Mean |SHAP|"),
            y=alt.Y("feature:N", sort="-x", title="Feature")
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("Mean absolute SHAP importance across all runs.")

        with st.expander("Feature descriptions"):
            desc_df = pd.DataFrame({
                "feature": available_features,
                "description": [FEATURE_DESCRIPTIONS.get(c, "") for c in available_features]
            })
            st.dataframe(desc_df)
    else:
        st.info("No SHAP explainer — global importance not available.")
        
    # Simple performance trend (example metric). Add more KPIs as needed.
    st.markdown("#### Performance trend (avg pace over time)")
    if "avg_pace_min_km" in summary_df.columns:
        trend = alt.Chart(summary_df).mark_line(point=True).encode(
            x="date:T", y=alt.Y("avg_pace_min_km:Q", title="Avg pace (min/km)")
        )
        st.altair_chart(trend, use_container_width=True)
    else:
        st.info("Column 'avg_pace_min_km' not found in data.")


# ---------------------------------------------------
# 🏁 Wrap-Up — Notebook 6 Summary
# ---------------------------------------------------
# ✅ In this notebook we transformed static model explanations into a fully
#    *interactive dashboard* — connecting predictions, SHAP insights, and
#    PostgreSQL data into one cohesive user experience.
#
# 🔹 Key Results:
# - Built a **Streamlit dashboard** with modular tabs for single-run, comparison,
#   and global insights.
# - Integrated **trained models + SHAP explainers** for live, local feature attributions.
# - Connected to the **PostgreSQL backend** (via SQLAlchemy) to read processed data dynamically.
# - Created reusable functions for consistent SHAP vector extraction and display.
#
# 🔹 Insights:
# - Users can now explore model decisions visually — understanding *why* a run
#   was classified as “easy,” “endurance,” or “interval.”
# - Comparing SHAP profiles between runs reveals subtle shifts in training load,
#   cadence, or elevation that influence predictions.
# - Global feature summaries highlight long-term performance drivers and training trends.
#
# 🔹 Deliverables:
# - A deployable **Streamlit app template** (notebook6.py) ready for Docker packaging.
# - Unified SHAP utilities (`xai_utils.py`) for local/global explanations.
# - Live database connection for real-time data exploration and updates.

