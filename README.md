# 🏃 Running Agent — Data Science & Explainable ML Project

[![CI](https://github.com/gommezen/running-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/gommezen/running-agent/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.46-purple)
![Ruff](https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white)

**Running Agent** analyzes Garmin/Strava running data to extract insights, track training load, and build predictive, explainable models.
It serves both as a **personal training analytics tool** and a **data-science portfolio project** showcasing reproducible pipelines, interpretable ML, and SQL-backed dashboards.

---

## 📊 Overview

### Purpose
- Understand and visualize individual running patterns
- Track key performance indicators (distance, pace, cadence, load)
- Cluster runs into natural categories (easy, tempo, hilly, intervals)
- Predict pace and fatigue using Random Forest models
- Prototype a *Tamagotchi-style running agent* that suggests training intensity

### Core Concepts
- End-to-end ML workflow: raw Garmin `.fit` → cleaned dataset → ML models
- Explainable AI (SHAP) for transparent model behaviour
- Interactive dashboard powered by **Streamlit**
- **PostgreSQL + SQLAlchemy** for structured, persistent data storage

---

## 📓 Notebook Workflow

| Notebook | Focus | Key Outputs |
|----------|--------|-------------|
| **01_explore_data** | Load & inspect Garmin/Strava data | Basic stats & visualizations |
| **02_feature_engineering** | Compute derived metrics (load, variability, cadence drift) | `runs_summary.csv` |
| **03_clustering_runs** | Unsupervised learning for run grouping | Cluster labels |
| **04_predictive_models** | Random Forest regression + classification | Pace & run-type models |
| **05_model_interpretation** | SHAP explainability | Global & local feature attributions |
| **06_interactive_dashboard** | Streamlit app | Interactive UI |
| **07_postgresql_storage** | Save data + SHAP results to PostgreSQL | Tables: `runs_summary`, `shap_importance_global`, `data_lineage` |

---

## 🗄️ Database Integration

- **PostgreSQL 16** for structured, durable storage
- **SQLAlchemy** for engine creation and ORM-style interactions

**Core tables**

- `runs_summary` — per-run feature set
- `shap_importance_global` — mean SHAP values across features
- `data_lineage` — timestamps, dataset versions, transformation logs

Example SQL use cases:
- Weekly summaries & training load trends
- Top SHAP features per model
- Reproducibility checks through lineage

---

## 🧱 Folder Structure

```text
running-agent/
│
├── data/
│   ├── raw/                  # raw Garmin/Strava exports (ignored in Git)
│   ├── interim/              # temporary intermediate outputs
│   ├── processed/            # derived CSV/Parquet files
│   └── sql/                  # database init scripts (ignored in Git)
│
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering_runs.ipynb
│   ├── 04_predictive_models.ipynb
│   ├── 05_model_interpretation.ipynb
│   ├── 07_postgresql_storage.ipynb
│   └── archive/
│
├── src/
│   ├── __init__.py
│   ├── db_utils.py           # PostgreSQL utilities
│   ├── xai_utils.py          # SHAP helper functions
│   ├── features/
│   │   └── engineering.py    # feature engineering pipeline
│   └── ingestion/
│       └── parse_fit.py      # .fit file parsing
│
├── models/                   # trained models (ignored in Git)
│   ├── model_rf_clf.joblib
│   └── shap_explainer_clf.pkl
│
├── 06_interactive_dashboard_humanized.py  # Streamlit dashboard
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md

---

## ⚙️ Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/gommezen/running-agent.git
cd running-agent

# 2. Create and activate environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .\.venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # linting, testing, pre-commit

# 4. Set up PostgreSQL
# Create a database, then create a .env file in the project root:
#   DATABASE_URL=postgresql://user:password@localhost:5432/running_agent

# 5. Test database connection
python -m src.db_utils

# 6. Run the dashboard
streamlit run 06_interactive_dashboard_humanized.py
```

---

🧩 Next Steps for this Project

1. **UX Update — Streamlit Dashboard (🔄 In Progress)**
   Refine layout, tabs, and visual hierarchy for a smoother user experience.
   Add filters, metric cards, and consistent color/label styling.

2. **Notebook 7 → PostgreSQL Storage (✅ Completed)**
   Data now stored persistently in PostgreSQL and queried live via SQLAlchemy.

3. **Notebook 8 → Monitoring & Automated Logging**
   Implement lineage tracking, model-version logging, and automated SHAP summaries.

4. **Dockerize the App**
   Containerize the Streamlit + PostgreSQL setup for portable, reproducible deployment.

5. **CI/CD Integration (GitHub Actions)**
   Automate testing, style checks, and build verification on every commit.

6. **API Integration (Garmin / Strava)**
   Enable automatic ingestion of new running data through connected APIs.

7. **Agent Iteration v0.3+**
   Extend toward an adaptive “Running Agent” that provides personalized training insights and recommendations.

---
