# 🏃 Running Agent — Data Science Project

This project analyzes running data (Garmin/Strava `.fit` files) to extract insights, track training load, and build predictive models.  
It serves both as a **personal training analytics tool** and a **data-science portfolio project** demonstrating reproducible pipelines, explainable ML, and SQL database integration.

---

## 📊 Overview

**Purpose**
- Understand and visualize running patterns  
- Track progress (distance, pace, cadence, training load)  
- Cluster runs into natural types (easy, tempo, intervals, hilly)  
- Build predictive models for pace and fatigue  
- Prototype a “Tamagotchi-style” agent that suggests training intensity

---

## 📓 Notebook Workflow

| Notebook | Focus | Key Outputs |
|-----------|--------|-------------|
| **01_explore_data** | Load & explore Garmin/Strava data | Basic stats & visualizations |
| **02_feature_engineering** | Compute derived features (load, variability, cadence drift) | `runs_summary.csv` |
| **03_clustering_runs** | Group runs by training type using unsupervised learning | Cluster labels |
| **04_predictive_models** | Random Forest regression & classification | Pace & cluster predictions |
| **05_model_interpretation** | SHAP explainability | Global + local feature contributions |
| **06_interactive_dashboard** *(next)* | Streamlit app for prediction & explainability | Live UI |
| **07_postgresql_storage** | Persist processed data & SHAP summaries to PostgreSQL | Tables: `runs_summary`, `shap_importance_global`, `data_lineage` |

---

## 🗄️ Database Integration

- **PostgreSQL 16** is used for structured, secure data storage  
- **SQLAlchemy** handles database connections and read/write operations  
- Created tables:
  - `runs_summary` → per-run features and training stats  
  - `shap_importance_global` → mean SHAP feature importances  
  - `data_lineage` → dataset version and timestamp logs
- Includes validation queries for:
  - Weekly statistics  
  - Top-importance SHAP features  
  - Metadata logging for reproducibility  

---

## 🧱 Folder Structure

running-agent/
│
├── data/
│ ├── raw/ # ignored
│ ├── interim/ # ignored
│ ├── processed/ # local only (ignored in Git)
│ └── .gitkeep
│
├── notebooks/
│ ├── 01_explore_data.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_clustering_runs.ipynb
│ ├── 04_predictive_models.ipynb
│ ├── 05_model_interpretation.ipynb
│ ├── 06_interactive_dashboard.ipynb
│ └── 07_postgresql_storage.ipynb
│
├── src/
│ ├── init.py
│ └── db_utils.py
│
├── requirements.txt
├── .gitignore
├── README.md
└── check_storage.sql


---

## ⚙️ Environment Setup

```bash
# Clone the repository
git clone https://github.com/<YOUR_USERNAME>/running-agent.git
cd running-agent

# Create environment and install dependencies
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Test database connection
python -m src.db_utils


🧩 Next Steps for this project

1. Notebook 6 → Build Streamlit dashboard for live predictions
2. Notebook 8 → Add monitoring and automatic database logging
3. Dockerize the app for portable deployment
4. Enable GitHub Actions for CI/CD checks
5. Integrate Garmin/Strava API for automated data ingestion

🪪 License

MIT License © 2025 Niels Jørgen Gommesen