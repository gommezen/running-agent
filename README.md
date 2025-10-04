# 🏃 Running Agent — Data Science Project

This project analyzes running data (Garmin/Strava `.fit` files) to extract insights, track training load, and build predictive models.  
It is both a personal improvement tool and a showcase of data science, ML, and analytics skills.

---

## 📊 Project Overview

**Purpose**:  
- Understand and visualize running patterns.  
- Track progress (distance, pace, cadence, training load).  
- Cluster runs into natural types (easy, tempo, intervals, hilly).  
- Build predictive models for pace and fatigue.  
- Prototype a "Tamagotchi-style agent" that suggests training intensity.

---

## 📂 Structure

- `data/` → raw `.fit.gz` files, parsed CSVs, and aggregated summaries (ignored in Git).  
- `notebooks/` → stepwise analysis:
  - `01_explore_data.ipynb` → per-run analysis & visualizations.
  - `02_feature_engineering.ipynb` → multi-run aggregation & features.
  - `03_clustering_runs.ipynb` → clustering runs into training types.
  - `04_predictive_models.ipynb` → predictive ML (pace, fatigue, recommendations).
- `src/` → Python modules for data ingestion, feature engineering, and clustering logic.
- `requirements.txt` → Python dependencies.

---

## 🚀 Getting Started

### 1. Clone Repo
```bash
git clone https://github.com/YOURUSERNAME/running-agent.git
cd running-agent
