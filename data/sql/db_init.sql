-- Table: runs_summary
-- Stores per-run statistics and clustering results.
-- Includes key performance metrics (distance, duration, pace, cadence, elevation, and rolling loads).
-- 'cluster' links to run type classification; 'run_id' uniquely identifies each run.
-- FLOAT fields capture decimal metrics; 'date' records when the run occurred.

CREATE TABLE runs_summary (
    run_id SERIAL PRIMARY KEY,
    date DATE,
    total_distance_km FLOAT,
    duration_min FLOAT,
    avg_pace_min_km FLOAT,
    avg_cadence FLOAT,
    total_elev_gain FLOAT,
    load_7d FLOAT,
    load_28d FLOAT,
    cluster FLOAT
);

CREATE TABLE shap_importance_global (
    feature VARCHAR(64),
    mean_abs_shap FLOAT
);
