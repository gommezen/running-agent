# TODO — Running Agent

## High Priority
- [ ] Dashboard UX + Themes: Cohesive theme, metric cards (weekly mileage, pace trend, streak), sidebar filters, consistent styling
- [ ] AI Model Selection: Toggle between RF classifier/regressor in dashboard, compare predictions, dynamic SHAP explanations per model
- [ ] Dockerize: docker-compose.yml with Streamlit + PostgreSQL, one-command setup

## Next Up
- [ ] Notebook 8 — Monitoring & Automated Logging: Lineage tracking, model-version logging, automated SHAP summaries
- [ ] API Integration (Garmin / Strava): Enable automatic ingestion of new running data
- [ ] Agent Iteration v0.3+: Adaptive "Running Agent" with personalized training insights and recommendations

## Completed
- [x] Notebook 7 — PostgreSQL Storage: Data stored persistently, queried live via SQLAlchemy
- [x] Pre-commit hooks: Ruff, Mypy, Pytest, YAML/whitespace hygiene
- [x] GitHub Actions CI pipeline
- [x] Humanized dashboard with SHAP explainability
