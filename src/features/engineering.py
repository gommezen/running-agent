from pathlib import Path
from typing import Any

import pandas as pd


def process_run(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a per-run CSV and return both:
      • df: the raw DataFrame
      • summary: a dict with aggregated stats (one row per run)

    Args:
        csv_path: Path or string pointing to a single run CSV file.

    Returns:
        Tuple of (df, summary):
            df (pd.DataFrame): Loaded and parsed run data.
            summary (dict): Aggregated run-level statistics.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Ensure required columns exist
    distance = df["distance"].max() if "distance" in df else None
    duration = (
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60
        if "timestamp" in df
        else None
    )
    avg_pace = None
    if distance and duration:
        avg_pace = duration / (distance / 1000)  # min per km

    summary: dict[str, Any] = {
        "filename": Path(csv_path).name,
        "date": df["timestamp"].min().date() if "timestamp" in df else None,
        "total_distance_km": distance / 1000 if distance else None,
        "duration_min": duration,
        "avg_pace_min_km": avg_pace,
        "avg_cadence": df["cadence"].mean() if "cadence" in df else None,
        "total_elev_gain": (
            df["altitude"].diff().clip(lower=0).sum() if "altitude" in df else None
        ),
    }

    return df, summary
