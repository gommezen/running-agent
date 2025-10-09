from __future__ import annotations

import os
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine

load_dotenv()

DB_URL: str = os.getenv("DB_URL", "postgresql://niels:securepwd@localhost:5432/running_agent")
engine: Engine = create_engine(DB_URL)


def write_table(
    df: pd.DataFrame, table_name: str, mode: Literal["replace", "append", "fail"] = "replace"
) -> None:
    """Write a DataFrame to PostgreSQL."""
    df.to_sql(table_name, engine, if_exists=mode, index=False)
    print(f"✅ Wrote {len(df)} rows to table: {table_name}")


def read_table(table_name: str) -> pd.DataFrame:
    """Read a full table from PostgreSQL."""
    return pd.read_sql_table(table_name, engine)


def get_engine() -> Engine:
    """Return the active SQLAlchemy engine."""
    return engine


if __name__ == "__main__":
    with engine.connect() as conn:
        print("✅ Connected to PostgreSQL successfully!")
