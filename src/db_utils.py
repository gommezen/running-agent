from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL", "postgresql://niels:securepwd@localhost:5432/running_agent")
engine = create_engine(DB_URL)

def write_table(df: pd.DataFrame, table_name: str, mode="replace"):
    """Write a DataFrame to PostgreSQL."""
    df.to_sql(table_name, engine, if_exists=mode, index=False)
    print(f"✅ Wrote {len(df)} rows to table: {table_name}")

def read_table(table_name: str) -> pd.DataFrame:
    """Read a full table from PostgreSQL."""
    return pd.read_sql_table(table_name, engine)


if __name__ == "__main__":
    with engine.connect() as conn:
        print("✅ Connected to PostgreSQL successfully!")