from pathlib import Path

import pandas as pd

from src.features.engineering import process_run
from src.ingestion.parse_fit import parse_fit_file
from src.ingestion.unpack_fit import unpack_fit_gz

raw_dir = Path("data/raw")
interim_dir = Path("data/interim")
processed_dir = Path("data/processed")

interim_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Unpack .fit.gz into .fit
gz_files = list(raw_dir.glob("*.fit.gz"))
print(f"Found {len(gz_files)} .fit.gz files")

for gz in gz_files:
    fit_path = unpack_fit_gz(gz, raw_dir)  # saves alongside gz
    print(f"Unpacked {gz.name} → {fit_path.name}")

# Step 2: Parse .fit into interim CSV
fit_files = list(raw_dir.glob("*.fit"))
print(f"Found {len(fit_files)} .fit files")

for f in fit_files:
    out_path = interim_dir / f"{f.stem}.csv"
    if out_path.exists():
        print(f"⏭️ Skipping {f.name} (already parsed)")
    else:
        try:
            df = parse_fit_file(f)
            df.to_csv(out_path, index=False)
            print(f"✅ Parsed {f.name} → {out_path.name}")
        except Exception as e:
            print(f"⚠️ Failed on {f.name}: {e}")

# Step 3: Create/update runs_summary.csv
summaries = []
for f in interim_dir.glob("*.csv"):
    df, summary = process_run(f)
    summaries.append(summary)

summary_df = pd.DataFrame(summaries)
out_summary = processed_dir / "runs_summary.csv"
summary_df.to_csv(out_summary, index=False)
print(f"💾 Saved {len(summary_df)} runs → {out_summary}")
