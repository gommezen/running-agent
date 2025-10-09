from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd
from fitparse import FitFile


def unpack_fit_gz(input_path: str | Path, output_dir: str | Path) -> Path:
    """Unpack a .fit.gz file into a .fit file and return the new Path."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / input_path.with_suffix("").name

    with gzip.open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return output_path


def parse_fit_file(fit_path: str | Path) -> pd.DataFrame:
    """Parse a .fit file into a DataFrame."""
    fit_path = Path(fit_path)
    fitfile = FitFile(str(fit_path))
    records: list[dict[str, object]] = []

    for record in fitfile.get_messages("record"):
        row = {data.name: data.value for data in record}
        records.append(row)

    return pd.DataFrame(records)


def fit_to_csv(fit_path: str | Path, output_dir: str | Path = "data/interim") -> pd.DataFrame:
    """Parse a .fit file and save it as a CSV in the interim folder."""
    fit_path = Path(fit_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{fit_path.stem}.csv"
    fitfile = FitFile(str(fit_path))
    records: list[dict[str, object]] = [
        {data.name: data.value for data in record} for record in fitfile.get_messages("record")
    ]

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    return df
