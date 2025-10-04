from fitparse import FitFile
import pandas as pd
from pathlib import Path

def parse_fit_file(fit_path: str, output_dir: str = "data/interim"):
    """
    Parse a .fit file into a DataFrame.
    """
    fitfile = FitFile(str(fit_path))   # ensure it's a string
    records = []
    for record in fitfile.get_messages("record"):
        row = {field.name: field.value for field in record}
        records.append(row)

    df = pd.DataFrame(records)
    return df



def fit_to_csv(fit_path: str, output_dir: str = "data/interim") -> pd.DataFrame:
    """
    Parse a .fit file and save it as a CSV in the interim folder.
    """
    df = parse_fit_file(fit_path)
    output_path = Path(output_dir) / (Path(fit_path).stem + ".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Parsed: {output_path} with {len(df)} rows")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse .fit into .csv")
    parser.add_argument("fit_path", help="Path to .fit file")
    parser.add_argument("--output_dir", default="data/interim", help="Output directory")

    args = parser.parse_args()
    fit_to_csv(args.fit_path, args.output_dir)
