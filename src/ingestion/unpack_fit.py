import gzip
import shutil
from pathlib import Path

def unpack_fit_gz(input_path: str, output_dir: str = "data/raw"):
    """
    Unpacks a .fit.gz file into .fit in the raw data folder.
    """
    input_path = Path(input_path)
    output_path = Path(output_dir) / input_path.with_suffix("").name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(input_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"✅ Unpacked: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack .fit.gz into .fit")
    parser.add_argument("input_path", help="Path to .fit.gz file")
    parser.add_argument("--output_dir", default="data/raw", help="Output directory")

    args = parser.parse_args()
    unpack_fit_gz(args.input_path, args.output_dir)
