import gzip
import shutil
from pathlib import Path


def unpack_fit_gz(input_path: str | Path, output_dir: str | Path) -> Path:
    """
    Unpack a .fit.gz file into a .fit file and return the new Path.

    Args:
        input_path: Path to the .fit.gz file.
        output_dir: Directory to extract the file into.

    Returns:
        Path to the unpacked .fit file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_path = output_dir / input_path.with_suffix("").name

    # Ensure destination exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open both gzip and output in a single context
    with gzip.open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack .fit.gz into .fit")
    parser.add_argument("input_path", help="Path to .fit.gz file")
    parser.add_argument("--output_dir", default="data/raw", help="Output directory")

    args = parser.parse_args()
    unpack_fit_gz(args.input_path, args.output_dir)
