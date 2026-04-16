"""Apply OCR simulation to a plain-text or CSV file.

Examples
--------
# Noise a plain-text file (one sentence per line):
python apply_ocr_to_file.py input.txt --language deu --condition distorted -o noised.txt

# Noise specific columns of a CSV:
python apply_ocr_to_file.py data.csv --columns deu fra --language deu --condition distorted -o noised.csv

# Try different conditions:
python apply_ocr_to_file.py input.txt --language ell --condition noisy -o noised.txt

Supported languages (requires matching Tesseract language pack):
    eng, deu, fra, spa, ltz, rus, ell, ara, heb, kat
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from ocr_simulator import OCRSimulator
from ocr_simulator.languages import LANGUAGE_CONFIGS


def ocr_noise_text(
    text: str,
    sim: OCRSimulator,
) -> str:
    """Run a single string through the OCR simulator and return the noised text."""
    result = sim.process_single_text(text)
    return result["ocr_text"] or text


def process_plain_text(
    in_path: Path,
    out_path: Path,
    sim: OCRSimulator,
) -> int:
    """Noise every non-empty line of a plain-text file. Returns the line count."""
    lines_written = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.rstrip("\n")
            if stripped:
                noised = ocr_noise_text(stripped, sim)
            else:
                noised = ""
            fout.write(noised + "\n")
            lines_written += 1
    return lines_written


def process_csv(
    in_path: Path,
    out_path: Path,
    columns: list[str],
    sim: OCRSimulator,
    suffix: str = "_ocr",
) -> int:
    """Add OCR-noised copies of *columns* to a CSV. Returns the row count."""
    import pandas as pd

    df = pd.read_csv(in_path)
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
        df[col + suffix] = df[col].astype(str).apply(lambda s: ocr_noise_text(s, sim))
    df.to_csv(out_path, index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply OCR simulation noise to a text or CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Path to the input file (.txt or .csv).")
    parser.add_argument(
        "--language",
        default="eng",
        choices=sorted(LANGUAGE_CONFIGS),
        help="Tesseract language code (default: eng).",
    )
    parser.add_argument(
        "--condition",
        default="distorted",
        choices=["simple", "blackletter", "distorted", "noisy"],
        help="OCR simulation condition (default: distorted).",
    )
    parser.add_argument("--font-size", type=int, default=12, help="Font size in pt (default: 12).")
    parser.add_argument("--dpi", type=int, default=300, help="Image DPI (default: 300).")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="CSV columns to noise (required for CSV input).",
    )
    parser.add_argument(
        "--suffix",
        default="_ocr",
        help="Suffix for noised CSV columns (default: _ocr).",
    )
    parser.add_argument("-o", "--output", default=None, help="Output file path.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Error: file not found: {in_path}")

    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_ocr")

    sim = OCRSimulator(
        condition=args.condition,
        language=args.language,
        font_size=args.font_size,
        dpi=args.dpi,
        save_images=False,
    )

    is_csv = in_path.suffix.lower() == ".csv"

    if is_csv:
        if not args.columns:
            sys.exit("Error: --columns is required for CSV input.")
        n = process_csv(in_path, out_path, args.columns, sim, suffix=args.suffix)
        print(f"Wrote {n} rows → {out_path}")
    else:
        n = process_plain_text(in_path, out_path, sim)
        print(f"Wrote {n} lines → {out_path}")


if __name__ == "__main__":
    main()
