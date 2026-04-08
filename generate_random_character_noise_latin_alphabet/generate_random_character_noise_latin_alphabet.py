"""Generate synthetic OCR noise using script-specific confusable character tables.

Each script type defines a pool of characters (including common OCR-confusable
glyphs) used for substitution and insertion errors. Deletion errors are script-agnostic.

Output columns follow the noisy_finetuning_data naming convention:
    {column}_{cer_suffix}
where cer_suffix is the integer CER percentage zero-padded to two digits,
e.g. CER=0.04 → suffix '_04', CER=0.10 → suffix '_10'.

Usage
-----
    python generate_random_character_noise.py input.csv \\
        --columns deu fra \\
        --script latin \\
        --cer 0.04 \\
        --seed 42 \\
        -o output.csv

This produces new columns ``deu_04``, ``fra_04`` in the output CSV,
matching the column naming convention of the noisy_finetuning_data folder.
"""

from __future__ import annotations

import argparse
import random
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Script type and confusable character tables
# ---------------------------------------------------------------------------

ScriptType = Literal[
    "latin", "cyrillic", "greek", "arabic", "hebrew", "georgian",
]

LATIN_CHARSET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "    "
    "öüäéèà ÜÄÖ"
)

CYRILLIC_CHARSET = (
    "абвгдежзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "    "
    "ёЁ"
)

GREEK_CHARSET = (
    "αβγδεζηθικλμνξοπρστυφχψω"
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    "    "
    "άέήίόύώ"
)

ARABIC_CHARSET = (
    "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
    "    "
    "ءآأؤإئ"
)

HEBREW_CHARSET = (
    "אבגדהוזחטיכלמנסעפצקרשת"
    "    "
    "ךםןףץ"
)

GEORGIAN_CHARSET = (
    "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"
    "    "
)

CHARSETS: dict[str, str] = {
    "latin":    LATIN_CHARSET,
    "cyrillic": CYRILLIC_CHARSET,
    "greek":    GREEK_CHARSET,
    "arabic":   ARABIC_CHARSET,
    "hebrew":   HEBREW_CHARSET,
    "georgian": GEORGIAN_CHARSET,
}


# ---------------------------------------------------------------------------
# Core noise function
# ---------------------------------------------------------------------------

def apply_ocr_noise(
    text: str,
    script: ScriptType = "latin",
    target_cer: float = 0.04,
) -> str:
    """Return *text* with synthetic OCR errors at roughly *target_cer*."""
    if not isinstance(text, str):
        text = str(text)
    charset = CHARSETS[script]
    n_changes = max(1, int(len(text) * target_cer))
    mutated = list(text)

    for _ in range(n_changes):
        op = random.choice(("substitution", "insertion", "deletion"))

        if op == "substitution" and mutated:
            idx = random.randrange(len(mutated))
            mutated[idx] = random.choice(charset)

        elif op == "insertion":
            idx = random.randrange(max(len(mutated), 1))
            mutated.insert(idx, random.choice(charset))

        elif op == "deletion" and mutated:
            idx = random.randrange(len(mutated))
            del mutated[idx]

    return "".join(mutated)


# ---------------------------------------------------------------------------
# DataFrame helper
# ---------------------------------------------------------------------------

def cer_to_suffix(target_cer: float) -> str:
    """Convert CER float to the dataset column suffix convention.

    Examples
    --------
    >>> cer_to_suffix(0.04)
    '_04'
    >>> cer_to_suffix(0.10)
    '_10'
    >>> cer_to_suffix(0.15)
    '_15'
    """
    return f"_{int(round(target_cer * 100)):02d}"


def noise_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    script: ScriptType = "latin",
    target_cer: float = 0.04,
) -> pd.DataFrame:
    """Add noised copies of *columns* using the dataset naming convention.

    The suffix is derived automatically from *target_cer*:
    CER 0.04 → column ``{col}_04``, CER 0.10 → ``{col}_10``, etc.
    """
    suffix = cer_to_suffix(target_cer)
    out = df.copy()
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input CSV.")
        out[col + suffix] = out[col].apply(
            lambda s: apply_ocr_noise(s, script=script, target_cer=target_cer)
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add synthetic OCR noise columns to a CSV. "
            "Output column names follow the noisy_finetuning_data convention: "
            "{column}_{cer_int}, e.g. deu_04 for German at CER=0.04."
        )
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--columns", nargs="+", required=True,
        help="Column names to noise (e.g. --columns deu fra).",
    )
    parser.add_argument(
        "--script",
        default="latin",
        choices=list(CHARSETS),
        help=(
            "Script / alphabet to use for confusable characters. "
            f"Choices: {', '.join(CHARSETS)}. Default: latin."
        ),
    )
    parser.add_argument(
        "--cer", type=float, default=0.04,
        help=(
            "Target character error rate (default: 0.04). "
            "Controls how many characters are perturbed. "
            "The suffix appended to output column names is derived automatically: "
            "CER 0.04 → _04, CER 0.10 → _10, CER 0.15 → _15."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path (default: overwrites input).",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    df = pd.read_csv(args.input_csv)
    df = noise_dataframe(
        df,
        columns=args.columns,
        script=args.script,
        target_cer=args.cer,
    )
    out_path = args.output or args.input_csv
    df.to_csv(out_path, index=False)
    suffix = cer_to_suffix(args.cer)
    new_cols = [c + suffix for c in args.columns]
    print(f"Wrote {len(df)} rows → {out_path}")
    print(f"New columns: {', '.join(new_cols)}")


if __name__ == "__main__":
    main()