# ocr_simulator/utils.py
import os


def ensure_directory(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
