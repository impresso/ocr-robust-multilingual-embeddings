# ocr_simulator/examples/demo.py
import os
from pathlib import Path
import shutil
from ocr_simulator import OCRSimulator
import pandas as pd


def ensure_output_dir(base_dir: str) -> str:
    """Create output directory with error handling."""
    try:
        output_dir = os.path.join(os.getcwd(), base_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Using output directory: {output_dir}")

        total, used, free = shutil.disk_usage(os.path.dirname(output_dir))
        if free < 1_000_000_000:  # Less than 1GB
            print("Warning: Low disk space available!")

        return output_dir
    except OSError as e:
        print(f"Error creating output directory: {e}")
        temp_dir = os.path.join(os.path.expanduser("~"), "temp_ocr_output")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Falling back to temporary directory: {temp_dir}")
        return temp_dir


def demo_simple_text():
    """Demo using single text input."""
    print("\n=== Demo 1: Simple Text Processing ===")

    output_dir = ensure_output_dir("ocr_output/simple")

    simulator = OCRSimulator(
        condition='simple',
        language='eng',
        save_images=True,
        output_dir=output_dir
    )

    text = "This is a sample text for OCR simulation."
    result = simulator.process_input(
        text,
        input_type='text',
        save_image=True,
        image_filename=os.path.join(output_dir, "sample.png")
    )

    print(f"Original text: {result['original_text']}")
    print(f"OCR output: {result['ocr_text']}")


def demo_blackletter():
    """Demo using blackletter font."""
    print("\n=== Demo 2: Blackletter Font Processing ===")

    output_dir = ensure_output_dir("ocr_output/blackletter")

    simulator = OCRSimulator(
        condition='blackletter',
        language='deu',
        font_path="/Library/Fonts/Canterbury.ttf",  # Update with your font path
        save_images=True,
        output_dir=output_dir
    )

    german_text = "Dies ist ein Beispieltext in deutscher Sprache."
    result = simulator.process_input(german_text, input_type='text')

    print(f"Original text: {result['original_text']}")
    print(f"OCR output: {result['ocr_text']}")


def demo_distorted():
    """Demo using distorted text."""
    print("\n=== Demo 3: Distorted Text Processing ===")

    output_dir = ensure_output_dir("ocr_output/distorted")

    simulator = OCRSimulator(
        condition='distorted',
        language='eng',
        save_images=True,
        output_dir=output_dir,
        config={
            'skew_range': (-0.06, 0.06),
            'incomplete_prob': 0.15,
            'gap_range': (1, 3),
            'text_noise_range': (-30, 30),
            'bg_noise_prob': 0.05,
            'bg_noise_range': (-10, 10)
        }
    )

    # Create DataFrame from list
    texts = [
        "First sample text with distortion.",
        "Second sample text to process.",
        "Third text with different content."
    ]
    df = pd.DataFrame({'text': texts})

    results = simulator.process_dataframe(
        df,
        output_csv=os.path.join(output_dir, "results.csv"),
        image_prefix="distorted"
    )
    print("\nProcessed multiple texts with distortion:")
    print(results)


def demo_noisy():
    """Demo using noisy background."""
    print("\n=== Demo 4: Noisy Background Processing ===")

    output_dir = ensure_output_dir("ocr_output/noisy")

    simulator = OCRSimulator(
        condition='noisy',
        language='eng',
        save_images=True,
        output_dir=output_dir,
        config={'dot_density': 0.0045},
        n_jobs=1
    )

    df = pd.DataFrame({
        'text': [
            "Sample text with noise.",
            "Another text to process.",
            "Testing noise effects."
        ]
    })

    results = simulator.process_dataframe(
        df,
        output_csv=os.path.join(output_dir, "results.csv"),
        image_prefix="noisy"
    )
    print("\nProcessed DataFrame with noise:")
    print(results)


def main():
    """Run all demos."""
    try:
        demo_simple_text()
        demo_blackletter()
        demo_distorted()
        demo_noisy()
    except Exception as e:
        print(f"\nError running demo: {e}")
        print("Please check disk space and permissions.")


if __name__ == "__main__":
    main()
