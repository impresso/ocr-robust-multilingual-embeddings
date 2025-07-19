# ocr_simulator/examples/demo2.py

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


def demo_with_custom_sizes():
    """Demo showing different size configurations."""
    print("\n=== Demo: Custom Size Configurations ===")

    output_dir = ensure_output_dir("ocr_output/custom_sizes")

    # Example 1: Large font with specific image size
    simulator_large = OCRSimulator(
        condition='simple',
        language='eng',
        font_size=24,  # Larger font
        image_width=800,  # Fixed width
        image_height=400,  # Fixed height
        save_images=True,
        output_dir=output_dir
    )

    # Example 2: Custom margin with auto-sized image
    simulator_margin = OCRSimulator(
        condition='simple',
        language='eng',
        font_size=12,
        margin=1.0,  # Larger margin
        save_images=True,
        output_dir=output_dir
    )

    # Example 3: Distorted text with fixed size
    simulator_distorted = OCRSimulator(
        condition='distorted',
        language='eng',
        font_size=16,
        image_width=1000,
        image_height=500,
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

    test_text = "Testing custom size configurations for OCR simulation."

    # Process with each configuration
    result1 = simulator_large.process_single_text(
        test_text,
        save_image=True,
        image_filename=os.path.join(output_dir, "large_font.png")
    )
    print("\nLarge Font Configuration:")
    print(f"Original text: {result1['original_text']}")
    print(f"OCR output: {result1['ocr_text']}")

    result2 = simulator_margin.process_single_text(
        test_text,
        save_image=True,
        image_filename=os.path.join(output_dir, "large_margin.png")
    )
    print("\nLarge Margin Configuration:")
    print(f"Original text: {result2['original_text']}")
    print(f"OCR output: {result2['ocr_text']}")

    result3 = simulator_distorted.process_single_text(
        test_text,
        save_image=True,
        image_filename=os.path.join(output_dir, "fixed_size_distorted.png")
    )
    print("\nDistorted Fixed Size Configuration:")
    print(f"Original text: {result3['original_text']}")
    print(f"OCR output: {result3['ocr_text']}")


def main():
    """Run the demo with error handling."""
    try:
        demo_with_custom_sizes()
    except Exception as e:
        print(f"\nError running demo: {e}")
        print("Please check disk space and permissions.")


if __name__ == "__main__":
    main()
