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
    """Demo using single text input in Luxembourgish."""
    print("\n=== Demo 1: Simple Luxembourgish Text Processing ===")

    output_dir = ensure_output_dir("ocr_output/simple")

    simulator = OCRSimulator(
        condition='simple',
        language='lb',  # Luxembourgish language code
        save_images=True,
        output_dir=output_dir
    )

    text = "Dëst ass en Test fir OCR Simulatioun."
    result = simulator.process_input(
        text,
        input_type='text',
        save_image=True,
        image_filename=os.path.join(output_dir, "sample.png")
    )

    print(f"Original text: {result['original_text']}")
    print(f"OCR output: {result['ocr_text']}")


def demo_blackletter():
    """Demo using blackletter font with Luxembourgish text."""
    print("\n=== Demo 2: Blackletter Font Processing ===")

    output_dir = ensure_output_dir("ocr_output/blackletter")

    simulator = OCRSimulator(
        condition='blackletter',
        language='lb',  # Luxembourgish language code
        font_path="/Library/Fonts/Canterbury.ttf",  # Update with your font path
        save_images=True,
        output_dir=output_dir
    )

    luxembourgish_text = "Mir wëlle bleiwe wat mir sinn."
    result = simulator.process_input(luxembourgish_text, input_type='text')

    print(f"Original text: {result['original_text']}")
    print(f"OCR output: {result['ocr_text']}")


def demo_distorted():
    """Demo using distorted Luxembourgish text."""
    print("\n=== Demo 3: Distorted Text Processing ===")

    output_dir = ensure_output_dir("ocr_output/distorted")

    simulator = OCRSimulator(
        condition='distorted',
        language='lb',  # Luxembourgish language code
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

    # Create DataFrame with Luxembourgish texts
    texts = [
        "Moien, wéi geet et Iech?",
        "Lëtzebuerg ass e schéint Land.",
        "Ech schwätzen Lëtzebuergesch."
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
    """Demo using noisy background with Luxembourgish text."""
    print("\n=== Demo 4: Noisy Background Processing ===")

    output_dir = ensure_output_dir("ocr_output/noisy")

    simulator = OCRSimulator(
        condition='noisy',
        language='lb',  # Luxembourgish language code
        save_images=True,
        output_dir=output_dir,
        config={'dot_density': 0.0045},
        n_jobs=1
    )

    df = pd.DataFrame({
        'text': [
            "D'Wieder ass haut schéin.",
            "Ech hunn Honger.",
            "Wann ech grouss sinn."
        ]
    })

    results = simulator.process_dataframe(
        df,
        output_csv=os.path.join(output_dir, "results.csv"),
        image_prefix="noisy"
    )
    print("\nProcessed DataFrame with noise:")
    print(results)


def run_blackletter_error_addition(texts):
    output_dir = ensure_output_dir("ocr_output/blackletter")
    simulator = OCRSimulator(
            condition='blackletter',
            language='lb',  # Luxembourgish language code
            font_path="/Library/Fonts/Canterbury.ttf",  # Update with your font path
            save_images=True,
            output_dir=output_dir
    )
    ocr_error_texts = [simulator.process_input(text, input_type='text')['ocr_text'] for text in texts]
    for i in range(5): # print first 5 samples
        print("ORIGINAL: " + texts[i])
        print("Blackletter ERROR: " + str(ocr_error_texts[i]))
        
    return ocr_error_texts

def main():
    """Run all demos."""
    try:
        texts = [
        "Moien, wéi geet et Iech?",
        "Lëtzebuerg ass e schéint Land.",
        "Ech schwätzen Lëtzebuergesch.",
        "Moien, wéi geet et Iech?",
        "Lëtzebuerg ass e schéint Land.",
        "Ech schwätzen Lëtzebuergesch."
        ]
        run_blackletter_error_addition(texts)
        # demo_simple_text()
        # demo_blackletter()
        # demo_distorted()
        # demo_noisy()
    except Exception as e:
        print(f"\nError running demo: {e}")
        print("Please check disk space and permissions.")


if __name__ == "__main__":
    main()
