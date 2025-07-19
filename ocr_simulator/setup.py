from setuptools import setup, find_packages

setup(
    name="ocr_simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "Pillow",
        "pytesseract",
        "pandas",
        "joblib",
        "tqdm"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="OCR Simulator with various text effects",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ocr-simulator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
