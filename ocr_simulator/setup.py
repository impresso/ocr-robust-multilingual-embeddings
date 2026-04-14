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
    author="Yining Wang, Stylianos Psychias",
    author_email="yining.wang@uzh.ch, stylianos.psychias@uzh.ch",
    description="OCR Simulator with various text effects",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/impresso/ocr-robust-multilingual-embeddings",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
