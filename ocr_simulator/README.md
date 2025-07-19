# OCR Simulator

A comprehensive OCR simulation library with various text effects and conditions.

## Features
- Multiple text effects (Simple, Blackletter, Distorted, Noisy)
- Customizable image sizes and font settings
- Multiple input formats support

Run in local: python -m ocr_simulator.examples.demo

Use Tesseract OCR

1. Check which language do you have 
tesseract --list-langs
2. Add the new languages
sudo curl -L https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata -o /opt/homebrew/share/tessdata/fra.traineddata
Change the language name