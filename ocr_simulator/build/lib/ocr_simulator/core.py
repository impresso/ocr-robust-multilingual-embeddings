# ocr_simulator/core.py
import numpy as np
import textwrap
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Optional, Dict, List, Union, Literal, Tuple
import pytesseract
import random
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from .languages import LANGUAGE_CONFIGS
from .effects import apply_effects
from .utils import ensure_directory


# Update the OCRSimulator class initialization in core.py

class OCRSimulator:
    """A comprehensive OCR simulator supporting multiple conditions."""

    SUPPORTED_CONDITIONS: List[str] = [
        'simple', 'blackletter', 'distorted', 'noisy']

    def __init__(
        self,
        condition: str = 'simple',
        language: str = "eng",
        font_path: Optional[str] = None,
        font_size: int = 10,
        dpi: int = 300,
        save_images: bool = False,
        output_dir: Optional[str] = None,
        n_jobs: int = -1,
        config: Optional[Dict] = None,
        # Add new parameters
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        margin: float = 0.5
    ):
        """Initialize the OCR simulator.

        Args:
            condition: Type of OCR simulation
            language: Language code
            font_path: Path to font file
            font_size: Font size in points
            dpi: DPI for image generation
            save_images: Whether to save generated images
            output_dir: Directory for saved images
            n_jobs: Number of parallel jobs
            config: Additional configuration
            image_width: Fixed width for generated images (optional)
            image_height: Fixed height for generated images (optional)
            margin: Margin as a fraction of DPI (default: 0.5)
        """
        if condition not in OCRSimulator.SUPPORTED_CONDITIONS:
            raise ValueError(
                f"Condition must be one of {OCRSimulator.SUPPORTED_CONDITIONS}")

        self.condition = condition
        self.language = language
        self.font_size = font_size
        self.dpi = dpi
        self.save_images = save_images
        self.output_dir = output_dir
        self.n_jobs = n_jobs

        # Store size-related parameters
        self.custom_width = image_width
        self.custom_height = image_height
        self.margin_multiplier = margin

        # Precompute constants
        self.font_size_px = int(font_size * dpi / 72)
        self.margin_px = int(self.margin_multiplier * self.dpi)
        self.max_width_px = (image_width if image_width else
                             int(8.5 * self.dpi) - 2 * self.margin_px)

        # Set default configuration
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        if save_images and output_dir:
            ensure_directory(output_dir)

        # Load language configuration
        self.lang_config = LANGUAGE_CONFIGS.get(
            language, LANGUAGE_CONFIGS['eng'])

        # Store font path instead of font object
        self.font_path = self._get_font_path(font_path)

        # Test font loading
        try:
            test_font = ImageFont.truetype(self.font_path, self.font_size_px)
        except Exception as e:
            print(
                f"Warning: Could not load font {self.font_path}. Using default. Error: {e}")

    def _get_default_config(self) -> Dict:
        """Get default configuration based on condition."""
        if self.condition == 'simple':
            return {}

        elif self.condition == 'blackletter':
            return {
                'font_multiplier': 1.2
            }

        elif self.condition == 'distorted':
            return {
                'skew_range': (-0.06, 0.06),
                'incomplete_prob': 0.15,
                'gap_range': (1, 3),
                'text_noise_range': (-30, 30),
                'bg_noise_prob': 0.05,
                'bg_noise_range': (-10, 10)
            }

        else:  # noisy
            return {
                'dot_density': 0.0045
            }

    def _get_font_path(self, font_path: Optional[str]) -> str:
        """Get the appropriate font path."""
        if font_path:
            return font_path
        if self.condition == 'blackletter':
            return self.lang_config['blackletter_font']
        return self.lang_config['default_font']

    def _initialize_font(self) -> ImageFont.ImageFont:
        """Initialize and return a font object."""
        try:
            return ImageFont.truetype(self.font_path, self.font_size_px)
        except Exception as e:
            print(
                f"Warning: Could not load font {self.font_path}. Using default. Error: {e}")
            return ImageFont.load_default()

    def text_to_image(
        self,
        text: str,
        output_path: Optional[str] = None,
        bg_color: str = "white",
        text_color: str = "black"
    ) -> Image.Image:
        """Convert text to image with optional effects."""
        font = self._initialize_font()

        wrapper = textwrap.TextWrapper(
            width=self.max_width_px // self.font_size_px,
            break_long_words=False
        )
        wrapped_text = wrapper.wrap(text) or ['']

        # Calculate dimensions
        text_width_px = max(font.getbbox(line)[2] for line in wrapped_text)
        text_height_px = sum(
            font.getbbox(line)[3] - font.getbbox(line)[1]
            for line in wrapped_text
        )

        # Use custom dimensions if provided, otherwise calculate based on text
        if self.custom_width:
            image_width_px = self.custom_width
        else:
            image_width_px = int(
                (text_width_px + 2 * self.margin_px) / self.dpi * self.dpi)
            if self.condition in ['distorted']:
                image_width_px = int(image_width_px * 1.2)

        if self.custom_height:
            image_height_px = self.custom_height
        else:
            image_height_px = int(
                (text_height_px + 2 * self.margin_px) / self.dpi * self.dpi)
            if self.condition in ['distorted']:
                image_height_px = int(image_height_px * 1.2)

        # Create image
        image = Image.new("RGB", (image_width_px, image_height_px), bg_color)
        draw = ImageDraw.Draw(image)

        # Center text if using custom dimensions
        if self.custom_width or self.custom_height:
            x_offset = (image_width_px - text_width_px) // 2
            y_offset = (image_height_px - text_height_px) // 2
        else:
            x_offset = self.margin_px
            y_offset = self.margin_px

        # Draw text
        y_text = y_offset
        for line in wrapped_text:
            x_text = x_offset

            if self.condition == 'distorted':
                skew_factor = random.uniform(
                    self.config['skew_range'][0],
                    self.config['skew_range'][1]
                )
                x_text += int(skew_factor * y_text)

            draw.text((x_text, y_text), line, font=font, fill=text_color)
            y_text += font.getbbox(line)[3] - font.getbbox(line)[1]

        # Apply effects
        if self.condition != 'simple':
            image = apply_effects(image, self.condition, self.config)

        # Save if requested
        if output_path and self.save_images:
            image.save(output_path, dpi=(self.dpi, self.dpi))

        return image

    def image_to_text(self, image: Union[str, Image.Image]) -> str:
        """Convert image back to text using OCR."""
        custom_config = f'--oem 3 --psm 6 -l {self.lang_config["tesseract_lang"]}'

        if isinstance(image, str):
            ocr_output = pytesseract.image_to_string(
                image, config=custom_config)
        else:
            ocr_output = pytesseract.image_to_string(
                image, config=custom_config)

        return ocr_output.replace('\n', ' ').strip()

    def process_single_text(
        self,
        text: str,
        save_image: bool = False,
        image_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """Process a single text input."""
        if save_image and not image_filename and self.output_dir:
            image_filename = os.path.join(
                self.output_dir,
                f"single_text_{random.randint(1000, 9999)}.png"
            )

        image = self.text_to_image(
            text, image_filename if save_image else None)
        ocr_text = self.image_to_text(image)

        return {
            'original_text': text,
            'ocr_text': ocr_text
        }

    def process_dataframe(
        self,
        df: pd.DataFrame,
        output_csv: Optional[str] = None,
        image_prefix: Optional[str] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """Process entire dataframe."""
        if image_prefix is None:
            image_prefix = f"{self.condition}_{self.language}"

        if self.n_jobs == 1:
            ocr_df = pd.DataFrame(index=df.index, columns=df.columns)
            for index, row in tqdm(df.iterrows(), total=len(df), disable=not show_progress):
                for column in df.columns:
                    sentence = str(row[column])
                    if self.save_images and self.output_dir:
                        image_filename = os.path.join(
                            self.output_dir,
                            f"{image_prefix}_{index}_{column}.png"
                        )
                    else:
                        image_filename = None

                    image = self.text_to_image(sentence, image_filename)
                    ocr_text = self.image_to_text(image)
                    ocr_df.at[index, column] = ocr_text
        else:
            tasks = [
                (index, column, str(row[column]))
                for index, row in df.iterrows()
                for column in df.columns
            ]

            results = []
            with tqdm(total=len(tasks), disable=not show_progress) as pbar:
                for args in tasks:
                    index, column, sentence = args
                    if self.save_images and self.output_dir:
                        image_filename = os.path.join(
                            self.output_dir,
                            f"{image_prefix}_{index}_{column}.png"
                        )
                    else:
                        image_filename = None

                    result = self._process_cell(
                        (index, column, sentence, image_filename))
                    results.append(result)
                    pbar.update(1)

            ocr_df = pd.DataFrame(index=df.index, columns=df.columns)
            for index, column, ocr_text in results:
                ocr_df.at[index, column] = ocr_text

        if output_csv:
            ocr_df.to_csv(output_csv, index=False)

        return ocr_df

    def _process_cell(self, args: tuple) -> tuple:
        """Process a single cell for parallel processing."""
        if len(args) == 4:
            index, column, sentence, image_filename = args
        else:
            index, column, sentence = args
            image_filename = None

        try:
            temp_simulator = OCRSimulator(
                condition=self.condition,
                language=self.language,
                font_path=self.font_path,
                font_size=self.font_size,
                dpi=self.dpi,
                config=self.config,
                save_images=self.save_images,
                output_dir=self.output_dir
            )
            image = temp_simulator.text_to_image(sentence, image_filename)
            ocr_text = temp_simulator.image_to_text(image)
        except Exception as e:
            print(f"Error processing index {index}, column {column}: {e}")
            ocr_text = ""
        return (index, column, ocr_text)

    def process_text_folder(
        self,
        input_folder: str,
        file_pattern: str = "*.txt",
        output_csv: Optional[str] = None,
        recursive: bool = False
    ) -> pd.DataFrame:
        """Process all text files in a folder."""
        input_path = Path(input_folder)
        if recursive:
            files = list(input_path.rglob(file_pattern))
        else:
            files = list(input_path.glob(file_pattern))

        results = []
        for file in tqdm(files, desc="Processing files"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                result = self.process_single_text(
                    text,
                    save_image=self.save_images,
                    image_filename=os.path.join(
                        self.output_dir, f"{file.stem}.png") if self.save_images else None
                )

                results.append({
                    'filename': file.name,
                    'original_text': result['original_text'],
                    'ocr_text': result['ocr_text']
                })
            except Exception as e:
                print(f"Error processing {file}: {e}")

        df = pd.DataFrame(results)
        if output_csv:
            df.to_csv(output_csv, index=False)
        return df

    def process_input(
        self,
        input_source: Union[str, pd.DataFrame, List[str]],
        input_type: str = 'auto',
        **kwargs
    ) -> Union[Dict[str, str], pd.DataFrame]:
        """Process different types of input sources."""
        if input_type == 'auto':
            if isinstance(input_source, str):
                if os.path.isfile(input_source) and input_source.endswith('.csv'):
                    input_type = 'csv'
                elif os.path.isdir(input_source):
                    input_type = 'folder'
                else:
                    input_type = 'text'
            elif isinstance(input_source, pd.DataFrame):
                input_type = 'dataframe'
            elif isinstance(input_source, list):
                input_type = 'list'

        if input_type == 'csv':
            df = pd.read_csv(input_source)
            return self.process_dataframe(df, **kwargs)

        elif input_type == 'folder':
            return self.process_text_folder(
                input_source,
                file_pattern=kwargs.get('file_pattern', '*.txt'),
                output_csv=kwargs.get('output_csv'),
                recursive=kwargs.get('recursive', False)
            )

        elif input_type == 'text':
            return self.process_single_text(
                input_source,
                save_image=kwargs.get('save_image', self.save_images),
                image_filename=kwargs.get('image_filename')
            )

        elif input_type == 'list':
            results = []
            for text in tqdm(input_source, desc="Processing texts"):
                result = self.process_single_text(text)
                results.append(result)
            df = pd.DataFrame(results)
            if 'output_csv' in kwargs:
                df.to_csv(kwargs['output_csv'], index=False)
            return df

        elif input_type == 'dataframe':
            return self.process_dataframe(input_source, **kwargs)

        else:
            raise ValueError(f"Unsupported input type: {input_type}")
