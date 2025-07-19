# ocr_simulator/effects.py
from PIL import Image, ImageDraw
import random
from typing import Dict


def apply_effects(
    image: Image.Image,
    condition: str,
    config: Dict
) -> Image.Image:
    """Apply effects based on condition."""
    if condition == 'noisy':
        return apply_salt_and_pepper(image, config['dot_density'])
    elif condition == 'distorted':
        return apply_distortions(image, config)
    return image


def apply_salt_and_pepper(
    image: Image.Image,
    dot_density: float
) -> Image.Image:
    """Apply salt and pepper noise."""
    width, height = image.size
    draw = ImageDraw.Draw(image)

    num_dots = int(width * height * dot_density)
    for _ in range(num_dots):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        draw.point((x, y), fill="black")

    return image


def apply_distortions(
    image: Image.Image,
    config: Dict
) -> Image.Image:
    """Apply complex distortions."""
    pixels = image.load()
    width, height = image.size

    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]

            # Text pixel processing
            if r < 100 and g < 100 and b < 100:
                if random.random() < config['incomplete_prob']:
                    gap_height = random.randint(
                        config['gap_range'][0],
                        config['gap_range'][1]
                    )
                    for k in range(max(0, j - gap_height), j):
                        pixels[i, k] = (255, 255, 255)

                noise = random.randint(
                    config['text_noise_range'][0],
                    config['text_noise_range'][1]
                )
                pixels[i, j] = tuple(
                    max(0, min(255, c + noise))
                    for c in (r, g, b)
                )
            else:
                if random.random() < config['bg_noise_prob']:
                    noise = random.randint(
                        config['bg_noise_range'][0],
                        config['bg_noise_range'][1]
                    )
                    pixels[i, j] = tuple(
                        max(0, min(255, c + noise))
                        for c in (r, g, b)
                    )

    return image
