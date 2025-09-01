"""Configuration constants for the AI Booth application."""

from typing import List

# Style configurations
STYLE_NAMES: List[str] = ["Cartoon", "Pencil", "EdgePaint"]
DEFAULT_STYLE: str = STYLE_NAMES[0]

# Directory configurations
SAVE_DIR: str = "captures"

# Camera and display configurations
TEXTURE_WIDTH: int = 640
TEXTURE_HEIGHT: int = 480
CAMERA_INDEX: int = 0

# Window configurations
DEFAULT_WINDOW_WIDTH: int = 1500  # Increased for side-by-side images
DEFAULT_WINDOW_HEIGHT: int = 850
MIN_WINDOW_WIDTH: int = 1200  # Increased minimum width for two images
MIN_WINDOW_HEIGHT: int = 600

# UI configurations
CONTROL_HEIGHT: int = 100
PADDING: int = 40
ASPECT_RATIO: float = 4.0 / 3.0

# Image processing parameters
CARTOON_PARAMS = {
    "median_blur_ksize": 7,
    "adaptive_threshold_max_value": 255,
    "adaptive_threshold_block_size": 9,
    "adaptive_threshold_c": 9,
    "bilateral_d": 9,
    "bilateral_sigma_color": 150,
    "bilateral_sigma_space": 150,
}

PENCIL_PARAMS = {
    "sigma_s": 60,
    "sigma_r": 0.1,
    "shade_factor": 0.04,
}

EDGE_PAINT_PARAMS = {
    "sigma_s": 60,
    "sigma_r": 0.5,
}
