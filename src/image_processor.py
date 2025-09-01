"""Image processing and stylization for the AI Booth application."""

import cv2 as cv
import numpy as np
from typing import Optional
from .config import CARTOON_PARAMS, PENCIL_PARAMS, EDGE_PAINT_PARAMS


class ImageProcessor:
    """Handles image stylization and processing operations."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.supported_styles = ["Cartoon", "Pencil", "EdgePaint"]
    
    def stylize(self, frame: np.ndarray, style: str) -> Optional[np.ndarray]:
        """Apply the specified style to the input frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            style: Style name to apply
            
        Returns:
            Stylized image as numpy array, or None if processing failed
        """
        if frame is None:
            return None
            
        try:
            if style == "Cartoon":
                return self._apply_cartoon_style(frame)
            elif style == "Pencil":
                return self._apply_pencil_style(frame)
            elif style == "EdgePaint":
                return self._apply_edge_paint_style(frame)
            else:
                print(f"Unknown style: {style}")
                return frame
        except Exception as e:
            print(f"Error applying style {style}: {e}")
            return frame
    
    def _apply_cartoon_style(self, frame: np.ndarray) -> np.ndarray:
        """Apply cartoon-style effect using bilateral filter and edge detection.
        
        Args:
            frame: Input image
            
        Returns:
            Cartoon-stylized image
        """
        # Convert to grayscale for edge detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, CARTOON_PARAMS["median_blur_ksize"])
        
        # Create edge mask
        edges = cv.adaptiveThreshold(
            gray,
            CARTOON_PARAMS["adaptive_threshold_max_value"],
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,
            CARTOON_PARAMS["adaptive_threshold_block_size"],
            CARTOON_PARAMS["adaptive_threshold_c"]
        )
        
        # Apply bilateral filter for smooth color regions
        color = cv.bilateralFilter(
            frame,
            CARTOON_PARAMS["bilateral_d"],
            CARTOON_PARAMS["bilateral_sigma_color"],
            CARTOON_PARAMS["bilateral_sigma_space"]
        )
        
        # Combine color and edges
        return cv.bitwise_and(color, color, mask=edges)
    
    def _apply_pencil_style(self, frame: np.ndarray) -> np.ndarray:
        """Apply pencil sketch effect.
        
        Args:
            frame: Input image
            
        Returns:
            Pencil sketch styled image
        """
        dst, _ = cv.pencilSketch(
            frame,
            sigma_s=PENCIL_PARAMS["sigma_s"],
            sigma_r=PENCIL_PARAMS["sigma_r"],
            shade_factor=PENCIL_PARAMS["shade_factor"]
        )
        # Convert grayscale to BGR for consistency
        return cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    
    def _apply_edge_paint_style(self, frame: np.ndarray) -> np.ndarray:
        """Apply edge-preserving paint-like effect.
        
        Args:
            frame: Input image
            
        Returns:
            Paint-style image
        """
        return cv.stylization(
            frame,
            sigma_s=EDGE_PAINT_PARAMS["sigma_s"],
            sigma_r=EDGE_PAINT_PARAMS["sigma_r"]
        )
    
    def prepare_for_display(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Prepare image for display in the UI.
        
        Args:
            frame: Input image (BGR format)
            target_width: Target width for display
            target_height: Target height for display
            
        Returns:
            RGBA image data as flat array for DearPyGUI
        """
        # Resize to target dimensions
        frame_resized = cv.resize(frame, (target_width, target_height))
        
        # Convert BGR to RGB
        rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
        
        # Add alpha channel to create RGBA
        rgba = np.dstack((rgb, np.full((target_height, target_width), 255, dtype=np.uint8)))
        
        # Normalize to 0-1 range and flatten for DearPyGUI
        return (rgba / 255.0).astype(np.float32).flatten()
