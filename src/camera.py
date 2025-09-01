"""Camera operations for the AI Booth application."""

import cv2 as cv
from typing import Optional, Tuple
import numpy as np


class Camera:
    """Handles camera operations and frame capture."""
    
    def __init__(self, camera_index: int = 0):
        """Initialize the camera.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.cap: Optional[cv.VideoCapture] = None
        self.is_opened = False
        
    def open(self) -> bool:
        """Open the camera connection.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        try:
            self.cap = cv.VideoCapture(self.camera_index)
            self.is_opened = self.cap.isOpened()
            return self.is_opened
        except Exception as e:
            print(f"Error opening camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame) where success is bool and frame is numpy array or None
        """
        if not self.cap or not self.is_opened:
            return False, None
            
        try:
            ret, frame = self.cap.read()
            if ret:
                # Mirror the frame for a more natural selfie experience
                frame = cv.flip(frame, 1)
            return ret, frame
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
    
    def close(self) -> None:
        """Close the camera connection."""
        if self.cap:
            try:
                self.cap.release()
                self.is_opened = False
            except Exception as e:
                print(f"Error closing camera: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        if self.open():
            return self
        else:
            raise RuntimeError("Failed to open camera")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
