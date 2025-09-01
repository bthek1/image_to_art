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
        
    def _try_camera_indices(self, indices: list) -> bool:
        """Try to open camera using different indices.
        
        Args:
            indices: List of camera indices to try
            
        Returns:
            True if any camera opened successfully, False otherwise
        """
        for idx in indices:
            try:
                print(f"Trying camera index {idx}...")
                cap = cv.VideoCapture(idx)
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Successfully opened camera index {idx}")
                        self.cap = cap
                        self.camera_index = idx
                        self.is_opened = True
                        return True
                    else:
                        cap.release()
                else:
                    cap.release()
            except Exception as e:
                print(f"Failed to open camera index {idx}: {e}")
                continue
        return False
        
    def open(self) -> bool:
        """Open the camera connection.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        try:
            # Try the specified index first, then fall back to common indices
            indices_to_try = [self.camera_index]
            
            # Add other common camera indices if not already in list
            for idx in [0, 1, 2, 3]:
                if idx not in indices_to_try:
                    indices_to_try.append(idx)
            
            return self._try_camera_indices(indices_to_try)
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
