"""File operations for the AI Booth application."""

import os
import time
import cv2 as cv
import numpy as np
from typing import Optional


class FileManager:
    """Handles file operations for saving snapshots."""
    
    def __init__(self, save_directory: str):
        """Initialize the file manager.
        
        Args:
            save_directory: Directory to save snapshots
        """
        self.save_directory = save_directory
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Create the save directory if it doesn't exist."""
        try:
            os.makedirs(self.save_directory, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {self.save_directory}: {e}")
    
    def save_snapshot(self, image: np.ndarray, style: str) -> Optional[str]:
        """Save a snapshot image with timestamp and style name.
        
        Args:
            image: Image to save (BGR format)
            style: Style name to include in filename
            
        Returns:
            Path of saved file, or None if saving failed
        """
        if image is None:
            print("Cannot save empty image")
            return None
        
        try:
            # Generate timestamp-based filename
            timestamp = int(time.time() * 1000)
            filename = f"booth_{style}_{timestamp}.jpg"
            file_path = os.path.join(self.save_directory, filename)
            
            # Save the image
            success = cv.imwrite(file_path, image)
            
            if success:
                print(f"Saved {file_path}")
                return file_path
            else:
                print(f"Failed to save image to {file_path}")
                return None
                
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            return None
    
    def get_save_directory(self) -> str:
        """Get the current save directory path.
        
        Returns:
            Save directory path
        """
        return self.save_directory
    
    def list_snapshots(self) -> list:
        """List all snapshot files in the save directory.
        
        Returns:
            List of snapshot filenames
        """
        try:
            if not os.path.exists(self.save_directory):
                return []
            
            files = []
            for filename in os.listdir(self.save_directory):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    files.append(filename)
            
            return sorted(files, reverse=True)  # Most recent first
            
        except Exception as e:
            print(f"Error listing snapshots: {e}")
            return []
