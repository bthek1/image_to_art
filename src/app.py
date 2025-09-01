"""Main application class for the AI Booth."""

import numpy as np
import time
from typing import Optional
from .camera import Camera
from .image_processor import ImageProcessor
from .file_manager import FileManager
from .ui import UI
from .config import SAVE_DIR, CAMERA_INDEX, TEXTURE_WIDTH, TEXTURE_HEIGHT


class AIBoothApp:
    """Main application class that coordinates all components."""
    
    def __init__(self):
        """Initialize the AI Booth application."""
        # Initialize components
        self.camera = Camera(CAMERA_INDEX)
        self.image_processor = ImageProcessor()
        self.file_manager = FileManager(SAVE_DIR)
        self.ui = UI()
        
        # Current frame storage
        self.current_frame: Optional[np.ndarray] = None
        self.current_original_frame: Optional[np.ndarray] = None
        
        # Set up UI callbacks
        self.ui.set_style_change_callback(self._on_style_change)
        self.ui.set_snapshot_callback(self._on_snapshot_request)
    
    def _on_style_change(self, new_style: str) -> None:
        """Handle style change from UI.
        
        Args:
            new_style: The newly selected style
        """
        # The UI already handles the style change internally
        # This could be used for additional logic if needed
        pass
    
    def _on_snapshot_request(self) -> None:
        """Handle snapshot request from UI."""
        if self.current_frame is not None:
            current_style = self.ui.get_current_style()
            self.file_manager.save_snapshot(self.current_frame, current_style)
    
    def _process_frame(self) -> None:
        """Process a single frame from the camera."""
        frame_start_time = time.time()
        
        # Read frame from camera
        camera_start_time = time.time()
        success, frame = self.camera.read_frame()
        camera_end_time = time.time()
        
        if not success or frame is None:
            # Record dropped frame
            self.ui.record_dropped_frame()
            return
        
        # Store original frame
        self.current_original_frame = frame
        
        # Apply current style
        processing_start_time = time.time()
        current_style = self.ui.get_current_style()
        stylized_frame = self.image_processor.stylize(frame, current_style)
        processing_end_time = time.time()
        
        if stylized_frame is not None:
            # Store current styled frame for snapshot capability
            self.current_frame = stylized_frame
            
            # Prepare both images for display
            original_display_data = self.image_processor.prepare_for_display(
                frame, TEXTURE_WIDTH, TEXTURE_HEIGHT
            )
            styled_display_data = self.image_processor.prepare_for_display(
                stylized_frame, TEXTURE_WIDTH, TEXTURE_HEIGHT
            )
            
            # Update UI with both images
            self.ui.update_images(original_display_data, styled_display_data)
            
            # Calculate timing metrics
            frame_end_time = time.time()
            camera_time = camera_end_time - camera_start_time
            processing_time = processing_end_time - processing_start_time
            total_time = frame_end_time - frame_start_time
            
            # Update performance stats with timing information
            self.ui.update_stats(processing_time, camera_time, total_time)
        else:
            # Record dropped frame if stylization failed
            self.ui.record_dropped_frame()
    
    def run(self) -> None:
        """Run the AI Booth application."""
        print("Starting AI Booth application...")
        
        # Open camera
        if not self.camera.open():
            print("Error: Could not open camera")
            return
        
        try:
            # Setup UI
            self.ui.setup()
            
            print("AI Booth is running. Close the window to exit.")
            
            # Main application loop
            while self.ui.is_running():
                self._process_frame()
                self.ui.render_frame()
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up application resources."""
        print("Cleaning up...")
        self.camera.close()
        self.ui.cleanup()
        print("Cleanup complete.")


def main() -> None:
    """Main entry point for the application."""
    app = AIBoothApp()
    app.run()


if __name__ == "__main__":
    main()
