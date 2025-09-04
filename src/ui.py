"""User interface for the AI Booth application."""

import dearpygui.dearpygui as dpg
import numpy as np
import time
import psutil
import os
from typing import Callable, Tuple, Optional
from .config import (
    STYLE_NAMES, DEFAULT_STYLE, TEXTURE_WIDTH, TEXTURE_HEIGHT,
    DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT,
    CONTROL_HEIGHT, PADDING, ASPECT_RATIO
)


class UI:
    """Handles the user interface using DearPyGUI."""
    
    def __init__(self, available_styles: Optional[list] = None):
        """Initialize the UI.
        
        Args:
            available_styles: List of available styles, defaults to STYLE_NAMES from config
        """
        self.available_styles = available_styles or STYLE_NAMES
        self.current_style = self.available_styles[0] if self.available_styles else DEFAULT_STYLE
        self.style_change_callback: Optional[Callable] = None
        self.snapshot_callback: Optional[Callable] = None
        
        # FPS tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.frame_count = 0
        
        # Latency tracking
        self.processing_latencies = []
        self.camera_latencies = []
        self.total_latencies = []
        self.avg_processing_latency = 0.0
        self.avg_camera_latency = 0.0
        self.avg_total_latency = 0.0
        
        # Performance metrics
        self.dropped_frames = 0
        self.last_stats_update = time.time()
        self.stats_update_interval = 0.5  # Update stats every 0.5 seconds
        
        # Initialize DearPyGUI
        dpg.create_context()
        self._setup_texture()
        self._create_viewport()
        self._create_main_window()
    
    def _setup_texture(self) -> None:
        """Set up the textures for image display."""
        # Create a fixed-size texture with placeholder data
        default_data = np.ones((TEXTURE_HEIGHT, TEXTURE_WIDTH, 4), dtype=np.float32) * 0.5
        
        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                width=TEXTURE_WIDTH,
                height=TEXTURE_HEIGHT,
                default_value=default_data.flatten(),
                tag="original_texture_id"
            )
            dpg.add_dynamic_texture(
                width=TEXTURE_WIDTH,
                height=TEXTURE_HEIGHT,
                default_value=default_data.flatten(),
                tag="styled_texture_id"
            )
    
    def _create_viewport(self) -> None:
        """Create and configure the viewport."""
        dpg.create_viewport(
            title="AI Booth",
            width=DEFAULT_WINDOW_WIDTH,
            height=DEFAULT_WINDOW_HEIGHT,
            resizable=True,
            min_width=MIN_WINDOW_WIDTH,
            min_height=MIN_WINDOW_HEIGHT
        )
        
        # Set up viewport resize callback
        dpg.set_viewport_resize_callback(self._resize_callback)
    
    def _create_main_window(self) -> None:
        """Create the main application window."""
        with dpg.window(
            label="AI Booth",
            tag="main_window",
            no_title_bar=True,
            no_resize=True,
            no_move=True
        ):
            dpg.add_text("Pick a style")
            dpg.add_combo(
                self.available_styles,
                default_value=self.current_style,
                callback=self._on_style_change,
                tag="style_combo"
            )
            dpg.add_button(label="Snapshot", callback=self._on_snapshot_click)
            dpg.add_separator()
            
            # Get initial display size for side-by-side images
            display_width, display_height = self._get_image_display_size()
            
            # Create horizontal group for side-by-side images
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Original", color=(255, 255, 255))
                    dpg.add_image(
                        "original_texture_id",
                        width=display_width,
                        height=display_height,
                        tag="original_image_display"
                    )
                
                dpg.add_spacer(width=10)
                
                with dpg.group():
                    dpg.add_text("Styled", color=(255, 255, 255))
                    dpg.add_image(
                        "styled_texture_id",
                        width=display_width,
                        height=display_height,
                        tag="styled_image_display"
                    )
            
            # Add stats section at the bottom
            dpg.add_separator()
            dpg.add_text("Performance Stats:", color=(200, 200, 200))
            
            # First row of stats
            with dpg.group(horizontal=True):
                dpg.add_text("FPS: 0.0", tag="fps_text", color=(150, 255, 150))
                dpg.add_spacer(width=15)
                dpg.add_text("Frames: 0", tag="frame_count_text", color=(150, 150, 255))
                dpg.add_spacer(width=15)
                dpg.add_text("Style: Cartoon", tag="current_style_text", color=(255, 255, 150))
            
            # Second row of stats - Latency metrics
            with dpg.group(horizontal=True):
                dpg.add_text("Total Latency: 0.0ms", tag="total_latency_text", color=(255, 150, 150))
                dpg.add_spacer(width=15)
                dpg.add_text("Processing: 0.0ms", tag="processing_latency_text", color=(255, 200, 100))
                dpg.add_spacer(width=15)
                dpg.add_text("Camera: 0.0ms", tag="camera_latency_text", color=(100, 255, 200))
            
            # Third row of stats - Additional metrics
            with dpg.group(horizontal=True):
                dpg.add_text("Dropped: 0", tag="dropped_frames_text", color=(255, 100, 100))
                dpg.add_spacer(width=15)
                dpg.add_text("CPU Load: 0%", tag="cpu_load_text", color=(200, 150, 255))
                dpg.add_spacer(width=15)
                dpg.add_text("Mem: 0MB", tag="memory_text", color=(150, 255, 200))
        
        # Set as primary window
        dpg.set_primary_window("main_window", True)
    
    def _get_image_display_size(self) -> Tuple[int, int]:
        """Calculate the image display size based on current window size for side-by-side images.
        
        Returns:
            Tuple of (width, height) for each image display
        """
        try:
            if dpg.does_item_exist("main_window"):
                window_width = dpg.get_item_width("main_window")
                window_height = dpg.get_item_height("main_window")
                
                # Reserve space for controls and account for two images side by side
                # Include space for the spacer between images (10px) and text labels
                spacer_width = 10
                text_height = 20  # Approximate height for text labels
                available_width = max(640, window_width - PADDING - spacer_width) // 2
                available_height = max(240, window_height - CONTROL_HEIGHT - PADDING - text_height)
                
                # Maintain aspect ratio for each image
                if available_width / available_height > ASPECT_RATIO:
                    # Height is limiting factor
                    display_height = available_height
                    display_width = int(display_height * ASPECT_RATIO)
                else:
                    # Width is limiting factor
                    display_width = available_width
                    display_height = int(display_width / ASPECT_RATIO)
                
                return display_width, display_height
        except Exception:
            pass
        
        # Default size for each image (half width for side-by-side)
        return TEXTURE_WIDTH // 2, TEXTURE_HEIGHT
    
    def _resize_callback(self) -> None:
        """Handle window resize events."""
        try:
            viewport_width = dpg.get_viewport_client_width()
            viewport_height = dpg.get_viewport_client_height()
            dpg.configure_item("main_window", width=viewport_width, height=viewport_height)
            
            # Update image display sizes for both images
            display_width, display_height = self._get_image_display_size()
            dpg.configure_item("original_image_display", width=display_width, height=display_height)
            dpg.configure_item("styled_image_display", width=display_width, height=display_height)
        except Exception:
            pass
    
    def _on_style_change(self, sender, app_data) -> None:
        """Handle style change event."""
        self.current_style = app_data
        print(f"Style changed to: {self.current_style}")
        
        # Update the stats display immediately
        try:
            if dpg.does_item_exist("current_style_text"):
                dpg.set_value("current_style_text", f"Style: {self.current_style}")
        except Exception:
            pass
        
        if self.style_change_callback:
            self.style_change_callback(app_data)
    
    def _on_snapshot_click(self, sender, app_data) -> None:
        """Handle snapshot button click."""
        if self.snapshot_callback:
            self.snapshot_callback()
    
    def set_style_change_callback(self, callback: Callable[[str], None]) -> None:
        """Set the callback for style changes.
        
        Args:
            callback: Function to call when style changes
        """
        self.style_change_callback = callback
    
    def set_snapshot_callback(self, callback: Callable[[], None]) -> None:
        """Set the callback for snapshot requests.
        
        Args:
            callback: Function to call when snapshot is requested
        """
        self.snapshot_callback = callback
    
    def update_images(self, original_data: np.ndarray, styled_data: np.ndarray) -> None:
        """Update both original and styled displayed images.
        
        Args:
            original_data: Flattened RGBA image data for original image
            styled_data: Flattened RGBA image data for styled image
        """
        try:
            if dpg.does_item_exist("original_texture_id"):
                dpg.set_value("original_texture_id", original_data)
            if dpg.does_item_exist("styled_texture_id"):
                dpg.set_value("styled_texture_id", styled_data)
        except Exception as e:
            print(f"Texture update error: {e}")

    def update_image(self, image_data: np.ndarray) -> None:
        """Update the styled image display (legacy method for compatibility).
        
        Args:
            image_data: Flattened RGBA image data for display
        """
        try:
            if dpg.does_item_exist("styled_texture_id"):
                dpg.set_value("styled_texture_id", image_data)
        except Exception as e:
            print(f"Texture update error: {e}")
    
    def _update_fps(self) -> None:
        """Update FPS calculation and display."""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Keep a rolling window of frame times (last 30 frames)
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Calculate average FPS
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        self.frame_count += 1
    
    def _update_latency(self, processing_time: float, camera_time: float, total_time: float) -> None:
        """Update latency calculations.
        
        Args:
            processing_time: Time spent on image processing in seconds
            camera_time: Time spent on camera operations in seconds  
            total_time: Total frame processing time in seconds
        """
        # Convert to milliseconds
        processing_ms = processing_time * 1000
        camera_ms = camera_time * 1000
        total_ms = total_time * 1000
        
        # Keep a rolling window of latencies (last 30 measurements)
        self.processing_latencies.append(processing_ms)
        if len(self.processing_latencies) > 30:
            self.processing_latencies.pop(0)
            
        self.camera_latencies.append(camera_ms)
        if len(self.camera_latencies) > 30:
            self.camera_latencies.pop(0)
            
        self.total_latencies.append(total_ms)
        if len(self.total_latencies) > 30:
            self.total_latencies.pop(0)
        
        # Calculate averages
        if self.processing_latencies:
            self.avg_processing_latency = sum(self.processing_latencies) / len(self.processing_latencies)
        if self.camera_latencies:
            self.avg_camera_latency = sum(self.camera_latencies) / len(self.camera_latencies)
        if self.total_latencies:
            self.avg_total_latency = sum(self.total_latencies) / len(self.total_latencies)
    
    def _get_system_stats(self) -> Tuple[float, float]:
        """Get system CPU and memory statistics.
        
        Returns:
            Tuple of (cpu_percent, memory_mb)
        """
        try:
            # Get CPU usage for current process
            process = psutil.Process(os.getpid())
            cpu_percent = process.cpu_percent()
            
            # Get memory usage for current process in MB
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            return cpu_percent, memory_mb
        except Exception:
            return 0.0, 0.0
    
    def update_stats(self, processing_time: float = 0.0, camera_time: float = 0.0, total_time: float = 0.0) -> None:
        """Update the performance stats display.
        
        Args:
            processing_time: Time spent on image processing in seconds
            camera_time: Time spent on camera operations in seconds
            total_time: Total frame processing time in seconds
        """
        self._update_fps()
        
        # Update latency if provided
        if total_time > 0:
            self._update_latency(processing_time, camera_time, total_time)
        
        # Only update UI elements periodically to avoid excessive UI updates
        current_time = time.time()
        if current_time - self.last_stats_update >= self.stats_update_interval:
            self.last_stats_update = current_time
            
            # Get system stats
            cpu_percent, memory_mb = self._get_system_stats()
            
            try:
                # First row stats
                if dpg.does_item_exist("fps_text"):
                    dpg.set_value("fps_text", f"FPS: {self.fps:.1f}")
                if dpg.does_item_exist("frame_count_text"):
                    dpg.set_value("frame_count_text", f"Frames: {self.frame_count}")
                if dpg.does_item_exist("current_style_text"):
                    dpg.set_value("current_style_text", f"Style: {self.current_style}")
                
                # Second row - Latency stats
                if dpg.does_item_exist("total_latency_text"):
                    dpg.set_value("total_latency_text", f"Total Latency: {self.avg_total_latency:.1f}ms")
                if dpg.does_item_exist("processing_latency_text"):
                    dpg.set_value("processing_latency_text", f"Processing: {self.avg_processing_latency:.1f}ms")
                if dpg.does_item_exist("camera_latency_text"):
                    dpg.set_value("camera_latency_text", f"Camera: {self.avg_camera_latency:.1f}ms")
                
                # Third row - System stats
                if dpg.does_item_exist("dropped_frames_text"):
                    dpg.set_value("dropped_frames_text", f"Dropped: {self.dropped_frames}")
                if dpg.does_item_exist("cpu_load_text"):
                    dpg.set_value("cpu_load_text", f"CPU Load: {cpu_percent:.1f}%")
                if dpg.does_item_exist("memory_text"):
                    dpg.set_value("memory_text", f"Mem: {memory_mb:.1f}MB")
                    
            except Exception as e:
                print(f"Stats update error: {e}")
    
    def record_dropped_frame(self) -> None:
        """Record a dropped frame for statistics."""
        self.dropped_frames += 1
    
    def get_current_style(self) -> str:
        """Get the currently selected style.
        
        Returns:
            Current style name
        """
        return self.current_style
    
    def setup(self) -> None:
        """Setup the UI for rendering."""
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        # Initial resize to fit viewport
        self._resize_callback()
    
    def is_running(self) -> bool:
        """Check if the UI is still running.
        
        Returns:
            True if UI should continue running
        """
        return dpg.is_dearpygui_running()
    
    def render_frame(self) -> None:
        """Render a single frame."""
        dpg.render_dearpygui_frame()
    
    def cleanup(self) -> None:
        """Clean up UI resources."""
        try:
            dpg.destroy_context()
        except Exception as e:
            print(f"Error during UI cleanup: {e}")
