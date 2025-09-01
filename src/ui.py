"""User interface for the AI Booth application."""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Callable, Tuple, Optional
from .config import (
    STYLE_NAMES, DEFAULT_STYLE, TEXTURE_WIDTH, TEXTURE_HEIGHT,
    DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT,
    CONTROL_HEIGHT, PADDING, ASPECT_RATIO
)


class UI:
    """Handles the user interface using DearPyGUI."""
    
    def __init__(self):
        """Initialize the UI."""
        self.current_style = DEFAULT_STYLE
        self.style_change_callback: Optional[Callable] = None
        self.snapshot_callback: Optional[Callable] = None
        
        # Initialize DearPyGUI
        dpg.create_context()
        self._setup_texture()
        self._create_viewport()
        self._create_main_window()
    
    def _setup_texture(self) -> None:
        """Set up the texture for image display."""
        # Create a fixed-size texture with placeholder data
        default_data = np.ones((TEXTURE_HEIGHT, TEXTURE_WIDTH, 4), dtype=np.float32) * 0.5
        
        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                width=TEXTURE_WIDTH,
                height=TEXTURE_HEIGHT,
                default_value=default_data.flatten(),
                tag="texture_id"
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
                STYLE_NAMES,
                default_value=self.current_style,
                callback=self._on_style_change,
                tag="style_combo"
            )
            dpg.add_button(label="Snapshot", callback=self._on_snapshot_click)
            dpg.add_separator()
            
            # Get initial display size
            display_width, display_height = self._get_image_display_size()
            
            # Add image display
            dpg.add_image(
                "texture_id",
                width=display_width,
                height=display_height,
                tag="image_display"
            )
        
        # Set as primary window
        dpg.set_primary_window("main_window", True)
    
    def _get_image_display_size(self) -> Tuple[int, int]:
        """Calculate the image display size based on current window size.
        
        Returns:
            Tuple of (width, height) for image display
        """
        try:
            if dpg.does_item_exist("main_window"):
                window_width = dpg.get_item_width("main_window")
                window_height = dpg.get_item_height("main_window")
                
                # Reserve space for controls
                available_width = max(320, window_width - PADDING)
                available_height = max(240, window_height - CONTROL_HEIGHT - PADDING)
                
                # Maintain aspect ratio
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
        
        return TEXTURE_WIDTH, TEXTURE_HEIGHT
    
    def _resize_callback(self) -> None:
        """Handle window resize events."""
        try:
            viewport_width = dpg.get_viewport_client_width()
            viewport_height = dpg.get_viewport_client_height()
            dpg.configure_item("main_window", width=viewport_width, height=viewport_height)
            
            # Update image display size
            display_width, display_height = self._get_image_display_size()
            dpg.configure_item("image_display", width=display_width, height=display_height)
        except Exception:
            pass
    
    def _on_style_change(self, sender, app_data) -> None:
        """Handle style change event."""
        self.current_style = app_data
        print(f"Style changed to: {self.current_style}")
        
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
    
    def update_image(self, image_data: np.ndarray) -> None:
        """Update the displayed image.
        
        Args:
            image_data: Flattened RGBA image data for display
        """
        try:
            if dpg.does_item_exist("texture_id"):
                dpg.set_value("texture_id", image_data)
        except Exception as e:
            print(f"Texture update error: {e}")
    
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
