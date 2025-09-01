# pip/uv install: opencv-python dearpygui
import cv2 as cv
import time
import dearpygui.dearpygui as dpg
import numpy as np
import os

STYLE_NAMES = ["Cartoon", "Pencil", "EdgePaint"]
current_style = STYLE_NAMES[0]
save_dir = "captures"
cap = None
last_frame = [None]

# Fixed texture size - we'll use this consistently
TEXTURE_WIDTH = 640
TEXTURE_HEIGHT = 480

def stylize(frame, style):
    if style == "Cartoon":
        # Fast bilateral + edge boost
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 7)
        edges = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                     cv.THRESH_BINARY, 9, 9)
        color = cv.bilateralFilter(frame, 9, 150, 150)
        return cv.bitwise_and(color, color, mask=edges)
    elif style == "Pencil":
        dst, _ = cv.pencilSketch(frame, sigma_s=60, sigma_r=0.1, shade_factor=0.04)
        return cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    else:  # EdgePaint (simple stylization)
        return cv.stylization(frame, sigma_s=60, sigma_r=0.5)

def change_style(sender, app_data):
    global current_style
    current_style = app_data
    print(f"Style changed to: {current_style}")

def snapshot(sender, app_data):
    if last_frame[0] is not None:
        ts = int(time.time()*1000)
        path = f"{save_dir}/booth_{current_style}_{ts}.jpg"
        cv.imwrite(path, last_frame[0])
        print(f"Saved {path}")

def get_image_display_size():
    """Calculate the image display size based on current window size."""
    if dpg.does_item_exist("main_window"):
        try:
            window_width = dpg.get_item_width("main_window")
            window_height = dpg.get_item_height("main_window")
            
            # Reserve space for controls
            control_height = 100
            available_width = max(320, window_width - 40)  # Padding
            available_height = max(240, window_height - control_height - 40)  # Padding
            
            # Maintain aspect ratio (4:3)
            aspect_ratio = 4.0 / 3.0
            
            if available_width / available_height > aspect_ratio:
                # Height is limiting factor
                display_height = available_height
                display_width = int(display_height * aspect_ratio)
            else:
                # Width is limiting factor
                display_width = available_width
                display_height = int(display_width / aspect_ratio)
            
            return display_width, display_height
        except Exception:
            return 640, 480
    return 640, 480

def resize_callback():
    """Handle window resize events."""
    try:
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        dpg.configure_item("main_window", width=viewport_width, height=viewport_height)
        
        # Update image display size
        display_width, display_height = get_image_display_size()
        dpg.configure_item("image_display", width=display_width, height=display_height)
    except Exception:
        pass

def render_loop():
    global cap, last_frame
    ret, frame = cap.read()
    if not ret:
        return
    
    frame = cv.flip(frame, 1)  # mirror
    out = stylize(frame, current_style)
    last_frame[0] = out
    draw_image("canvas", out)

def draw_image(tag, frame_bgr):
    # Always resize to fixed texture size
    frame_resized = cv.resize(frame_bgr, (TEXTURE_WIDTH, TEXTURE_HEIGHT))
    
    # DearPyGui expects RGBA format for textures
    rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
    # Convert to RGBA by adding alpha channel
    rgba = np.dstack((rgb, np.full((TEXTURE_HEIGHT, TEXTURE_WIDTH), 255, dtype=np.uint8)))
    # Normalize to 0-1 range and flatten
    data = (rgba / 255.0).astype(np.float32).flatten()
    
    # Simply update the texture data (size never changes)
    try:
        if dpg.does_item_exist("texture_id"):
            dpg.set_value("texture_id", data)
    except Exception as e:
        print(f"Texture update error: {e}")

def main():
    global cap
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize camera
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Create DearPyGUI context
    dpg.create_context()
    
    # Create a fixed-size texture with placeholder data
    default_data = np.ones((TEXTURE_HEIGHT, TEXTURE_WIDTH, 4), dtype=np.float32) * 0.5
    
    with dpg.texture_registry():
        dpg.add_dynamic_texture(width=TEXTURE_WIDTH, height=TEXTURE_HEIGHT, 
                               default_value=default_data.flatten(), tag="texture_id")
    
    # Create viewport (make it resizable)
    dpg.create_viewport(title="AI Booth", width=1300, height=850, resizable=True, 
                       min_width=800, min_height=600)
    
    # Create main window (make it auto-resize to viewport)
    with dpg.window(label="AI Booth", tag="main_window", no_title_bar=True, 
                   no_resize=True, no_move=True):
        dpg.add_text("Pick a style")
        dpg.add_combo(STYLE_NAMES, default_value=current_style, callback=change_style, tag="style_combo")
        dpg.add_button(label="Snapshot", callback=snapshot)
        dpg.add_separator()
        
        # Get initial display size
        display_width, display_height = get_image_display_size()
        
        # Add image display - this will scale the fixed-size texture
        dpg.add_image("texture_id", width=display_width, height=display_height, tag="image_display")
    
    # Set primary window
    dpg.set_primary_window("main_window", True)
    
    # Set up viewport resize callback
    dpg.set_viewport_resize_callback(resize_callback)
    
    # Setup and show viewport
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Initial resize to fit viewport
    resize_callback()
    
    # Render loop
    while dpg.is_dearpygui_running():
        render_loop()
        dpg.render_dearpygui_frame()
    
    # Cleanup
    cap.release()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
