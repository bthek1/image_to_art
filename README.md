# AI Booth - Real-time Camera Stylization

A real-time camera application that applies artistic styles to webcam feeds using OpenCV and DearPyGUI.

## Features

- **Real-time stylization**: Apply artistic effects to your webcam feed in real-time
- **Multiple styles**: Choose from Cartoon, Pencil sketch, and Edge-preserving paint effects
- **Snapshot capture**: Save stylized images with timestamps
- **Responsive UI**: Resizable interface that adapts to different screen sizes
- **Modular architecture**: Clean, maintainable code structure

## Installation

### Requirements
- Python 3.7+
- Webcam/Camera device

### Dependencies
Install using pip or uv:
```bash
pip install opencv-python dearpygui
```

Or with uv:
```bash
uv add opencv-python dearpygui
```

## Usage

Run the application:
```bash
python main.py
```

### Controls
1. **Style Selection**: Use the dropdown menu to choose between Cartoon, Pencil, and EdgePaint styles
2. **Snapshot**: Click the "Snapshot" button to save the current stylized frame
3. **Window Resize**: The interface automatically adapts to window size changes

### Saved Images
Snapshots are saved in the `captures/` directory with filenames like:
```
booth_Cartoon_1756736589378.jpg
booth_Pencil_1756736612445.jpg
```

## Project Structure

The application follows a modular architecture for better maintainability and testing:

```
├── main.py                    # Entry point (uses new modular structure)
├── src/                       # Source code package
│   ├── __init__.py           # Package initialization
│   ├── app.py                # Main application coordinator
│   ├── camera.py             # Camera operations and frame capture
│   ├── image_processor.py    # Image stylization and processing
│   ├── ui.py                 # User interface (DearPyGUI)
│   ├── file_manager.py       # File operations and snapshot saving
│   └── config.py             # Configuration constants
├── captures/                 # Directory for saved snapshots
├── pyproject.toml           # Project configuration
├── uv.lock                  # Dependency lock file
└── README.md                # This file
```

### Architecture Components

#### `src/app.py` - Application Coordinator
- Main `AIBoothApp` class that orchestrates all components
- Handles the main application loop and coordinates between camera, image processing, UI, and file operations
- Manages application lifecycle and cleanup

#### `src/camera.py` - Camera Operations
- `Camera` class for webcam interaction
- Handles camera initialization, frame capture, and cleanup
- Includes context manager support for proper resource management

#### `src/image_processor.py` - Image Processing
- `ImageProcessor` class for applying artistic styles
- Supports Cartoon, Pencil sketch, and EdgePaint effects
- Handles image format conversion for UI display

#### `src/ui.py` - User Interface
- `UI` class managing the DearPyGUI interface
- Handles window resizing, user interactions, and image display
- Provides callback system for user actions

#### `src/file_manager.py` - File Operations
- `FileManager` class for saving and managing snapshots
- Automatic directory creation and timestamp-based naming
- Includes listing and management of saved files

#### `src/config.py` - Configuration
- Centralized configuration constants
- Style parameters, window dimensions, file paths
- Easy customization of application behavior

## Artistic Styles

### Cartoon Style
- Uses bilateral filtering for smooth color regions
- Applies adaptive threshold edge detection
- Combines smooth colors with sharp edges for a cartoon-like effect

### Pencil Style
- Converts image to pencil sketch using OpenCV's pencil sketch function
- Adjustable parameters for sketch intensity and shading

### EdgePaint Style
- Edge-preserving stylization that maintains important image structures
- Creates a paint-like artistic effect while preserving edges

## Configuration

Modify `src/config.py` to customize:
- Image processing parameters for each style
- Window dimensions and UI layout
- File paths and naming conventions
- Camera settings

## Development

### Code Quality Features
- **Type hints**: Full type annotation for better IDE support and documentation
- **Error handling**: Comprehensive exception handling throughout the application
- **Modular design**: Separated concerns for easier testing and maintenance
- **Context managers**: Proper resource management with context managers
- **Documentation**: Comprehensive docstrings following Python conventions

### Testing
The modular structure makes it easy to unit test individual components:
- Test camera operations independently
- Mock UI interactions for automated testing
- Test image processing algorithms with known inputs
- Verify file operations without camera dependency

## Troubleshooting

### Camera Issues
- Ensure your webcam is not being used by another application
- Try changing the camera index in `src/config.py` if you have multiple cameras
- Check camera permissions on your system

### Performance Issues
- Lower the resolution by modifying `TEXTURE_WIDTH` and `TEXTURE_HEIGHT` in `src/config.py`
- Adjust image processing parameters in the same file
- Close other applications that might be using system resources

### UI Issues
- If the window doesn't display correctly, try resizing it
- On high-DPI displays, the interface might appear small - this is a DearPyGUI limitation

## License

This project is open source. Feel free to modify and distribute according to your needs.