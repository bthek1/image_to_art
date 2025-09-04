"""Image processing and stylization for the AI Booth application."""

import cv2 as cv
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path
from typing import Optional, Dict, Any
from .config import CARTOON_PARAMS, PENCIL_PARAMS, EDGE_PAINT_PARAMS


class ImageProcessor:
    """Handles image stylization and processing operations."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.supported_styles = ["Cartoon", "Pencil", "EdgePaint"]
        self.onnx_sessions: Dict[str, Any] = {}
        self.onnx_styles: Dict[str, str] = {}
        self.models_dir = Path("models")
        
        # Load available ONNX models
        self._load_onnx_models()
        
        # Show loaded models count
        if self.onnx_sessions:
            print(f"Loaded {len(self.onnx_sessions)} ONNX style transfer models")
    
    def _load_onnx_models(self) -> None:
        """Load available ONNX models from the models directory."""
        if not self.models_dir.exists():
            return
            
        onnx_files = list(self.models_dir.glob("*.onnx"))
        if not onnx_files:
            return
        
        for model_file in onnx_files:
            try:
                # Test if model can be loaded
                session = ort.InferenceSession(str(model_file))
                
                # Validate model inputs/outputs
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                
                if not inputs or not outputs:
                    continue
                
                # Create style name and register
                style_name = f"ONNX_{model_file.stem}"
                self.onnx_sessions[style_name] = session
                self.onnx_styles[style_name] = str(model_file)
                self.supported_styles.append(style_name)
                
            except Exception as e:
                print(f"Failed to load {model_file.name}: {e}")
                continue
    
    def get_onnx_model_info(self, style_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an ONNX model.
        
        Args:
            style_name: Name of the ONNX style
            
        Returns:
            Dictionary with model info or None if not found
        """
        if style_name not in self.onnx_sessions:
            return None
            
        session = self.onnx_sessions[style_name]
        input_info = []
        output_info = []
        
        for input_meta in session.get_inputs():
            input_info.append({
                'name': input_meta.name,
                'type': input_meta.type,
                'shape': input_meta.shape
            })
            
        for output_meta in session.get_outputs():
            output_info.append({
                'name': output_meta.name,
                'type': output_meta.type,
                'shape': output_meta.shape
            })
            
        return {
            'inputs': input_info,
            'outputs': output_info,
            'model_path': self.onnx_styles[style_name]
        }
    
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
            elif style.startswith("ONNX_"):
                return self._apply_onnx_style(frame, style)
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
    
    def _apply_onnx_style(self, frame: np.ndarray, style: str) -> Optional[np.ndarray]:
        """Apply ONNX model style transfer.
        
        Args:
            frame: Input image (BGR format)
            style: ONNX style name
            
        Returns:
            Stylized image as numpy array, or None if processing failed
        """
        if style not in self.onnx_sessions:
            return frame
            
        try:
            session = self.onnx_sessions[style]
            
            # Get input details
            input_meta = session.get_inputs()[0]
            input_name = input_meta.name
            input_shape = input_meta.shape
            
            # Prepare input image
            processed_frame = self._preprocess_for_onnx(frame, input_shape)
            
            # Run inference
            outputs = session.run(None, {input_name: processed_frame})
            
            # Post-process output
            result = self._postprocess_from_onnx(outputs[0], frame.shape)
            
            return result
            
        except Exception as e:
            print(f"Error applying ONNX style {style}: {e}")
            return frame
    
    def _preprocess_for_onnx(self, frame: np.ndarray, input_shape: tuple) -> np.ndarray:
        """Preprocess image for ONNX model input.
        
        Args:
            frame: Input BGR image
            input_shape: Expected input shape from model metadata
            
        Returns:
            Preprocessed image array
        """
        try:
            # Analyze the input shape to determine the expected format
            if len(input_shape) == 4:
                batch_dim, dim1, dim2, dim3 = input_shape
                
                # Determine if this is NCHW or NHWC format
                # NHWC: channels should be 3 and be the last dimension
                # NCHW: channels should be 3 and be the second dimension
                
                if dim3 == 3:  # NHWC format (batch, height, width, channels)
                    target_format = "NHWC"
                    # Use default size for dynamic dimensions
                    target_height = int(dim1) if isinstance(dim1, int) and dim1 > 0 else 256
                    target_width = int(dim2) if isinstance(dim2, int) and dim2 > 0 else 256
                elif dim1 == 3:  # NCHW format (batch, channels, height, width)
                    target_format = "NCHW"
                    target_height = int(dim2) if isinstance(dim2, int) and dim2 > 0 else 256
                    target_width = int(dim3) if isinstance(dim3, int) and dim3 > 0 else 256
                else:
                    # Unknown format, assume NHWC with default size
                    target_format = "NHWC"
                    target_height = target_width = 256
            else:
                # Fallback for unexpected input shapes
                target_format = "NHWC"
                target_height = target_width = 256
            
            # Convert BGR to RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Resize to target dimensions
            resized = cv.resize(rgb_frame, (target_width, target_height))
            
            # Normalize to [0, 1] range
            normalized = resized.astype(np.float32) / 255.0
            
            # Format according to expected input layout
            if target_format == "NCHW":
                # Transpose HWC to CHW and add batch dimension
                processed = np.transpose(normalized, (2, 0, 1))
                processed = np.expand_dims(processed, 0)
            else:  # NHWC
                # Just add batch dimension (keep HWC format)
                processed = np.expand_dims(normalized, 0)
            
            return processed
            
        except Exception:
            # Fallback: simple NHWC preprocessing
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            resized = cv.resize(rgb_frame, (256, 256))
            normalized = resized.astype(np.float32) / 255.0
            return np.expand_dims(normalized, 0)
    
    def _postprocess_from_onnx(self, output: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Post-process ONNX model output.
        
        Args:
            output: Model output array
            original_shape: Original image shape (height, width, channels)
            
        Returns:
            Post-processed BGR image
        """
        try:
            # Remove batch dimension if present
            if len(output.shape) == 4 and output.shape[0] == 1:
                output = output[0]
            
            # Handle different output formats
            if len(output.shape) == 3:
                # Check if it's CHW (channels first) or HWC (channels last)
                if output.shape[0] == 3 and output.shape[2] != 3:
                    # CHW format - transpose to HWC
                    output = np.transpose(output, (1, 2, 0))
                # If shape[2] == 3, it's already in HWC format
            
            # Handle different value ranges
            if output.max() <= 1.0 and output.min() >= -1.0:
                # Values in [-1, 1] or [0, 1] range
                if output.min() < 0:
                    # Convert from [-1, 1] to [0, 1]
                    output = (output + 1.0) / 2.0
                # Values are now in [0, 1] range
                output = np.clip(output, 0, 1)
                output = (output * 255).astype(np.uint8)
            elif output.max() > 1.0:
                # Values might be in [0, 255] range already
                output = np.clip(output, 0, 255).astype(np.uint8)
            else:
                # Fallback: normalize to [0, 255]
                output = ((output - output.min()) / (output.max() - output.min()) * 255).astype(np.uint8)
            
            # Resize to original dimensions
            original_height, original_width = original_shape[:2]
            if output.shape[:2] != (original_height, original_width):
                output = cv.resize(output, (original_width, original_height))
            
            # Convert RGB back to BGR for OpenCV compatibility
            if len(output.shape) == 3 and output.shape[2] == 3:
                bgr_output = cv.cvtColor(output, cv.COLOR_RGB2BGR)
            else:
                # If not 3-channel, return as-is
                bgr_output = output
            
            return bgr_output
            
        except Exception as e:
            print(f"Error in ONNX post-processing: {e}")
            # Return a placeholder in case of error
            return np.zeros(original_shape, dtype=np.uint8)
    
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
