# ONNX Style Transfer Models

This directory contains ONNX models for neural style transfer. Place your `.onnx` model files here, and they will automatically be loaded when the application starts.

## How to Add ONNX Models

1. **Download or convert your style transfer model to ONNX format**
2. **Place the `.onnx` file in this `models/` directory**
3. **Restart the application**
4. **The model will appear as `ONNX_[filename]` in the style dropdown**

## Model Requirements

- **Input format**: RGB images (typically 3 channels)
- **Input shape**: Common shapes are `[1, 3, 256, 256]` or `[1, 3, 224, 224]`
- **Output format**: RGB images with same dimensions as input
- **Value range**: Input and output should be normalized to [0, 1] range

## Popular Style Transfer Models

You can find pre-trained ONNX models from:

### 1. ONNX Model Zoo
- Fast Neural Style Transfer models
- Various artistic styles available
- URL: https://github.com/onnx/models/tree/main/vision/style_transfer

### 2. PyTorch Hub
- Convert PyTorch models to ONNX using `torch.onnx.export()`

### 3. TensorFlow Hub
- Convert TensorFlow models to ONNX using `tf2onnx`

## Example Models to Try

1. **Fast Neural Style Transfer**
   - Candy, Mosaic, Rain Princess, Udnie styles
   - Input: `[1, 3, 224, 224]`
   - Real-time performance

2. **AdaIN Style Transfer**
   - Adaptive Instance Normalization
   - Can transfer style from any reference image

## Model Conversion Example

If you have a PyTorch model, convert it to ONNX:

```python
import torch
import torch.onnx

# Load your PyTorch model
model = YourStyleTransferModel()
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 256, 256)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "your_style_model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

## Testing Your Model

1. Place your `.onnx` file in this directory
2. Run the application
3. Check the console for loading messages
4. Select your model from the style dropdown
5. Test with live camera feed

## Troubleshooting

- **Model not appearing**: Check console for error messages
- **Poor quality results**: Verify input/output preprocessing
- **Slow performance**: Consider using smaller input sizes or optimized models
- **Memory errors**: Reduce batch size or image resolution

## Performance Tips

- Use models with input size 256x256 or smaller for real-time performance
- GPU acceleration available if ONNX Runtime GPU is installed
- Consider quantized models for better performance on CPU
