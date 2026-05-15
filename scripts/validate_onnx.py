import sys
import torch
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO

def validate_models():
    print("Loading PyTorch and ONNX models for validation...")
    
    # 1. Load both models
    pt_model = YOLO('yolov8s.pt')
    onnx_session = ort.InferenceSession('yolov8s.onnx')

    # 2. Create a dummy input tensor matching the YOLOv8 input signature
    # Shape: (Batch=1, Channels=3, Height=640, Width=640)
    dummy_input = torch.randn(1, 3, 640, 640)

    # 3. PyTorch Inference
    # We use pt_model.model to bypass Ultralytics pre/post processing wrappers 
    # and get the raw output tensor to match the ONNX graph directly.
    with torch.no_grad():
        pt_output = pt_model.model(dummy_input)[0].detach().numpy()

    # 4. ONNX Inference
    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name
    
    onnx_output = onnx_session.run(
        [onnx_output_name], 
        {onnx_input_name: dummy_input.numpy()}
    )[0]

    # 5. Compare outputs using the strictly required tolerance
    # atol=1e-4 accounts for standard floating-point drift during ONNX conversion
    is_close = np.allclose(pt_output, onnx_output, atol=1e-4)

    if is_close:
        # 6. Print success message and exit 0 (Required for Automated Evaluation)
        print("✓ ONNX model validation successful. Outputs are numerically equivalent.")
        sys.exit(0)
    else:
        print("❌ Validation failed: ONNX model output does not match PyTorch model output!")
        sys.exit(1)

if __name__ == "__main__":
    validate_models()