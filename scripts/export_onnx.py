import onnx
from ultralytics import YOLO

def export_and_patch():
    print("1. Exporting base YOLOv8 model...")
    model = YOLO('yolov8s.pt')
    # Export using Ultralytics
    model.export(format='onnx', opset=12, dynamic=True)

    print("2. Patching ONNX graph for Triton compatibility...")
    # Load the exported graph into memory
    onnx_model = onnx.load("yolov8s.onnx")

    # FIX 1: Downgrade IR version to 9 for Triton 23.12
    onnx_model.ir_version = 9

    # FIX 2: Force the first dimension (batch size) of inputs and outputs to be dynamic
    for input_tensor in onnx_model.graph.input:
        input_tensor.type.tensor_type.shape.dim[0].dim_param = 'batch'
        
    for output_tensor in onnx_model.graph.output:
        output_tensor.type.tensor_type.shape.dim[0].dim_param = 'batch'

    # Save the patched model over the old one
    onnx.save(onnx_model, "yolov8s.onnx")
    print("✓ Model successfully patched: IR Version 9 + Dynamic Axes.")

if __name__ == "__main__":
    export_and_patch()