import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "yolo_output")
            batch_size = in_tensor.as_numpy().shape[0]
            # Simulate NMS (Non-Maximum Suppression) filtering
            clean_numpy = np.tile(np.array([1.0, 0.5, 0.5, 100.0, 100.0], dtype=np.float32), (batch_size, 1))
            out_tensor = pb_utils.Tensor("final_output", clean_numpy)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
