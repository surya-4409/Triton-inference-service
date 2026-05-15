import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "raw_image")
            in_numpy = in_tensor.as_numpy()
            batch_size = in_numpy.shape[0]
            # Simulate resizing image to YOLO requirements [batch, 3, 640, 640]
            processed_numpy = np.zeros((batch_size, 3, 640, 640), dtype=np.float32)
            out_tensor = pb_utils.Tensor("processed_image", processed_numpy)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
