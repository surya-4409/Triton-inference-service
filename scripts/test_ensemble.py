import urllib.request
import json

payload = {
    "inputs": [{
        "name": "INPUT",
        "shape": [1, 100],
        "datatype": "FP32",
        "data": [0.5] * 100
    }]
}

print("Sending simulated image data to ensemble_yolo endpoint...")
req = urllib.request.Request("http://localhost:8000/v2/models/ensemble_yolo/infer")
req.add_header('Content-Type', 'application/json')
try:
    response = urllib.request.urlopen(req, json.dumps(payload).encode('utf-8'))
    print("✓ Ensemble Inference Successful!\nResponse JSON:")
    print(json.dumps(json.loads(response.read()), indent=2))
except Exception as e:
    print(f"❌ Failed: {e}")
