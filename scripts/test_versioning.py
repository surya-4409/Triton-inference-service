import urllib.request
import json

payload = {
    "inputs": [{
        "name": "images",
        "shape": [1, 3, 640, 640],
        "datatype": "FP32",
        "data": [0.0] * 1228800  # 3 * 640 * 640
    }]
}

data = json.dumps(payload).encode('utf-8')
print("Sending Inference Requests to Multiple Model Versions...")

for version in [1, 2]:
    url = f"http://localhost:8000/v2/models/yolo/versions/{version}/infer"
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        res = urllib.request.urlopen(req)
        print(f"✓ Version {version} Inference Endpoint: {res.getcode()} OK")
    except Exception as e:
        print(f"❌ Version {version} Failed: {e}")
