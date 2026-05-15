
# YOLOv8 Triton Inference Service: Ensemble Pipeline & Zero-Downtime Deployment

## 🎯 Problem Statement
Deploying raw machine learning models into production environments presents several challenges: raw models cannot handle image byte decoding or JSON formatting, they struggle with optimized concurrent traffic, and updating them traditionally requires taking the server offline. 

This project solves these challenges by deploying a **YOLOv8 Object Detection Model** using the **NVIDIA Triton Inference Server**. It implements an end-to-end Ensemble Pipeline (encapsulating pre-processing and post-processing), utilizes Dynamic Batching for high throughput, and demonstrates Zero-Downtime model hot-swapping.

---

## 🏗️ Project Overview & Architecture

This deployment is built around a **3-Step Triton Ensemble Scheduler**, which allows multiple models to execute in a single gRPC/HTTP request without the data ever leaving the server's memory.

1. **Preprocess Node (`python` backend):** Intercepts raw input tensors, extracts the batch size, and simulates the normalization and resizing of image data to the `[batch, 3, 640, 640]` format required by YOLO.
2. **YOLOv8 Node (`onnxruntime` backend):** The core computer vision model. Exported to ONNX (IR Version 9) with explicit dynamic batching enabled on the 0th axis to maximize GPU/CPU utilization.
3. **Postprocess Node (`python` backend):** Intercepts the raw `[batch, 84, 8400]` output tensor, simulates Non-Maximum Suppression (NMS), and formats the final bounding box data.

### Key MLOps Features Implemented:
* **Dynamic Batching:** Triton automatically groups incoming requests within a specified time window to maximize compute efficiency.
* **Zero-Downtime Versioning:** Configured with `MODE_POLL`, allowing new model versions (e.g., `/yolo/2/`) to be dropped into the repository and loaded instantly without dropping live traffic.
* **Performance Benchmarking:** Rigorous load testing conducted via `perf_analyzer`, capturing latency and throughput metrics from concurrency levels 1 through 32.

---

## 📂 Repository Structure

```text
.
├── docker-compose.yml             # Server orchestration configuration
├── README.md                      # Project documentation
├── results/
│   └── benchmark.csv              # Load testing artifacts (Concurrency 1 to 32)
├── scripts/
│   ├── export_onnx.py             # Script to export YOLOv8 with dynamic axes & IR9
│   ├── run_benchmark.sh           # Automated perf_analyzer benchmarking script
│   ├── test_ensemble.py           # Verification script for the 3-step pipeline
│   └── test_versioning.sh         # Script to test concurrent multi-version inference
└── model_repository/
    ├── ensemble_yolo/             # Ensemble router and scheduler config
    ├── preprocess/                # Python preprocessing model & config
    ├── postprocess/               # Python postprocessing model & config
    └── yolo/                      # ONNX YOLOv8 models (Versions 1 & 2)

```

---

## 🚀 Execution Guide (From Scratch)

Follow these steps to spin up the server, run the benchmarks, and test the pipeline.

### Step 1: Start the Triton Server

Ensure Docker and Docker Compose are installed. Launch the server in detached mode:

```bash
docker-compose up -d

```

You can monitor the model loading process (wait until all models show `READY`):

```bash
docker-compose logs -f triton

```

### Step 2: Verify Server Health

Run a quick sanity check to ensure the HTTP endpoint is alive and accepting traffic:

```bash
curl -i http://localhost:8000/v2/health/ready

```

*Expected Output:* `HTTP/1.1 200 OK`

### Step 3: Run the Ensemble Pipeline

Send a simulated image payload through the full `preprocess -> yolo -> postprocess` pipeline:

```bash
python scripts/test_ensemble.py

```

*Expected Output:* A JSON response containing the formatted `[1.0, 0.5, 0.5, 100.0, 100.0]` bounding box array.

### Step 4: Test Zero-Downtime Versioning

Verify that Triton is successfully routing traffic to multiple live versions of the YOLO model simultaneously:

```bash
./scripts/test_versioning.sh

```

*Expected Output:*

```text
✓ Version 1 Inference Endpoint: 200 OK
✓ Version 2 Inference Endpoint: 200 OK

```

### Step 5: Performance Benchmarking

The `results/benchmark.csv` file has already been generated using Triton's `perf_analyzer`. If you need to regenerate it, run:

```bash
./scripts/run_benchmark.sh

```

*Note: This script tests multi-level concurrency including the baseline (1) and max stress (32) loads.*

---

## 🧑‍💻 Author

**BILLAKURTI VENKATA SURYANARAYANA**
* B.Tech --Data Science*

