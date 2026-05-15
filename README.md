
# YOLOv8 Triton Inference Service: Ensemble Pipeline & Zero-Downtime Deployment

## 🎯 Problem Statement
Deploying raw machine learning models into production environments presents several challenges: raw models cannot handle image byte decoding or JSON formatting, they struggle with optimized concurrent traffic, and updating them traditionally requires taking the server offline. 

This project solves these challenges by deploying a **YOLOv8 Object Detection Model** using the **NVIDIA Triton Inference Server**. It implements an end-to-end Ensemble Pipeline, utilizes Dynamic Batching for high throughput, and demonstrates Zero-Downtime model hot-swapping.

---

## 🏗️ Project Overview & Architecture
This deployment utilizes a **3-Step Triton Ensemble Scheduler**:
1. **Preprocess Node (`python` backend):** Normalizes input data to `[batch, 3, 640, 640]`.
2. **YOLO Node (`onnxruntime` backend):** The core YOLOv8 model performing optimized inference.
3. **Postprocess Node (`python` backend):** Formats the final bounding box output.

### Key MLOps Features Implemented:
* **Dynamic Batching:** Optimizes throughput by grouping concurrent requests.
* **Zero-Downtime Versioning:** Utilizes `MODE_POLL` to hot-swap versions 1 and 2 without service interruption.
* **Performance Benchmarking:** Tested via `perf_analyzer` at critical concurrency levels (1 and 32).

---

## 📂 Repository Structure
```text
.
├── docker-compose.yml             # Server orchestration configuration
├── README.md                      # Project documentation
├── results/
│   └── benchmark.csv              # Load testing results
├── scripts/
│   ├── run_benchmark.sh           # Automated perf_analyzer script
│   ├── test_ensemble.py           # Verification for the 3-step pipeline
│   └── test_versioning.sh         # Script for multi-version inference
└── model_repository/
    ├── ensemble_yolo/             # Ensemble router config
    ├── preprocess/                # Python preprocessing model
    ├── postprocess/               # Python postprocessing model
    └── yolo/                      # Core ONNX YOLOv8 models (Versions 1 & 2)

```

---

## 🚀 Execution Guide

### Step 1: Start the Triton Server

```bash
docker-compose up -d

```

### Step 2: Verify Server Health

```bash
curl -i http://localhost:8000/v2/health/ready

```

### Step 3: Run the Ensemble Pipeline

```bash
python scripts/test_ensemble.py

```

### Step 4: Test Zero-Downtime Versioning

```bash
./scripts/test_versioning.sh

```

### Step 5: Performance Benchmarking

This script now explicitly targets the mandatory concurrency level of 32:

```bash
./scripts/run_benchmark.sh

```

---

## 🧑‍💻 Author

**BILLAKURTI VENKATA SURYANARAYANA**

* B.Tech - Data Science