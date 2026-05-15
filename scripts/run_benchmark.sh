#!/bin/bash
mkdir -p results
echo "Starting Performance Benchmarking..."
docker run --rm --net=host nvcr.io/nvidia/tritonserver:23.12-py3-sdk \
    perf_analyzer -m yolo -u localhost:8000 \
    --concurrency-range 1:32:10 \
    -f results/benchmark.csv
echo "✓ Benchmark complete!"
