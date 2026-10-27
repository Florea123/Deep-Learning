#!/bin/bash
set -e


docker build -t libtorch_benchmark . || docker build -f Dockerfile.cpu -t libtorch_benchmark .

CHECKPOINTS_DIR="$(cd "$(dirname "$0")/../checkpoints" && pwd)"

echo ""
echo "Running C++ LibTorch Inference Benchmark in Docker..."
docker run --rm -v "${CHECKPOINTS_DIR}:/app/checkpoints" libtorch_benchmark checkpoints/best_resnet34_traced.pt cpu 50 200
