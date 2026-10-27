# PowerShell script to build and run LibTorch Docker benchmark

docker build -t libtorch_benchmark .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed! Trying lightweight CPU Dockerfile..." -ForegroundColor Yellow
    docker build -f Dockerfile.cpu -t libtorch_benchmark .
}

$CHECKPOINTS_DIR = Join-Path (Split-Path $PSScriptRoot -Parent) "checkpoints"

docker run --rm -v "${CHECKPOINTS_DIR}:/app/checkpoints" libtorch_benchmark checkpoints/best_resnet34_traced.pt cpu 50 200
