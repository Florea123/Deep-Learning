
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV CMAKE_PREFIX_PATH=/opt/conda/lib/python3.11/site-packages/torch

COPY CMakeLists.txt /app/
COPY main.cpp /app/

RUN cmake -B build -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc)

VOLUME ["/app/checkpoints"]

ENTRYPOINT ["./build/libtorch_benchmark"]
CMD ["checkpoints/best_resnet34_traced.pt", "cpu", "50", "200"]
