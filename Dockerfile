# Multi-arch (amd64/arm64) GPU stack used by the SDPO reproducibility setup.
# It provides matched builds of CUDA, PyTorch, vLLM, FlashAttention, FlashInfer,
# xFormers, and Triton; do not reinstall those packages independently.
FROM nvcr.io/nvidia/vllm:25.12.post1-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_INDEX_URL=https://pypi.org/simple

# Some clusters export both CUDA and ROCm visibility variables. verl expects a
# single accelerator family, so keep the ROCm variable empty in this CUDA image.
ENV ROCR_VISIBLE_DEVICES="" \
    NCCL_NET=Socket

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only the upper-layer Python dependencies. The GPU stack comes from
# the NGC base image and is deliberately absent from this lock file.
COPY requirements-ngc.txt /app/requirements-ngc.txt
RUN python -m pip install --no-cache-dir \
    --index-url "${PIP_INDEX_URL}" \
    -r /app/requirements-ngc.txt

COPY . /app

# Match SDPO's installation order: preserve the prebuilt GPU stack and overlay
# this repository's verl/GCPO sources without dependency re-resolution.
RUN python -m pip install -e /app --no-deps --no-build-isolation \
    && python -m pip check

CMD ["/bin/bash"]
