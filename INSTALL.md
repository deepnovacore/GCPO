# Installation

GCPO is built on verl and uses vLLM for rollout generation. A practical setup
requires Linux, Python 3.10+, NVIDIA GPUs, and a CUDA-compatible PyTorch build.

## Python environment

Create an isolated environment, install PyTorch for your CUDA driver, and then
install this repository:

```bash
conda create -n gcpo python=3.12 -y
conda activate gcpo

# Select the PyTorch command for your CUDA version from pytorch.org.
pip install torch
pip install -e ".[vllm,math]"
```

`vllm`, PyTorch, and CUDA must be mutually compatible. Install a different
PyTorch build first when the default package does not match the target CUDA
driver.

FlashAttention is optional and may be installed after PyTorch when supported:

```bash
pip install flash-attn --no-build-isolation
```

## Verify

```bash
pip install pytest
pytest -q tests/utils/test_gcpo.py
DRY_RUN=1 bash experiments/gcpo/train_gcpo.sh
```

For a containerized setup, adapt the CUDA base image in `Dockerfile` to the
target driver and GPU architecture before building it.
