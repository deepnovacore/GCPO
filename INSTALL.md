# Installation

GCPO is built on verl and uses vLLM for rollout generation. A practical setup
requires Linux, Python 3.10-3.13, NVIDIA GPUs, and a CUDA-compatible PyTorch
build. Python 3.12 is the recommended default.

## Python environment

Create an isolated environment, install PyTorch for your CUDA driver, and then
install this repository:

```bash
conda create -n gcpo python=3.12 -y
conda activate gcpo

# Select the PyTorch command for your CUDA version from pytorch.org.
pip install torch

# Install GCPO with the rollout and math extras used by the README examples.
pip install -e ".[vllm,math]"
```

`vllm`, PyTorch, and CUDA must be mutually compatible. Choose the PyTorch wheel
for the target CUDA stack first, then install GCPO. If the default `pip install
torch` command does not match the target driver/toolkit, replace it with the
appropriate command from pytorch.org before installing this repository.

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
target driver and GPU architecture before building it. The root Dockerfile
currently targets CUDA 12.4.1, Python 3.12, and a matching PyTorch 2.5.1
install, so verify that this stack also matches the host driver and the vLLM
version you plan to use.
