# Installation

GCPO is built on verl and uses vLLM for rollout generation. The recommended
installation is the repository Docker image, which follows the environment
strategy used by [SDPO](https://github.com/lasgroup/SDPO): start from a
prebuilt NVIDIA vLLM stack, add an exact upper-layer dependency lock, and
install the repository with `--no-deps` so pip cannot replace the GPU stack.

## Recommended: Docker

Requirements:

- Linux with NVIDIA GPUs;
- Docker with the NVIDIA Container Toolkit; and
- a host driver compatible with CUDA 13.1.

Build the image from the repository root:

```bash
docker build -t gcpo:latest .
```

The base image is `nvcr.io/nvidia/vllm:25.12.post1-py3` and publishes both
`linux/amd64` and `linux/arm64` variants. The environment tested for this
repository contains:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| CUDA runtime | 13.1 |
| PyTorch | `2.10.0a0+b4e4ee81d3.nv25.12` |
| vLLM | `0.12.0+35a9f223.nv25.12.post1` |
| FlashAttention | `2.7.4.post1` |
| NVIDIA CUTLASS DSL | `4.3.4` |

Run an interactive container with writable caches and output directories:

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/GCPO:/app \
  -v /path/to/huggingface-cache:/root/.cache/huggingface \
  -v /path/to/datasets:/app/datasets \
  -v /path/to/artifacts:/app/artifacts \
  -v /path/to/checkpoints:/app/checkpoints \
  -w /app \
  gcpo:latest \
  /bin/bash
```

`requirements-ngc.txt` follows SDPO's NGC upper-layer lock. PyTorch, vLLM,
FlashAttention, FlashInfer, xFormers, Triton, and CUDA Python bindings are
provided by the base image and intentionally absent from that file; compatible
CUTLASS DSL and other upper-layer packages remain pinned in the lock. Do not
upgrade or reinstall the base-image packages independently inside the
container.

If access to the default Python package index is slow, the Dockerfile accepts
an alternate PEP 503-compatible index for the upper-layer packages:

```bash
docker build \
  --build-arg PIP_INDEX_URL=https://your-python-index.example/simple \
  -t gcpo:latest .
```

## Advanced: native Python environment

The package metadata supports Python 3.10-3.13, with Python 3.12 recommended.
A native installation is appropriate only when the machine already has a
mutually compatible CUDA, PyTorch, vLLM, FlashAttention, FlashInfer, CUTLASS,
xFormers, and Triton stack. Installing or upgrading those packages separately
can produce an importable environment that still fails when vLLM loads a model
or launches a CUDA kernel.

After provisioning a compatible GPU stack, install the same upper layer and
overlay GCPO without dependency re-resolution:

```bash
conda create -n gcpo python=3.12 -y
conda activate gcpo

# Provision the mutually compatible GPU stack before these commands.
python -m pip install -r requirements-ngc.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
```

`requirements-ngc.txt` is validated against the NGC version matrix above. If
the native GPU stack differs, derive and test a complete lock for that stack
instead of mixing individual wheels into this one. The optional dependency
bounds in `setup.py` prevent the known vLLM/FlashAttention incompatibility, but
they are not a substitute for an end-to-end environment lock.

## Verify

Inside the container or activated native environment, run:

```bash
python -m pip check
python -c "import torch, vllm, flash_attn; print(torch.__version__, vllm.__version__, flash_attn.__version__)"
python -m pip install pytest
pytest -q tests/utils/test_gcpo.py
DRY_RUN=1 bash experiments/gcpo/train_gcpo.sh
```

Before a full run, follow the README to prepare the dataset and GCPO basis. A
dry run validates launcher configuration only; it does not load model weights
or execute GPU kernels.
