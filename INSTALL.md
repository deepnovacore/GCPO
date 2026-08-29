# Installation

GCPO is built on verl and uses vLLM for rollout generation. Two installation
paths are tested: a native Linux x86_64 / Python 3.12 environment based on the
public vLLM 0.12 release, and an NVIDIA NGC image following the environment
strategy used by [SDPO](https://github.com/lasgroup/SDPO). Keep their lock
files separate and install the repository with `--no-deps` so pip cannot
replace the selected GPU stack.

## Native Linux x86_64

Requirements:

- Linux x86_64 with NVIDIA GPUs and a driver compatible with CUDA 12.8;
- Conda; and
- outbound access to the Python package index and GitHub Releases during setup.

Create the environment and install the native lock in one resolution pass:

```bash
conda create -n gcpo python=3.12 -y
conda activate gcpo
python -m pip install -r requirements-native.txt
```

Install the official FlashAttention 2.8.1 wheel built for Python 3.12,
PyTorch 2.9, and CUDA 12. The checksum below is published by the GitHub
Release API and prevents a partial or altered download from being installed:

```bash
FLASH_ATTN_WHEEL=/tmp/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
curl -fL --retry 5 \
  -o "${FLASH_ATTN_WHEEL}" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
echo "88ea50d97200b1b0b74f100c5525ae8e1827aae5fd41b488f4fa7489725b4e56  ${FLASH_ATTN_WHEEL}" | sha256sum --check
python -m pip install --no-deps "${FLASH_ATTN_WHEEL}"
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
```

The native environment validated for this repository contains:

| Component | Version |
| --- | --- |
| Python | `3.12` |
| CUDA runtime packaged with PyTorch | `12.8` |
| PyTorch | `2.9.0` |
| vLLM | `0.12.0` |
| FlashAttention | `2.8.1` |
| FlashInfer | `0.5.3` |
| NVIDIA CUTLASS DSL | `4.3.4` |
| Numba / llvmlite | `0.61.2` / `0.44.0` |

The prebuilt FlashAttention filename is specific to Linux x86_64, Python
3.12, and PyTorch 2.9. For another Python version or architecture, use a
matching official release asset or use the Docker path rather than building
against a mismatched local CUDA toolkit.

Hugging Face libraries need a writable cache for model, dataset, lock, and
generated-module files. On a shared machine, set it explicitly before data
preparation or training:

```bash
export HF_HOME=/path/to/huggingface-cache
```

## Docker

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

`requirements-ngc.txt` is only for the NGC base image. Public vLLM 0.12 pins
Numba 0.61.2, whose compatible llvmlite version in the native lock is 0.44.0,
while the NGC/SDPO upper layer uses newer versions. Installing the NGC lock
into the native environment creates a real dependency conflict. Use
`requirements-native.txt` for native installation.

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
