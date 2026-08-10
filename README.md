# GCPO

GCPO (Geometry-Constrained Policy Optimization) is an efficient and stable
post-training method for constraining policy updates away from a frozen base
model's principal parameter subspaces. This repository implements GCPO on top
of [verl](https://github.com/volcengine/verl) and supports multiple
training objectives.

## Installation

GCPO requires Linux, Python 3.10+, PyTorch, and CUDA GPUs for practical LLM
training. Install a PyTorch build suitable for your CUDA driver, then run:

```bash
git clone <repository-url>
cd PriorPost
pip install -e ".[vllm,math]"
```

The training path uses vLLM by default. PyTorch, vLLM, and CUDA must be mutually
compatible; see `INSTALL.md` for additional environment notes.

## Data

Place a task under `datasets/<task>/` with `train.json` and `test.json`, then
convert it to the parquet format consumed by verl:

```bash
python data/preprocess.py --data_source datasets/<task>
```

The resulting directory must contain `train.parquet` and `test.parquet`.
Datasets, checkpoints, generated bases, and credentials are intentionally not
tracked by Git.

## Run GCPO

### 1. Precompute the principal-subspace basis

The model and `TARGET_MODULES` must match the subsequent training command.

```bash
MODEL_PATH=Qwen/Qwen3-8B \
GCPO_TOPK=16 \
TARGET_MODULES=all-linear \
bash experiments/gcpo/prepare_basis.sh
```

By default the basis is written to
`artifacts/gcpo_basis/<model>/k16/basis.pt`. Run this step on a GPU allocation;
set `DEVICE=cpu` only for small models.

### 2. Train with GRPO + GCPO

```bash
MODEL_PATH=Qwen/Qwen3-8B \
TASK=datasets/math500 \
CONFIG_NAME=baseline_grpo \
N_GPUS_PER_NODE=4 \
bash experiments/gcpo/train_gcpo.sh
```

To use GCPO with SDPO instead:

```bash
MODEL_PATH=Qwen/Qwen3-8B \
TASK=datasets/math500 \
CONFIG_NAME=sdpo \
SDPO_ALPHA=0.5 \
N_GPUS_PER_NODE=4 \
bash experiments/gcpo/train_gcpo.sh
```

For Slurm, adapt the resource directives in
`experiments/gcpo/submit_gcpo.sbatch`, then export the same variables before
calling `sbatch`.

### Main overrides

| Variable | Default | Meaning |
| --- | --- | --- |
| `GCPO_TOPK` | `16` | Number of protected singular directions |
| `LORA_RANK` | `32` | Trainable LoRA rank |
| `LORA_ALPHA` | `16` | LoRA scaling parameter |
| `TRAIN_BATCH_SIZE` | `32` | Prompt batch size |
| `ROLLOUT_N` | `8` | Samples generated per prompt |
| `LR` | `1e-5` | Actor learning rate |
| `TOTAL_TRAINING_STEPS` | `300` | Number of policy updates |
| `OUTPUT_DIR` | repository checkpoint directory | Checkpoint destination |

W&B is optional. The launcher logs only to the console unless
`WANDB_API_KEY` is already present in the environment. No credentials or local
machine paths are read from repository files.

Set `DRY_RUN=1` on the training command to print the fully resolved launch
command without checking data/model artifacts or starting Ray.

## Implementation

- `verl/utils/gcpo.py`: constrained LoRA parameterization and rollout export.
- `scripts/prepare_gcpo_basis.py`: frozen-model SVD basis preparation.
- `verl/workers/fsdp_workers.py`: actor and rollout integration.
- `experiments/gcpo/`: portable preparation and training launchers.

## Acknowledgements

This codebase builds on verl and the SDPO implementation. Their original
licenses and notices are preserved in the repository.
