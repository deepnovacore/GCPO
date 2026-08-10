#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
# export RAY_DEBUG=1
ulimit -c 0

# Optional: export WANDB_ENTITY=your-team before running. If unset, wandb uses the API key's default workspace.
export EXPERIMENT=${1:-"experiment"}
CONFIG_NAME=${2:-"ppo_trainer"}
export TASK=${3:-"datasets/math500"}

# removes the first three arguments from the command line
if [ "$#" -ge 3 ]; then
    shift 3
else
    echo "Usage: $0 <experiment_name> <config_name> <data_path>"
    echo "Example: $0 test ppo_trainer datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"
    exit 1
fi

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG_NAME"
echo "Task: $TASK"
echo "Arguments: $@"

python -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" "$@"
