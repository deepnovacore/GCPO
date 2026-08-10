#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
TASK="${TASK:-datasets/math500}"
CONFIG_NAME="${CONFIG_NAME:-baseline_grpo}"
GCPO_TOPK="${GCPO_TOPK:-16}"
GCPO_BASIS_PATH="${GCPO_BASIS_PATH:-${PROJECT_ROOT}/artifacts/gcpo_basis/${MODEL_TAG}/k${GCPO_TOPK}/basis.pt}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
ROLLOUT_N="${ROLLOUT_N:-8}"
VAL_N="${VAL_N:-16}"
LR="${LR:-1e-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-5}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
SDPO_ALPHA="${SDPO_ALPHA:-0.5}"
WANDB_PROJECT="${WANDB_PROJECT:-GCPO}"
DRY_RUN="${DRY_RUN:-0}"

case "${CONFIG_NAME}" in
  baseline_grpo) ALGORITHM_TAG="grpo" ;;
  sdpo) ALGORITHM_TAG="sdpo" ;;
  *)
    echo "CONFIG_NAME must be baseline_grpo or sdpo, got: ${CONFIG_NAME}" >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -f "${GCPO_BASIS_PATH}" ]]; then
    echo "Missing GCPO basis: ${GCPO_BASIS_PATH}" >&2
    echo "Run experiments/gcpo/prepare_basis.sh first." >&2
    exit 2
  fi
  if [[ ! -f "${PROJECT_ROOT}/${TASK}/train.parquet" || ! -f "${PROJECT_ROOT}/${TASK}/test.parquet" ]]; then
    echo "Expected train.parquet and test.parquet under ${PROJECT_ROOT}/${TASK}." >&2
    exit 2
  fi
fi

TASK_TAG="$(basename "${TASK}")"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gcpo-${ALGORITHM_TAG}-${TASK_TAG}-${MODEL_TAG}-r${LORA_RANK}-k${GCPO_TOPK}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/${TASK_TAG}/${EXPERIMENT_NAME}}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER_CONFIG='["console","wandb"]'
else
  LOGGER_CONFIG='["console"]'
fi

ARGS=(
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.project_name=${WANDB_PROJECT}"
  "trainer.group_name=GCPO-${ALGORITHM_TAG}"
  "trainer.logger=${LOGGER_CONFIG}"
  "trainer.default_local_dir=${OUTPUT_DIR}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.max_actor_ckpt_to_keep=1"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
  "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
  "actor_rollout_ref.model.target_modules=${TARGET_MODULES}"
  "actor_rollout_ref.model.gcpo_enabled=True"
  "actor_rollout_ref.model.gcpo_basis_path=${GCPO_BASIS_PATH}"
  "actor_rollout_ref.model.gcpo_topk=${GCPO_TOPK}"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.rollout.load_format=safetensors"
  "actor_rollout_ref.rollout.layered_summon=True"
)

if [[ "${CONFIG_NAME}" == "sdpo" ]]; then
  ARGS+=(
    "actor_rollout_ref.actor.self_distillation.distillation_topk=100"
    "actor_rollout_ref.actor.self_distillation.alpha=${SDPO_ALPHA}"
    "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
    "algorithm.rollout_correction.rollout_is=token"
  )
fi

export TASK
echo "GCPO configuration: algorithm=${ALGORITHM_TAG} model=${MODEL_PATH} task=${TASK} rank=${LORA_RANK} topk=${GCPO_TOPK}"
echo "Basis: ${GCPO_BASIS_PATH}"
echo "Output: ${OUTPUT_DIR}"

COMMAND=(
  bash "${PROJECT_ROOT}/training/verl_training.sh"
  "${EXPERIMENT_NAME}"
  "${CONFIG_NAME}"
  "${TASK}"
  "${ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

"${COMMAND[@]}"
