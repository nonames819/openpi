#!/usr/bin/env bash
set -e

# 所有要跑的 LIBERO benchmark
TASK_SUITES=(
  libero_spatial
  libero_object
  libero_goal
  libero_10
)

export MUJOCO_GL=osmesa
export PYTHONPATH=$PWD/third_party/libero

for TASK in "${TASK_SUITES[@]}"; do
  echo "========================================"
  echo "Running LIBERO benchmark: $TASK"
  echo "========================================"

  python examples/libero/eval_with_log.py \
    --task-suite-name "$TASK"

  echo "Finished: $TASK"
  echo
done
