#!/bin/bash

GPU_ID=4
START_PORT=8000
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for i in "${!SUITES[@]}"; do
    PORT=$((START_PORT + i))
    GPU=$((GPU_ID + i))
    TASK_SUITE=${SUITES[$i]}
    echo "Starting tmux session for $TASK_SUITE on GPU $GPU, port $PORT"
    bash bash_scripts/eval_libero_tmux.sh $GPU $PORT $TASK_SUITE
done