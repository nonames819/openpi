#!/bin/bash

set -euo pipefail

# =========================
# 参数解析
# =========================
CHECKPOINT_DIR=${1:-}
GPU_ID=${2:-0}
START_PORT=${3:-8000}
TASK_SUITE=${4:-all}
POLICY_CONFIG=${5:-pi05_libero}

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
SESSION_NAME="libero_${GPU_ID}_${START_PORT}"

# 兼容单 suite 或全部 suite
TARGET_SUITES=()
if [[ "$TASK_SUITE" == "all" ]]; then
    TARGET_SUITES=("${SUITES[@]}")
else
    FOUND=0
    for suite in "${SUITES[@]}"; do
        if [[ "$suite" == "$TASK_SUITE" ]]; then
            TARGET_SUITES=("$TASK_SUITE")
            FOUND=1
            break
        fi
    done
    if [[ "$FOUND" -eq 0 ]]; then
        echo "Invalid TASK_SUITE: $TASK_SUITE"
        echo "Supported: all ${SUITES[*]}"
        exit 1
    fi
fi

# serve_policy 额外参数：checkpoint 分支
SERVE_POLICY_EXTRA_ARGS=()
if [[ -n "$CHECKPOINT_DIR" ]]; then
    SERVE_POLICY_EXTRA_ARGS+=("policy:checkpoint")
    SERVE_POLICY_EXTRA_ARGS+=("--policy.config=$POLICY_CONFIG")
    SERVE_POLICY_EXTRA_ARGS+=("--policy.dir=$CHECKPOINT_DIR")
fi

echo "Using base GPU_ID: $GPU_ID"
echo "Using START_PORT: $START_PORT"
echo "Target suites: ${TARGET_SUITES[*]}"
echo "Tmux Session: $SESSION_NAME"
if [[ -n "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint mode: enabled"
    echo "Policy config: $POLICY_CONFIG"
    echo "Policy dir: $CHECKPOINT_DIR"
else
    echo "Checkpoint mode: disabled (use default policy for --env LIBERO)"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists."
    echo "Please kill it first: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

for i in "${!TARGET_SUITES[@]}"; do
    PORT=$((START_PORT + i))
    GPU=$((GPU_ID + i))
    SUITE="${TARGET_SUITES[$i]}"
    WINDOW_NAME="$SUITE"

    SERVE_POLICY_ARGS=(--env LIBERO --port "$PORT" "${SERVE_POLICY_EXTRA_ARGS[@]}")
    printf -v SERVE_POLICY_ARGS_STR '%q ' "${SERVE_POLICY_ARGS[@]}"

    if [[ "$i" -eq 0 ]]; then
        tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME"
    else
        tmux new-window -t "$SESSION_NAME" -n "$WINDOW_NAME"
    fi

    # 左侧 pane：serve policy
    tmux send-keys -t "${SESSION_NAME}:${i}.0" "
CUDA_VISIBLE_DEVICES=$GPU \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
uv run scripts/serve_policy.py \
    $SERVE_POLICY_ARGS_STR
" C-m

    # 右侧 pane：eval
    tmux split-window -h -t "${SESSION_NAME}:${i}"
    tmux send-keys -t "${SESSION_NAME}:${i}.1" "
source examples/libero/.venv/bin/activate
CUDA_VISIBLE_DEVICES=$GPU \
MUJOCO_GL=osmesa \
PYTHONPATH=\$PWD/third_party/libero \
python examples/libero/eval_with_log.py \
    --port $PORT \
    --task_suite_name $SUITE
" C-m

    tmux select-layout -t "${SESSION_NAME}:${i}" even-horizontal
done

tmux select-window -t "${SESSION_NAME}:0"
echo "All windows started in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
