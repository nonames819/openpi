#!/bin/bash

# =========================
# 参数解析
# =========================
GPU_ID=$1
PORT=$2
TASK_SUITE=$3

# 默认值
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-8000}
TASK_SUITE=${TASK_SUITE:-libero_10}

SESSION_NAME="libero_${GPU_ID}_${PORT}"

echo "Using GPU: $GPU_ID"
echo "Using PORT: $PORT"
echo "Task Suite: $TASK_SUITE"
echo "Tmux Session: $SESSION_NAME"

# =========================
# 创建 tmux session
# =========================
tmux new-session -d -s $SESSION_NAME

# =========================
# 左侧：启动 policy server
# =========================
tmux send-keys -t $SESSION_NAME "
CUDA_VISIBLE_DEVICES=$GPU_ID \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
uv run scripts/serve_policy.py \
    --env LIBERO \
    --port $PORT
" C-m

# =========================
# 右侧：split panel
# =========================
tmux split-window -h -t $SESSION_NAME

# =========================
# 右侧：启动 eval
# =========================
tmux send-keys -t $SESSION_NAME "
source examples/libero/.venv/bin/activate
CUDA_VISIBLE_DEVICES=$GPU_ID \
MUJOCO_GL=osmesa \
PYTHONPATH=\$PWD/third_party/libero \
python examples/libero/eval_with_log.py \
    --port $PORT \
    --task_suite_name $TASK_SUITE
" C-m

# =========================
# 自动调整布局
# =========================
tmux select-layout -t $SESSION_NAME tiled

# =========================
# attach session
# =========================
tmux attach -t $SESSION_NAME